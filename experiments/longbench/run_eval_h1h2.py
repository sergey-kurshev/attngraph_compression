"""H1 + H2 evaluation on the LongBench-En subset, at three compression ratios.

For each (example, ratio):
1. Extract the source sentence graph G once per example (streaming extractor).
2. Run Sentinel at this ratio → V'.
3. Re-extract G_hat on the compressed text C'.
4. Align sentences (greedy text match) and compute the H1 metric panel.
5. Compute the H2 metric panel against G at budget B = |V'|.
6. Persist one record per (example, ratio).

Memory note: the streaming extractor caps total prompt length at
``MAX_SEQ_LEN`` (10000 here). Examples sampled by ``select_examples.py``
are filtered to ≤ 10000 reported tokens, but the prompt template adds a
few tens of tokens so very-long examples may hit the cap. Records at the
cap are valid but their N reflects the truncation.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR = os.path.dirname(THIS_DIR)
ROOT = os.path.dirname(EXPS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from attention_compressor import AttentionCompressor  # noqa: E402
from spectral.streaming_extractor import StreamingExtractor  # noqa: E402
from spectral.h1_test import (  # noqa: E402
    align_sentences_by_text,
    run_subgraph_hypothesis,
    spectral_entropy,
    spectrum_distance,
)
from spectral.h2_test import run_optimal_cut_hypothesis  # noqa: E402
from spectral.laplacian import induced_subgraph  # noqa: E402


# -- configuration --
SUBSET_PATH = os.path.join(THIS_DIR, "data_subset", "longbench_en_subset.jsonl")
RESULTS_DIR = os.path.join(THIS_DIR, "results_h1h2")
COMPRESSION_RATES = [0.5, 0.67, 0.8]   # 2x, 3x, 5x per plan §7.1
# NOTE: AttentionCompressor's compression_rate is the *removed* fraction —
# actual kept = 1 - compression_rate. So input 0.5/0.67/0.8 -> keep 50/33/20%
# of the context which corresponds to {2x, 3x, 5x} compression.
POOLING = "mean"
# Lowered from 10000 to 8000 after the first run died from sustained-OOM
# state degradation. Per-layer attention at N=8000 peaks at 3.6 GB instead
# of 5.6 GB, well clear of contention with the loaded extractor+compressor.
MAX_SEQ_LEN = 8000
ALPHA = 1.0
EXACT_MAX_SUBSETS = 500_000  # H2 exact will skip for large K, which is fine
N_RANDOM_SEEDS = 20
# local_search at K=200 with default max_iter=200 would take ~5 min per call.
# At LongBench scale we cap aggressively — even one well-chosen swap from
# Sentinel is informative about how local-minimum-y its selection is.
LOCAL_MAX_ITER = 10


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _load_examples() -> List[Dict[str, Any]]:
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _h1_metrics_one(
    W_full: np.ndarray,
    kept: List[int],
    sentences_full: List[str],
    extractor: StreamingExtractor,
    compressed_text: str,
    question: str,
) -> Dict[str, Any]:
    """H1 metric panel for a single (example, ratio)."""
    out: Dict[str, Any] = {}
    W_induced = induced_subgraph(W_full, kept)

    # Re-extract on compressed text
    try:
        hat = extractor.extract_graph(compressed_text, question, context_type="english", pooling=POOLING)
    except torch.cuda.OutOfMemoryError as e:
        out["status_h1"] = "h1_oom_on_chat"
        out["h1_error"] = "OOM: " + str(e).splitlines()[0][:200]
        return out

    W_hat = hat.W
    sentences_hat = hat.sentences
    out["K_hat"] = int(W_hat.shape[0])
    out["n_tokens_hat"] = int(hat.n_tokens)
    out["sentences_kept"] = [sentences_full[i] for i in kept]
    out["sentences_hat"] = sentences_hat

    # Align sentences
    induced_sents = [sentences_full[i] for i in kept]
    aligned = False
    if W_hat.shape == W_induced.shape:
        perm = align_sentences_by_text(induced_sents, sentences_hat)
        if perm is not None and perm != list(range(len(perm))):
            inv = np.argsort(perm)
            W_hat = W_hat[np.ix_(inv, inv)]
            out["alignment"] = "reordered"
        else:
            out["alignment"] = "identity_or_failed"
        aligned = True
    else:
        perm = align_sentences_by_text(induced_sents, sentences_hat)
        if perm is not None:
            inv = np.argsort(perm)
            W_hat = W_hat[np.ix_(inv, inv)]
            aligned = True
            out["alignment"] = "aligned_after_resegment"
        else:
            out["alignment"] = "failed_cardinality_mismatch"

    if aligned and W_hat.shape == W_induced.shape:
        k_eig = max(1, min(2, W_induced.shape[0] - 1))
        try:
            metrics = run_subgraph_hypothesis(
                W_induced, W_hat,
                k_eig=k_eig,
                k_edges=min(5, W_induced.shape[0] * (W_induced.shape[0] - 1) // 2),
            )
            out["h1_metrics"] = metrics
            out["status_h1"] = "ok"
        except Exception as exc:
            out["status_h1"] = "h1_metrics_error"
            out["h1_error"] = repr(exc)
    else:
        out["h1_metrics_size_agnostic"] = {
            "spectral_l2": spectrum_distance(W_induced, W_hat, metric="l2"),
            "spectral_wass": spectrum_distance(W_induced, W_hat, metric="wasserstein"),
            "spectral_entropy_diff": abs(
                spectral_entropy(W_induced) - spectral_entropy(W_hat)
            ),
        }
        out["status_h1"] = "ok_size_agnostic"
    return out


def _h2_metrics_one(
    W_full: np.ndarray,
    q_sent: np.ndarray,
    kept: List[int],
    seed: int,
) -> Dict[str, Any]:
    """H2 metric panel for a single (example, ratio)."""
    try:
        h2 = run_optimal_cut_hypothesis(
            W=W_full, a_q=q_sent, V_sentinel=kept,
            alpha=ALPHA, n_random_seeds=N_RANDOM_SEEDS,
            exact_max_subsets=EXACT_MAX_SUBSETS,
            random_seed=seed,
            local_max_iter=LOCAL_MAX_ITER,
        )
        h2["status_h2"] = "ok"
        return h2
    except Exception as exc:
        return {"status_h2": "h2_error", "h2_error": repr(exc)}


def _eval_one(
    example: Dict[str, Any],
    extractor: StreamingExtractor,
    compressor: AttentionCompressor,
    rates: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Build one record per compression ratio. Reuses the full-context graph.

    ``rates`` defaults to ``COMPRESSION_RATES``; pass a subset to skip ratios
    that are already covered by a resume-source.
    """
    if rates is None:
        rates = COMPRESSION_RATES
    ctx = example["context"]
    q = example["question"]
    records: List[Dict[str, Any]] = []

    # ----- 1. Build G on full context (ONCE per example) ------------------
    t0 = time.time()
    try:
        full = extractor.extract_graph(ctx, q, context_type="english", pooling=POOLING)
    except torch.cuda.OutOfMemoryError as e:
        for rate in COMPRESSION_RATES:
            records.append({
                "id": example["id"], "task": example["task"], "rate": rate,
                "status": "oom_on_full", "error": "OOM: " + str(e).splitlines()[0][:200],
            })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return records

    W_full = full.W
    q_sent = full.q_sent
    K_full = int(W_full.shape[0])
    t_extract_full = time.time() - t0

    if K_full < 3:
        for rate in rates:
            records.append({
                "id": example["id"], "task": example["task"], "rate": rate,
                "K_full": K_full, "status": "skip_too_few_sentences",
            })
        return records

    seed = int(np.uint32(hash(example["id"]) & 0xFFFFFFFF))

    # Release the streaming extractor's per-example CUDA cache before the
    # compressor's heavy attention forward — otherwise the allocator can't
    # find the ~3-5 GB contiguous block the compressor's per-layer Q@K^T
    # needs, even when total allocations look fine.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----- 2. Per-rate: Sentinel → V' → H1 + H2 ---------------------------
    for rate in rates:
        rec: Dict[str, Any] = {
            "id": example["id"], "task": example["task"], "rate": rate,
            "n_chars": example["n_chars"], "length_tokens": example["length"],
            "K_full": K_full, "n_tokens_full": int(full.n_tokens),
            "extract_full_s": t_extract_full,
            "config": {
                "pooling": POOLING, "alpha": ALPHA,
                "max_seq_len": MAX_SEQ_LEN, "extractor": "streaming",
            },
        }
        try:
            t0 = time.time()
            res = compressor.compress(
                context=ctx, question=q, target_token=-1,
                compression_rate=rate, context_type="english",
            )
            rec["compress_s"] = time.time() - t0
        except torch.cuda.OutOfMemoryError as e:
            rec["status"] = "oom_on_compress"
            rec["error"] = "OOM: " + str(e).splitlines()[0][:200]
            records.append(rec)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            continue

        kept = sorted(int(i) for i in res["preserved_indices"])
        rec["kept_indices"] = kept
        rec["n_kept"] = len(kept)
        rec["compression_ratio_tokens"] = res["compression_ratio"]
        # Persist the compressed text for downstream QA — drives the reader run.
        rec["compressed_text"] = res["compressed_text"]

        if res["sentences"] != full.sentences:
            rec["status"] = "skip_segmentation_drift"
            records.append(rec)
            continue
        if len(kept) < 3 or len(kept) >= K_full:
            rec["status"] = "skip_degenerate_budget"
            records.append(rec)
            continue

        # ----- 3a. H1 ------------------------------------------------------
        t0 = time.time()
        h1 = _h1_metrics_one(W_full, kept, full.sentences,
                              extractor, res["compressed_text"], q)
        rec["h1_time_s"] = time.time() - t0
        rec.update(h1)

        # ----- 3b. H2 ------------------------------------------------------
        t0 = time.time()
        h2 = _h2_metrics_one(W_full, q_sent, kept, seed=seed)
        rec["h2_time_s"] = time.time() - t0
        rec.update(h2)

        # Overall record status is "ok" if both panels succeeded.
        rec["status"] = "ok" if (rec.get("status_h1", "").startswith("ok") and
                                  rec.get("status_h2") == "ok") else "partial"
        records.append(rec)

    return records


def _load_resume_keys(path: str) -> set:
    """Return the set of (id, rate) tuples already covered by a prior run.

    Tuple-level granularity (rather than per-example) lets us change
    COMPRESSION_RATES mid-experiment and still carry forward usable records
    for any rates that overlap with the new sweep.
    """
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ex_id = r.get("id")
            rate = r.get("rate")
            if ex_id is None or rate is None:
                continue
            done.add((ex_id, float(rate)))
    return done


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-from", default=None,
                    help="JSONL of a previous run; examples found there are skipped.")
    args = ap.parse_args()

    examples = _load_examples()
    print(f"Loaded {len(examples)} examples from {SUBSET_PATH}")
    print(f"Tasks       : {sorted(set(e['task'] for e in examples))}")
    print(f"Ratios      : {COMPRESSION_RATES}")
    print(f"max_seq_len : {MAX_SEQ_LEN}")

    done_keys: set = set()
    carried_records: List[Dict[str, Any]] = []
    target_rate_set = set(float(r) for r in COMPRESSION_RATES)
    if args.resume_from:
        done_keys_all = _load_resume_keys(args.resume_from)
        # Only carry forward records at rates we're still sweeping over —
        # records at superseded rates (e.g. the old 0.33/0.2 inputs) are dropped
        # so the new file is consistent with COMPRESSION_RATES.
        done_keys = {(eid, r) for (eid, r) in done_keys_all if r in target_rate_set}
        print(f"Resume      : {len(done_keys_all)} (id,rate) tuples in source; "
              f"{len(done_keys)} match the current rate set {sorted(target_rate_set)}")

        with open(args.resume_from, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("id"), float(r.get("rate", -1)))
                if key in done_keys:
                    carried_records.append(r)
        print(f"Resume      : carried forward {len(carried_records)} previous records")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy = _proxy_path()
    print(f"Device      : {device}")
    print(f"Proxy       : {proxy}\n")

    extractor = StreamingExtractor(
        attention_model_path=proxy,
        eval_tokenizer_path=proxy,
        max_seq_len=MAX_SEQ_LEN,
        device=device,
    )
    compressor = AttentionCompressor(
        attention_model_path=proxy,
        detector_path=None,
        use_raw_attention=True,
        use_last_layer_only=False,
        use_all_queries=False,
        max_seq_len=MAX_SEQ_LEN,
        device=device,
        print_sentence_scores=False,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"longbench_h1h2_{timestamp}.jsonl")
    latest_path = os.path.join(RESULTS_DIR, "longbench_h1h2_latest.jsonl")

    total_records_target = len(examples) * len(COMPRESSION_RATES)
    n_ok = n_skip = n_partial = n_err = 0
    t_total = time.time()
    # For each example, determine which rates still need to be run.
    todo: List[Tuple[Dict[str, Any], List[float]]] = []
    for ex in examples:
        missing = [r for r in COMPRESSION_RATES if (ex["id"], r) not in done_keys]
        if missing:
            todo.append((ex, missing))
    n_fully_done = len(examples) - len(todo)
    print(f"To run      : {len(todo)} examples have missing rates "
          f"({n_fully_done} fully covered by resume source)\n")

    with open(out_path, "w", encoding="utf-8") as f:
        # First, replay any carried-forward records so the new file is canonical.
        for rec in carried_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tag = rec.get("status", "?")
            if tag == "ok":
                n_ok += 1
            elif tag.startswith("skip"):
                n_skip += 1
            elif tag.startswith("partial"):
                n_partial += 1
            else:
                n_err += 1
        f.flush()
        if carried_records:
            print(f"  Carried forward {len(carried_records)} records "
                  f"(ok={n_ok} skip={n_skip} partial={n_partial} err={n_err})\n")

        for i, (ex, missing_rates) in enumerate(todo, 1):
            try:
                recs = _eval_one(ex, extractor, compressor, rates=missing_rates)
            except torch.cuda.OutOfMemoryError as exc:
                recs = [{
                    "id": ex["id"], "task": ex["task"], "rate": rate,
                    "status": "oom_outer", "error": str(exc)[:200],
                } for rate in missing_rates]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as exc:
                recs = [{
                    "id": ex["id"], "task": ex["task"], "rate": rate,
                    "status": "error", "error": repr(exc),
                    "traceback": traceback.format_exc(),
                } for rate in missing_rates]

            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                tag = rec.get("status", "?")
                if tag == "ok":
                    n_ok += 1
                elif tag.startswith("skip"):
                    n_skip += 1
                elif tag.startswith("partial"):
                    n_partial += 1
                else:
                    n_err += 1

            n_kept_str = "/".join(str(r.get("n_kept", "-")) for r in recs)
            statuses = ",".join(r.get("status", "?")[:14] for r in recs)
            K_full = recs[0].get("K_full", "-")
            elapsed = time.time() - t_total
            eta = elapsed / max(i, 1) * (len(todo) - i)
            print(f"  [{i:>3}/{len(todo)}] {ex['task']:<18s} "
                  f"K={K_full:<4} kept={n_kept_str:<14s} "
                  f"status={statuses:<48s}  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
                  flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed/60:.1f}m.  records={n_ok+n_skip+n_partial+n_err}/{total_records_target}  "
          f"ok={n_ok}  partial={n_partial}  skip={n_skip}  err={n_err}")
    print(f"Wrote: {out_path}")

    with open(latest_path, "w", encoding="utf-8") as out, open(out_path, "r", encoding="utf-8") as src:
        out.write(src.read())
    print(f"Updated: {latest_path}")


if __name__ == "__main__":
    main()
