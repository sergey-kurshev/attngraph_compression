"""Run the H2 (optimal-cut hypothesis) panel across the H1 SQuAD v2 subset.

For each example:
1. Extract attention on the full source context and build the sentence graph G.
2. Run Sentinel compression to obtain V'_Sentinel.
3. At budget B = |V'_Sentinel|, run all H2 baselines (exact / spectral /
   query-anchored spectral / local search / top-attention / random).
4. Persist the full metric panel to JSONL with per-example details for later
   inspection.

We reuse the same 50-example SQuAD subset as H1 (``experiments/data/squad_v2_h1_subset.jsonl``)
for direct comparability with the H1 results in ``experiments/results/``.

Outputs
-------
- ``experiments/h2/results/h2_eval_<timestamp>.jsonl`` — one row per example.
- ``experiments/h2/results/h2_eval_latest.jsonl`` — plain copy pointer.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR = os.path.dirname(THIS_DIR)
ROOT = os.path.dirname(EXPS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from attention_compressor import AttentionCompressor  # noqa: E402
from spectral.attention_extraction import SpectralExtractor  # noqa: E402
from spectral.graph_builder import build_sentence_graph  # noqa: E402
from spectral.h2_test import run_optimal_cut_hypothesis  # noqa: E402


# -- configuration: keep aligned with H1 for apples-to-apples comparison --
SUBSET_PATH = os.path.join(EXPS_DIR, "data", "squad_v2_h1_subset.jsonl")
RESULTS_DIR = os.path.join(THIS_DIR, "results")
COMPRESSION_RATE = 0.5
AGGREGATION = "mean"
POOLING = "mean"
SPARSIFY = "none"   # tiny graphs (K=6-20)
MAX_SEQ_LEN = 1024
ALPHA = 1.0          # NCut_q query-anchor weight
EXACT_MAX_SUBSETS = 500_000  # K=20, C(20,10)=184_756 — comfortably under
N_RANDOM_SEEDS = 20


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _load_examples() -> List[Dict[str, Any]]:
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _h2_for_example(
    example: Dict[str, Any],
    extractor: SpectralExtractor,
    compressor: AttentionCompressor,
) -> Dict[str, Any]:
    ctx = example["context"]
    q = example["question"]

    record: Dict[str, Any] = {
        "id": example["id"],
        "n_sentences_source": example["n_sentences"],
        "n_chars": example["n_chars"],
        "config": {
            "compression_rate": COMPRESSION_RATE,
            "aggregation": AGGREGATION,
            "pooling": POOLING,
            "sparsify": SPARSIFY,
            "alpha": ALPHA,
        },
    }

    # ----- 1. Build G on full context --------------------------------------
    t0 = time.time()
    ex_full = extractor.extract(ctx, q, context_type="english")
    W_full, q_sent = build_sentence_graph(
        ex_full.attn,
        ex_full.sentence_spans,
        aggregation=AGGREGATION,
        pooling=POOLING,
        sparsify_method=SPARSIFY,
        final_token_idx=ex_full.query_token_idx,
    )
    t_extract = time.time() - t0
    record["K_full"] = int(W_full.shape[0])
    record["n_tokens_full"] = int(ex_full.n_tokens)

    # ----- 2. Run Sentinel -------------------------------------------------
    t0 = time.time()
    res = compressor.compress(
        context=ctx,
        question=q,
        target_token=-1,
        compression_rate=COMPRESSION_RATE,
        context_type="english",
    )
    t_compress = time.time() - t0
    kept = sorted(int(i) for i in res["preserved_indices"])
    record["kept_indices"] = kept
    record["n_kept"] = len(kept)
    record["compression_ratio_tokens"] = res["compression_ratio"]

    # Sentinel and the extractor must agree on segmentation, else the indices
    # don't refer to the same sentences as the graph.
    if res["sentences"] != ex_full.sentences:
        record["status"] = "skip_segmentation_drift"
        return record

    if len(kept) < 2 or len(kept) >= W_full.shape[0]:
        record["status"] = "skip_degenerate_budget"
        return record

    # ----- 3. Run the H2 panel --------------------------------------------
    t0 = time.time()
    h2 = run_optimal_cut_hypothesis(
        W=W_full,
        a_q=q_sent,
        V_sentinel=kept,
        alpha=ALPHA,
        n_random_seeds=N_RANDOM_SEEDS,
        exact_max_subsets=EXACT_MAX_SUBSETS,
        random_seed=int(np.uint32(hash(example["id"]) & 0xFFFFFFFF)),
    )
    t_h2 = time.time() - t0
    record.update(h2)
    record["status"] = "ok"
    record["timings_s"] = {
        "extract": t_extract,
        "compress": t_compress,
        "h2_panel": t_h2,
    }
    return record


def main():
    examples = _load_examples()
    print(f"Loaded {len(examples)} examples from {SUBSET_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy = _proxy_path()
    print(f"Device: {device}")
    print(f"Proxy : {proxy}\n")

    extractor = SpectralExtractor(
        attention_model_path=proxy,
        eval_tokenizer_path=proxy,
        max_seq_len=MAX_SEQ_LEN,
        device=device,
        print_sentence_scores=False,
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
    out_path = os.path.join(RESULTS_DIR, f"h2_eval_{timestamp}.jsonl")
    latest_path = os.path.join(RESULTS_DIR, "h2_eval_latest.jsonl")

    n_ok = n_skip = n_err = 0
    t_total = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples, 1):
            try:
                rec = _h2_for_example(ex, extractor, compressor)
            except torch.cuda.OutOfMemoryError as exc:
                rec = {"id": ex["id"], "status": "oom", "error": str(exc)}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as exc:  # pragma: no cover
                rec = {
                    "id": ex["id"],
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            tag = rec.get("status", "?")
            n_kept = rec.get("n_kept", "-")
            if tag == "ok":
                n_ok += 1
            elif tag.startswith("skip"):
                n_skip += 1
            else:
                n_err += 1
            gap = rec.get("normalized_gap")
            gap_s = f"gap={gap:.3f}" if isinstance(gap, float) else "gap=-"
            print(f"  [{i:>2}/{len(examples)}] {tag:<28s} n_kept={n_kept}  {gap_s}  ({ex['id']})")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.1f}s. ok={n_ok}  skip={n_skip}  err={n_err}")
    print(f"Wrote: {out_path}")

    with open(latest_path, "w", encoding="utf-8") as out, open(out_path, "r", encoding="utf-8") as src:
        out.write(src.read())
    print(f"Updated: {latest_path}")


if __name__ == "__main__":
    main()
