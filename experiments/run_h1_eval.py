"""Run the H1 metric panel across the sampled SQuAD v2 subset.

For each example:
1. Build the full sentence graph G from the source context.
2. Run Sentinel compression to get V'.
3. Form G[V'] by slicing G.
4. Re-extract attention on the compressed text C' to build G_hat.
5. Try to align sentences (greedy text match) so the graphs share a coordinate system.
6. Record the full metric panel + metadata.

Outputs
-------
- ``experiments/results/h1_eval_<timestamp>.jsonl`` -- one row per example.
- ``experiments/results/h1_eval_latest.jsonl`` -- a symlink-style copy.
- A summary on stdout. ``summarize_h1.py`` does the proper reduction.
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
ROOT = os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from attention_compressor import AttentionCompressor  # noqa: E402
from spectral.attention_extraction import SpectralExtractor  # noqa: E402
from spectral.graph_builder import build_sentence_graph  # noqa: E402
from spectral.h1_test import (  # noqa: E402
    align_sentences_by_text,
    run_subgraph_hypothesis,
    spectral_entropy,
    spectrum_distance,
)
from spectral.laplacian import induced_subgraph  # noqa: E402


# -- configuration --
SUBSET_PATH = os.path.join(THIS_DIR, "data", "squad_v2_h1_subset.jsonl")
RESULTS_DIR = os.path.join(THIS_DIR, "results")
COMPRESSION_RATE = 0.5
AGGREGATION = "mean"
POOLING = "mean"
SPARSIFY = "none"   # tiny graphs (K=6-20), so no sparsification
MAX_SEQ_LEN = 1024


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _load_examples() -> List[Dict[str, Any]]:
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _h1_for_example(
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
        },
    }

    # ----- 1. Build G on full context --------------------------------------
    t0 = time.time()
    ex_full = extractor.extract(ctx, q, context_type="english")
    W_full, _ = build_sentence_graph(
        ex_full.attn,
        ex_full.sentence_spans,
        aggregation=AGGREGATION,
        pooling=POOLING,
        sparsify_method=SPARSIFY,
        final_token_idx=ex_full.query_token_idx,
    )
    t_extract_full = time.time() - t0
    record["K_full"] = W_full.shape[0]
    record["n_tokens_full"] = ex_full.n_tokens

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
    kept = sorted(res["preserved_indices"])
    record["kept_indices"] = kept
    record["n_kept"] = len(kept)
    record["compression_ratio_tokens"] = res["compression_ratio"]
    record["compressed_text"] = res["compressed_text"]

    # Compressor and extractor must agree on segmentation of source context.
    if res["sentences"] != ex_full.sentences:
        record["status"] = "skip_segmentation_drift"
        return record

    if len(kept) < 3:
        # Most metrics degenerate; skip.
        record["status"] = "skip_too_few_kept"
        return record

    # ----- 3. Induced subgraph --------------------------------------------
    W_induced = induced_subgraph(W_full, kept)

    # ----- 4. Re-extract on C' ---------------------------------------------
    t0 = time.time()
    ex_hat = extractor.extract(res["compressed_text"], q, context_type="english")
    W_hat, _ = build_sentence_graph(
        ex_hat.attn,
        ex_hat.sentence_spans,
        aggregation=AGGREGATION,
        pooling=POOLING,
        sparsify_method=SPARSIFY,
        final_token_idx=ex_hat.query_token_idx,
    )
    t_extract_hat = time.time() - t0
    record["K_hat"] = W_hat.shape[0]
    record["n_tokens_hat"] = ex_hat.n_tokens
    record["sentences_kept"] = [ex_full.sentences[i] for i in kept]
    record["sentences_hat"] = ex_hat.sentences

    # ----- 5. Align if needed ----------------------------------------------
    aligned_full = False
    if W_hat.shape == W_induced.shape:
        # Always try greedy text match in case the order differs.
        induced_sents = [ex_full.sentences[i] for i in kept]
        perm = align_sentences_by_text(induced_sents, ex_hat.sentences)
        if perm is not None and perm != list(range(len(perm))):
            inv = np.argsort(perm)
            W_hat = W_hat[np.ix_(inv, inv)]
            record["alignment"] = "reordered"
        else:
            record["alignment"] = "identity_or_failed"
        aligned_full = True
    else:
        induced_sents = [ex_full.sentences[i] for i in kept]
        perm = align_sentences_by_text(induced_sents, ex_hat.sentences)
        if perm is not None:
            inv = np.argsort(perm)
            W_hat = W_hat[np.ix_(inv, inv)]
            aligned_full = True
            record["alignment"] = "aligned_after_resegment"
        else:
            record["alignment"] = "failed_cardinality_mismatch"

    # ----- 6. Compute metrics ----------------------------------------------
    if aligned_full and W_hat.shape == W_induced.shape:
        k_eig = max(1, min(2, W_induced.shape[0] - 1))
        try:
            metrics = run_subgraph_hypothesis(
                W_induced, W_hat,
                k_eig=k_eig,
                k_edges=min(5, W_induced.shape[0] * (W_induced.shape[0] - 1) // 2),
            )
            record["metrics"] = metrics
            record["status"] = "ok"
        except Exception as exc:  # pragma: no cover - safety net
            record["status"] = "metrics_error"
            record["error"] = repr(exc)
    else:
        # Size-agnostic only.
        record["metrics_size_agnostic"] = {
            "spectral_l2": spectrum_distance(W_induced, W_hat, metric="l2"),
            "spectral_wass": spectrum_distance(W_induced, W_hat, metric="wasserstein"),
            "spectral_entropy_diff": abs(
                spectral_entropy(W_induced) - spectral_entropy(W_hat)
            ),
        }
        record["status"] = "ok_size_agnostic"

    record["timings_s"] = {
        "extract_full": t_extract_full,
        "compress": t_compress,
        "extract_hat": t_extract_hat,
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
    out_path = os.path.join(RESULTS_DIR, f"h1_eval_{timestamp}.jsonl")
    latest_path = os.path.join(RESULTS_DIR, "h1_eval_latest.jsonl")

    records: List[Dict[str, Any]] = []
    n_ok = n_skip = n_err = 0
    t_total = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples, 1):
            try:
                rec = _h1_for_example(ex, extractor, compressor)
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
            records.append(rec)

            tag = rec.get("status", "?")
            n_kept = rec.get("n_kept", "-")
            if tag == "ok":
                n_ok += 1
            elif tag.startswith("ok_"):
                n_ok += 1
            elif tag.startswith("skip"):
                n_skip += 1
            else:
                n_err += 1
            print(f"  [{i:>2}/{len(examples)}] {tag:<28s} n_kept={n_kept}  ({ex['id']})")

            # Aggressive cleanup between examples - we OOM at the upper end of
            # context lengths if we let allocator fragmentation accumulate.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.1f}s. ok={n_ok}  skip={n_skip}  err={n_err}")
    print(f"Wrote: {out_path}")

    # latest pointer (plain copy on Windows; symlinks need admin rights)
    with open(latest_path, "w", encoding="utf-8") as out, open(out_path, "r", encoding="utf-8") as src:
        out.write(src.read())
    print(f"Updated: {latest_path}")


if __name__ == "__main__":
    main()
