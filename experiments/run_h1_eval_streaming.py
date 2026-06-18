"""Streaming-extractor version of ``run_h1_eval.py``.

Same algorithm and configuration as the original, but uses
``StreamingExtractor.extract_graph()`` to build the sentence graphs without
materializing the full ``[L, H, N, N]`` attention tensor. This unblocks
LongBench-scale contexts and *also* uses the corrected sentence-span
coordinates (see ``experiments/longbench/results/streaming_refactor_findings.md``).

Outputs
-------
- ``experiments/results_streaming/h1_eval_<timestamp>.jsonl``
- ``experiments/results_streaming/h1_eval_latest.jsonl``
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
ROOT = os.path.dirname(THIS_DIR)
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
from spectral.laplacian import induced_subgraph  # noqa: E402


# -- configuration (matches the original run_h1_eval.py exactly) --
SUBSET_PATH = os.path.join(THIS_DIR, "data", "squad_v2_h1_subset.jsonl")
RESULTS_DIR = os.path.join(THIS_DIR, "results_streaming")
COMPRESSION_RATE = 0.5
POOLING = "mean"
MAX_SEQ_LEN = 1024


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _load_examples() -> List[Dict[str, Any]]:
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _h1_for_example(
    example: Dict[str, Any],
    extractor: StreamingExtractor,
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
            "aggregation": "mean",
            "pooling": POOLING,
            "sparsify": "none",
            "extractor": "streaming",
        },
    }

    # ----- 1. Build G on full context --------------------------------------
    t0 = time.time()
    full = extractor.extract_graph(ctx, q, context_type="english", pooling=POOLING)
    W_full = full.W
    t_extract_full = time.time() - t0
    record["K_full"] = int(W_full.shape[0])
    record["n_tokens_full"] = int(full.n_tokens)

    # ----- 2. Run Sentinel -------------------------------------------------
    t0 = time.time()
    res = compressor.compress(
        context=ctx, question=q, target_token=-1,
        compression_rate=COMPRESSION_RATE, context_type="english",
    )
    t_compress = time.time() - t0
    kept = sorted(int(i) for i in res["preserved_indices"])
    record["kept_indices"] = kept
    record["n_kept"] = len(kept)
    record["compression_ratio_tokens"] = res["compression_ratio"]
    record["compressed_text"] = res["compressed_text"]

    if res["sentences"] != full.sentences:
        record["status"] = "skip_segmentation_drift"
        return record

    if len(kept) < 3:
        record["status"] = "skip_too_few_kept"
        return record

    # ----- 3. Induced subgraph --------------------------------------------
    W_induced = induced_subgraph(W_full, kept)

    # ----- 4. Re-extract on C' ---------------------------------------------
    t0 = time.time()
    hat = extractor.extract_graph(res["compressed_text"], q, context_type="english", pooling=POOLING)
    W_hat = hat.W
    t_extract_hat = time.time() - t0
    record["K_hat"] = int(W_hat.shape[0])
    record["n_tokens_hat"] = int(hat.n_tokens)
    record["sentences_kept"] = [full.sentences[i] for i in kept]
    record["sentences_hat"] = hat.sentences

    # ----- 5. Align if needed ----------------------------------------------
    aligned_full = False
    if W_hat.shape == W_induced.shape:
        induced_sents = [full.sentences[i] for i in kept]
        perm = align_sentences_by_text(induced_sents, hat.sentences)
        if perm is not None and perm != list(range(len(perm))):
            inv = np.argsort(perm)
            W_hat = W_hat[np.ix_(inv, inv)]
            record["alignment"] = "reordered"
        else:
            record["alignment"] = "identity_or_failed"
        aligned_full = True
    else:
        induced_sents = [full.sentences[i] for i in kept]
        perm = align_sentences_by_text(induced_sents, hat.sentences)
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
        except Exception as exc:
            record["status"] = "metrics_error"
            record["error"] = repr(exc)
    else:
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
    print(f"Proxy : {proxy}")
    print(f"Extractor: StreamingExtractor (memory-efficient, corrected coords)\n")

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
    out_path = os.path.join(RESULTS_DIR, f"h1_eval_{timestamp}.jsonl")
    latest_path = os.path.join(RESULTS_DIR, "h1_eval_latest.jsonl")

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
            except Exception as exc:
                rec = {
                    "id": ex["id"], "status": "error",
                    "error": repr(exc), "traceback": traceback.format_exc(),
                }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            tag = rec.get("status", "?")
            n_kept = rec.get("n_kept", "-")
            if tag == "ok" or tag.startswith("ok_"):
                n_ok += 1
            elif tag.startswith("skip"):
                n_skip += 1
            else:
                n_err += 1
            print(f"  [{i:>2}/{len(examples)}] {tag:<28s} n_kept={n_kept}  ({ex['id']})")

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
