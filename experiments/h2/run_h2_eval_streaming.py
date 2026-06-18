"""Streaming-extractor version of ``run_h2_eval.py``.

Same algorithm and configuration as the original, but uses
``StreamingExtractor.extract_graph()`` to build the source-context sentence
graph without materializing the full ``[L, H, N, N]`` attention tensor.
This unblocks LongBench-scale contexts and uses the corrected sentence-span
coordinates (see ``experiments/longbench/results/streaming_refactor_findings.md``).

Outputs
-------
- ``experiments/h2/results_streaming/h2_eval_<timestamp>.jsonl``
- ``experiments/h2/results_streaming/h2_eval_latest.jsonl``
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
from spectral.streaming_extractor import StreamingExtractor  # noqa: E402
from spectral.h2_test import run_optimal_cut_hypothesis  # noqa: E402


# -- configuration: aligned with run_h2_eval.py exactly --
SUBSET_PATH = os.path.join(EXPS_DIR, "data", "squad_v2_h1_subset.jsonl")
RESULTS_DIR = os.path.join(THIS_DIR, "results_streaming")
COMPRESSION_RATE = 0.5
POOLING = "mean"
MAX_SEQ_LEN = 1024
ALPHA = 1.0
EXACT_MAX_SUBSETS = 500_000
N_RANDOM_SEEDS = 20


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _load_examples() -> List[Dict[str, Any]]:
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _h2_for_example(
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
            "alpha": ALPHA,
            "extractor": "streaming",
        },
    }

    # ----- 1. Build G on full context --------------------------------------
    t0 = time.time()
    full = extractor.extract_graph(ctx, q, context_type="english", pooling=POOLING)
    t_extract = time.time() - t0
    W_full = full.W
    q_sent = full.q_sent
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

    if res["sentences"] != full.sentences:
        record["status"] = "skip_segmentation_drift"
        return record

    if len(kept) < 2 or len(kept) >= W_full.shape[0]:
        record["status"] = "skip_degenerate_budget"
        return record

    # ----- 3. Run the H2 panel --------------------------------------------
    t0 = time.time()
    h2 = run_optimal_cut_hypothesis(
        W=W_full, a_q=q_sent, V_sentinel=kept,
        alpha=ALPHA, n_random_seeds=N_RANDOM_SEEDS,
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
            except Exception as exc:
                rec = {
                    "id": ex["id"], "status": "error",
                    "error": repr(exc), "traceback": traceback.format_exc(),
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
