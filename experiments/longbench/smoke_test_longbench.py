"""Smoke test: can the existing pipeline handle a LongBench-En context?

Two paths to test separately, since they have very different memory profiles:

* **Sentinel-only path** (``AttentionCompressor`` with ``use_all_queries=False``)
  uses last-token-only attention via monkey-patch; memory is ``O(L * H * N)``
  and shouldn't OOM at any realistic context length.

* **Spectral-extractor path** (``SpectralExtractor``, ``use_all_queries=True``)
  materializes the full ``[L, H, N, N]`` attention tensor. At N=1024 that's
  ~1.4 GiB; at N=2048 it's ~5.6 GiB; at N=4096 it's ~22 GiB — beyond our
  12 GiB card.

The test sweeps ``max_seq_len ∈ {1024, 2048, 4096, 8192}`` for each path
separately, reporting wall-time, peak GPU memory, and OOM points.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Dict, List

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


LONGBENCH_DIR = os.path.join(THIS_DIR, "data", "data")


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _load_longbench_task(task: str = "multifieldqa_en"):
    path = os.path.join(LONGBENCH_DIR, f"{task}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def _peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def test_sentinel_only(ctx: str, q: str, max_seq_len: int) -> Dict:
    """The cheap path: monkey-patched last-token attention, no [L,H,N,N] tensor."""
    info = {"path": "sentinel", "max_seq_len": max_seq_len}
    _reset_gpu()
    try:
        comp = AttentionCompressor(
            attention_model_path=_proxy_path(), detector_path=None,
            use_raw_attention=True, use_last_layer_only=False,
            use_all_queries=False, max_seq_len=max_seq_len,
            device="cuda", print_sentence_scores=False,
        )
        t0 = time.time()
        res = comp.compress(
            context=ctx, question=q, target_token=-1,
            compression_rate=0.5, context_type="english",
        )
        info["compress_s"] = time.time() - t0
        info["K"] = len(res["sentences"])
        info["n_kept"] = len(res["preserved_indices"])
        info["compression_ratio_tokens"] = res["compression_ratio"]
        info["ok"] = True
        info["peak_mb"] = _peak_mb()
    except torch.cuda.OutOfMemoryError as e:
        info["ok"] = False
        info["error"] = "OOM: " + str(e).splitlines()[0][:300]
        info["peak_mb"] = _peak_mb()
    except Exception as e:
        info["ok"] = False
        info["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if "comp" in locals():
            del comp
        _reset_gpu()
    return info


def test_extractor(ctx: str, q: str, max_seq_len: int) -> Dict:
    """The expensive path: full [L, H, N, N] attention tensor."""
    info = {"path": "extractor", "max_seq_len": max_seq_len}
    _reset_gpu()
    try:
        ext = SpectralExtractor(
            attention_model_path=_proxy_path(), eval_tokenizer_path=_proxy_path(),
            max_seq_len=max_seq_len, device="cuda", print_sentence_scores=False,
        )
        t0 = time.time()
        ex = ext.extract(ctx, q, context_type="english")
        info["extract_s"] = time.time() - t0
        info["N_tokens"] = int(ex.n_tokens)
        info["K"] = len(ex.sentences)
        # Build graph too — cheap but verifies the full path
        t0 = time.time()
        W, _ = build_sentence_graph(
            ex.attn, ex.sentence_spans,
            aggregation="mean", pooling="mean", sparsify_method="none",
            final_token_idx=ex.query_token_idx,
        )
        info["build_graph_s"] = time.time() - t0
        info["attn_tensor_mb"] = float(ex.attn.nbytes) / 1024**2
        info["ok"] = True
        info["peak_mb"] = _peak_mb()
    except torch.cuda.OutOfMemoryError as e:
        info["ok"] = False
        info["error"] = "OOM: " + str(e).splitlines()[0][:300]
        info["peak_mb"] = _peak_mb()
    except Exception as e:
        info["ok"] = False
        info["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if "ext" in locals():
            del ext
        _reset_gpu()
    return info


def main():
    print("=" * 72)
    print("LongBench smoke test")
    print("=" * 72)

    print("\n[1] Loading LongBench multifieldqa_en from local JSONL ...")
    ds = _load_longbench_task("multifieldqa_en")
    print(f"    {len(ds)} examples")

    # Sweep representative lengths: p25, median, p75, p95.
    # Indices computed in advance (see length-distribution check).
    bench_indices = [
        ("p25", 48),
        ("p50", 5),
        ("p75", 44),
        ("p95", 149),
    ]

    # The current full-attn extractor caps at max_seq_len. To characterize the
    # OOM frontier we pick the EXTRACTOR'S working max as the sweep length,
    # so the only variable that changes between examples is N (actual prompt
    # length).
    EXTRACT_MAX = 8192

    all_sentinel: List[Dict] = []
    all_extractor: List[Dict] = []

    for label, idx in bench_indices:
        ex = ds[idx]
        n_tokens = ex.get("length")
        print("")
        print(f"--- Example {label} (idx={idx}, reported tokens={n_tokens}) ---")
        print(f"    Question : {ex['input'][:100]}")
        print(f"    Chars    : {len(ex['context'])}")

        info_s = test_sentinel_only(ex["context"], ex["input"], max_seq_len=EXTRACT_MAX)
        info_s["bench_label"] = label
        info_s["example_length"] = n_tokens
        all_sentinel.append(info_s)
        if info_s["ok"]:
            print(f"    Sentinel  : K={info_s['K']:>3}  V'={info_s['n_kept']:>3}  "
                  f"compress={info_s['compress_s']:.2f}s  peak={info_s['peak_mb']/1024:.2f}GiB")
        else:
            print(f"    Sentinel  : FAILED: {info_s.get('error', '?')[:120]}")

        info_e = test_extractor(ex["context"], ex["input"], max_seq_len=EXTRACT_MAX)
        info_e["bench_label"] = label
        info_e["example_length"] = n_tokens
        all_extractor.append(info_e)
        if info_e["ok"]:
            print(f"    Extractor : N={info_e['N_tokens']:>5}  K={info_e['K']:>3}  "
                  f"extract={info_e['extract_s']:.2f}s  attn={info_e['attn_tensor_mb']/1024:.2f}GiB  "
                  f"peak={info_e['peak_mb']/1024:.2f}GiB")
        else:
            print(f"    Extractor : FAILED: {info_e.get('error', '?')[:140]}")

    sentinel_results = all_sentinel
    extractor_results = all_extractor

    print("\n[4] Summary")
    print("-" * 72)

    def _ok_max_len(results):
        ok = [r for r in results if r.get("ok")]
        return max((r["example_length"] for r in ok), default=None)

    sentinel_top = _ok_max_len(sentinel_results)
    extractor_top = _ok_max_len(extractor_results)
    print(f"    Sentinel-only path  : works up to ~{sentinel_top} tokens (this dataset)")
    print(f"    Spectral-extractor  : works up to ~{extractor_top} tokens (this dataset)")
    print("")
    print("    The H1/H2 eval needs the spectral-extractor path. If its limit is")
    print("    below the LongBench median, options are:")
    print("    a) Truncate contexts to the extractor's limit — defeats the point.")
    print("    b) Refactor: fold attention layer-by-layer into the sentence graph,")
    print("       avoiding the full [L,H,N,N] tensor (saves a factor of L*H in mem).")
    print("    c) Sentence-pool inside the model forward, never materializing the")
    print("       token-level [N,N] either.")

    out_path = os.path.join(THIS_DIR, "results", "smoke_longbench_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "sentinel_path": sentinel_results,
            "extractor_path": extractor_results,
        }, f, indent=2)
    print(f"\n    Wrote: {out_path}")


if __name__ == "__main__":
    main()
