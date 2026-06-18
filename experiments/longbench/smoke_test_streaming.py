"""Smoke test: does the streaming extractor handle LongBench scale?

Companion to ``smoke_test_longbench.py``. Same four representative lengths
(p25 / p50 / p75 / p95 of multifieldqa_en), but using the new
``StreamingExtractor`` instead of the OOMing ``SpectralExtractor``.

What we want to see
-------------------
- Every example finishes without OOM.
- Peak GPU memory stays well under 12 GiB.
- Wall time per example is comparable to (or better than) the existing
  Sentinel-only path.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR = os.path.dirname(THIS_DIR)
ROOT = os.path.dirname(EXPS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from spectral.streaming_extractor import StreamingExtractor  # noqa: E402


LONGBENCH_DIR = os.path.join(THIS_DIR, "data", "data")


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _load_task(task: str):
    with open(os.path.join(LONGBENCH_DIR, f"{task}.jsonl"), "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _peak_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def main():
    print("=" * 72)
    print("StreamingExtractor smoke test on LongBench")
    print("=" * 72)

    ds = _load_task("multifieldqa_en")
    print(f"\nLoaded {len(ds)} examples\n")

    bench_indices = [
        ("p25", 48),
        ("p50", 5),
        ("p75", 44),
        ("p95", 149),
    ]

    print("Building one StreamingExtractor at max_seq_len=10000 (used for all examples).")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ext = StreamingExtractor(
        attention_model_path=_proxy_path(),
        max_seq_len=10000,
        device="cuda",
    )
    print(f"    loaded in {time.time() - t0:.1f}s, peak after load = {_peak_gib():.2f} GiB\n")

    results = []
    for label, idx in bench_indices:
        ex = ds[idx]
        n_tokens = ex.get("length")
        print(f"--- {label}: idx={idx}, reported tokens={n_tokens} ---")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

        try:
            t0 = time.time()
            sg = ext.extract_graph(ex["context"], ex["input"], context_type="english")
            dt = time.time() - t0
            info = {
                "label": label,
                "idx": idx,
                "reported_tokens": n_tokens,
                "ok": True,
                "N": sg.n_tokens,
                "K": len(sg.sentences),
                "extract_s": dt,
                "peak_gib": _peak_gib(),
            }
            print(f"    N={info['N']:>5}  K={info['K']:>4}  extract={info['extract_s']:.2f}s  "
                  f"peak={info['peak_gib']:.2f} GiB")
        except torch.cuda.OutOfMemoryError as e:
            info = {
                "label": label, "idx": idx, "reported_tokens": n_tokens,
                "ok": False,
                "error": "OOM: " + str(e).splitlines()[0][:200],
                "peak_gib": _peak_gib(),
            }
            print(f"    OOM at peak {_peak_gib():.2f} GiB")
        except Exception as e:
            info = {
                "label": label, "idx": idx, "reported_tokens": n_tokens,
                "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }
            print(f"    FAILED: {type(e).__name__}: {str(e)[:120]}")
        results.append(info)

    print("\nSummary")
    print("-" * 72)
    ok = [r for r in results if r.get("ok")]
    if len(ok) == len(results):
        max_peak = max(r["peak_gib"] for r in ok)
        max_tok = max(r["N"] for r in ok)
        max_time = max(r["extract_s"] for r in ok)
        print(f"    All {len(results)} examples succeeded.")
        print(f"    Max N = {max_tok} tokens  (LongBench full range covered)")
        print(f"    Max peak GPU = {max_peak:.2f} GiB on a 12 GiB card")
        print(f"    Max extract time = {max_time:.2f}s")
    else:
        print(f"    {len(ok)}/{len(results)} examples succeeded.")
        for r in results:
            if not r.get("ok"):
                print(f"      {r['label']} (tokens={r['reported_tokens']}): {r.get('error', '?')[:150]}")

    out_path = os.path.join(THIS_DIR, "results", "smoke_streaming_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
