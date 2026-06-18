"""Efficient H2 rerun adding the anchor-constrained cut (option D).

The anchored cut is a pure function of ``(W, a_q, budget)``, and H1 is
independent of any H2 solver, so we don't need the full
``run_eval_h1h2.py`` pipeline (compressor + per-rate H1 re-extraction, the
OOM-prone part). Instead we:

  1. re-extract each example's sentence graph ``W``/``a_q`` ONCE (GPU,
     rate-independent — same as the original eval, which extracts once per
     example and reuses across rates),
  2. recompute the *entire* H2 panel — including the new ``anchored`` column —
     from the existing records' ``V_sentinel`` (== Sentinel's selection),
  3. carry every H1 field forward unchanged (graphs reproduce bit-identically,
     verified at jaccard 1.0).

Output goes to a fresh ``results_h1h2_anchored/`` folder. Non-``ok`` records
(OOM/skip) are copied through untouched. Resumable at (id, rate) granularity.

Run: ``.venv/Scripts/python.exe experiments/longbench/rerun_h2_anchored.py``
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, ROOT)

from spectral.streaming_extractor import StreamingExtractor  # noqa: E402
from spectral.h2_test import run_optimal_cut_hypothesis  # noqa: E402

SUBSET_PATH = os.path.join(THIS_DIR, "data_subset", "longbench_en_subset.jsonl")
SRC_PATH = os.path.join(THIS_DIR, "results_h1h2", "longbench_h1h2_latest.jsonl")
OUT_DIR = os.path.join(THIS_DIR, "results_h1h2_anchored")

# Match run_eval_h1h2.py's H2 configuration exactly.
POOLING = "mean"
MAX_SEQ_LEN = 8000
ALPHA = 1.0
ANCHOR_FRAC = 0.25
N_RANDOM_SEEDS = 20
EXACT_MAX_SUBSETS = 500_000
LOCAL_MAX_ITER = 10


def _proxy_path() -> str:
    local = os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct")
    return os.path.abspath(local) if os.path.exists(local) else "Qwen/Qwen2.5-0.5B-Instruct"


def _seed(example_id: str) -> int:
    """Deterministic per-example seed (hash() is salted per process)."""
    h = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _load_resume_keys(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") is not None and r.get("rate") is not None:
                done.add((r["id"], float(r["rate"])))
    return done


def main() -> None:
    import argparse
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-from", default=None,
                    help="prior output JSONL; (id,rate) tuples there are carried and skipped")
    args = ap.parse_args()

    # ----- load source records + examples ---------------------------------
    examples: Dict[str, Dict[str, Any]] = {}
    with open(SUBSET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                examples[r["id"]] = r

    src_records: List[Dict[str, Any]] = []
    with open(SRC_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                src_records.append(json.loads(line))
    print(f"Loaded {len(src_records)} source records, {len(examples)} examples")

    # group records by id, preserving order
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in src_records:
        by_id[r["id"]].append(r)

    done_keys = set()
    carried: List[Dict[str, Any]] = []
    if args.resume_from:
        done_keys = _load_resume_keys(args.resume_from)
        with open(args.resume_from, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rr = json.loads(line)
                    if (rr.get("id"), float(rr.get("rate", -1))) in done_keys:
                        carried.append(rr)
        print(f"Resume: {len(done_keys)} (id,rate) tuples carried forward")

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"longbench_h1h2_{timestamp}.jsonl")
    latest_path = os.path.join(OUT_DIR, "longbench_h1h2_latest.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proxy = _proxy_path()
    print(f"Device: {device}\nProxy : {proxy}\n")
    extractor = StreamingExtractor(
        attention_model_path=proxy, eval_tokenizer_path=proxy,
        max_seq_len=MAX_SEQ_LEN, device=device,
    )

    n_recomputed = n_carried = n_copied = n_mismatch = n_extract_fail = 0
    t0 = time.time()
    out_records: List[Dict[str, Any]] = list(carried)

    ids = list(by_id.keys())
    for idx, ex_id in enumerate(ids, 1):
        t_ex = time.time()
        recs = by_id[ex_id]
        # all rates for this id already carried by resume?
        if all((ex_id, float(r.get("rate", -1))) in done_keys for r in recs):
            continue

        ok_recs = [r for r in recs if r.get("status") == "ok" and r.get("V_sentinel") is not None]

        W = a_q = None
        if ok_recs:
            ex = examples.get(ex_id)
            if ex is None:
                print(f"  [{idx}/{len(ids)}] {ex_id}: no example in subset, copying records")
            else:
                try:
                    g = extractor.extract_graph(
                        ex["context"], ex["question"],
                        context_type="english", pooling=POOLING,
                    )
                    W = np.asarray(g.W, dtype=np.float64)
                    a_q = np.asarray(g.q_sent, dtype=np.float64)
                except Exception as exc:  # noqa: BLE001
                    n_extract_fail += 1
                    print(f"  [{idx}/{len(ids)}] {ex_id}: extract FAILED ({exc!r}); copying records")
                if device == "cuda":
                    torch.cuda.empty_cache()

        for r in recs:
            key = (ex_id, float(r.get("rate", -1)))
            if key in done_keys:
                continue
            out = dict(r)
            if W is not None and r.get("status") == "ok" and r.get("V_sentinel") is not None:
                V_sent = r["V_sentinel"]
                if max(V_sent) >= W.shape[0]:
                    n_mismatch += 1
                    out["h2_recompute"] = "skip_index_mismatch"
                else:
                    h2 = run_optimal_cut_hypothesis(
                        W=W, a_q=a_q, V_sentinel=V_sent,
                        alpha=ALPHA, n_random_seeds=N_RANDOM_SEEDS,
                        exact_max_subsets=EXACT_MAX_SUBSETS,
                        random_seed=_seed(ex_id), local_max_iter=LOCAL_MAX_ITER,
                        anchor_frac=ANCHOR_FRAC,
                    )
                    out.update(h2)        # overwrite all H2 fields, keep H1
                    out["status_h2"] = "ok"
                    out["h2_recompute"] = "anchored_v1"
                    n_recomputed += 1
            else:
                n_copied += 1
            out_records.append(out)

        K = W.shape[0] if W is not None else 0
        print(f"  [{idx}/{len(ids)}] {ex_id.split(':')[0]:16s} K={K:4d} "
              f"n_ok={len(ok_recs)}  ({time.time() - t_ex:.1f}s)  "
              f"[recomp={n_recomputed} copy={n_copied} mism={n_mismatch} "
              f"xfail={n_extract_fail}]", flush=True)

        # incremental flush every few examples so a crash is resumable
        if idx % 5 == 0 or idx == len(ids):
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in out_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(latest_path, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"  recomputed H2 (with anchored): {n_recomputed}")
    print(f"  carried (resume)             : {len(carried)}")
    print(f"  copied unchanged (non-ok)    : {n_copied}")
    print(f"  index mismatch (skipped)     : {n_mismatch}")
    print(f"  extract failures             : {n_extract_fail}")
    print(f"  total records written        : {len(out_records)}")
    print(f"\nWrote: {out_path}\nWrote: {latest_path}")


if __name__ == "__main__":
    main()
