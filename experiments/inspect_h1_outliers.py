"""Inspect the worst-performing H1 cases.

For each metric where "high = worse H1" or "low = worse H1", pick the bottom-k
records and print enough detail (sentences kept, sentences in C' after
re-segmentation, alignment status, all metrics) to characterize the failure mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEFAULT_PATH = os.path.join(THIS_DIR, "results", "h1_eval_latest.jsonl")


def _load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# (metric, direction). "high" -> sort descending = worst at top.
WORST_DIRECTION = {
    "davis_kahan_deg":      "high",
    "spectral_l2":          "high",
    "spectral_wass":        "high",
    "spectral_entropy_diff":"high",
    "eff_resistance_drift": "high",
    "edge_spearman":        "low",   # 1 = perfect; -1 = anticorrelated
    "edge_recall_at_k":     "low",
}


def _print_record(idx: int, rank: int, r: Dict, focus: str):
    print(f"\n{'='*72}")
    print(f"[#{rank}]  worst on  '{focus}'   record_idx={idx}")
    print(f"id          : {r.get('id')}")
    print(f"K_full      : {r.get('K_full')}")
    print(f"|V'|        : {r.get('n_kept')}")
    print(f"K_hat       : {r.get('K_hat')}")
    print(f"alignment   : {r.get('alignment')}")
    print(f"status      : {r.get('status')}")
    m = r.get("metrics") or {}
    if m:
        print("metrics:")
        for k, v in m.items():
            star = "  <--" if k == focus else ""
            print(f"  {k:<24s} {v:>10.4f}{star}")
    kept = r.get("sentences_kept", [])
    hat = r.get("sentences_hat", [])
    if kept:
        print(f"\nSentences kept by Sentinel ({len(kept)}):")
        for i, s in enumerate(kept):
            print(f"  V'[{i}] {s.strip()}")
    if hat:
        print(f"\nSentences in C' after re-segmentation ({len(hat)}):")
        for i, s in enumerate(hat):
            print(f"  C'[{i}] {s.strip()}")


def inspect(path: str, k: int = 5, metrics: Optional[List[str]] = None):
    records = _load(path)
    print(f"Loaded {len(records)} records from {path}\n")

    if metrics is None:
        metrics = ["davis_kahan_deg", "edge_spearman", "eff_resistance_drift"]

    for metric in metrics:
        direction = WORST_DIRECTION[metric]
        scored = []
        for i, r in enumerate(records):
            m = r.get("metrics") or {}
            if metric not in m:
                continue
            v = m[metric]
            if v is None:
                continue
            scored.append((v, i, r))
        if not scored:
            continue
        # Sort so the worst is first.
        scored.sort(key=lambda t: t[0], reverse=(direction == "high"))

        print("\n" + "#" * 72)
        print(f"# Worst {k} on '{metric}' ({direction} = worse)")
        print("#" * 72)
        for rank, (val, i, r) in enumerate(scored[:k], 1):
            _print_record(i, rank, r, focus=metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--metrics", nargs="+", default=None)
    args = ap.parse_args()
    inspect(args.path, k=args.k, metrics=args.metrics)


if __name__ == "__main__":
    main()
