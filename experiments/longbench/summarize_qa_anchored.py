"""Merge the anchored-baseline QA run with the existing QA results and compare.

The anchored reader run scored only the new ``anchored`` baseline (the other
baselines reconstruct identical text, so their F1 is unchanged). This merges:

  - ``results_qa_anchored/qa_eval_latest.jsonl``  (anchored)
  - ``results_qa/qa_eval_latest.jsonl``           (sentinel/spec/top_attn/uncompressed)

joined on (id, rate, baseline), and reports per-rate / per-task F1 with the
head-to-head questions: does the anchored cut close the spec<->Sentinel gap at
aggressive rates while keeping the NCut_q win?
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ANCHORED = os.path.join(THIS_DIR, "results_qa_anchored", "qa_eval_latest.jsonl")
EXISTING = os.path.join(THIS_DIR, "results_qa", "qa_eval_latest.jsonl")
ORDER = ["uncompressed", "sentinel", "top_attn", "spec", "anchored"]


def _load(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def summarize(anchored_path: str, existing_path: str) -> str:
    rows = _load(existing_path) + _load(anchored_path)
    # de-dup: prefer anchored file for the anchored baseline; key by (id,rate,baseline)
    by_key: Dict[tuple, Dict] = {}
    for r in rows:
        k = (r["id"], float(r["rate"]), r["baseline"])
        by_key[k] = r           # later (anchored) overwrites if duplicate
    rows = list(by_key.values())

    baselines = [b for b in ORDER if any(r["baseline"] == b for r in rows)]
    rates = sorted({float(r["rate"]) for r in rows}, reverse=True)
    tasks = sorted({r["task"] for r in rows})

    # (rate, baseline) -> list of f1
    f1s: Dict[tuple, List[float]] = defaultdict(list)
    f1s_task: Dict[tuple, List[float]] = defaultdict(list)
    for r in rows:
        if r.get("f1") is None:
            continue
        f1s[(float(r["rate"]), r["baseline"])].append(r["f1"])
        f1s_task[(r["task"], float(r["rate"]), r["baseline"])].append(r["f1"])

    L: List[str] = []
    L.append("# Anchored-cut reader-QA comparison")
    L.append("")
    L.append(f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    L.append(f"- anchored : `{anchored_path}`")
    L.append(f"- existing : `{existing_path}`")
    L.append(f"- baselines: {baselines}")
    L.append("")

    # --- pooled F1 per rate ---
    L.append("## Pooled F1 by rate (mean across all tasks)")
    L.append("")
    L.append("| rate | " + " | ".join(baselines) + " |")
    L.append("|" + "|".join(["---"] * (len(baselines) + 1)) + "|")
    for rate in rates:
        cells = [f"{rate:.2f}"]
        for b in baselines:
            v = f1s.get((rate, b), [])
            cells.append(f"{np.mean(v):.3f}" if v else "-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # --- anchored vs spec and vs sentinel head-to-head ---
    by_idrate: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r.get("f1") is not None:
            by_idrate[(r["id"], float(r["rate"]))][r["baseline"]] = r["f1"]

    for opp in ("spec", "sentinel"):
        L.append(f"## Anchored vs {opp} head-to-head (per-record F1)")
        L.append("")
        L.append(f"| rate | {opp} wins | tie | anchored wins | mean(anchored - {opp}) |")
        L.append("|---|---|---|---|---|")
        for rate in rates:
            a_w = t = o_w = 0
            deltas = []
            for (rid, rr), sc in by_idrate.items():
                if rr != rate or "anchored" not in sc or opp not in sc:
                    continue
                d = sc["anchored"] - sc[opp]
                deltas.append(d)
                if d > 1e-6:
                    a_w += 1
                elif d < -1e-6:
                    o_w += 1
                else:
                    t += 1
            md = np.mean(deltas) if deltas else 0.0
            L.append(f"| {rate:.2f} | {o_w} | {t} | {a_w} | {md:+.3f} |")
        L.append("")

    # --- per-task F1 at each rate ---
    for rate in rates:
        L.append(f"## Per-task F1 at rate = {rate}")
        L.append("")
        L.append("| task | " + " | ".join(baselines) + " |")
        L.append("|" + "|".join(["---"] * (len(baselines) + 1)) + "|")
        for task in tasks:
            cells = [task]
            for b in baselines:
                v = f1s_task.get((task, rate, b), [])
                cells.append(f"{np.mean(v):.3f}" if v else "-")
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    return "\n".join(L)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchored", default=ANCHORED)
    ap.add_argument("--existing", default=EXISTING)
    args = ap.parse_args()
    text = summarize(args.anchored, args.existing)
    print(text)
    out = os.path.join(os.path.dirname(args.anchored), "qa_anchored_comparison.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
