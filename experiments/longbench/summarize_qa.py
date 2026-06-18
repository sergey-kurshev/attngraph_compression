"""Aggregate ``qa_eval_*.jsonl`` into per-task / per-rate / per-baseline tables."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def summarize(jsonl_path: str) -> str:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    # Group by (task, rate, baseline)
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("task", "?"), float(r.get("rate", -1)), r.get("baseline", "?"))
        grouped[key].append(r)

    tasks = sorted({k[0] for k in grouped})
    rates = sorted({k[1] for k in grouped}, reverse=True)
    baselines = sorted({k[2] for k in grouped})

    lines: List[str] = []
    lines.append("# LongBench QA evaluation summary")
    lines.append("")
    lines.append(f"- Source: `{jsonl_path}`")
    lines.append(f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Rows: {len(rows)}")
    lines.append(f"- Tasks: {tasks}")
    lines.append(f"- Rates: {rates}")
    lines.append(f"- Baselines: {baselines}")
    lines.append("")

    # Overall mean F1 per (rate, baseline) — pooled across tasks
    lines.append("## Mean F1 pooled across all tasks (per rate, per baseline)")
    lines.append("")
    headers = ["rate"] + baselines
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for rate in rates:
        cells = [f"{rate:.2f}"]
        for b in baselines:
            f1s = [r["f1"] for k, vs in grouped.items() if k[1] == rate and k[2] == b
                   for r in vs if r.get("f1") is not None]
            cells.append(f"{np.mean(f1s):.3f} (n={len(f1s)})" if f1s else "-")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Mean F1 per (task, baseline) at each rate
    for rate in rates:
        lines.append(f"## Mean F1 per task at rate = {rate}")
        lines.append("")
        headers = ["task"] + baselines
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for task in tasks:
            cells = [task]
            for b in baselines:
                vs = grouped.get((task, rate, b), [])
                f1s = [r["f1"] for r in vs if r.get("f1") is not None]
                cells.append(f"{np.mean(f1s):.3f}" if f1s else "-")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Same but EM
    lines.append("## Mean EM pooled across all tasks (per rate, per baseline)")
    lines.append("")
    headers = ["rate"] + baselines
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for rate in rates:
        cells = [f"{rate:.2f}"]
        for b in baselines:
            ems = [r["em"] for k, vs in grouped.items() if k[1] == rate and k[2] == b
                   for r in vs if r.get("em") is not None]
            cells.append(f"{np.mean(ems):.3f}" if ems else "-")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Sentinel vs spectral head-to-head: per record, which wins?
    if "sentinel" in baselines and "spec" in baselines:
        lines.append("## Sentinel vs spectral-cut head-to-head (per record F1 comparison)")
        lines.append("")
        # Build (id, rate) -> {baseline -> f1}
        by_key: Dict[tuple, Dict[str, float]] = defaultdict(dict)
        for r in rows:
            k = (r["id"], float(r["rate"]))
            by_key[k][r["baseline"]] = r.get("f1")
        lines.append("| rate | spec_better | tie | sentinel_better | spec−sentinel (mean) |")
        lines.append("|---|---|---|---|---|")
        for rate in rates:
            spec_better = tie = sent_better = 0
            deltas = []
            for k, scores in by_key.items():
                if k[1] != rate:
                    continue
                if "sentinel" not in scores or "spec" not in scores:
                    continue
                s = scores["sentinel"]; sp = scores["spec"]
                if s is None or sp is None:
                    continue
                delta = sp - s
                deltas.append(delta)
                if delta > 1e-6:
                    spec_better += 1
                elif delta < -1e-6:
                    sent_better += 1
                else:
                    tie += 1
            mean_delta = np.mean(deltas) if deltas else 0.0
            lines.append(f"| {rate:.2f} | {spec_better} | {tie} | {sent_better} | {mean_delta:+.3f} |")
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_PATH = os.path.join(THIS_DIR, "results_qa", "qa_eval_latest.jsonl")
    ap.add_argument("--path", default=DEFAULT_PATH)
    args = ap.parse_args()

    text = summarize(args.path)
    print(text)

    out_dir = os.path.dirname(os.path.abspath(args.path))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"qa_summary_{timestamp}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    latest = os.path.join(out_dir, "qa_summary_latest.md")
    with open(latest, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nWrote: {out_path}\nWrote: {latest}")


if __name__ == "__main__":
    main()
