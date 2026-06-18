"""Summarize the H2 eval JSONL into per-metric distributions + a markdown report.

Reads ``experiments/h2/results/h2_eval_latest.jsonl`` (or ``--path``) and emits:
- ``experiments/h2/results/h2_summary_<timestamp>.md`` and ``h2_summary_latest.md``
  with min / 25th / median / 75th / max per metric, plus rank tables.
- Stdout: the same content.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


METRIC_KEYS = [
    "ncut_sentinel",
    "ncut_exact",
    "ncut_spec",
    "ncut_spec_q",
    "ncut_local",
    "ncut_top_attn",
    "ncut_random_mean",
    "delta_cut",
    "normalized_gap",
    "cheeger_sentinel",
    "lambda_2",
    "jaccard_exact",
    "jaccard_spec",
    "jaccard_spec_q",
    "jaccard_local",
    "jaccard_top_attn",
    "jaccard_random_mean",
]


def _percentiles(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _format_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["metric", "n", "min", "p25", "median", "p75", "max", "mean", "std"]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = [r["metric"]]
        for h in headers[1:]:
            v = r.get(h)
            if v is None:
                cells.append("-")
            elif h == "n":
                cells.append(str(v))
            else:
                cells.append(f"{v:.4g}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _rank_table(records: List[Dict[str, Any]]) -> str:
    """How often does each strategy beat Sentinel on NCut_q?"""
    rows = []
    methods = [
        ("ncut_exact", "exact (lower bound)"),
        ("ncut_spec", "spectral (no query)"),
        ("ncut_spec_q", "spectral (query-anchored)"),
        ("ncut_local", "local search from Sentinel"),
        ("ncut_top_attn", "top-K by query attention"),
        ("ncut_random_mean", "random (mean)"),
    ]
    for key, label in methods:
        n = 0
        n_beats = 0
        n_ties = 0
        for r in records:
            if r.get("status") != "ok":
                continue
            sent = r.get("ncut_sentinel")
            other = r.get(key)
            if sent is None or other is None:
                continue
            n += 1
            if other < sent - 1e-9:
                n_beats += 1
            elif abs(other - sent) <= 1e-9:
                n_ties += 1
        if n == 0:
            continue
        rows.append({
            "label": label,
            "beats_sentinel": f"{n_beats}/{n}",
            "ties": f"{n_ties}/{n}",
            "beats_pct": f"{100.0 * n_beats / n:.1f}%",
        })

    headers = ["alternative", "beats Sentinel", "ties", "beats %"]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append(f"| {row['label']} | {row['beats_sentinel']} | {row['ties']} | {row['beats_pct']} |")
    return "\n".join(out)


def summarize(jsonl_path: str) -> str:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    status_counts: Dict[str, int] = {}
    metric_lists: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    n_kept_list: List[int] = []
    k_full_list: List[int] = []

    for r in records:
        status = r.get("status", "?")
        status_counts[status] = status_counts.get(status, 0) + 1
        if "n_kept" in r:
            n_kept_list.append(r["n_kept"])
        if "K_full" in r:
            k_full_list.append(r["K_full"])
        for k in METRIC_KEYS:
            if k in r and r[k] is not None:
                metric_lists[k].append(r[k])

    lines: List[str] = []
    lines.append(f"# H2 evaluation summary")
    lines.append("")
    lines.append(f"- Source: `{jsonl_path}`")
    lines.append(f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Records: {len(records)}")
    lines.append("")
    lines.append("## Status breakdown")
    lines.append("")
    for status, n in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{status}`: {n}")
    lines.append("")
    lines.append("## Graph sizes")
    lines.append("")
    if k_full_list:
        lines.append(f"- `K_full` (source-context sentences): "
                     f"min={min(k_full_list)} median={int(np.median(k_full_list))} max={max(k_full_list)}")
    if n_kept_list:
        lines.append(f"- `|V'|`  (Sentinel kept = budget B): "
                     f"min={min(n_kept_list)} median={int(np.median(n_kept_list))} max={max(n_kept_list)}")
    lines.append("")
    lines.append("## NCut_q across methods (status `ok`)")
    lines.append("")
    rows = []
    for k in METRIC_KEYS:
        stats = _percentiles(metric_lists[k])
        if stats.get("n", 0) == 0:
            continue
        row = {"metric": k}
        row.update(stats)
        rows.append(row)
    if rows:
        lines.append(_format_table(rows))
    lines.append("")
    lines.append("## How often does each alternative beat Sentinel on NCut_q?")
    lines.append("")
    lines.append(_rank_table(records))
    lines.append("")
    lines.append("Notes:")
    lines.append("- `exact` is the global minimum at budget B — by construction it cannot lose.")
    lines.append("- `local search from Sentinel` is Sentinel + ≥0 improving swaps; it cannot lose either.")
    lines.append("- The interesting comparisons are `spectral`, `spectral (query-anchored)`, `top-K by query attention`, and `random`.")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default=os.path.join(THIS_DIR, "results", "h2_eval_latest.jsonl"),
        help="JSONL of per-example H2 records.",
    )
    args = ap.parse_args()

    text = summarize(args.path)
    print(text)

    # Write summary next to the input JSONL so it ends up in the same folder.
    out_dir = os.path.dirname(os.path.abspath(args.path))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"h2_summary_{timestamp}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    latest = os.path.join(out_dir, "h2_summary_latest.md")
    with open(latest, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nWrote: {out_path}\nWrote: {latest}")


if __name__ == "__main__":
    main()
