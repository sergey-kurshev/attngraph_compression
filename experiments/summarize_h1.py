"""Summarize the H1 eval JSONL into per-metric distributions + a markdown report.

Reads ``experiments/results/h1_eval_latest.jsonl`` (or ``--path``) and emits:
- ``experiments/results/h1_summary_<timestamp>.md`` with min / 25th / median /
  75th / max per metric, plus a status breakdown.
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
ROOT = os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


METRIC_KEYS = [
    "spectral_l2",
    "spectral_wass",
    "davis_kahan_deg",
    "edge_spearman",
    "edge_recall_at_k",
    "eff_resistance_drift",
    "spectral_entropy_diff",
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


def summarize(jsonl_path: str) -> str:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    status_counts: Dict[str, int] = {}
    metric_lists: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    size_agnostic_lists: Dict[str, List[float]] = {
        "spectral_l2": [],
        "spectral_wass": [],
        "spectral_entropy_diff": [],
    }
    n_kept_list: List[int] = []
    k_full_list: List[int] = []
    k_hat_list: List[int] = []

    for r in records:
        status = r.get("status", "?")
        status_counts[status] = status_counts.get(status, 0) + 1
        if "n_kept" in r:
            n_kept_list.append(r["n_kept"])
        if "K_full" in r:
            k_full_list.append(r["K_full"])
        if "K_hat" in r:
            k_hat_list.append(r["K_hat"])
        m = r.get("metrics")
        if isinstance(m, dict):
            for k in METRIC_KEYS:
                if k in m and m[k] is not None:
                    metric_lists[k].append(m[k])
        m_size = r.get("metrics_size_agnostic")
        if isinstance(m_size, dict):
            for k in size_agnostic_lists:
                if k in m_size and m_size[k] is not None:
                    size_agnostic_lists[k].append(m_size[k])

    # Build markdown.
    lines: List[str] = []
    lines.append(f"# H1 evaluation summary")
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
        lines.append(f"- `|V'|`  (Sentinel kept): "
                     f"min={min(n_kept_list)} median={int(np.median(n_kept_list))} max={max(n_kept_list)}")
    if k_hat_list:
        lines.append(f"- `K_hat` (re-segmented C'): "
                     f"min={min(k_hat_list)} median={int(np.median(k_hat_list))} max={max(k_hat_list)}")
    lines.append("")
    lines.append("## Aligned-graph metrics (status `ok`)")
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
    else:
        lines.append("_no aligned-graph records — all examples fell back to size-agnostic._")
    lines.append("")
    lines.append("## Size-agnostic fallback metrics (status `ok_size_agnostic`)")
    lines.append("")
    rows_sa = []
    for k in size_agnostic_lists:
        stats = _percentiles(size_agnostic_lists[k])
        if stats.get("n", 0) == 0:
            continue
        row = {"metric": k}
        row.update(stats)
        rows_sa.append(row)
    if rows_sa:
        lines.append(_format_table(rows_sa))
    else:
        lines.append("_no size-agnostic records._")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default=os.path.join(THIS_DIR, "results", "h1_eval_latest.jsonl"),
        help="JSONL of per-example H1 records.",
    )
    args = ap.parse_args()

    text = summarize(args.path)
    print(text)

    # Write summary next to the input JSONL so it ends up in the same folder.
    out_dir = os.path.dirname(os.path.abspath(args.path))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"h1_summary_{timestamp}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    latest = os.path.join(out_dir, "h1_summary_latest.md")
    with open(latest, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nWrote: {out_path}\nWrote: {latest}")


if __name__ == "__main__":
    main()
