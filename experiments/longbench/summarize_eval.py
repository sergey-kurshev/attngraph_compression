"""Aggregate ``longbench_h1h2_*.jsonl`` into per-task / per-ratio summaries.

For each (task, ratio) cell, reports median + IQR for the H1 panel
(spectral_l2, Wasserstein, Davis–Kahan, edge Spearman, edge_recall, eff
resistance drift, spectral entropy diff) and the H2 panel (ncut_sentinel,
ncut_spec, ncut_local, ncut_top_attn, ncut_random_mean, jaccard_top_attn,
normalized_gap if available, cheeger, lambda_2).

Writes markdown to ``<results_dir>/longbench_h1h2_summary_latest.md`` and
``<results_dir>/longbench_h1h2_summary_<timestamp>.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

H1_METRIC_KEYS = [
    "spectral_l2",
    "spectral_wass",
    "davis_kahan_deg",
    "edge_spearman",
    "edge_recall_at_k",
    "eff_resistance_drift",
    "spectral_entropy_diff",
]
H2_METRIC_KEYS = [
    "ncut_sentinel",
    "ncut_spec",
    "ncut_spec_q",
    "ncut_local",
    "ncut_top_attn",
    "ncut_random_mean",
    "delta_cut",
    "normalized_gap",
    "cheeger_sentinel",
    "lambda_2",
    "jaccard_spec",
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


def _flat_metric(rec: Dict[str, Any], key: str):
    """Pull a metric from either the H1 nested dict or the H2 flat dict."""
    if key in rec and rec[key] is not None:
        return rec[key]
    h1 = rec.get("h1_metrics")
    if isinstance(h1, dict) and key in h1:
        return h1[key]
    return None


def _format_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h)
            if v is None:
                cells.append("-")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(v))
            elif isinstance(v, str):
                cells.append(v)
            else:
                cells.append(f"{v:.3g}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def summarize(jsonl_path: str) -> str:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Group by (task, rate)
    groups: Dict[Tuple[str, float], List[Dict]] = {}
    statuses: Dict[str, int] = {}
    for r in records:
        key = (r.get("task", "?"), r.get("rate", -1.0))
        groups.setdefault(key, []).append(r)
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1

    tasks = sorted(set(t for t, _ in groups))
    rates = sorted(set(r for _, r in groups), reverse=True)

    lines: List[str] = []
    lines.append("# LongBench-En H1 + H2 evaluation summary")
    lines.append("")
    lines.append(f"- Source: `{jsonl_path}`")
    lines.append(f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Records: {len(records)}")
    lines.append(f"- Tasks: {tasks}")
    lines.append(f"- Ratios: {rates}")
    lines.append("")
    lines.append("## Status breakdown")
    lines.append("")
    for s, n in sorted(statuses.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{s}`: {n}")
    lines.append("")

    # ---- Headline median tables ----
    lines.append("## H1 medians per (task, ratio)")
    lines.append("")
    h1_headline_keys = ["davis_kahan_deg", "edge_spearman", "edge_recall_at_k", "spectral_l2"]
    for key in h1_headline_keys:
        lines.append(f"### {key} (median)")
        lines.append("")
        rows = []
        for t in tasks:
            row = {"task": t}
            for rate in rates:
                vals = [_flat_metric(r, key) for r in groups.get((t, rate), [])]
                stats = _percentiles(vals)
                row[f"rate={rate}"] = stats.get("median")
            rows.append(row)
        headers = ["task"] + [f"rate={r}" for r in rates]
        lines.append(_format_table(rows, headers))
        lines.append("")

    lines.append("## H2 medians per (task, ratio)")
    lines.append("")
    h2_headline_keys = ["ncut_sentinel", "ncut_spec", "ncut_top_attn", "ncut_random_mean",
                        "jaccard_top_attn", "jaccard_spec", "normalized_gap"]
    for key in h2_headline_keys:
        lines.append(f"### {key} (median)")
        lines.append("")
        rows = []
        for t in tasks:
            row = {"task": t}
            for rate in rates:
                vals = [_flat_metric(r, key) for r in groups.get((t, rate), [])]
                stats = _percentiles(vals)
                row[f"rate={rate}"] = stats.get("median")
            rows.append(row)
        headers = ["task"] + [f"rate={r}" for r in rates]
        lines.append(_format_table(rows, headers))
        lines.append("")

    # ---- Beats-Sentinel rates ----
    lines.append("## How often does each H2 alternative beat Sentinel? (per ratio, all tasks pooled)")
    lines.append("")
    alts = [
        ("ncut_spec", "spectral (no query)"),
        ("ncut_spec_q", "spectral (query-anchored)"),
        ("ncut_local", "local search"),
        ("ncut_top_attn", "top-K by attention"),
        ("ncut_random_mean", "random (mean)"),
    ]
    for rate in rates:
        lines.append(f"### rate = {rate}")
        lines.append("")
        rows = []
        ok_records = [r for r in records if r.get("rate") == rate and r.get("status_h2") == "ok"]
        n_total = len(ok_records)
        for key, label in alts:
            n_beats = sum(1 for r in ok_records
                          if r.get(key) is not None and r.get("ncut_sentinel") is not None
                          and r[key] < r["ncut_sentinel"] - 1e-9)
            n_ties = sum(1 for r in ok_records
                         if r.get(key) is not None and r.get("ncut_sentinel") is not None
                         and abs(r[key] - r["ncut_sentinel"]) <= 1e-9)
            rows.append({
                "alternative": label,
                "beats_sentinel": f"{n_beats}/{n_total}",
                "ties": f"{n_ties}/{n_total}",
                "beats_pct": f"{100.0 * n_beats / n_total:.1f}%" if n_total else "-",
            })
        lines.append(_format_table(rows, ["alternative", "beats_sentinel", "ties", "beats_pct"]))
        lines.append("")

    # ---- Pooled distributions per ratio ----
    lines.append("## Pooled distributions per ratio (all tasks)")
    lines.append("")
    all_keys = h1_headline_keys + ["normalized_gap", "ncut_sentinel", "ncut_top_attn",
                                     "jaccard_top_attn"]
    for rate in rates:
        lines.append(f"### rate = {rate}")
        lines.append("")
        rows = []
        rate_recs = [r for r in records if r.get("rate") == rate]
        for key in all_keys:
            vals = [_flat_metric(r, key) for r in rate_recs]
            stats = _percentiles(vals)
            if stats.get("n", 0) == 0:
                continue
            row = {"metric": key}
            row.update(stats)
            rows.append(row)
        headers = ["metric", "n", "min", "p25", "median", "p75", "max", "mean", "std"]
        lines.append(_format_table(rows, headers))
        lines.append("")

    # ---- Graph-size stats per task ----
    lines.append("## Graph-size stats (K_full, n_kept per task)")
    lines.append("")
    rows = []
    for t in tasks:
        recs_t = [r for r in records if r.get("task") == t]
        Ks = [r.get("K_full") for r in recs_t if r.get("K_full") is not None]
        Ns = [r.get("n_tokens_full") for r in recs_t if r.get("n_tokens_full") is not None]
        row = {
            "task": t,
            "K_full_med": int(np.median(Ks)) if Ks else None,
            "K_full_max": int(np.max(Ks)) if Ks else None,
            "N_tokens_med": int(np.median(Ns)) if Ns else None,
            "N_tokens_max": int(np.max(Ns)) if Ns else None,
        }
        for rate in rates:
            kepts = [r.get("n_kept") for r in recs_t if r.get("rate") == rate and r.get("n_kept") is not None]
            row[f"kept_med@{rate}"] = int(np.median(kepts)) if kepts else None
        rows.append(row)
    headers = ["task", "K_full_med", "K_full_max", "N_tokens_med", "N_tokens_max"] + \
              [f"kept_med@{r}" for r in rates]
    lines.append(_format_table(rows, headers))
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_PATH = os.path.join(THIS_DIR, "results_h1h2", "longbench_h1h2_latest.jsonl")
    ap.add_argument("--path", default=DEFAULT_PATH)
    args = ap.parse_args()

    text = summarize(args.path)
    print(text)

    out_dir = os.path.dirname(os.path.abspath(args.path))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"longbench_h1h2_summary_{timestamp}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    latest = os.path.join(out_dir, "longbench_h1h2_summary_latest.md")
    with open(latest, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nWrote: {out_path}\nWrote: {latest}")


if __name__ == "__main__":
    main()
