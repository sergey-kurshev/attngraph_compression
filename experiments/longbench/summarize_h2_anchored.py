"""Summarize the anchored-cut H2 rerun: NCut_q and top-attention coverage.

Compares the new ``anchored`` cut against Sentinel, plain spectral (``spec``),
query-anchored spectral (``spec_q``) and ``top_attn`` per compression rate:

- median NCut_q (lower = better on the graph objective),
- top-m attention coverage (the failure the anchored cut fixes),
- how often each beats Sentinel on NCut_q.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(THIS_DIR, "results_h1h2_anchored", "longbench_h1h2_latest.jsonl")


def summarize(path: str) -> str:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok" and r.get("ncut_anchored") is not None]
    by_rate: Dict[float, List[Dict]] = defaultdict(list)
    for r in ok:
        by_rate[float(r["rate"])].append(r)
    rates = sorted(by_rate)

    L: List[str] = []
    L.append("# H2 anchored-cut rerun summary")
    L.append("")
    L.append(f"- Source: `{path}`")
    L.append(f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    L.append(f"- ok records with anchored column: {len(ok)} / {len(rows)} total")
    L.append("")

    def med(vals):
        return float(np.median(vals)) if vals else float("nan")

    # --- median NCut_q per rate ---
    L.append("## Median NCut_q by rate (lower = better)")
    L.append("")
    L.append("| rate | n | Sentinel | spec | spec_q | **anchored** | top_attn |")
    L.append("|---|---|---|---|---|---|---|")
    for rate in rates:
        rs = by_rate[rate]
        L.append(
            f"| {rate:.2f} | {len(rs)} | "
            f"{med([r['ncut_sentinel'] for r in rs]):.3f} | "
            f"{med([r['ncut_spec'] for r in rs]):.3f} | "
            f"{med([r['ncut_spec_q'] for r in rs]):.3f} | "
            f"**{med([r['ncut_anchored'] for r in rs]):.3f}** | "
            f"{med([r['ncut_top_attn'] for r in rs]):.3f} |"
        )
    L.append("")

    # --- top-m attention coverage ---
    L.append("## Mean top-m attention coverage by rate (1.0 = all top sentences kept)")
    L.append("")
    L.append("| rate | m_avg | spec | spec_q | **anchored** |")
    L.append("|---|---|---|---|---|")
    for rate in rates:
        rs = by_rate[rate]
        L.append(
            f"| {rate:.2f} | {np.mean([r.get('m_anchor', 0) for r in rs]):.1f} | "
            f"{np.mean([r['topm_cov_spec'] for r in rs]):.2f} | "
            f"{np.mean([r['topm_cov_spec_q'] for r in rs]):.2f} | "
            f"**{np.mean([r['topm_cov_anchored'] for r in rs]):.2f}** |"
        )
    L.append("")

    # --- beats Sentinel on NCut_q ---
    L.append("## % of records that beat Sentinel on NCut_q")
    L.append("")
    L.append("| rate | spec | spec_q | **anchored** | top_attn |")
    L.append("|---|---|---|---|---|")
    for rate in rates:
        rs = by_rate[rate]
        n = len(rs)
        def pct(key):
            return 100.0 * sum(1 for r in rs if r[key] < r["ncut_sentinel"]) / n if n else 0.0
        L.append(
            f"| {rate:.2f} | {pct('ncut_spec'):.0f}% | {pct('ncut_spec_q'):.0f}% | "
            f"**{pct('ncut_anchored'):.0f}%** | {pct('ncut_top_attn'):.0f}% |"
        )
    L.append("")

    # --- anchored vs spec: NCut cost of the inclusion guarantee ---
    L.append("## Anchored vs plain spec: NCut_q delta (anchored - spec)")
    L.append("")
    L.append("| rate | median Δ | anchored lower | spec lower |")
    L.append("|---|---|---|---|")
    for rate in rates:
        rs = by_rate[rate]
        deltas = [r["ncut_anchored"] - r["ncut_spec"] for r in rs]
        lower = sum(1 for d in deltas if d < -1e-9)
        higher = sum(1 for d in deltas if d > 1e-9)
        L.append(f"| {rate:.2f} | {med(deltas):+.3f} | {lower} | {higher} |")
    L.append("")

    return "\n".join(L)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT)
    args = ap.parse_args()
    text = summarize(args.path)
    print(text)
    out = os.path.join(os.path.dirname(args.path), "h2_anchored_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
