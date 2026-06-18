"""Anchor-constrained cut test on the 5 worst spec cases.

Background
----------
On LongBench, the plain ``spectral_cut`` (Fiedler vector + sweep) decides
membership purely by Fiedler rank, so high query-attention ("answer-bearing")
sentences in the interior of that ordering are dropped. These five examples
are the worst offenders — the spectral cut shares almost no sentences with the
top-attention set, and the reader-QA F1 collapses (Sentinel 1.0 vs spec 0.0).

What this test checks, in three phases:

1. **Phase 1 — the failure is real.** For each cached graph, ``spectral_cut``
   does *not* contain the top-``m`` sentences by ``a_q``. Coverage is low.

2. **Phase 2 — alpha doesn't reliably fix it.** Sweep ``alpha`` for
   ``query_anchored_spectral_cut`` (option B) and report top-``m`` coverage per
   example. The eval ran ``alpha=1.0``; we show it changes behaviour
   inconsistently across the 5 examples (no single alpha guarantees inclusion).

3. **Phase 3 — option D guarantees inclusion.** ``anchored_spectral_cut`` pins
   the top-``m`` attention sentences and fills the rest spectrally with an
   anchor-constrained local search. Top-``m`` coverage is exactly 1.0 on every
   example, and we report its NCut_q vs the plain spectral cut.

The graphs are cached by ``build_anchor_fixtures.py`` (a one-time GPU step) so
this test runs on CPU alone. If the fixtures are missing, the test is skipped
with a clear message.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from spectral.cut_solvers import (
    anchored_spectral_cut,
    query_anchored_spectral_cut,
    spectral_cut,
)
from spectral.ncut import ncut_q

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIX_DIR = os.path.join(THIS_DIR, "fixtures", "anchor")
MANIFEST = os.path.join(FIX_DIR, "manifest.json")

# Fraction of the budget reserved for guaranteed top-attention sentences.
# m = ceil(ANCHOR_FRAC * budget). Same m is used to *measure* coverage in all
# three phases so the before/after comparison is apples-to-apples.
ANCHOR_FRAC = 0.25
ALPHA_SWEEP = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]


def _anchor_m(budget: int) -> int:
    return max(1, min(int(math.ceil(ANCHOR_FRAC * budget)), budget))


def _top_m(a_q: np.ndarray, m: int) -> set:
    return set(int(i) for i in np.argsort(-a_q)[:m])


def _coverage(kept, top_set) -> float:
    """Fraction of the top-attention set that survives in ``kept``."""
    if not top_set:
        return 1.0
    return len(set(int(i) for i in kept) & top_set) / len(top_set)


def _load_cases():
    if not os.path.exists(MANIFEST):
        pytest.skip(
            "anchor fixtures missing — run "
            "`.venv/Scripts/python.exe tests/test_spectral/build_anchor_fixtures.py` first"
        )
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    cases = []
    for entry in manifest:
        npz = np.load(os.path.join(FIX_DIR, entry["npz"]))
        cases.append({
            "id": entry["id"],
            "task": entry["task"],
            "rate": entry["rate"],
            "budget": int(entry["budget"]),
            "W": npz["W"],
            "a_q": npz["a_q"],
        })
    return cases


@pytest.fixture(scope="module")
def cases():
    return _load_cases()


# --------------------------------------------------------------------------
# Phase 1: the plain spectral cut drops the top-attention sentences
# --------------------------------------------------------------------------

def test_spectral_cut_drops_top_attention(cases, capsys):
    coverages = []
    print("\n=== Phase 1: plain spectral_cut top-attention coverage ===")
    print(f"{'task':16s} {'K':>4s} {'B':>4s} {'m':>3s} {'cov(top-m)':>11s} {'ncut_spec':>10s}")
    for c in cases:
        W, a_q, B = c["W"], c["a_q"], c["budget"]
        m = _anchor_m(B)
        top = _top_m(a_q, m)
        V_spec, ncut_spec = spectral_cut(W, a_q, budget=B, alpha=1.0)
        cov = _coverage(V_spec, top)
        coverages.append(cov)
        print(f"{c['task']:16s} {W.shape[0]:4d} {B:4d} {m:3d} {cov:11.2f} {ncut_spec:10.3f}")

    mean_cov = float(np.mean(coverages))
    n_incomplete = sum(1 for cov in coverages if cov < 1.0)
    print(f"mean top-m coverage = {mean_cov:.2f}; {n_incomplete}/{len(cases)} examples miss >=1 top sentence")

    # The whole premise: the structural cut drops answer-bearing sentences.
    assert n_incomplete >= 4, "expected >=4/5 examples to drop a top-attention sentence"
    assert mean_cov < 0.6, f"expected low coverage, got {mean_cov:.2f}"


# --------------------------------------------------------------------------
# Phase 2: alpha sweep on query_anchored_spectral_cut (option B)
# --------------------------------------------------------------------------

def test_query_anchored_alpha_sweep(cases, capsys):
    print("\n=== Phase 2: query_anchored_spectral_cut top-m coverage vs alpha ===")
    header = f"{'task':16s} " + " ".join(f"a={a:g}".rjust(7) for a in ALPHA_SWEEP)
    print(header)

    # rows[case_idx][alpha_idx] = coverage
    rows = []
    for c in cases:
        W, a_q, B = c["W"], c["a_q"], c["budget"]
        m = _anchor_m(B)
        top = _top_m(a_q, m)
        row = []
        for alpha in ALPHA_SWEEP:
            V, _ = query_anchored_spectral_cut(W, a_q, budget=B, alpha=alpha)
            row.append(_coverage(V, top))
        rows.append(row)
        print(f"{c['task']:16s} " + " ".join(f"{v:7.2f}" for v in row))

    arr = np.array(rows)  # [n_cases, n_alpha]
    print("\nper-alpha mean coverage across the 5 examples:")
    for j, alpha in enumerate(ALPHA_SWEEP):
        full = int((arr[:, j] >= 0.999).sum())
        print(f"  alpha={alpha:6g}: mean={arr[:, j].mean():.2f}  full-coverage={full}/{len(cases)}")

    # "Consistent across all 5?" — find any alpha that guarantees every example.
    alpha_guarantees_all = [
        ALPHA_SWEEP[j] for j in range(len(ALPHA_SWEEP))
        if bool((arr[:, j] >= 0.999).all())
    ]
    # At the eval's alpha=1.0, coverage is not full for every example.
    j1 = ALPHA_SWEEP.index(1.0)
    assert not bool((arr[:, j1] >= 0.999).all()), \
        "alpha=1.0 unexpectedly already guarantees top-m inclusion on all 5"
    print(f"\nalphas giving full coverage on ALL 5 examples: {alpha_guarantees_all or 'NONE in sweep'}")
    print("=> option B is a soft bias, not a guarantee; motivates option D.")


# --------------------------------------------------------------------------
# Phase 3: option D guarantees top-attention inclusion
# --------------------------------------------------------------------------

def test_anchored_cut_guarantees_top_attention(cases, capsys):
    print("\n=== Phase 3: anchored_spectral_cut (D) - guaranteed inclusion ===")
    print(f"{'task':16s} {'m':>3s} {'cov_D':>6s} {'ncut_spec':>10s} {'ncut_D':>8s} {'ncut_top':>9s}")
    for c in cases:
        W, a_q, B = c["W"], c["a_q"], c["budget"]
        m = _anchor_m(B)
        top = _top_m(a_q, m)

        V_spec, ncut_spec = spectral_cut(W, a_q, budget=B, alpha=1.0)
        V_d, ncut_d = anchored_spectral_cut(
            W, a_q, budget=B, alpha=1.0, anchor_frac=ANCHOR_FRAC, refine=True
        )
        # NCut_q of the naive top-B-by-attention cut, for context.
        V_top_b = set(int(i) for i in np.argsort(-a_q)[:B])
        ncut_top = ncut_q(W, a_q, V_top_b, alpha=1.0)

        cov_d = _coverage(V_d, top)
        print(f"{c['task']:16s} {m:3d} {cov_d:6.2f} {ncut_spec:10.3f} {ncut_d:8.3f} {ncut_top:9.3f}")

        # The guarantee: every pinned top-m sentence is present, budget exact.
        assert top.issubset(V_d), f"{c['id']}: D failed to include all top-{m} attention sentences"
        assert len(V_d) == B, f"{c['id']}: |V_D|={len(V_d)} != budget {B}"
        assert cov_d == 1.0


if __name__ == "__main__":
    # Standalone run with full prints (pytest swallows stdout unless -s).
    cs = _load_cases()

    class _Cap:  # no-op shim so the test fns run outside pytest
        pass

    test_spectral_cut_drops_top_attention(cs, _Cap())
    test_query_anchored_alpha_sweep(cs, _Cap())
    test_anchored_cut_guarantees_top_attention(cs, _Cap())
    print("\nAll phases passed.")
