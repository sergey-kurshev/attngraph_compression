"""Hypothesis H2: Sentinel's V' is a near-optimal query-anchored cut.

For a fixed example, we compare Sentinel's selection ``V'_Sentinel`` against
five alternatives, all evaluated under the *same* cardinality budget
``B = |V'_Sentinel|``:

- ``V'_exact``     : exhaustive minimizer of NCut_q at budget B (gold standard
                      when ``C(K, B)`` is enumerable).
- ``V'_spec``      : plain Fiedler-vector sweep cut at budget B.
- ``V'_spec_q``    : query-anchored Fiedler-vector sweep cut at budget B.
- ``V'_local``     : single-swap local minimum starting from V'_Sentinel.
- ``V'_top_attn``  : top-B sentences by query attention.
- ``V'_random``    : uniform random size-B subset (averaged over multiple seeds).

The headline figure is the "near-optimality gap"
``ncut_sentinel - ncut_exact`` (when exact is available), normalized by the
random-baseline gap so 0 = perfect, 1 = no better than random.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from spectral.cut_solvers import (
    anchored_spectral_cut,
    exact_min_ncut,
    local_search,
    query_anchored_spectral_cut,
    random_cut,
    spectral_cut,
    top_query_attention_cut,
)
from spectral.laplacian import eigh_laplacian, normalized_laplacian
from spectral.ncut import (
    cheeger_ratio,
    cut_value,
    jaccard,
    ncut_q,
    query_mass,
    volume,
)


def algebraic_connectivity(W: np.ndarray, normalized: bool = True) -> float:
    """The second-smallest eigenvalue of the (normalized) Laplacian, a.k.a. lambda_2."""
    L = normalized_laplacian(W) if normalized else (np.diag(W.sum(1)) - W)
    w, _ = eigh_laplacian(L)
    if len(w) < 2:
        return 0.0
    return float(w[1])


def run_optimal_cut_hypothesis(
    W: np.ndarray,
    a_q: np.ndarray,
    V_sentinel: Iterable[int],
    alpha: float = 1.0,
    n_random_seeds: int = 20,
    exact_max_subsets: int = 500_000,
    random_seed: int = 0,
    local_max_iter: int = 200,
    anchor_frac: float = 0.25,
) -> Dict[str, Any]:
    """Run the full H2 metric panel and return a flat dict suitable for JSONL.

    The cardinality budget is taken as ``len(V_sentinel)`` so every solver
    competes at exactly Sentinel's compression ratio.
    """
    K = W.shape[0]
    V_sent = frozenset(int(i) for i in V_sentinel)
    B = len(V_sent)

    # --- Sentinel ---
    ncut_sent = ncut_q(W, a_q, V_sent, alpha=alpha)
    cut_sent = cut_value(W, V_sent)
    vol_sent = volume(W, V_sent)
    qmass_sent = query_mass(a_q, V_sent)
    cheeger_sent = cheeger_ratio(W, V_sent)

    # --- Exact (gold standard) ---
    V_star, ncut_star = exact_min_ncut(
        W, a_q, budget=B, alpha=alpha, max_subsets=exact_max_subsets
    )
    exact_feasible = V_star is not None

    # --- Spectral (plain & query-anchored) ---
    V_spec, ncut_spec = spectral_cut(W, a_q, budget=B, alpha=alpha)
    V_spec_q, ncut_spec_q = query_anchored_spectral_cut(W, a_q, budget=B, alpha=alpha)

    # --- Anchor-constrained spectral cut (option D): guarantees the top-m
    #     query-attention sentences are kept, then minimizes NCut_q on the rest.
    V_anchored, ncut_anchored = anchored_spectral_cut(
        W, a_q, budget=B, alpha=alpha, anchor_frac=anchor_frac,
        refine=True, max_iter=local_max_iter,
    )
    # Top-m attention set used by the anchor (for coverage diagnostics).
    m_anchor = max(1, min(int(np.ceil(anchor_frac * B)), B))
    top_m = set(int(i) for i in np.argsort(-a_q)[:m_anchor])

    def _cov(V):
        return (len(top_m & set(int(i) for i in V)) / len(top_m)) if top_m else 1.0

    # --- Local search from Sentinel ---
    V_local, ncut_local = local_search(W, a_q, V_sent, alpha=alpha, max_iter=local_max_iter)

    # --- Top-attention baseline ---
    V_top = top_query_attention_cut(a_q, budget=B)
    ncut_top = ncut_q(W, a_q, V_top, alpha=alpha)

    # --- Random baseline (mean over seeds) ---
    rng = np.random.default_rng(random_seed)
    ncut_rand_samples = []
    jaccard_rand_samples = []
    for _ in range(n_random_seeds):
        V_r = random_cut(K, B, rng)
        ncut_rand_samples.append(ncut_q(W, a_q, V_r, alpha=alpha))
        jaccard_rand_samples.append(jaccard(V_sent, V_r))
    ncut_random_mean = float(np.mean(ncut_rand_samples))
    ncut_random_std = float(np.std(ncut_rand_samples))
    jaccard_random_mean = float(np.mean(jaccard_rand_samples))

    # --- Normalized gap: 0 = optimal, 1 = random ---
    # gap = (ncut_sentinel - ncut_star) / (ncut_random_mean - ncut_star)
    normalized_gap: Optional[float] = None
    if exact_feasible:
        denom = ncut_random_mean - ncut_star
        if abs(denom) > 1e-12:
            normalized_gap = float((ncut_sent - ncut_star) / denom)

    return {
        "K": int(K),
        "budget": int(B),
        "alpha": float(alpha),
        # Sentinel diagnostics
        "ncut_sentinel": float(ncut_sent),
        "cut_sentinel": float(cut_sent),
        "vol_sentinel": float(vol_sent),
        "qmass_sentinel": float(qmass_sent),
        "cheeger_sentinel": float(cheeger_sent),
        # Baselines
        "ncut_exact": (float(ncut_star) if exact_feasible else None),
        "ncut_spec": float(ncut_spec),
        "ncut_spec_q": float(ncut_spec_q),
        "ncut_anchored": float(ncut_anchored),
        "ncut_local": float(ncut_local),
        "ncut_top_attn": float(ncut_top),
        "ncut_random_mean": ncut_random_mean,
        "ncut_random_std": ncut_random_std,
        # Agreement (Jaccard with Sentinel)
        "jaccard_exact": (jaccard(V_sent, V_star) if exact_feasible else None),
        "jaccard_spec": jaccard(V_sent, V_spec),
        "jaccard_spec_q": jaccard(V_sent, V_spec_q),
        "jaccard_anchored": jaccard(V_sent, V_anchored),
        "jaccard_local": jaccard(V_sent, V_local),
        "jaccard_top_attn": jaccard(V_sent, V_top),
        # Top-m attention coverage: plain spectral drops these; anchored keeps all.
        "anchor_frac": float(anchor_frac),
        "m_anchor": int(m_anchor),
        "topm_cov_spec": float(_cov(V_spec)),
        "topm_cov_spec_q": float(_cov(V_spec_q)),
        "topm_cov_anchored": float(_cov(V_anchored)),
        "jaccard_random_mean": jaccard_random_mean,
        # Headline near-optimality measure
        "delta_cut": (float(ncut_sent - ncut_star) if exact_feasible else None),
        "normalized_gap": normalized_gap,
        # Graph diagnostics
        "lambda_2": algebraic_connectivity(W, normalized=True),
        # Chosen sets for downstream inspection
        "V_sentinel": sorted(V_sent),
        "V_exact": (sorted(V_star) if exact_feasible else None),
        "V_spec": sorted(V_spec),
        "V_spec_q": sorted(V_spec_q),
        "V_anchored": sorted(V_anchored),
        "V_local": sorted(V_local),
        "V_top_attn": sorted(V_top),
        "exact_feasible": bool(exact_feasible),
    }
