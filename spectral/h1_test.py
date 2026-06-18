"""Hypothesis H1: compressed text as subgraph (IMPLEMENTATION_PLAN.md §4).

Given two sentence-level graphs:
- ``G_induced``: the induced subgraph ``G[V']`` where ``G`` is built from the
  source context ``C`` and ``V'`` is Sentinel's kept-sentence set.
- ``G_hat``: the graph built from scratch by re-running the proxy on the
  compressed text ``C'`` alone.

This module reports the metrics from §4.1 (weak) and §4.2 (strict). The
``test_subgraph_hypothesis`` entry point bundles them into a dict.

Note on graph sizes
-------------------
Many metrics (Davis–Kahan, edge correlations, recall@k) require the two
graphs to live in the same ``K``-dimensional space. When sentence
re-segmentation of ``C'`` produces ``K_hat != |V'|``, the caller must
align sentences first; see ``align_sentences_by_text``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import scipy.linalg
import scipy.stats

from spectral.laplacian import (
    eigh_laplacian,
    laplacian,
    normalized_laplacian,
    pseudoinverse,
)


# ---------------------------------------------------------------------------
# Spectrum-based metrics (§4.1)
# ---------------------------------------------------------------------------

def laplacian_spectrum(
    W: np.ndarray,
    normalized: bool = True,
) -> np.ndarray:
    """Ascending eigenvalues of the (normalized) Laplacian."""
    L = normalized_laplacian(W) if normalized else laplacian(W)
    w, _ = eigh_laplacian(L)
    return w


def spectrum_distance(
    W1: np.ndarray,
    W2: np.ndarray,
    metric: str = "l2",
    normalized: bool = True,
    k: Optional[int] = None,
) -> float:
    """Distance between two Laplacian spectra.

    Parameters
    ----------
    metric :
        - "l2": Euclidean distance between the (top-k or all) sorted eigenvalues.
          Vectors are zero-padded to a common length if sizes differ.
        - "wasserstein" / "w1": 1-Wasserstein between the two eigenvalue
          empirical distributions. Naturally handles different lengths via
          ``scipy.stats.wasserstein_distance``.
    k :
        If given, restrict to the smallest-``k`` eigenvalues (the soft community
        structure). Useful when only the bottom of the spectrum is interpretable.
    """
    s1 = laplacian_spectrum(W1, normalized=normalized)
    s2 = laplacian_spectrum(W2, normalized=normalized)
    if k is not None:
        s1 = s1[:k]
        s2 = s2[:k]

    if metric == "l2":
        n = max(len(s1), len(s2))
        a = np.zeros(n)
        b = np.zeros(n)
        a[: len(s1)] = s1
        b[: len(s2)] = s2
        return float(np.linalg.norm(a - b))

    if metric in ("wasserstein", "w1"):
        return float(scipy.stats.wasserstein_distance(s1, s2))

    raise ValueError(f"unknown metric: {metric!r}")


def davis_kahan_angle(
    W1: np.ndarray,
    W2: np.ndarray,
    k: int,
    normalized: bool = True,
    drop_first: bool = True,
    units: str = "degrees",
) -> float:
    """Largest principal angle between bottom-``k`` eigenspaces.

    Per §4.1, this is the *actual* subspace angle, not the Davis–Kahan upper
    bound. Requires ``W1`` and ``W2`` to be the same shape (graphs over the
    same K sentences).

    Parameters
    ----------
    drop_first :
        Skip the trivial constant eigenvector (eigenvalue 0 for connected
        graphs). Recommended: this eigenvector adds noise unrelated to
        community structure.
    units : "degrees" or "radians".
    """
    if W1.shape != W2.shape:
        raise ValueError(
            f"davis_kahan_angle requires same shape: W1={W1.shape}, W2={W2.shape}. "
            "Align sentences first via align_sentences_by_text."
        )
    N = W1.shape[0]
    L1 = normalized_laplacian(W1) if normalized else laplacian(W1)
    L2 = normalized_laplacian(W2) if normalized else laplacian(W2)
    _, V1 = eigh_laplacian(L1)
    _, V2 = eigh_laplacian(L2)

    start = 1 if drop_first else 0
    end = min(start + k, N)
    if end - start < 1:
        raise ValueError(f"k={k} leaves no eigenvectors (N={N}, drop_first={drop_first})")
    U1 = V1[:, start:end]
    U2 = V2[:, start:end]

    angles_rad = scipy.linalg.subspace_angles(U1, U2)
    largest = float(angles_rad.max())
    if units == "radians":
        return largest
    if units == "degrees":
        return float(np.degrees(largest))
    raise ValueError(f"units must be 'degrees' or 'radians', got {units!r}")


def spectral_entropy(
    W: np.ndarray,
    normalized: bool = True,
    eps: float = 1e-12,
) -> float:
    """Spectral / von Neumann entropy of the (normalized) Laplacian.

    ``H = -sum_i lam~_i log lam~_i`` with ``lam~_i = lam_i / sum_j lam_j``.
    Eigenvalues below ``eps`` are skipped (zero contribution).
    """
    s = laplacian_spectrum(W, normalized=normalized)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# Edge-level metrics (§4.2)
# ---------------------------------------------------------------------------

def _upper_triangle_offdiag(W: np.ndarray) -> np.ndarray:
    """Return the strict-upper-triangle off-diagonal entries as a 1-D vector."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got {W.shape}")
    iu = np.triu_indices(W.shape[0], k=1)
    return W[iu]


def edge_rank_corr(W1: np.ndarray, W2: np.ndarray) -> float:
    """Spearman rank correlation of edge weights between two graphs.

    Requires the same shape. Returns NaN if either graph has constant edges
    (no variance for Spearman).
    """
    if W1.shape != W2.shape:
        raise ValueError(f"shape mismatch: {W1.shape} vs {W2.shape}")
    a = _upper_triangle_offdiag(W1)
    b = _upper_triangle_offdiag(W2)
    if a.size < 2:
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    rho, _ = scipy.stats.spearmanr(a, b)
    return float(rho)


def top_edge_recall(W1: np.ndarray, W2: np.ndarray, k: int) -> float:
    """Recall@k: fraction of top-``k`` edges in W1 that also appear in top-``k`` of W2.

    Edges are off-diagonal upper-triangle entries (graphs are symmetric).
    """
    if W1.shape != W2.shape:
        raise ValueError(f"shape mismatch: {W1.shape} vs {W2.shape}")
    a = _upper_triangle_offdiag(W1)
    b = _upper_triangle_offdiag(W2)
    if k <= 0 or a.size == 0:
        return float("nan")
    k = min(k, a.size)
    top_a = set(np.argpartition(-a, kth=k - 1)[:k].tolist())
    top_b = set(np.argpartition(-b, kth=k - 1)[:k].tolist())
    return len(top_a & top_b) / k


# ---------------------------------------------------------------------------
# Effective resistance (§4.2 + §6.2)
# ---------------------------------------------------------------------------

def effective_resistance_matrix(W: np.ndarray) -> np.ndarray:
    """Pairwise effective resistance ``R[i,j] = L^+_{ii} + L^+_{jj} - 2 L^+_{ij}``.

    Diagonal is zero; off-diagonal is non-negative.
    """
    L = laplacian(W)
    Lp = pseudoinverse(L)
    diag = np.diag(Lp)
    R = diag[:, None] + diag[None, :] - 2.0 * Lp
    # Numerical cleanup.
    np.fill_diagonal(R, 0.0)
    R = np.maximum(R, 0.0)
    return R


def effective_resistance_drift(
    W1: np.ndarray,
    W2: np.ndarray,
    aggregator: str = "median_abs",
    eps_frac: float = 1e-3,
) -> float:
    """How much do pairwise resistances change between W1 and W2?

    Aggregators:

    - "median_abs" (default): median of ``|R1 - R2|``. Robust and in the
      same units as resistance. **Use this for real attention graphs** —
      relative metrics blow up on weakly-connected pairs where the
      pseudoinverse-derived resistance is numerically tiny.
    - "mean_abs": mean of ``|R1 - R2|``.
    - "median_relative" / "mean_relative": median or mean of ``|R1-R2|/max(R1, eps)``
      where ``eps = eps_frac * max(R1)``. These are numerically unstable on
      graphs where R values span many orders of magnitude — keep only for
      sanity-checking against the absolute aggregators.
    """
    R1 = effective_resistance_matrix(W1)
    R2 = effective_resistance_matrix(W2)
    iu = np.triu_indices(R1.shape[0], k=1)
    r1 = R1[iu]
    r2 = R2[iu]
    if r1.size == 0:
        return float("nan")
    diff = np.abs(r1 - r2)
    if aggregator == "mean_abs":
        return float(diff.mean())
    if aggregator == "median_abs":
        return float(np.median(diff))
    if aggregator in ("mean_relative", "median_relative"):
        floor = max(eps_frac * float(r1.max()), 1e-12)
        denom = np.maximum(r1, floor)
        ratios = diff / denom
        return float(np.mean(ratios) if aggregator == "mean_relative" else np.median(ratios))
    raise ValueError(f"unknown aggregator: {aggregator!r}")


# ---------------------------------------------------------------------------
# Sentence alignment helper (for the K_induced != K_hat case)
# ---------------------------------------------------------------------------

def align_sentences_by_text(
    sentences_induced: Sequence[str],
    sentences_hat: Sequence[str],
) -> Optional[List[int]]:
    """Greedy 1:1 alignment from ``sentences_hat`` to ``sentences_induced``.

    Returns a permutation ``perm`` such that ``sentences_hat[i]`` corresponds
    to ``sentences_induced[perm[i]]``, or ``None`` if alignment fails
    (different cardinalities or any unmatched sentence after greedy pass).
    The match is exact substring containment (after whitespace normalization);
    we don't reach for fuzzy matching because Sentinel concatenates verbatim.
    """
    if len(sentences_hat) != len(sentences_induced):
        return None
    norm = lambda s: " ".join(s.split())
    induced_norm = [norm(s) for s in sentences_induced]
    used = [False] * len(induced_norm)
    perm: List[int] = []
    for s in sentences_hat:
        sn = norm(s)
        match = -1
        for j, t in enumerate(induced_norm):
            if used[j]:
                continue
            if sn == t or sn in t or t in sn:
                match = j
                break
        if match < 0:
            return None
        used[match] = True
        perm.append(match)
    return perm


# ---------------------------------------------------------------------------
# Bundled entry point (Algorithm 4.3)
# ---------------------------------------------------------------------------

def run_subgraph_hypothesis(
    W_induced: np.ndarray,
    W_hat: np.ndarray,
    k_eig: int = 5,
    k_edges: int = 10,
    normalized: bool = True,
) -> Dict[str, float]:
    """Run the full H1 metric panel between two same-shape sentence graphs.

    Returns a dict mirroring the example in IMPLEMENTATION_PLAN.md §4.3.
    """
    if W_induced.shape != W_hat.shape:
        raise ValueError(
            f"H1 requires aligned graphs of the same shape; got "
            f"{W_induced.shape} vs {W_hat.shape}. Use align_sentences_by_text."
        )

    return {
        "spectral_l2": spectrum_distance(W_induced, W_hat, metric="l2", normalized=normalized),
        "spectral_wass": spectrum_distance(W_induced, W_hat, metric="wasserstein", normalized=normalized),
        "davis_kahan_deg": davis_kahan_angle(W_induced, W_hat, k=k_eig, normalized=normalized),
        "edge_spearman": edge_rank_corr(W_induced, W_hat),
        "edge_recall_at_k": top_edge_recall(W_induced, W_hat, k=k_edges),
        "eff_resistance_drift": effective_resistance_drift(W_induced, W_hat),
        "spectral_entropy_diff": abs(
            spectral_entropy(W_induced, normalized=normalized)
            - spectral_entropy(W_hat, normalized=normalized)
        ),
    }
