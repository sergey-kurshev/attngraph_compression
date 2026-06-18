"""H2 cut solvers: spectral relaxation, local search, exact enumeration.

All solvers take ``(W, a_q, budget, alpha)`` and return ``(V', ncut_value)``
where ``V'`` is a frozenset of kept-sentence indices and ``ncut_value`` is the
NCut_q on that set (see ``spectral.ncut``). All operate under the *hard*
cardinality constraint ``|V'| == budget``.

Three solvers
-------------
- ``spectral_cut``    : Fiedler vector + sweep at the budget. Continuous
                        relaxation of NCut, no query term — fast and gives a
                        purely structural cut to compare against.
- ``local_search``    : single-swap greedy. Initialized from any V' (Sentinel's
                        selection is the natural seed). Returns a local minimum
                        of NCut_q at the same cardinality.
- ``exact_min_ncut``  : exhaustive enumeration over all ``C(K, B)`` subsets.
                        Only viable for small graphs; we cap subset count for
                        safety. Returns the global minimum at budget ``B``.

The plan's §5.1.3 suggested ILP via pulp/CBC. We bypass that dependency
because our SQuAD subset has ``K <= 20`` so ``C(20, 10) <= 184_756`` — well
within an in-process numpy loop and far cheaper than spinning up a solver.
"""

from __future__ import annotations

import itertools
from math import comb
from typing import FrozenSet, Iterable, Optional, Tuple

import numpy as np

from spectral.laplacian import (
    eigh_laplacian,
    laplacian,
    normalized_laplacian,
)
from spectral.ncut import ncut_q


# ---------------------------------------------------------------------------
# Spectral relaxation (§5.1.1)
# ---------------------------------------------------------------------------

def fiedler_vector(W: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Eigenvector of the second-smallest eigenvalue of the (normalized) Laplacian."""
    L = normalized_laplacian(W) if normalized else laplacian(W)
    w, V = eigh_laplacian(L)
    if V.shape[1] < 2:
        # Single-node graph — return a trivial constant.
        return np.zeros(W.shape[0])
    return V[:, 1]


def _sweep_at_budget(
    scores: np.ndarray,
    W: np.ndarray,
    a_q: np.ndarray,
    budget: int,
    alpha: float,
) -> Tuple[FrozenSet[int], float]:
    """Sort by ``scores`` and try the two budget-sized prefixes (low end / high end)."""
    order = np.argsort(scores)
    candidates = [
        order[:budget].tolist(),
        order[-budget:].tolist(),
    ]
    best_val = float("inf")
    best_set: FrozenSet[int] = frozenset()
    for cand in candidates:
        val = ncut_q(W, a_q, cand, alpha=alpha)
        if val < best_val:
            best_val = val
            best_set = frozenset(int(i) for i in cand)
    return best_set, best_val


def spectral_cut(
    W: np.ndarray,
    a_q: np.ndarray,
    budget: int,
    alpha: float = 1.0,
    normalized: bool = True,
) -> Tuple[FrozenSet[int], float]:
    """Fiedler-vector relaxation + sweep cut at ``|V'| = budget``."""
    fv = fiedler_vector(W, normalized=normalized)
    return _sweep_at_budget(fv, W, a_q, budget=budget, alpha=alpha)


def query_anchored_spectral_cut(
    W: np.ndarray,
    a_q: np.ndarray,
    budget: int,
    alpha: float = 1.0,
    normalized: bool = True,
) -> Tuple[FrozenSet[int], float]:
    """Spectral relaxation with a query-attention prior.

    The plan §5.1.1 calls for solving ``Lx = lambda D x + mu a_q``. Rather than
    chase a clean closed form for ``mu``, we use the principled and equivalent
    practical surrogate: sweep by the *combined* score
    ``score_i = fiedler_i - alpha * a_q[i] / sqrt(deg_i + eps)``.

    The intuition: the Fiedler value scores each node by structural side; the
    query term biases the sweep toward high-a_q nodes. Both are continuous
    relaxations of the NCut_q objective and the sweep finds the best
    cardinality-``budget`` partition along that ordering.
    """
    fv = fiedler_vector(W, normalized=normalized)
    deg = W.sum(axis=1)
    norm = np.sqrt(np.maximum(deg, 1e-12))
    score = fv - alpha * (a_q / norm)
    return _sweep_at_budget(score, W, a_q, budget=budget, alpha=alpha)


# ---------------------------------------------------------------------------
# Local search (§5.1.2)
# ---------------------------------------------------------------------------

def local_search(
    W: np.ndarray,
    a_q: np.ndarray,
    V_init: Iterable[int],
    alpha: float = 1.0,
    max_iter: int = 200,
    fixed: Optional[Iterable[int]] = None,
) -> Tuple[FrozenSet[int], float]:
    """Greedy best-improvement single-swap local search at fixed cardinality.

    Each step tries every ``(i in V', j not in V')`` swap and applies the
    single swap that most decreases NCut_q. Stops at a local minimum or
    ``max_iter``. With ``K <= 20`` the per-iteration cost is ``O(K^2)``
    NCut evaluations which is trivially cheap.

    ``fixed`` is an optional set of indices that must remain in ``V'`` — those
    nodes are never swapped out. Used by ``anchored_spectral_cut`` to pin the
    top-attention sentences while optimizing the remaining budget.
    """
    K = W.shape[0]
    V = set(int(i) for i in V_init)
    not_V = set(range(K)) - V
    locked = set(int(i) for i in fixed) if fixed is not None else set()
    val = ncut_q(W, a_q, V, alpha=alpha)

    for _ in range(max_iter):
        best_swap: Optional[Tuple[int, int]] = None
        best_val = val
        for i in V - locked:           # never swap out a pinned node
            for j in not_V:
                cand = (V - {i}) | {j}
                cv = ncut_q(W, a_q, cand, alpha=alpha)
                if cv < best_val - 1e-12:
                    best_val = cv
                    best_swap = (i, j)
        if best_swap is None:
            break
        i, j = best_swap
        V = (V - {i}) | {j}
        not_V = (not_V - {j}) | {i}
        val = best_val

    return frozenset(V), val


# ---------------------------------------------------------------------------
# Anchor-constrained spectral cut (option D)
# ---------------------------------------------------------------------------

def anchored_spectral_cut(
    W: np.ndarray,
    a_q: np.ndarray,
    budget: int,
    alpha: float = 1.0,
    anchor_frac: float = 0.25,
    n_anchors: Optional[int] = None,
    normalized: bool = True,
    refine: bool = True,
    max_iter: int = 200,
) -> Tuple[FrozenSet[int], float]:
    """Spectral cut that *guarantees* the top query-attention sentences are kept.

    The plain ``spectral_cut`` orders nodes by the Fiedler value alone, so a
    high-``a_q`` sentence sitting in the interior of that 1-D ordering is never
    swept in — exactly the failure mode where the structural cut drops
    answer-bearing sentences.

    This solver instead:

    1. **Pins** the top ``m`` sentences by ``a_q`` (``m = n_anchors`` if given,
       else ``ceil(anchor_frac * budget)``, capped at ``budget``). These are
       guaranteed to appear in ``V'``.
    2. **Fills** the remaining ``budget - m`` slots from the non-anchor nodes by
       a Fiedler sweep (low/high prefix, whichever gives lower ``NCut_q`` on the
       combined set).
    3. Optionally **refines** with anchor-constrained ``local_search`` — swapping
       only the non-pinned slots — to reach a local ``NCut_q`` minimum that still
       respects the inclusion guarantee.

    Returns ``(V', ncut_value)`` with ``|V'| == budget`` and the top-``m``
    attention sentences ⊆ ``V'``.
    """
    K = W.shape[0]
    if budget <= 0 or budget >= K:
        return frozenset(range(K)), ncut_q(W, a_q, range(K), alpha=alpha)

    m = n_anchors if n_anchors is not None else int(np.ceil(anchor_frac * budget))
    m = max(1, min(m, budget))

    order_attn = np.argsort(-a_q)
    anchors = set(int(i) for i in order_attn[:m])
    free_budget = budget - m

    if free_budget == 0:
        V = frozenset(anchors)
        return V, ncut_q(W, a_q, V, alpha=alpha)

    # Fill the remaining slots from non-anchor nodes via a Fiedler sweep.
    fv = fiedler_vector(W, normalized=normalized)
    non_anchor = np.array([i for i in range(K) if i not in anchors], dtype=int)
    order = non_anchor[np.argsort(fv[non_anchor])]

    best_val = float("inf")
    best_set: FrozenSet[int] = frozenset()
    for fill in (order[:free_budget].tolist(), order[-free_budget:].tolist()):
        cand = anchors | set(int(i) for i in fill)
        val = ncut_q(W, a_q, cand, alpha=alpha)
        if val < best_val:
            best_val = val
            best_set = frozenset(cand)

    if refine:
        best_set, best_val = local_search(
            W, a_q, best_set, alpha=alpha, max_iter=max_iter, fixed=anchors
        )

    return best_set, best_val


# ---------------------------------------------------------------------------
# Exact enumeration (§5.1.3, simpler than ILP for small K)
# ---------------------------------------------------------------------------

DEFAULT_EXACT_MAX_SUBSETS = 500_000


def exact_min_ncut(
    W: np.ndarray,
    a_q: np.ndarray,
    budget: int,
    alpha: float = 1.0,
    max_subsets: int = DEFAULT_EXACT_MAX_SUBSETS,
) -> Tuple[Optional[FrozenSet[int]], Optional[float]]:
    """Brute-force minimum NCut_q over all subsets of size ``budget``.

    Returns ``(None, None)`` if ``C(K, budget) > max_subsets`` — the caller
    should fall back to ``local_search`` or ``spectral_cut`` in that case.
    """
    K = W.shape[0]
    if budget <= 0 or budget >= K:
        return None, None
    n = comb(K, budget)
    if n > max_subsets:
        return None, None

    best_val = float("inf")
    best_set: Optional[FrozenSet[int]] = None
    for combo in itertools.combinations(range(K), budget):
        val = ncut_q(W, a_q, combo, alpha=alpha)
        if val < best_val:
            best_val = val
            best_set = frozenset(combo)
    return best_set, best_val


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------

def top_query_attention_cut(
    a_q: np.ndarray,
    budget: int,
) -> FrozenSet[int]:
    """The top-``budget`` sentences by query attention — Sentinel-esque baseline."""
    K = len(a_q)
    if budget <= 0 or budget >= K:
        return frozenset(range(K))
    order = np.argsort(-a_q)
    return frozenset(int(i) for i in order[:budget])


def random_cut(
    K: int,
    budget: int,
    rng: np.random.Generator,
) -> FrozenSet[int]:
    """A uniformly random size-``budget`` subset of ``{0, ..., K-1}``."""
    if budget <= 0 or budget >= K:
        return frozenset(range(K))
    return frozenset(int(i) for i in rng.choice(K, size=budget, replace=False))
