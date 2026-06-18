"""Query-anchored normalized cut objective (IMPLEMENTATION_PLAN.md §5.1).

For a symmetric sentence graph ``W`` with non-negative weights, a
per-sentence query-attention vector ``a_q``, and a kept set ``V' ⊆ V``:

    NCut_q(V') = cut(V', V\\V') / vol(V')   -   alpha * att_q(V') / vol(V')

with

    cut(V', V\\V') = sum_{i in V', j not in V'} W[i, j]
    vol(V')        = sum_{i in V'} deg(i)        (deg = row sum of W)
    att_q(V')      = sum_{i in V'} a_q[i]

The budget is enforced as a hard constraint by the callers (we restrict
candidate ``V'`` to a fixed cardinality); we therefore omit the soft
budget penalty ``beta * 1[|V'| > B]`` from the plan and just refuse to
evaluate sets that violate the budget.

Conventions
-----------
- ``V'`` is passed as an iterable of ints (or a numpy bool mask of size K).
- Empty or full ``V'`` returns ``+inf`` — the cut is ill-defined.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np

Indices = Union[Iterable[int], np.ndarray]


def _to_mask(K: int, indices: Indices) -> np.ndarray:
    """Coerce an indices iterable (or a bool mask of length K) to a bool mask."""
    arr = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices)
    if arr.dtype == bool:
        if arr.shape != (K,):
            raise ValueError(f"bool mask shape {arr.shape} != ({K},)")
        return arr
    mask = np.zeros(K, dtype=bool)
    if arr.size > 0:
        mask[arr.astype(int)] = True
    return mask


def cut_value(W: np.ndarray, indices: Indices) -> float:
    """Sum of edge weights between V' and V\\V'."""
    mask = _to_mask(W.shape[0], indices)
    if not mask.any() or mask.all():
        return 0.0
    return float(W[np.ix_(mask, ~mask)].sum())


def volume(W: np.ndarray, indices: Indices) -> float:
    """Sum of degrees of nodes in V'."""
    mask = _to_mask(W.shape[0], indices)
    deg = W.sum(axis=1)
    return float(deg[mask].sum())


def query_mass(a_q: np.ndarray, indices: Indices) -> float:
    """Sum of query-attention mass on V'."""
    mask = _to_mask(len(a_q), indices)
    return float(a_q[mask].sum())


def ncut_q(
    W: np.ndarray,
    a_q: np.ndarray,
    indices: Indices,
    alpha: float = 1.0,
) -> float:
    """Query-anchored normalized cut.

    Returns ``+inf`` for the degenerate ``|V'| ∈ {0, K}`` or zero-volume cases.
    Smaller is better.
    """
    K = W.shape[0]
    mask = _to_mask(K, indices)
    n_in = int(mask.sum())
    if n_in == 0 or n_in == K:
        return float("inf")
    vol = volume(W, mask)
    if vol <= 1e-12:
        return float("inf")
    cut = cut_value(W, mask)
    att = query_mass(a_q, mask)
    return cut / vol - alpha * (att / vol)


def cheeger_ratio(W: np.ndarray, indices: Indices) -> float:
    """Cheeger / conductance of ``V'``:
    ``h(V') = cut(V', V\\V') / min(vol(V'), vol(V\\V'))``.

    By the Cheeger inequality (§5.2), ``lambda_2 / 2 <= h*(G) <= sqrt(2 lambda_2)``
    so this bounds Sentinel against the optimal min-conductance set without
    solving for it.
    """
    K = W.shape[0]
    mask = _to_mask(K, indices)
    n_in = int(mask.sum())
    if n_in == 0 or n_in == K:
        return float("inf")
    cut = cut_value(W, mask)
    deg = W.sum(axis=1)
    vol_in = deg[mask].sum()
    vol_out = deg[~mask].sum()
    denom = min(vol_in, vol_out)
    if denom <= 1e-12:
        return float("inf")
    return float(cut / denom)


def jaccard(A: Indices, B: Indices) -> float:
    """Jaccard index of two index sets, in [0, 1]."""
    sa = set(int(x) for x in A)
    sb = set(int(x) for x in B)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union)
