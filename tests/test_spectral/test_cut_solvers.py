"""Unit tests for the H2 cut solvers."""

from __future__ import annotations

import numpy as np
import pytest

from spectral.cut_solvers import (
    exact_min_ncut,
    fiedler_vector,
    local_search,
    query_anchored_spectral_cut,
    random_cut,
    spectral_cut,
    top_query_attention_cut,
)
from spectral.ncut import ncut_q


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blocks():
    """K=6, two tight 3-blocks with a bridge — same as test_ncut.py."""
    W = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            if i != j:
                W[i, j] = 1.0
                W[i + 3, j + 3] = 1.0
    W[2, 3] = W[3, 2] = 0.1
    return W


@pytest.fixture
def two_blocks_with_query(two_blocks):
    """Same graph; query attends to the second block."""
    a_q = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return two_blocks, a_q


# ---------------------------------------------------------------------------
# fiedler_vector
# ---------------------------------------------------------------------------

def test_fiedler_vector_separates_blocks(two_blocks):
    """The Fiedler vector should change sign between the two blocks."""
    fv = fiedler_vector(two_blocks, normalized=True)
    # Within-block signs should be consistent, across-block they should flip.
    sign_block_a = np.sign(fv[:3])
    sign_block_b = np.sign(fv[3:])
    assert np.all(sign_block_a == sign_block_a[0])
    assert np.all(sign_block_b == sign_block_b[0])
    assert sign_block_a[0] != sign_block_b[0]


# ---------------------------------------------------------------------------
# spectral_cut (plain Fiedler, no query)
# ---------------------------------------------------------------------------

def test_spectral_cut_recovers_block_partition(two_blocks):
    """At budget 3, the sweep cut along Fiedler should find {0,1,2} or {3,4,5}."""
    a_q = np.zeros(6)
    V, val = spectral_cut(two_blocks, a_q, budget=3, alpha=0.0)
    assert V in (frozenset({0, 1, 2}), frozenset({3, 4, 5}))


def test_spectral_cut_returns_finite_ncut(two_blocks):
    a_q = np.zeros(6)
    _, val = spectral_cut(two_blocks, a_q, budget=3, alpha=0.0)
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# query_anchored_spectral_cut
# ---------------------------------------------------------------------------

def test_query_anchored_pulls_toward_query_block(two_blocks_with_query):
    W, a_q = two_blocks_with_query
    V, _ = query_anchored_spectral_cut(W, a_q, budget=3, alpha=10.0)
    # With strong query weight, the query-attended block wins.
    assert V == frozenset({3, 4, 5})


# ---------------------------------------------------------------------------
# local_search
# ---------------------------------------------------------------------------

def test_local_search_improves_or_holds(two_blocks_with_query):
    """A bad seed should be improved by local search (lower or equal NCut)."""
    W, a_q = two_blocks_with_query
    bad_seed = [0, 1, 2]  # zero query mass
    init_val = ncut_q(W, a_q, bad_seed, alpha=1.0)
    V_final, final_val = local_search(W, a_q, bad_seed, alpha=1.0)
    assert final_val <= init_val + 1e-12


def test_local_search_preserves_cardinality(two_blocks_with_query):
    W, a_q = two_blocks_with_query
    V_final, _ = local_search(W, a_q, [0, 1, 2], alpha=1.0)
    assert len(V_final) == 3


def test_local_search_stops_at_local_minimum(two_blocks):
    """Starting from a local minimum, no swap should improve."""
    a_q = np.zeros(6)
    V_final, val = local_search(W=two_blocks, a_q=a_q, V_init=[0, 1, 2], alpha=0.0)
    # Already at the natural block partition (or equivalent), shouldn't move.
    assert V_final == frozenset({0, 1, 2})


# ---------------------------------------------------------------------------
# exact_min_ncut
# ---------------------------------------------------------------------------

def test_exact_min_ncut_finds_global_min(two_blocks):
    a_q = np.zeros(6)
    V_star, val_star = exact_min_ncut(two_blocks, a_q, budget=3, alpha=0.0)
    assert V_star in (frozenset({0, 1, 2}), frozenset({3, 4, 5}))
    # No subset of size 3 should beat the global min.
    import itertools
    for combo in itertools.combinations(range(6), 3):
        assert ncut_q(two_blocks, a_q, combo, alpha=0.0) >= val_star - 1e-12


def test_exact_min_ncut_respects_subset_cap():
    """Too many subsets -> returns (None, None) for the caller to fall back."""
    W = np.eye(50)  # 50-node graph, C(50,25) >> 500_000
    a_q = np.zeros(50)
    V, val = exact_min_ncut(W, a_q, budget=25, alpha=0.0, max_subsets=10)
    assert V is None and val is None


def test_exact_no_better_than_local_for_two_blocks(two_blocks_with_query):
    """Local search from Sentinel-like seed should match exact on this toy."""
    W, a_q = two_blocks_with_query
    V_star, val_star = exact_min_ncut(W, a_q, budget=3, alpha=1.0)
    V_loc, val_loc = local_search(W, a_q, [3, 4, 5], alpha=1.0)
    assert val_loc == pytest.approx(val_star, abs=1e-9)


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------

def test_top_query_attention_cut_picks_largest():
    a_q = np.array([0.1, 0.5, 0.3, 0.05, 0.9])
    V = top_query_attention_cut(a_q, budget=2)
    assert V == frozenset({1, 4})


def test_random_cut_correct_cardinality():
    rng = np.random.default_rng(42)
    V = random_cut(K=10, budget=4, rng=rng)
    assert len(V) == 4
    assert all(0 <= i < 10 for i in V)


def test_random_cut_reproducible():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    assert random_cut(20, 5, rng1) == random_cut(20, 5, rng2)
