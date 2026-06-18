"""Unit tests for the query-anchored NCut objective."""

from __future__ import annotations

import numpy as np
import pytest

from spectral.ncut import (
    cheeger_ratio,
    cut_value,
    jaccard,
    ncut_q,
    query_mass,
    volume,
)


# ---------------------------------------------------------------------------
# Fixtures: a clean two-block toy graph
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blocks():
    """K=6, two tight 3-node blocks with a single weak bridge.

    Block A: 0,1,2 — all pairwise edge weight 1.0
    Block B: 3,4,5 — all pairwise edge weight 1.0
    Bridge: W[2,3] = W[3,2] = 0.1
    """
    W = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            if i != j:
                W[i, j] = 1.0
                W[i + 3, j + 3] = 1.0
    W[2, 3] = W[3, 2] = 0.1
    return W


# ---------------------------------------------------------------------------
# cut_value / volume / query_mass
# ---------------------------------------------------------------------------

def test_cut_value_two_blocks_perfect_cut(two_blocks):
    # Cutting {0,1,2} from {3,4,5}: only the bridge crosses
    assert cut_value(two_blocks, [0, 1, 2]) == pytest.approx(0.1)


def test_cut_value_empty_and_full_are_zero(two_blocks):
    assert cut_value(two_blocks, []) == 0.0
    assert cut_value(two_blocks, list(range(6))) == 0.0


def test_volume_matches_degree_sum(two_blocks):
    deg = two_blocks.sum(axis=1)
    assert volume(two_blocks, [0, 1, 2]) == pytest.approx(deg[:3].sum())


def test_volume_bool_mask_input(two_blocks):
    mask = np.array([True, True, True, False, False, False])
    deg = two_blocks.sum(axis=1)
    assert volume(two_blocks, mask) == pytest.approx(deg[:3].sum())


def test_query_mass_sums_correctly():
    a_q = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0])
    assert query_mass(a_q, [0, 1, 2]) == pytest.approx(0.6)
    assert query_mass(a_q, [3, 4, 5]) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# ncut_q
# ---------------------------------------------------------------------------

def test_ncut_q_block_cut_is_lower_than_split_cut(two_blocks):
    """Cutting along the bridge should beat cutting through a block."""
    a_q = np.zeros(6)
    block_cut = ncut_q(two_blocks, a_q, [0, 1, 2], alpha=0.0)
    # Same size but mixing the blocks creates many cross-edges:
    mixed_cut = ncut_q(two_blocks, a_q, [0, 1, 3], alpha=0.0)
    assert block_cut < mixed_cut


def test_ncut_q_empty_or_full_returns_inf(two_blocks):
    a_q = np.zeros(6)
    assert ncut_q(two_blocks, a_q, [], alpha=0.0) == float("inf")
    assert ncut_q(two_blocks, a_q, list(range(6)), alpha=0.0) == float("inf")


def test_ncut_q_alpha_pulls_toward_query_mass(two_blocks):
    """With high alpha, the term -alpha * att/vol dominates and high-q_q sets win."""
    a_q = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    nb_unfavored = ncut_q(two_blocks, a_q, [0, 1, 2], alpha=10.0)  # zero query mass
    nb_favored = ncut_q(two_blocks, a_q, [3, 4, 5], alpha=10.0)    # all query mass
    assert nb_favored < nb_unfavored


def test_ncut_q_alpha_zero_is_pure_cut_over_volume(two_blocks):
    a_q = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cut_a0 = ncut_q(two_blocks, a_q, [0, 1, 2], alpha=0.0)
    deg = two_blocks.sum(axis=1)
    expected = 0.1 / deg[:3].sum()
    assert cut_a0 == pytest.approx(expected)


def test_ncut_q_zero_volume_returns_inf():
    W = np.zeros((4, 4))
    a_q = np.zeros(4)
    # All-zero graph -> volume(V') = 0
    assert ncut_q(W, a_q, [0, 1], alpha=0.0) == float("inf")


# ---------------------------------------------------------------------------
# cheeger_ratio
# ---------------------------------------------------------------------------

def test_cheeger_ratio_block_cut_is_small(two_blocks):
    """The natural block cut should have low conductance (small Cheeger)."""
    h = cheeger_ratio(two_blocks, [0, 1, 2])
    # Cut weight = 0.1, min volume = sum of degrees of either block.
    deg = two_blocks.sum(axis=1)
    expected = 0.1 / min(deg[:3].sum(), deg[3:].sum())
    assert h == pytest.approx(expected)


def test_cheeger_ratio_degenerate_returns_inf(two_blocks):
    assert cheeger_ratio(two_blocks, []) == float("inf")
    assert cheeger_ratio(two_blocks, list(range(6))) == float("inf")


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------

def test_jaccard_identical_sets_is_one():
    assert jaccard([1, 2, 3], [3, 2, 1]) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert jaccard([1, 2], [3, 4]) == 0.0


def test_jaccard_partial_overlap():
    # |A ∩ B| = 1, |A ∪ B| = 3
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)


def test_jaccard_empty_with_empty_is_one():
    assert jaccard([], []) == 1.0
