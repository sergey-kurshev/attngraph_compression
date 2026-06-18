"""Unit tests for spectral.h1_test and spectral.laplacian.

Hand-crafted graphs with known spectra and an identical-graph sanity baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral.h1_test import (
    align_sentences_by_text,
    davis_kahan_angle,
    edge_rank_corr,
    effective_resistance_drift,
    effective_resistance_matrix,
    laplacian_spectrum,
    spectral_entropy,
    spectrum_distance,
    run_subgraph_hypothesis,
    top_edge_recall,
)
from spectral.laplacian import (
    degree_vector,
    eigh_laplacian,
    induced_subgraph,
    laplacian,
    normalized_laplacian,
    pseudoinverse,
)


# ---------------------------------------------------------------------------
# Laplacian basics
# ---------------------------------------------------------------------------

def _path_graph(n: int) -> np.ndarray:
    """Weighted path graph: 1-2-3-...-n with unit edges."""
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    return W


def _disconnected_two_blocks(n_per_block: int = 3) -> np.ndarray:
    """Two disconnected complete graphs of size n_per_block each."""
    n = 2 * n_per_block
    W = np.zeros((n, n))
    for block_start in (0, n_per_block):
        for i in range(block_start, block_start + n_per_block):
            for j in range(block_start, block_start + n_per_block):
                if i != j:
                    W[i, j] = 1.0
    return W


def test_degree_vector():
    W = _path_graph(4)
    d = degree_vector(W)
    np.testing.assert_array_equal(d, np.array([1, 2, 2, 1]))


def test_combinatorial_laplacian_path_graph():
    W = _path_graph(3)
    L = laplacian(W)
    expected = np.array(
        [
            [1, -1, 0],
            [-1, 2, -1],
            [0, -1, 1],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(L, expected)


def test_laplacian_has_zero_eigenvalue_connected():
    W = _path_graph(5)
    w, _ = eigh_laplacian(laplacian(W))
    # Smallest eigenvalue must be ~0 (constant null vector).
    np.testing.assert_allclose(w[0], 0.0, atol=1e-10)
    # Second smallest > 0 (connected).
    assert w[1] > 1e-6


def test_laplacian_multiplicity_equals_components():
    W = _disconnected_two_blocks(3)
    w, _ = eigh_laplacian(laplacian(W))
    # Two zero eigenvalues -> two connected components.
    np.testing.assert_allclose(w[:2], 0.0, atol=1e-10)
    assert w[2] > 1e-6


def test_normalized_laplacian_eigenvalues_in_unit_interval():
    rng = np.random.default_rng(7)
    A = rng.uniform(0, 1, size=(6, 6))
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)
    w, _ = eigh_laplacian(normalized_laplacian(W))
    # Normalized Laplacian eigenvalues are in [0, 2] (bipartite graphs achieve 2).
    assert (w >= -1e-8).all() and (w <= 2 + 1e-8).all()


def test_induced_subgraph_correct():
    W = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    S = induced_subgraph(W, [0, 2])
    np.testing.assert_array_equal(S, np.array([[0, 2], [2, 0]]))


def test_pseudoinverse_recovers_for_invertible_part():
    W = _path_graph(4)
    L = laplacian(W)
    Lp = pseudoinverse(L)
    # L · L+ · L == L  (defining property)
    np.testing.assert_allclose(L @ Lp @ L, L, atol=1e-8)


# ---------------------------------------------------------------------------
# spectrum_distance / davis_kahan / entropy
# ---------------------------------------------------------------------------

def test_spectrum_distance_identical_is_zero():
    W = _path_graph(5)
    assert spectrum_distance(W, W, metric="l2") == pytest.approx(0.0, abs=1e-10)
    assert spectrum_distance(W, W, metric="wasserstein") == pytest.approx(0.0, abs=1e-10)


def test_spectrum_distance_unequal_size_l2_pads():
    W1 = _path_graph(5)
    W2 = _path_graph(4)
    d = spectrum_distance(W1, W2, metric="l2")
    # Result must be finite and non-negative.
    assert d >= 0 and np.isfinite(d)


def test_spectrum_distance_unequal_size_wasserstein():
    W1 = _path_graph(5)
    W2 = _path_graph(4)
    d = spectrum_distance(W1, W2, metric="wasserstein")
    assert d >= 0 and np.isfinite(d)


def test_spectrum_distance_top_k():
    W = _path_graph(6)
    # Same graph compared to itself, restricted to bottom-3, is still 0.
    assert spectrum_distance(W, W, metric="l2", k=3) == pytest.approx(0.0, abs=1e-10)


def test_davis_kahan_identical_is_zero():
    W = _path_graph(6)
    angle = davis_kahan_angle(W, W, k=3)
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_davis_kahan_perturbation_small():
    W = _path_graph(6)
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.001, W.shape)
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)
    angle = davis_kahan_angle(W, W + noise, k=3)
    # Tiny perturbation -> small angle.
    assert angle < 10.0  # degrees


def test_davis_kahan_orthogonal_graphs_high():
    # Two graphs whose bottom eigenspaces are very different.
    W1 = _disconnected_two_blocks(3)             # block structure {0,1,2}|{3,4,5}
    # Swap block membership: connect 0-3 strongly, break original blocks
    W2 = np.zeros_like(W1)
    pairs = [(0, 3), (1, 4), (2, 5), (0, 4), (1, 5)]
    for i, j in pairs:
        W2[i, j] = 1.0
        W2[j, i] = 1.0
    angle = davis_kahan_angle(W1, W2, k=2)
    # Communities are differently aligned -> angle should be substantial.
    assert angle > 20.0


def test_davis_kahan_shape_mismatch_raises():
    with pytest.raises(ValueError):
        davis_kahan_angle(_path_graph(4), _path_graph(5), k=2)


def test_spectral_entropy_non_negative():
    W = _path_graph(5)
    H = spectral_entropy(W)
    assert H >= 0


def test_spectral_entropy_uniform_spectrum_is_log_n():
    # If all eigenvalues are equal, entropy = log(N).
    # Construct a graph whose normalized Laplacian has all eigenvalues equal:
    # the complete graph K_n with uniform weights has L_sym with eigenvalues
    # 0 (mult 1) and N/(N-1) (mult N-1). Skip the zero eigenvalue;
    # remaining ones are uniform -> entropy = log(N-1).
    N = 5
    W = np.ones((N, N)) - np.eye(N)
    H = spectral_entropy(W)
    np.testing.assert_allclose(H, np.log(N - 1), atol=1e-6)


# ---------------------------------------------------------------------------
# Edge-level metrics
# ---------------------------------------------------------------------------

def test_edge_rank_corr_identical_is_one():
    W = _path_graph(5) + 0.5 * np.eye(5)  # arbitrary, just to have varied edges
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    rho = edge_rank_corr(W, W)
    assert rho == pytest.approx(1.0, abs=1e-10)


def test_edge_rank_corr_negated_is_minus_one():
    rng = np.random.default_rng(1)
    W = rng.uniform(0, 1, (5, 5))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    # Note: -W has all-negative off-diag; the RANK is reversed.
    rho = edge_rank_corr(W, -W)
    assert rho == pytest.approx(-1.0, abs=1e-10)


def test_edge_rank_corr_constant_is_nan():
    W1 = np.zeros((4, 4))
    rng = np.random.default_rng(2)
    W2 = rng.uniform(0, 1, (4, 4))
    W2 = 0.5 * (W2 + W2.T)
    np.fill_diagonal(W2, 0.0)
    rho = edge_rank_corr(W1, W2)
    assert np.isnan(rho)


def test_top_edge_recall_identical_is_one():
    rng = np.random.default_rng(3)
    W = rng.uniform(0, 1, (6, 6))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    r = top_edge_recall(W, W, k=5)
    assert r == 1.0


def test_top_edge_recall_disjoint_top_zero():
    # W1: highest edges in (0,1), (2,3); W2: highest in (4,5), (0,2).
    W1 = np.zeros((6, 6))
    W1[0, 1] = W1[1, 0] = 1.0
    W1[2, 3] = W1[3, 2] = 0.9
    W1[4, 5] = W1[5, 4] = 0.05  # tiny

    W2 = np.zeros((6, 6))
    W2[4, 5] = W2[5, 4] = 1.0
    W2[0, 2] = W2[2, 0] = 0.9
    W2[0, 1] = W2[1, 0] = 0.05
    W2[2, 3] = W2[3, 2] = 0.05

    r = top_edge_recall(W1, W2, k=2)
    # Top-2 of W1: {(0,1), (2,3)}; top-2 of W2: {(4,5), (0,2)}; intersection empty.
    assert r == 0.0


# ---------------------------------------------------------------------------
# Effective resistance
# ---------------------------------------------------------------------------

def test_effective_resistance_path_graph_distance():
    # On an unweighted path of length n, effective resistance between adjacent
    # nodes is 1 (unit edge), and between endpoints is n-1 (resistors in series).
    n = 4
    W = _path_graph(n)
    R = effective_resistance_matrix(W)
    np.testing.assert_allclose(np.diag(R), 0.0)
    np.testing.assert_allclose(R[0, 1], 1.0, atol=1e-8)
    np.testing.assert_allclose(R[1, 2], 1.0, atol=1e-8)
    np.testing.assert_allclose(R[0, n - 1], n - 1, atol=1e-8)
    # Symmetric.
    np.testing.assert_allclose(R, R.T, atol=1e-12)


def test_effective_resistance_drift_zero_for_identical_graphs():
    W = _path_graph(5)
    d = effective_resistance_drift(W, W)
    assert d == pytest.approx(0.0, abs=1e-10)


def test_effective_resistance_drift_positive_under_perturbation():
    W = _path_graph(5)
    W2 = W.copy()
    W2[0, 1] = W2[1, 0] = 0.1  # weaken first edge
    d_rel = effective_resistance_drift(W, W2, aggregator="mean_relative")
    d_abs = effective_resistance_drift(W, W2, aggregator="mean_abs")
    assert d_rel > 0 and d_abs > 0


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------

def test_align_sentences_identity():
    s = ["a b c.", "d e f.", "g h i."]
    perm = align_sentences_by_text(s, s)
    assert perm == [0, 1, 2]


def test_align_sentences_reordered():
    induced = ["alpha beta.", "gamma delta.", "epsilon zeta."]
    hat = ["gamma delta.", "alpha beta.", "epsilon zeta."]
    perm = align_sentences_by_text(induced, hat)
    # hat[0] matches induced[1]; hat[1] matches induced[0]; hat[2] matches induced[2]
    assert perm == [1, 0, 2]


def test_align_sentences_cardinality_mismatch_returns_none():
    assert align_sentences_by_text(["a."], ["a.", "b."]) is None


def test_align_sentences_no_match_returns_none():
    assert align_sentences_by_text(["a."], ["q."]) is None


# ---------------------------------------------------------------------------
# Bundled test_subgraph_hypothesis
# ---------------------------------------------------------------------------

def test_subgraph_hypothesis_identical_graphs():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1, (6, 6))
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)
    metrics = run_subgraph_hypothesis(W, W, k_eig=2, k_edges=5)
    assert metrics["spectral_l2"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["spectral_wass"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["davis_kahan_deg"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["edge_spearman"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["edge_recall_at_k"] == 1.0
    assert metrics["eff_resistance_drift"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["spectral_entropy_diff"] == pytest.approx(0.0, abs=1e-8)


def test_subgraph_hypothesis_perturbation_monotone():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1, (6, 6))
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)

    # Two perturbed copies with increasing noise.
    def perturb(scale):
        n = rng.normal(0, scale, W.shape)
        n = 0.5 * (n + n.T)
        np.fill_diagonal(n, 0.0)
        return np.maximum(W + n, 0.0)

    m_small = run_subgraph_hypothesis(W, perturb(0.01), k_eig=2, k_edges=5)
    m_large = run_subgraph_hypothesis(W, perturb(0.5), k_eig=2, k_edges=5)
    assert m_small["spectral_l2"] < m_large["spectral_l2"]
    assert m_small["davis_kahan_deg"] < m_large["davis_kahan_deg"]


def test_subgraph_hypothesis_shape_mismatch_raises():
    with pytest.raises(ValueError):
        run_subgraph_hypothesis(_path_graph(4), _path_graph(5))
