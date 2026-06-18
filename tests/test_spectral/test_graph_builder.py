"""Unit tests for spectral.graph_builder.

Toy attention tensors with hand-checkable expected outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral.graph_builder import (
    aggregate_attention,
    build_sentence_graph,
    final_token_attention,
    pool_sentence_graph,
    query_attention_per_sentence,
    sentence_membership,
    sparsify,
    symmetrize,
)


# ---------------------------------------------------------------------------
# Fixtures: a tiny 4-token, 2-layer, 2-head attention tensor with known structure.
# Row-stochastic per (l, h) so it really looks like post-softmax attention.
# ---------------------------------------------------------------------------

def _row_stochastic(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.uniform(0.0, 1.0, size=(n, n))
    return a / a.sum(axis=1, keepdims=True)


@pytest.fixture
def attn_4x4() -> np.ndarray:
    """[L=2, H=2, N=4, N=4] random row-stochastic tensor."""
    rng = np.random.default_rng(0)
    attn = np.stack(
        [
            np.stack([_row_stochastic(rng, 4) for _ in range(2)], axis=0)
            for _ in range(2)
        ],
        axis=0,
    )
    return attn  # shape (2, 2, 4, 4)


# ---------------------------------------------------------------------------
# aggregate_attention
# ---------------------------------------------------------------------------

def test_aggregate_mean_matches_manual(attn_4x4):
    out = aggregate_attention(attn_4x4, method="mean")
    expected = attn_4x4.mean(axis=(0, 1))
    assert out.shape == (4, 4)
    np.testing.assert_allclose(out, expected)


def test_aggregate_mean_rows_sum_to_one(attn_4x4):
    # Row-stochastic in, row-stochastic out (mean preserves it).
    out = aggregate_attention(attn_4x4, method="mean")
    np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-6)


def test_aggregate_rollout_includes_identity_skip(attn_4x4):
    out = aggregate_attention(attn_4x4, method="rollout")
    # Rollout starts from I and multiplies tilde_A = 0.5*(head_avg + I) for each layer.
    # With 2 layers, the diagonal should be strictly positive because every tilde
    # carries an identity skip.
    assert out.shape == (4, 4)
    assert (np.diag(out) > 0).all()


def test_aggregate_rollout_manual_one_layer():
    # Single layer, single head, identity attention -> tilde = 0.5*(I+I) = I -> rollout = I.
    attn = np.eye(3)[None, None, :, :].astype(np.float64)
    out = aggregate_attention(attn, method="rollout")
    np.testing.assert_allclose(out, np.eye(3))


def test_aggregate_final_token_rollout_only_last_row(attn_4x4):
    out = aggregate_attention(attn_4x4, method="final_token_rollout", final_token_idx=3)
    # All rows except the final-token row must be zero.
    assert np.all(out[:3, :] == 0)
    assert np.any(out[3, :] > 0)


def test_aggregate_rejects_wrong_shape():
    with pytest.raises(ValueError):
        aggregate_attention(np.zeros((4, 4)))


def test_aggregate_unknown_method(attn_4x4):
    with pytest.raises(ValueError):
        aggregate_attention(attn_4x4, method="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# final_token_attention
# ---------------------------------------------------------------------------

def test_final_token_attention_shape_and_default_idx(attn_4x4):
    q = final_token_attention(attn_4x4)
    assert q.shape == (4,)
    # default is last token
    q_explicit = final_token_attention(attn_4x4, final_token_idx=3)
    np.testing.assert_allclose(q, q_explicit)


def test_final_token_attention_layer_reduce_modes(attn_4x4):
    q_mean = final_token_attention(attn_4x4, layer_reduce="mean")
    q_last = final_token_attention(attn_4x4, layer_reduce="last")
    # last-layer only differs from full mean unless all layers equal — they don't here
    assert not np.allclose(q_mean, q_last)


# ---------------------------------------------------------------------------
# sentence_membership
# ---------------------------------------------------------------------------

def test_sentence_membership_disjoint():
    M = sentence_membership([(0, 2), (2, 4), (4, 6)], n_tokens=6)
    assert M.shape == (6, 3)
    # Each token belongs to exactly one sentence.
    np.testing.assert_array_equal(M.sum(axis=1), np.ones(6))
    # Column sums equal sentence sizes.
    np.testing.assert_array_equal(M.sum(axis=0), np.array([2, 2, 2]))


def test_sentence_membership_partial_coverage():
    # Sentence covers a subset of tokens; uncovered tokens are zero rows.
    M = sentence_membership([(1, 3)], n_tokens=5)
    assert M.shape == (5, 1)
    np.testing.assert_array_equal(M[:, 0], np.array([0, 1, 1, 0, 0]))


def test_sentence_membership_invalid_span():
    with pytest.raises(ValueError):
        sentence_membership([(2, 1)], n_tokens=5)
    with pytest.raises(ValueError):
        sentence_membership([(0, 6)], n_tokens=5)


# ---------------------------------------------------------------------------
# pool_sentence_graph
# ---------------------------------------------------------------------------

def test_pool_sum_matches_quadratic_form():
    W_tok = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 1, 1],
            [2, 1, 0, 0],
            [3, 1, 0, 0],
        ],
        dtype=np.float64,
    )
    # Two sentences: tokens {0,1} and {2,3}.
    M = sentence_membership([(0, 2), (2, 4)], n_tokens=4)
    W = pool_sentence_graph(M, W_tok, method="sum", zero_diag=False)
    # By hand: W[0,1] = sum over i in {0,1}, j in {2,3} of W_tok[i,j]
    #        = (2 + 3) + (1 + 1) = 7
    assert W[0, 1] == 7
    assert W[1, 0] == 7  # symmetric input
    # Diagonal: W[0,0] = sum over {0,1}^2 = 0+1+1+0 = 2
    assert W[0, 0] == 2


def test_pool_mean_removes_size_bias():
    # If sentence 0 is twice as long as sentence 1 but tokens have identical
    # attention behaviour, mean pooling should NOT systematically favour
    # the longer sentence.
    n = 6
    W_tok = np.ones((n, n)) - np.eye(n)  # 1 everywhere off-diagonal
    M = sentence_membership([(0, 4), (4, 6)], n_tokens=n)
    W_sum = pool_sentence_graph(M, W_tok, method="sum", zero_diag=True)
    W_mean = pool_sentence_graph(M, W_tok, method="mean", zero_diag=True)
    # Sum: W_sum[0,1] = 4*2 = 8, W_sum[1,0] = 2*4 = 8 (same here, but uneven for asymmetric)
    assert W_sum[0, 1] == 8
    # Mean: row-normalized by |S_k| of the SOURCE sentence (k = row).
    # Row 0 divides by |S_0|=4: 8/4 = 2. Row 1 divides by |S_1|=2: 8/2 = 4.
    np.testing.assert_allclose(W_mean[0, 1], 2.0)
    np.testing.assert_allclose(W_mean[1, 0], 4.0)


def test_pool_zero_diag_on_by_default():
    W_tok = np.ones((4, 4))
    M = sentence_membership([(0, 2), (2, 4)], n_tokens=4)
    W = pool_sentence_graph(M, W_tok)
    assert np.all(np.diag(W) == 0)


def test_pool_shape_mismatch():
    with pytest.raises(ValueError):
        pool_sentence_graph(np.zeros((3, 2)), np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# symmetrize & sparsify
# ---------------------------------------------------------------------------

def test_symmetrize_idempotent():
    W = np.array([[0, 2, 1], [4, 0, 3], [1, 5, 0]], dtype=np.float64)
    S = symmetrize(W)
    np.testing.assert_allclose(S, S.T)
    # Twice yields the same matrix.
    np.testing.assert_allclose(S, symmetrize(S))


def test_symmetrize_requires_square():
    with pytest.raises(ValueError):
        symmetrize(np.zeros((3, 4)))


def test_sparsify_knn_keeps_top_k_per_row():
    W = np.array(
        [
            [0.0, 0.9, 0.1, 0.5, 0.2],
            [0.9, 0.0, 0.8, 0.1, 0.0],
            [0.1, 0.8, 0.0, 0.3, 0.4],
            [0.5, 0.1, 0.3, 0.0, 0.7],
            [0.2, 0.0, 0.4, 0.7, 0.0],
        ]
    )
    out = sparsify(W, method="knn", k=2, symmetric=False)
    # Each row has at most 2 non-zero entries (ties may exist but here weights are unique).
    nnz_per_row = (out > 0).sum(axis=1)
    assert (nnz_per_row <= 2).all()
    assert (nnz_per_row >= 1).all()
    # Diagonal must be zero.
    assert np.all(np.diag(out) == 0)


def test_sparsify_knn_symmetric_input_stays_symmetric():
    # The intended usage in the pipeline is symmetrize -> sparsify. With a
    # symmetric input, the output must also be symmetric.
    W = symmetrize(np.array(
        [
            [0.0, 1.0, 0.1, 0.3],
            [0.5, 0.0, 0.9, 0.2],
            [0.0, 0.6, 0.0, 0.4],
            [0.7, 0.1, 0.8, 0.0],
        ]
    ))
    out = sparsify(W, method="knn", k=2, symmetric=True)
    np.testing.assert_allclose(out, out.T)
    # Sanity: each row keeps at LEAST k entries (more if the union added some).
    assert ((out > 0).sum(axis=1) >= 2).all()


def test_sparsify_knn_large_k_returns_input_with_zero_diag():
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = sparsify(W, method="knn", k=5)
    np.testing.assert_allclose(np.diag(out), 0.0)


def test_sparsify_threshold():
    W = np.array(
        [
            [0.0, 1.0, 0.1],
            [1.0, 0.0, 0.2],
            [0.1, 0.2, 0.0],
        ]
    )
    out = sparsify(W, method="threshold", tau=0.5, symmetric=True)
    # Max is 1.0, cutoff = 0.5. Only (0,1) and (1,0) survive.
    assert out[0, 1] == 1.0
    assert out[1, 0] == 1.0
    assert out[0, 2] == 0.0
    assert out[1, 2] == 0.0


def test_sparsify_unknown_method():
    with pytest.raises(ValueError):
        sparsify(np.eye(3), method="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# query_attention_per_sentence
# ---------------------------------------------------------------------------

def test_query_attention_per_sentence_mean_vs_sum():
    q_tok = np.array([1.0, 1.0, 2.0, 2.0, 4.0])
    M = sentence_membership([(0, 2), (2, 5)], n_tokens=5)
    q_sum = query_attention_per_sentence(q_tok, M, method="sum")
    q_mean = query_attention_per_sentence(q_tok, M, method="mean")
    np.testing.assert_allclose(q_sum, np.array([2.0, 8.0]))
    np.testing.assert_allclose(q_mean, np.array([1.0, 8.0 / 3.0]))


def test_query_attention_per_sentence_shape_mismatch():
    with pytest.raises(ValueError):
        query_attention_per_sentence(np.zeros(3), np.zeros((4, 2)))


# ---------------------------------------------------------------------------
# build_sentence_graph (end-to-end)
# ---------------------------------------------------------------------------

def test_build_sentence_graph_smoke(attn_4x4):
    # 4 tokens partitioned into 2 sentences of size 2.
    W, q, meta = build_sentence_graph(
        attn_4x4,
        sentence_spans=[(0, 2), (2, 4)],
        aggregation="mean",
        pooling="mean",
        sparsify_method="knn",
        k=1,
        return_intermediate=True,
    )
    assert W.shape == (2, 2)
    assert q.shape == (2,)
    np.testing.assert_allclose(W, W.T)
    # Sparsified diagonal must be zero.
    np.testing.assert_allclose(np.diag(W), 0.0)
    # Intermediate exposes token-level objects.
    assert meta["W_tok"].shape == (4, 4)
    assert meta["q_tok"].shape == (4,)
    assert meta["M"].shape == (4, 2)


def test_build_sentence_graph_known_community_structure():
    """A block-diagonal token attention should yield a near-disconnected
    sentence graph when the sentence partition aligns with the blocks."""
    # 6 tokens, two clear blocks {0,1,2} and {3,4,5}.
    block = np.full((3, 3), 0.3)
    np.fill_diagonal(block, 0.0)
    off = np.full((3, 3), 0.01)
    W_tok = np.block([[block, off], [off, block]])
    # Make it look like a single (L=1, H=1) attention.
    attn = W_tok[None, None, :, :]
    # Row-normalize so it's stochastic.
    row_sums = attn.sum(axis=-1, keepdims=True)
    attn = attn / np.where(row_sums > 0, row_sums, 1.0)

    W_sent, q_sent = build_sentence_graph(
        attn,
        sentence_spans=[(0, 3), (3, 6)],
        aggregation="mean",
        pooling="mean",
        sparsify_method="none",
    )
    # Cross-block weight must be much smaller than within-block self-weight
    # would have been if not zeroed. Since diagonal is zeroed, we just check
    # cross-block is small in absolute terms.
    assert W_sent[0, 1] < 0.05
    assert W_sent[0, 1] == W_sent[1, 0]
