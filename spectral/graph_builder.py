"""Token-level attention -> sentence-level graph (IMPLEMENTATION_PLAN.md §2).

All functions take and return numpy arrays. The proxy model's attention is
extracted elsewhere (see spectral.attention_extraction) and handed in as an
``[L, H, N, N]`` tensor (post-softmax weights).

Conventions
-----------
- ``A[l, h, i, j]`` is the weight token ``i`` places on token ``j``.
- Sentence spans are inclusive-exclusive ``(start, end)`` token indices into
  the same ``N``-token coordinate system the attention was extracted in.
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 2.2  Token-level aggregation
# ---------------------------------------------------------------------------

def aggregate_attention(
    attn_stack: np.ndarray,
    method: Literal["mean", "rollout", "final_token_rollout"] = "mean",
    final_token_idx: int | None = None,
) -> np.ndarray:
    """Reduce ``[L, H, N, N]`` attention to a single ``[N, N]`` weight matrix.

    Parameters
    ----------
    attn_stack : ndarray, shape (L, H, N, N)
        Post-softmax attention weights from every layer and head.
    method :
        - "mean": average over layers and heads.
        - "rollout": Abnar & Zuidema (2020). For each layer mix per-head
          attention with identity and accumulate by matrix product across
          layers.
        - "final_token_rollout": rollout, then keep only the row of the final
          (query-anchored) token propagated backward; useful for constructing
          query-anchored sentence graphs.
    final_token_idx :
        Position of the query-anchored token. Required for
        ``"final_token_rollout"``; ignored otherwise.

    Returns
    -------
    ndarray, shape (N, N)
        Aggregated attention weight matrix.
    """
    if attn_stack.ndim != 4:
        raise ValueError(f"attn_stack must be 4-D [L,H,N,N], got {attn_stack.shape}")

    L, H, N, _ = attn_stack.shape

    if method == "mean":
        return attn_stack.mean(axis=(0, 1))

    if method in ("rollout", "final_token_rollout"):
        # tilde_A^(l) = 0.5 * (mean_h A^(l,h) + I)
        head_avg = attn_stack.mean(axis=1)  # [L, N, N]
        eye = np.eye(N, dtype=head_avg.dtype)
        rollout = eye.copy()
        for l in range(L):
            tilde = 0.5 * (head_avg[l] + eye)
            rollout = tilde @ rollout
        if method == "rollout":
            return rollout
        if final_token_idx is None:
            final_token_idx = N - 1
        # Project to the query-anchored row: keep edges feeding into final_token_idx.
        # The result is a matrix whose only non-zero row is the final-token row;
        # downstream pooling treats this as "directed flow into q".
        out = np.zeros_like(rollout)
        out[final_token_idx, :] = rollout[final_token_idx, :]
        return out

    raise ValueError(f"unknown method: {method!r}")


def final_token_attention(
    attn_stack: np.ndarray,
    final_token_idx: int | None = None,
    layer_reduce: Literal["mean", "last"] = "mean",
) -> np.ndarray:
    """Per-token attention mass placed on context by the final query token.

    This is the signal Sentinel itself consumes. Shape ``[N]``.
    """
    if attn_stack.ndim != 4:
        raise ValueError(f"attn_stack must be 4-D [L,H,N,N], got {attn_stack.shape}")
    L, H, N, _ = attn_stack.shape
    if final_token_idx is None:
        final_token_idx = N - 1

    if layer_reduce == "mean":
        a = attn_stack.mean(axis=(0, 1))  # [N, N]
    elif layer_reduce == "last":
        a = attn_stack[-1].mean(axis=0)  # average heads of last layer
    else:
        raise ValueError(f"layer_reduce must be 'mean' or 'last', got {layer_reduce!r}")
    return a[final_token_idx, :]


# ---------------------------------------------------------------------------
# 2.3  Sentence-level reduction
# ---------------------------------------------------------------------------

def sentence_membership(
    sentence_spans: Sequence[Tuple[int, int]],
    n_tokens: int,
) -> np.ndarray:
    """Build the ``[N, K]`` membership matrix ``M[i,k] = 1`` iff token ``i`` in sentence ``k``.

    Spans are inclusive-exclusive ``(start, end)``. Tokens not covered by any
    span (e.g. prompt scaffolding) contribute zero rows and are silently
    excluded from the sentence reduction — by construction they cannot belong
    to a "sentence" of the source context.
    """
    K = len(sentence_spans)
    M = np.zeros((n_tokens, K), dtype=np.float32)
    for k, (start, end) in enumerate(sentence_spans):
        if start < 0 or end > n_tokens or start >= end:
            raise ValueError(
                f"sentence {k} has invalid span ({start}, {end}) for N={n_tokens}"
            )
        M[start:end, k] = 1.0
    return M


def pool_sentence_graph(
    M: np.ndarray,
    W_tok: np.ndarray,
    method: Literal["sum", "mean"] = "mean",
    zero_diag: bool = True,
) -> np.ndarray:
    """Reduce token-level weights ``[N, N]`` to sentence-level ``[K, K]``.

    - sum: ``W = M^T W_tok M`` (mass-preserving).
    - mean: ``W = D_S^{-1} M^T W_tok M`` with ``D_S = diag(|S_k|)``
      (size-normalized — removes the length bias; required by the plan §2.3).
    """
    if M.shape[0] != W_tok.shape[0] or W_tok.shape[0] != W_tok.shape[1]:
        raise ValueError(
            f"shape mismatch: M={M.shape}, W_tok={W_tok.shape}"
        )

    raw = M.T @ W_tok @ M  # [K, K]

    if method == "sum":
        W = raw
    elif method == "mean":
        sizes = M.sum(axis=0)  # [K]
        # Avoid div-by-zero on empty sentences (shouldn't happen given the
        # validation in sentence_membership, but be defensive).
        inv = np.where(sizes > 0, 1.0 / sizes, 0.0)
        W = raw * inv[:, None]  # divide rows by source-sentence size
    else:
        raise ValueError(f"method must be 'sum' or 'mean', got {method!r}")

    if zero_diag:
        np.fill_diagonal(W, 0.0)
    return W


# ---------------------------------------------------------------------------
# 2.4  Symmetrize & sparsify
# ---------------------------------------------------------------------------

def symmetrize(W: np.ndarray) -> np.ndarray:
    """Return ``(W + W.T) / 2``."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square 2-D, got {W.shape}")
    return 0.5 * (W + W.T)


def sparsify(
    W: np.ndarray,
    method: Literal["knn", "threshold"] = "knn",
    k: int = 10,
    tau: float = 0.05,
    symmetric: bool = True,
) -> np.ndarray:
    """Sparsify a weight matrix.

    - "knn": for each row, keep the top-``k`` largest entries (excluding the
      diagonal), zero the rest. If ``symmetric=True``, take the union of the
      sparsity pattern with its transpose (an edge survives if either
      endpoint nominates it).
    - "threshold": zero entries below ``tau * max(W)``.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square 2-D, got {W.shape}")
    N = W.shape[0]

    if method == "threshold":
        cutoff = tau * W.max()
        out = np.where(W >= cutoff, W, 0.0)
        if symmetric:
            mask = (out > 0) | (out.T > 0)
            out = np.where(mask, W, 0.0)
        return out

    if method == "knn":
        if k >= N - 1:
            # Nothing to prune; just zero the diagonal.
            out = W.copy()
            np.fill_diagonal(out, 0.0)
            return out

        # Per-row top-k indices, ignoring self-loops.
        masked = W.copy()
        np.fill_diagonal(masked, -np.inf)
        # argpartition picks an unordered top-k; that's all we need for masking.
        idx = np.argpartition(-masked, kth=k - 1, axis=1)[:, :k]

        mask = np.zeros_like(W, dtype=bool)
        rows = np.repeat(np.arange(N), k)
        mask[rows, idx.reshape(-1)] = True

        if symmetric:
            mask = mask | mask.T

        out = np.where(mask, W, 0.0)
        np.fill_diagonal(out, 0.0)
        return out

    raise ValueError(f"unknown method: {method!r}")


# ---------------------------------------------------------------------------
# 2.5  Query anchoring
# ---------------------------------------------------------------------------

def query_attention_per_sentence(
    q_tok: np.ndarray,
    M: np.ndarray,
    method: Literal["sum", "mean"] = "mean",
) -> np.ndarray:
    """Reduce the per-token query-attention vector ``[N]`` to per-sentence ``[K]``.

    Mean is the natural default to remove sentence-length bias (matching the
    sentence-graph pooling).
    """
    if q_tok.ndim != 1 or q_tok.shape[0] != M.shape[0]:
        raise ValueError(f"shape mismatch: q_tok={q_tok.shape}, M={M.shape}")
    raw = M.T @ q_tok  # [K]
    if method == "sum":
        return raw
    sizes = M.sum(axis=0)
    return np.where(sizes > 0, raw / sizes, 0.0)


# ---------------------------------------------------------------------------
# Convenience: end-to-end token-graph -> sentence-graph
# ---------------------------------------------------------------------------

def build_sentence_graph(
    attn_stack: np.ndarray,
    sentence_spans: Sequence[Tuple[int, int]],
    aggregation: Literal["mean", "rollout", "final_token_rollout"] = "mean",
    pooling: Literal["sum", "mean"] = "mean",
    sparsify_method: Literal["knn", "threshold", "none"] = "knn",
    k: int = 10,
    tau: float = 0.05,
    final_token_idx: int | None = None,
    return_intermediate: bool = False,
):
    """End-to-end token graph -> symmetric, sparsified sentence graph.

    Returns the ``[K, K]`` sentence weight matrix and the per-sentence query
    attention vector ``[K]``. With ``return_intermediate=True``, also returns
    the dict of token-level objects (``W_tok``, ``q_tok``, ``M``) — useful for
    tests and diagnostics.
    """
    L, H, N, _ = attn_stack.shape
    W_tok = aggregate_attention(attn_stack, method=aggregation, final_token_idx=final_token_idx)
    q_tok = final_token_attention(attn_stack, final_token_idx=final_token_idx)
    M = sentence_membership(sentence_spans, n_tokens=N)

    W_sent = pool_sentence_graph(M, W_tok, method=pooling)
    W_sent = symmetrize(W_sent)
    if sparsify_method != "none":
        W_sent = sparsify(W_sent, method=sparsify_method, k=k, tau=tau)

    q_sent = query_attention_per_sentence(q_tok, M, method=pooling)

    if return_intermediate:
        return W_sent, q_sent, {"W_tok": W_tok, "q_tok": q_tok, "M": M}
    return W_sent, q_sent
