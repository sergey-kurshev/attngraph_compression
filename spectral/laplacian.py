"""Laplacian construction + spectral utilities for small sentence graphs.

For phase-2 H1 testing the sentence graphs are tiny (K ≤ 200), so we use
dense numpy/scipy throughout. Switch to sparse routines in Phase 3 if
needed for token-level graphs.

Conventions
-----------
- All inputs are symmetric non-negative weight matrices with zero diagonal.
- ``laplacian()`` returns the COMBINATORIAL Laplacian ``L = D - W``.
- ``normalized_laplacian()`` returns the SYMMETRIC normalized Laplacian
  ``L_sym = I - D^{-1/2} W D^{-1/2}``.
- Eigenvalues are returned in ascending order with matching eigenvectors as
  columns (numpy convention).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def degree_vector(W: np.ndarray) -> np.ndarray:
    """Weighted degree per node: ``d_i = sum_j W_{ij}``."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square 2-D, got {W.shape}")
    return W.sum(axis=1)


def laplacian(W: np.ndarray) -> np.ndarray:
    """Combinatorial Laplacian ``L = D - W``."""
    d = degree_vector(W)
    return np.diag(d) - W


def normalized_laplacian(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Symmetric normalized Laplacian ``L_sym = I - D^{-1/2} W D^{-1/2}``.

    Isolated nodes (degree 0) get an inverse-sqrt-degree of 0 to avoid div-by-zero;
    their row/column of ``L_sym`` ends up as the identity row, which is the
    standard convention (they become trivial eigenvectors with eigenvalue 1).
    """
    d = degree_vector(W)
    d_inv_sqrt = np.where(d > eps, 1.0 / np.sqrt(d + eps), 0.0)
    N = W.shape[0]
    DinvHalf_W_DinvHalf = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    return np.eye(N) - DinvHalf_W_DinvHalf


def eigh_laplacian(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dense symmetric eigendecomposition.

    Returns (eigenvalues sorted ascending, eigenvectors as columns).
    """
    # Symmetrize to wash out tiny numerical asymmetry from float math.
    Lsym = 0.5 * (L + L.T)
    w, V = np.linalg.eigh(Lsym)
    # np.linalg.eigh already returns ascending eigenvalues; be explicit.
    order = np.argsort(w)
    return w[order], V[:, order]


def pseudoinverse(L: np.ndarray, tol: float | None = None) -> np.ndarray:
    """Moore–Penrose pseudoinverse of a (PSD) Laplacian.

    The smallest eigenvalue of L is 0 (with multiplicity = number of connected
    components) so we cannot invert; ``np.linalg.pinv`` handles that for us.
    """
    Lsym = 0.5 * (L + L.T)
    return np.linalg.pinv(Lsym, rcond=tol if tol is not None else 1e-12, hermitian=True)


def induced_subgraph(W: np.ndarray, indices) -> np.ndarray:
    """Return ``W[indices, :][:, indices]``."""
    idx = np.asarray(list(indices), dtype=int)
    return W[np.ix_(idx, idx)].copy()
