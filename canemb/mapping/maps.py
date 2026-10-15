"""OLS linear and Wahba–Kabsch rotation maps (paper Eqs. 2–3)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def fit_linear(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Least-squares M minimizing ||src @ M^T - tgt|| (rows are samples).

    Returns M of shape (d_tgt, d_src) so mapped = normalize(src @ M.T).
    """
    # src: (N, d_s), tgt: (N, d_t)  →  M such that src @ M.T ≈ tgt
    # Solve M.T = lstsq(src, tgt) → M = lstsq(...).T
    sol, *_ = np.linalg.lstsq(src, tgt, rcond=None)
    return sol.T.astype(np.float64)


def fit_rotation(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Wahba–Kabsch optimal rotation (no centering; unit-sphere embeddings)."""
    # C = src.T @ tgt  (d_s, d_t); for equal dims use SVD
    c = src.T @ tgt
    u, _s, vh = np.linalg.svd(c, full_matrices=False)
    # I' with det fix for proper rotation
    d = np.ones(u.shape[1], dtype=np.float64)
    d[-1] = np.sign(np.linalg.det(u) * np.linalg.det(vh))
    r = u @ np.diag(d) @ vh
    # mapped = src @ R  if R maps src→tgt in row space; paper maps f_A → f_B
    # We return M with mapped = normalize(src @ M.T) so M = R.T
    return r.T.astype(np.float64)


def apply_map(emb: torch.Tensor, m: np.ndarray) -> torch.Tensor:
    """Apply map matrix and re-L2-normalize. emb (N,d), m (d_out, d_in)."""
    mt = torch.as_tensor(m, device=emb.device, dtype=emb.dtype)
    out = emb @ mt.T
    return F.normalize(out, p=2, dim=1)


def fit_maps(src: np.ndarray, tgt: np.ndarray) -> Dict[str, np.ndarray]:
    return {"linear": fit_linear(src, tgt), "rotation": fit_rotation(src, tgt)}
