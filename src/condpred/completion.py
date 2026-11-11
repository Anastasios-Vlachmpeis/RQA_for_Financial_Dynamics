"""Gap filling for missing observations (causal use: pass data ≤ t only in forecasting)."""
from __future__ import annotations

import numpy as np


def ffill(x):
    x = np.asarray(x, float).copy()
    mask = np.isfinite(x)
    if not mask.any():
        return x
    first = np.flatnonzero(mask)[0]
    x[:first] = x[first]
    for i in range(first + 1, len(x)):
        if not np.isfinite(x[i]):
            x[i] = x[i - 1]
    return x


def linear(x):
    x = np.asarray(x, float).copy()
    idx = np.arange(len(x))
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return ffill(x)
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    if not ok[0]:
        x[: np.flatnonzero(ok)[0]] = x[np.flatnonzero(ok)[0]]
    return x


def _hankel(x, m, tau):
    n = len(x) - (m - 1) * tau
    return np.stack([x[k * tau : k * tau + n] for k in range(m)])


def _anti_diag_avg(H):
    """Replace every entry by the mean of its anti-diagonal (Hankel projection), NaN entries ignored."""
    m, n = H.shape
    s = (np.arange(m)[:, None] + np.arange(n)[None, :]).ravel()
    vals = H.ravel()
    ok = np.isfinite(vals)
    sums = np.bincount(s[ok], weights=vals[ok], minlength=m + n - 1)
    counts = np.bincount(s[ok], minlength=m + n - 1)
    means = np.divide(sums, counts, out=np.full(m + n - 1, np.nan), where=counts > 0)
    return means[s].reshape(m, n)


def hankel_complete(x, m, tau=1, rank=None, n_iter=200, tol=1e-6):
    """Iterative low-rank Hankel completion (approximate Boussé–Dreesen-style; not Tensorlab SDF)."""
    x = np.asarray(x, float).copy()
    known = np.isfinite(x)
    filled = linear(x)
    H = _hankel(filled, m, tau)
    K = _hankel(np.where(known, x, np.nan), m, tau)
    known_h = np.isfinite(K)
    if rank is None:
        u0, s0, _ = np.linalg.svd(np.nan_to_num(H), full_matrices=False)
        cum = np.cumsum(s0**2) / (s0**2).sum()
        rank = int(np.searchsorted(cum, 0.99) + 1)
    for _ in range(n_iter):
        u, s, vt = np.linalg.svd(np.nan_to_num(H), full_matrices=False)
        H = (u[:, :rank] * s[:rank]) @ vt[:rank]
        H[known_h] = K[known_h]
        H_new = _anti_diag_avg(H)
        if np.linalg.norm(H_new - H) / max(np.linalg.norm(H), 1e-12) < tol:
            H = H_new
            break
        H = H_new
    n = H.shape[1]
    recon = np.zeros(len(x))
    counts = np.zeros(len(x))
    for k in range(m):
        for j in range(n):
            recon[k * tau + j] += H[k, j]
            counts[k * tau + j] += 1
    counts = np.maximum(counts, 1)
    out = recon / counts
    out[known] = x[known]
    return out
