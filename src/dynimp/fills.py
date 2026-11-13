"""Gap-filling methods that need no training: recurrence (analog) fill and local low-rank Hankel completion."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import CubicSpline

from condpred import completion


def nan_runs(v) -> list[tuple[int, int]]:
    """Maximal runs of NaN as (start, end) with end exclusive."""
    m = np.isnan(np.asarray(v, float)).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def recurrence_fill(test, lib, c, C=20, k=8, min_overlap=0.3, self_lib=False, weights=None, runs=None):
    """Fill the NaN runs of test[:, c] from the library segments whose state recurs with the state around the gap.

    test, lib: [n, F] with NaN. The state of a gap [a, b) is the window [a - C, b + C) of all channels. It is
    compared with every library window of the same length on the entries observed in both (mean squared
    difference, channel weights optional). Donors must be fully observed in channel c over the gap. The fill is the
    mean of the k nearest donors, weighted by exp(-d / d_min). ``self_lib``: lib is test; overlapping windows are excluded.
    """
    test, lib = np.asarray(test, float), np.asarray(lib, float)
    n, F = test.shape
    out = test[:, c].copy()
    w_ch = np.ones(F) if weights is None else np.asarray(weights, float)
    fin_c = np.concatenate([[0], np.cumsum(np.isfinite(lib[:, c]))])
    by_g: dict[int, list] = {}
    for a, b in nan_runs(test[:, c]) if runs is None else runs:
        by_g.setdefault(b - a, []).append((a, b))
    for g, group in by_g.items():
        Lw = g + 2 * C
        if len(lib) < Lw + 1:
            continue
        cand = sliding_window_view(lib, Lw, axis=0)  # [P, F, Lw]
        starts = np.arange(cand.shape[0])
        valid = (fin_c[starts + C + g] - fin_c[starts + C]) == g
        for a, b in group:
            q = np.full((F, Lw), np.nan)
            lo, hi = max(a - C, 0), min(b + C, n)
            q[:, lo - (a - C) : lo - (a - C) + hi - lo] = test[lo:hi].T
            ok = valid & (np.abs(starts - (a - C)) >= Lw) if self_lib else valid
            idx = np.flatnonzero(ok)
            if len(idx) == 0:
                continue
            sq = (cand[idx] - q) ** 2 * w_ch[None, :, None]
            cnt = np.isfinite(sq).sum(axis=(1, 2))
            dist = np.where(cnt >= min_overlap * F * Lw, np.nansum(sq, axis=(1, 2)) / np.maximum(cnt, 1), np.inf)
            kk = min(k, int(np.isfinite(dist).sum()))
            if kk == 0:
                continue
            nn = np.argpartition(dist, kk - 1)[:kk]
            w = np.exp(-dist[nn] / max(dist[nn].min(), 1e-12))
            segs = np.stack([lib[idx[j] + C : idx[j] + C + g, c] for j in nn])
            out[a:b] = (w[:, None] * segs).sum(0) / w.sum()
    return out


def _complete(w: np.ndarray, rows: int, rank: int, n_iter: int) -> np.ndarray:
    if np.isfinite(w).sum() < 2 or len(w) < rows + 2:
        return completion.linear(w)
    return completion.hankel_complete(w, rows, tau=1, rank=min(rank, rows), n_iter=n_iter, tol=1e-6)


def lowrank_fill_tiled(x, window: int, rank: int, n_iter: int = 200) -> np.ndarray:
    """Low-rank Hankel completion (rows = window // 3, tau 1) of consecutive windows of a 1-D series."""
    x = np.asarray(x, float)
    out = x.copy()
    for s in range(0, len(x), window):
        w = x[s : s + window]
        if np.isnan(w).any():
            out[s : s + window] = _complete(w, max(2, len(w) // 3), rank, n_iter)
    return out


def lowrank_fill_gaps(x, ctx: int, rank: int, n_iter: int = 100, runs=None, max_rows: int = 150) -> np.ndarray:
    """Low-rank Hankel completion of the window [a - ctx, b + ctx) around each NaN run; only the run is written back."""
    x = np.asarray(x, float)
    out = x.copy()
    for a, b in nan_runs(x) if runs is None else runs:
        lo, hi = max(a - ctx, 0), min(b + ctx, len(x))
        w = x[lo:hi]
        out[a:b] = _complete(w, min(max_rows, max(2, len(w) // 3)), rank, n_iter)[a - lo : b - lo]
    return out


def spline_fill(x) -> np.ndarray:
    """Cubic spline through the observed samples; outside their range the nearest observed value is used."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    t = np.arange(len(x))
    if ok.sum() < 4:
        return completion.linear(x)
    first, last = t[ok][0], t[ok][-1]
    out = np.where(ok, x, CubicSpline(t[ok], x[ok])(np.clip(t, first, last)))
    return out
