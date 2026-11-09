"""ρ_perc variants (secondary; not for primary tests until approved)."""
from __future__ import annotations

import numpy as np
from math import ceil

from recplay.connectivity import rr_critical


def rho_perc_corner(D, theiler=1, delta=0.05) -> float:
    return rr_critical(D, theiler=theiler, delta=delta)


def rho_perc_span(D, theiler=1, delta=0.05) -> float:
    """Early–early to late–late span (fresh union–find, not recplay)."""
    D = np.asarray(D, float)
    n = len(D)
    k = max(1, ceil(delta * n))
    I, J = np.triu_indices(n, theiler)
    if len(I) == 0:
        raise ValueError("empty domain")
    vals = D[I, J]
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]
    idx_i, idx_j = I[order], J[order]
    w = n + 2
    parent = list(range(w * w))
    active = bytearray(w * w)
    flags = bytearray(w * w)
    offsets = (-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1)
    for rank, (i, j) in enumerate(zip(idx_i, idx_j)):
        p = (i + 1) * w + (j + 1)
        active[p] = 1
        flags[p] = (int(i < k and j < k)) | ((int(i >= n - k and j >= n - k)) << 1)
        for o in offsets:
            q = p + o
            if active[q]:
                while parent[q] != q:
                    parent[q] = parent[parent[q]]
                    q = parent[q]
                if q != p:
                    parent[q] = p
                    flags[p] |= flags[q]
        if flags[p] == 3:
            return float(np.searchsorted(sorted_vals, sorted_vals[rank], side="right") / len(vals))
    return 1.0


def largest_component_jump(D, theiler=1, grid=None) -> float:
    if grid is None:
        grid = np.linspace(0.01, 0.5, 20)
    shares = []
    for q in grid:
        n = len(D)
        eps = np.quantile(D[np.triu_indices(n, theiler)], q)
        R = D <= eps
        band = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < theiler
        R[band] = False
        shares.append(_largest_share(R))
    shares = np.asarray(shares)
    jumps = np.diff(shares)
    if len(jumps) == 0:
        return np.nan
    i = int(np.argmax(jumps))
    return float(grid[i + 1])


def _largest_share(R):
    n = len(R)
    w = n + 2
    parent = list(range(w * w))
    active = bytearray(w * w)
    sizes = [0] * (w * w)
    offsets = (-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1)
    pixels = [(i, j) for i in range(n) for j in range(n) if R[i, j]]
    for i, j in pixels:
        p = (i + 1) * w + (j + 1)
        active[p] = 1
        sizes[p] = 1
        for o in offsets:
            q = p + o
            if active[q]:
                while parent[q] != q:
                    parent[q] = parent[parent[q]]
                    q = parent[q]
                if q != p:
                    parent[q] = p
                    sizes[p] += sizes[q]
                    sizes[q] = 0
    total = R.sum()
    if total == 0:
        return 0.0
    best = max(sizes)
    return best / total
