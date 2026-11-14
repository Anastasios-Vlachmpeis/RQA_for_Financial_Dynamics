"""Cover-rule rho_perc (00_CONSTITUTION.md Article 3, frozen after G0).

Both implementations read only the upper triangle of D at lag >= w.
"""
from __future__ import annotations

from math import ceil

import numpy as np
from scipy import ndimage


def _validate(D, theiler, delta, connectivity):
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D matrix")
    n = len(D)
    w = int(theiler)
    if w != theiler or not 1 <= w <= n // 2:
        raise ValueError("need 1 <= w <= N/2")
    if not 0 < delta < 0.5:
        raise ValueError("need 0 < delta < 1/2")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    return D, n, w


def _bounds(D, n, w, delta, I, J):
    k = max(1, ceil(delta * n))
    upper = np.zeros((n, n))
    upper[I, J] = D[I, J]
    sym = upper + upper.T
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    nearest = np.where(lag >= w, sym, np.inf).min(axis=1)
    eps_iso = float(nearest[k - 1 : n - k + 1].max())
    eps_lag = float(min(np.diagonal(D, ell).max() for ell in range(w, n // 2 + 1)))
    return eps_iso, eps_lag, k


def rho_perc(D, theiler, delta=0.05, connectivity=8):
    """Fast cover-rule rho_perc by bisection over sorted distinct distances in Omega."""
    D, n, w = _validate(D, theiler, delta, connectivity)
    I, J = np.triu_indices(n, w)
    vals = D[I, J].copy()
    vals_sorted = np.sort(vals)
    uniq = np.unique(vals_sorted)
    eps_iso, eps_lag, k = _bounds(D, n, w, delta, I, J)

    omega = np.zeros((n, n), bool)
    omega[I, J] = True
    structure = np.ones((3, 3), bool) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)

    def holds(eps):
        lab, _ = ndimage.label(omega & (D <= eps), structure=structure)
        a = np.unique(lab[:k, :])
        b = np.unique(lab[:, n - k:])
        cand = np.intersect1d(a[a > 0], b[b > 0])
        if cand.size == 0:
            return False
        slices = ndimage.find_objects(lab)
        for comp in cand:
            sl = slices[comp - 1]
            if sl is not None and sl[1].start <= sl[0].stop - 1 + 1:
                return True
        return False

    lo = int(np.searchsorted(uniq, eps_iso))
    # eps_lag bracket is valid only for 8-connectivity (a diagonal is 8- but not 4-connected)
    hi = int(np.searchsorted(uniq, eps_lag)) if connectivity == 8 else len(uniq) - 1
    lo = min(max(lo, 0), len(uniq) - 1)
    hi = min(max(hi, lo), len(uniq) - 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if holds(uniq[mid]):
            hi = mid
        else:
            lo = mid + 1
    eps = float(uniq[lo])

    def rate(e):
        return float(np.searchsorted(vals_sorted, e, side="right") / len(vals_sorted))

    return {"eps": eps, "rho": rate(eps), "eps_iso": eps_iso, "rho_iso": rate(eps_iso),
            "eps_lag": eps_lag, "rho_lag": rate(eps_lag), "n": int(n), "k": int(k), "theiler": int(w)}


def rho_perc_reference(D, theiler, delta=0.05, connectivity=8):
    """Slow union-find cover rule; returns rho (float)."""
    D, n, w = _validate(D, theiler, delta, connectivity)
    k = max(1, ceil(delta * n))
    I, J = np.triu_indices(n, w)
    vals = D[I, J].copy()
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]
    W = n + 2
    parent = list(range(W * W))
    active = bytearray(W * W)
    box = {}
    if connectivity == 8:
        offsets = (-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1)
    else:
        offsets = (-W, -1, 1, W)

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for rank, idx in enumerate(order.tolist()):
        i = int(I[idx])
        j = int(J[idx])
        p = (i + 1) * W + j + 1
        active[p] = 1
        parent[p] = p
        bx = [i, i, j, j]
        for o in offsets:
            q = p + o
            if active[q]:
                root = find(q)
                if root != p:
                    parent[root] = p
                    other = box.pop(root)
                    bx = [min(bx[0], other[0]), max(bx[1], other[1]),
                          min(bx[2], other[2]), max(bx[3], other[3])]
        box[p] = bx
        if bx[0] < k and bx[3] >= n - k and bx[2] <= bx[1] + 1:
            return float(np.searchsorted(sorted_vals, sorted_vals[rank], side="right") / len(vals))
    raise AssertionError("unreachable for w <= N/2")
