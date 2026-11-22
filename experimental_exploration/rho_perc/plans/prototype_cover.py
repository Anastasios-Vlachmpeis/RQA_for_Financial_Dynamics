"""Prototype of the corrected rho_perc (the "cover" rule of 00_CONSTITUTION.md, Article 3).

Verified on 13 Nov 2026. This is a starting point, not the deliverable: phase P0 rewrites it as ../src/rhoperc/core.py.
Run from RecPlay/:  python experimental_exploration/rho_perc/plans/prototype_cover.py
"""
from math import ceil

import numpy as np
from scipy import ndimage


def bounds(D, w=1, delta=0.05):
    """(eps_iso, eps_lag): eps_iso <= eps_perc <= eps_lag.

    eps_iso: the largest nearest-neighbour distance (lag >= w) among the times that must take part.
    eps_lag: the best single lag, i.e. the smallest over lags of the largest distance along that diagonal.
    Only the upper triangle of D at lag >= w is read (recplay.connectivity.omega_ranks fills nothing else).
    """
    D = np.asarray(D, float)
    n = len(D)
    k = max(1, ceil(delta * n))
    upper = np.triu(D, w)
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    nearest = np.where(lag >= w, upper + upper.T, np.inf).min(axis=1)
    eps_iso = nearest[k - 1 : n - k + 1].max()
    eps_lag = min(np.diagonal(D, l).max() for l in range(w, n // 2 + 1))
    return float(eps_iso), float(eps_lag)


def rho_perc_fast(D, w=1, delta=0.05, connectivity=8, rule="cover"):
    """Bisection over the distinct distances; each step labels the thresholded plot once.

    rule="cover": a component with a pixel in the first k rows, a pixel in the last k columns, and no time gap
                  (smallest column <= largest row + 1).
    rule="span":  the older candidate (a pixel with both indices early and a pixel with both late); comparison only.
    """
    D = np.asarray(D, float)
    n = len(D)
    if not 1 <= w <= n // 2:
        raise ValueError("need 1 <= w <= N/2")
    k = max(1, ceil(delta * n))
    I, J = np.triu_indices(n, w)
    vals = np.sort(D[I, J])
    uniq = np.unique(vals)
    omega = np.zeros((n, n), bool)
    omega[I, J] = True
    structure = np.ones((3, 3), bool) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)

    def holds(eps):
        lab, _ = ndimage.label(omega & (D <= eps), structure=structure)
        if rule == "span":
            a, b = np.unique(lab[:, :k]), np.unique(lab[n - k :, :])
            return np.intersect1d(a[a > 0], b[b > 0]).size > 0
        a, b = np.unique(lab[:k, :]), np.unique(lab[:, n - k :])
        for comp in np.intersect1d(a[a > 0], b[b > 0]):
            m = lab == comp
            max_row = np.flatnonzero(m.any(axis=1))[-1]
            min_col = np.flatnonzero(m.any(axis=0))[0]
            if min_col <= max_row + 1:
                return True
        return False

    eps_iso, eps_lag = bounds(D, w, delta)
    lo, hi = 0, len(uniq) - 1
    if rule == "cover":  # the bounds bracket the answer
        lo, hi = int(np.searchsorted(uniq, eps_iso)), int(np.searchsorted(uniq, eps_lag))
    while lo < hi:
        mid = (lo + hi) // 2
        if holds(uniq[mid]):
            hi = mid
        else:
            lo = mid + 1
    rate = lambda e: float(np.searchsorted(vals, e, side="right") / len(vals))
    return {"eps": float(uniq[lo]), "rho": rate(uniq[lo]), "eps_iso": eps_iso, "rho_iso": rate(eps_iso),
            "eps_lag": eps_lag, "rho_lag": rate(eps_lag), "n": n, "k": k, "w": w}


def rho_perc_reference(D, w=1, delta=0.05):
    """Cover rule, 8-neighbourhood, by union-find over pixels in order of increasing distance. Slow and plain."""
    D = np.asarray(D, float)
    n = len(D)
    k = max(1, ceil(delta * n))
    I, J = np.triu_indices(n, w)
    vals = D[I, J]
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]
    W = n + 2
    parent = list(range(W * W))
    active = bytearray(W * W)
    box = {}  # root -> [min row, max row, min col, max col]
    offsets = (-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1)
    for rank, (i, j) in enumerate(zip(I[order].tolist(), J[order].tolist())):
        p = (i + 1) * W + j + 1
        active[p] = 1
        bx = [i, i, j, j]
        for o in offsets:
            q = p + o
            if active[q]:
                while parent[q] != q:
                    parent[q] = parent[parent[q]]
                    q = parent[q]
                if q != p:
                    parent[q] = p
                    other = box.pop(q)
                    bx = [min(bx[0], other[0]), max(bx[1], other[1]), min(bx[2], other[2]), max(bx[3], other[3])]
        box[p] = bx
        if bx[0] < k and bx[3] >= n - k and bx[2] <= bx[1] + 1:
            return float(np.searchsorted(sorted_vals, sorted_vals[rank], side="right") / len(vals))
    raise AssertionError("unreachable for w <= N/2")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dist = lambda x: np.abs(np.subtract.outer(x, x))

    D = np.ones((100, 100)) + np.triu(rng.random((100, 100)) * 1e-3, 1)
    D = np.maximum(D, D.T)
    D[0, 99] = D[99, 0] = 0.01
    print("K1 single corner pixel:", round(rho_perc_fast(D)["rho"], 4), "(must not be near 0)")

    N = 300
    for p in (7, 40, 90):
        lags = np.arange(1, N)
        expected = (N - lags[lags % p == 0]).sum() / (N - lags).sum()
        got = rho_perc_fast(dist((np.arange(N) % p).astype(float)))
        span = rho_perc_fast(dist((np.arange(N) % p).astype(float)), rule="span")["rho"]
        print(f"K2 period {p:2d}: cover {got['rho']:.6f}  closed form {expected:.6f}  eps {got['eps']}  (span rule {span:.3f})")

    ok = True
    for trial in range(40):
        n = int(rng.integers(30, 160))
        x = rng.standard_normal(n) if trial % 2 else np.round(rng.standard_normal(n), 1)
        w = int(rng.choice([1, 3, 10]))
        r = rho_perc_fast(dist(x), w)
        ok &= np.isclose(r["rho"], rho_perc_reference(dist(x), w)) and r["eps_iso"] <= r["eps"] <= r["eps_lag"]
        ok &= np.isclose(r["rho"], rho_perc_fast(dist(x[::-1].copy()), w)["rho"])
        ok &= np.isclose(r["rho"], rho_perc_fast(dist(x) ** 2, w)["rho"])
        ok &= r["rho"] <= rho_perc_fast(dist(x), w, connectivity=4)["rho"] + 1e-12
    print("K3-K6 parity, bounds, time reversal, monotone transform, 8 <= 4 neighbourhood:", bool(ok))

    step = (np.arange(N) >= N // 2).astype(float)
    print("K7 noise-free step:", rho_perc_fast(dist(step))["rho"], "(must be 1.0)")
