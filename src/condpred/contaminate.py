"""Observation contamination (inputs are never modified in place)."""
from __future__ import annotations

import numpy as np


def add_jumps(log_price_or_returns, rate_per_day, size_k, scale, rng, bars_per_day=1):
    x = np.asarray(log_price_or_returns, float).copy()
    n = len(x)
    n_jumps = rng.poisson(rate_per_day * n / bars_per_day)
    if n_jumps:
        idx = rng.integers(0, n, size=n_jumps)
        sizes = rng.normal(0, size_k * scale, size=n_jumps)
        np.add.at(x, idx, sizes)
    return x


def add_obs_noise(log_price, omega, rng):
    x = np.asarray(log_price, float).copy()
    return x + rng.normal(0, omega, size=len(x))


def _block_mask(n, n_target, block_len, rng):
    """Non-overlapping blocks of ``block_len`` at uniformly random offsets, about ``n_target`` points in total."""
    block_len = max(1, int(block_len))
    nb = max(1, int(round(n_target / block_len)))
    nb = min(nb, n // block_len)
    out = np.zeros(n, dtype=bool)
    if nb <= 0:
        return out
    free = n - nb * block_len
    slots = np.sort(rng.choice(free + nb, size=nb, replace=False))
    for k, p in enumerate(slots):
        start = int(p) + k * (block_len - 1)
        out[start : start + block_len] = True
    return out


def mask_missing(x, frac, mode="random", block_len=10, rng=None):
    """Return (x with NaN, observed mask). ``mode="block"``: gaps of ``block_len`` totalling ``frac``."""
    if rng is None:
        raise TypeError("rng required")
    x = np.asarray(x, float).copy()
    mask = np.ones(len(x), dtype=bool)
    n_miss = int(frac * len(x))
    if n_miss <= 0:
        return x, mask
    if mode == "random":
        idx = rng.choice(len(x), size=n_miss, replace=False)
        mask[idx] = False
    else:
        mask[_block_mask(len(x), n_miss, block_len, rng)] = False
    x[~mask] = np.nan
    return x, mask


def make_stale(x, frac, mode, rng, block_len=10):
    """Stale indices repeat the last fresh value. ``mode="block"``: runs of ``block_len`` totalling ``frac``."""
    x = np.asarray(x, float).copy()
    n = len(x)
    n_stale = int(frac * n)
    stale = np.zeros(n, dtype=bool)
    if n_stale > 0:
        if mode == "random":
            stale[rng.choice(n, size=n_stale, replace=False)] = True
        else:
            stale = _block_mask(n, n_stale, block_len, rng)
    last = x[0]
    out = x.copy()
    for i in range(n):
        if stale[i]:
            out[i] = last
        else:
            last = x[i]
    return out
