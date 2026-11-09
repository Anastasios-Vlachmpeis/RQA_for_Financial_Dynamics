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


def mask_missing(x, frac, mode="random", block_len=10, rng=None):
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
        x[idx] = np.nan
    else:
        start = rng.integers(0, max(1, len(x) - block_len))
        end = min(len(x), start + max(block_len, n_miss))
        mask[start:end] = False
        x[start:end] = np.nan
    return x, mask


def make_stale(x, frac, mode, rng, block_len=10):
    x = np.asarray(x, float).copy()
    n = len(x)
    n_stale = int(frac * n)
    stale = np.zeros(n, dtype=bool)
    if n_stale > 0:
        if mode == "random":
            stale[rng.choice(n, size=n_stale, replace=False)] = True
        else:
            start = rng.integers(0, max(1, n - block_len))
            end = min(n, start + max(block_len, n_stale))
            stale[start:end] = True
    last = x[0]
    out = x.copy()
    for i in range(n):
        if stale[i]:
            out[i] = last
        else:
            last = x[i]
    return out
