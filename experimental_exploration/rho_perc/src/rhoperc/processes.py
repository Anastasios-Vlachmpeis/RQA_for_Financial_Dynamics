"""Article 4 process panel: generate(pid, n, rng) -> 1D series of length n."""
from __future__ import annotations

import numpy as np

from condpred import systems

PIDS = ["S1", "S2", "S3", "S4", "S5", "S6", "D1", "D2", "D3", "F1", "F2", "X1", "X2"]


def _ar2(n, rng, burn=500):
    phi1, phi2 = 1.5, -0.75
    e = rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    for t in range(2, n + burn):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + e[t]
    return x[burn:]


def generate(pid, n, rng):
    n = int(n)
    if pid == "S1":
        return rng.standard_normal(n)
    if pid == "S2":
        return systems.ar1(0.5, n, rng)
    if pid == "S3":
        return systems.ar1(0.9, n, rng)
    if pid == "S4":
        return _ar2(n, rng)
    if pid == "S5":
        return systems.garch11(0.97, n, rng)
    if pid == "S6":
        x, _ = systems.ms_ar1(phis=(0.2, 0.95), n=n, rng=rng)
        return x
    if pid == "D1":
        return systems.logistic(4.0, n, rng)
    if pid == "D2":
        return systems.logistic(3.7, n, rng)
    if pid == "D3":
        x, _ = systems.henon(n, rng)
        return np.asarray(x, dtype=float)
    if pid == "F1":
        raw = systems.rossler(n, rng)  # dt=0.05, sample_every=4 -> step 0.2
        return np.asarray(raw, dtype=float)[:, 0]
    if pid == "F2":
        _, x = systems.lorenz63(n, rng)  # dt=0.01, sample_every=5 -> step 0.05
        return np.asarray(x, dtype=float)
    if pid == "X1":
        return np.cumsum(rng.standard_normal(n))
    if pid == "X2":
        x = systems.ar1(0.9, n, rng)
        x = x.copy()
        x[n // 2:] += 4.0  # 4 sd (stationary sd of S3 is 1)
        return x
    raise ValueError(f"unknown pid {pid!r}")
