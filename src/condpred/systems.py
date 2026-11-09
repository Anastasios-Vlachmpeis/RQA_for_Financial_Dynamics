"""Synthetic dynamical and econometric generators (burn-in discarded)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _rk4(f, s, dt, n_steps, sub=5):
    h = dt / sub
    out = np.empty((n_steps, len(s)))
    for i in range(n_steps):
        for _ in range(sub):
            k1 = f(s)
            k2 = f(s + h / 2 * k1)
            k3 = f(s + h / 2 * k2)
            k4 = f(s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = s
    return out


def logistic(r, n, rng, burn=1000):
    x, out = float(rng.uniform(0.2, 0.8)), np.empty(n)
    for i in range(burn + n):
        x = r * x * (1 - x)
        if i >= burn:
            out[i - burn] = x
    return out


def henon(n, rng, a=1.4, b=0.3, burn=500):
    x, y = rng.normal(0, 0.1, 2)
    state = np.empty((n, 2))
    for i in range(burn + n):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        if i >= burn:
            state[i - burn] = (x, y)
    return state[:, 0], state


def lorenz63(n, rng, dt=0.01, sample_every=5, sigma=10, rho=28, beta=8 / 3, burn=2000):
    f = lambda s: np.array([sigma * (s[1] - s[0]), s[0] * (rho - s[2]) - s[1], s[0] * s[1] - beta * s[2]])
    s0 = rng.normal(0, 1, 3) + np.array([1.0, 1.0, 20.0])
    raw = _rk4(f, s0, dt, burn + n * sample_every, sub=5)[burn::sample_every][:n]
    return raw, raw[:, 0]


def rossler(n, rng, dt=0.05, sample_every=4, a=0.2, b=0.2, c=5.7, burn=2000):
    f = lambda s: np.array([-s[1] - s[2], s[0] + a * s[1], b + s[2] * (s[0] - c)])
    s0 = rng.normal(0, 0.1, 3) + np.array([0.1, 0.0, 0.1])
    raw = _rk4(f, s0, dt, burn + n * sample_every, sub=1)[burn::sample_every][:n]
    return raw


def ar1(phi, n, rng, sigma=None, burn=500):
    if sigma is None:
        sigma = np.sqrt(max(1 - phi**2, 0.0))
    e = sigma * rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def ms_ar1(phis=(0.2, 0.95), p_stay=(0.99, 0.99), n=None, rng=None, equal_variance=True, burn=500):
    if n is None or rng is None:
        raise TypeError("n and rng are required")
    phis = np.asarray(phis, float)
    sigmas = np.sqrt(1 - phis**2) if equal_variance else np.ones(2)
    state, x = 0, np.zeros(n + burn)
    labels = np.zeros(n + burn, dtype=int)
    for t in range(1, n + burn):
        if rng.random() < p_stay[state]:
            pass
        else:
            state = 1 - state
        labels[t] = state
        x[t] = phis[state] * x[t - 1] + sigmas[state] * rng.standard_normal()
    return x[burn:], labels[burn:]


def garch11(persistence, n, rng, alpha=0.08, dist="normal", burn=2000):
    beta = persistence - alpha
    omega = 1 - persistence
    z = rng.standard_normal(n + burn)
    if dist == "t5":
        z = stats.t.rvs(5, size=n + burn, random_state=rng) * np.sqrt(3 / 5)
    r, s2 = np.empty(n + burn), 1.0
    for t in range(n + burn):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r[burn:]


def ms_sv_intraday(
    n_days,
    rng,
    bars_per_day=78,
    calm=0.008,
    stress=0.02,
    P=None,
    logvol_phi=0.9,
    logvol_sd=0.15,
    shocks="normal",
    burn_days=50,
):
    if P is None:
        P = np.array([[0.99, 0.01], [0.03, 0.97]])
    n_all = burn_days + n_days
    reg = np.zeros(n_all, dtype=int)
    log_h = 0.0
    intraday = np.empty((n_all, bars_per_day))
    daily_iv = np.empty(n_all)
    bar_scale = 1.0 / np.sqrt(bars_per_day)
    for d in range(n_all):
        if d > 0 and rng.random() > P[reg[d - 1], reg[d - 1]]:
            reg[d] = 1 - reg[d - 1]
        else:
            reg[d] = reg[d - 1]
        log_h = logvol_phi * log_h + logvol_sd * rng.standard_normal()
        daily_vol = (calm if reg[d] == 0 else stress) * np.exp(log_h)
        per_bar_vol = daily_vol * bar_scale
        var_bar = per_bar_vol**2
        for b in range(bars_per_day):
            z = rng.standard_normal()
            if shocks == "t5":
                z = stats.t.rvs(5, random_state=rng) * np.sqrt(3 / 5)
            intraday[d, b] = per_bar_vol * z
        daily_iv[d] = var_bar * bars_per_day
    intraday = intraday[burn_days:]
    daily_iv = daily_iv[burn_days:]
    reg = reg[burn_days:]
    return {"intraday": intraday, "daily_iv": daily_iv, "regime": reg}
