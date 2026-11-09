"""Block bootstrap, HAC, Diebold–Mariano, Giacomini–White, Holm."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def block_bootstrap_idx(n, block, rng):
    block = max(1, int(block))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
    return idx


def panel_date_bootstrap(dates, block, rng):
    uniq = pd.Index(dates).unique()
    n = len(uniq)
    block = max(1, int(block))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    picked = np.concatenate([uniq[s : min(s + block, n)] for s in starts])[:n]
    return picked


def hac_se(x, lags=None) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lags = max(0, lags)
    x = x - x.mean()
    gamma0 = np.dot(x, x) / n
    w_sum = 0.0
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        gj = np.dot(x[j:], x[:-j]) / n
        w_sum += 2 * w * gj
    var = (gamma0 + w_sum) / n
    return float(np.sqrt(max(var, 0.0)))


def diebold_mariano(l1, l2, h=1) -> tuple[float, float]:
    d = np.asarray(l1, float) - np.asarray(l2, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return np.nan, np.nan
    d_bar = d.mean()
    se = hac_se(d, lags=max(h - 1, 1))
    if se == 0:
        return np.nan, np.nan
    stat = d_bar / se
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)


def _hac_gram(scores, lags: int) -> np.ndarray:
    n, _ = scores.shape
    scores = scores - scores.mean(axis=0, keepdims=True)
    lags = max(0, int(lags))
    meat = scores.T @ scores / n
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        cj = scores[j:].T @ scores[:-j] / n
        meat += w * (cj + cj.T)
    return meat * n


def giacomini_white(l1, l2, instruments, h=1) -> tuple[float, float]:
    d = np.asarray(l1, float) - np.asarray(l2, float)
    Z = np.asarray(instruments, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    ok = np.isfinite(d) & np.all(np.isfinite(Z), axis=1)
    d, Z = d[ok], Z[ok]
    if len(d) < Z.shape[1] + 3:
        return np.nan, np.nan
    Z = np.column_stack([np.ones(len(Z)), Z])
    beta = np.linalg.lstsq(Z, d, rcond=None)[0]
    resid = d - Z @ beta
    scores = Z * resid[:, np.newaxis]
    n = len(d)
    auto_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lags = max(h, auto_lags, 1)
    bread = np.linalg.inv(Z.T @ Z)
    hac_meat = _hac_gram(scores, lags)
    vcov = bread @ hac_meat @ bread
    wald = float(beta @ np.linalg.inv(vcov) @ beta)
    p = 1 - stats.chi2.cdf(wald, df=Z.shape[1])
    return wald, float(p)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, p[i] * (m - rank))
        prev = max(prev, val)
        adj[i] = prev
    return adj
