"""Non-RQA baseline features (row t uses information realised by t)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from recplay.embedding import embed
from recplay.recurrence import euclidean_distances


def recent_loss(loss_by_origin, h: int, halflife: float, t: int) -> float:
    """EWMA of losses whose targets are realised by ``t`` (origins s with s + h ≤ t)."""
    s_idx = np.arange(len(loss_by_origin))
    mask = s_idx + h <= t
    if not mask.any():
        return np.nan
    ages = t - s_idx[mask]
    w = 0.5 ** (ages / halflife)
    vals = np.asarray(loss_by_origin, float)[mask]
    ok = np.isfinite(vals)
    if not ok.any():
        return np.nan
    return float(np.average(vals[ok], weights=w[ok]))


def level_features(x, end: int, W: int) -> dict:
    win = np.asarray(x[max(0, end - W + 1) : end + 1], float)
    cur = float(x[end])
    med = float(np.median(win))
    return {"level_cur": cur, "level_mean": float(win.mean()), "level_vol_surprise": cur / med if med else np.nan}


def shape_features(returns, W: int, end: int) -> dict:
    win = np.asarray(returns[max(0, end - W + 1) : end + 1], float)
    if len(win) < 4:
        return {"shape_skew": np.nan, "shape_kurt": np.nan}
    return {"shape_skew": float(stats.skew(win)), "shape_kurt": float(stats.kurtosis(win))}


def jump_features(returns, W: int, end: int, ohlc=None) -> dict:
    win = np.asarray(returns[max(0, end - W + 1) : end + 1], float)
    sd = win.std()
    big = float(np.mean(np.abs(win) > 3 * sd)) if sd > 0 else np.nan
    out = {"jump_big_share": big}
    if ohlc is not None:
        out["jump_overnight_var_share"] = _overnight_var_share(ohlc, W, end)
    return out


def _overnight_var_share(ohlc, W: int, end: int) -> float:
    """Mean overnight / total Garman–Klass+overnight variance over the window (Tier A OHLC frame)."""
    o = np.asarray(ohlc["Open"], float)
    h = np.asarray(ohlc["High"], float)
    l = np.asarray(ohlc["Low"], float)
    c = np.asarray(ohlc["Close"], float)
    i0 = max(0, end - W + 1)
    sl = slice(i0, end + 1)
    o, h, l, c = o[sl], h[sl], l[sl], c[sl]
    if len(o) < 2:
        return np.nan
    c_prev = np.concatenate([[c[0]], c[:-1]])
    co = np.log(o / c_prev)
    overnight = co[1:] ** 2
    total = co[1:] ** 2 + 0.5 * (np.log(h[1:] / l[1:]) ** 2) - (2 * np.log(2) - 1) * (np.log(c[1:] / o[1:]) ** 2)
    total = np.clip(total, 1e-10, None)
    return float(np.mean(overnight / total))


def perm_entropy(x, W: int, end: int, order=4, delay=1) -> float:
    win = np.asarray(x[max(0, end - W + 1) : end + 1], float)
    n = len(win) - (order - 1) * delay
    if n < order:
        return np.nan
    patterns = np.argsort(np.stack([win[k * delay : k * delay + n] for k in range(order)], 1), 1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    from math import factorial

    return float(-(p * np.log(p)).sum() / np.log(factorial(order)))


def knn_spread(
    x,
    end: int,
    m: int,
    tau: int,
    k: int,
    h: int,
    lookback: int,
    *,
    lib_end: int | None = None,
) -> float:
    """Std of h-step successors of k nearest embedded past states (successors realised by ``end``).

    If ``lib_end`` is set, neighbours are drawn from ``x[0:lib_end+1]`` only (E1 library ensemble),
  while the query state still uses ``x[:end+1]``.
    """
    x = np.asarray(x, float)
    qhist = x[: end + 1]
    off = (m - 1) * tau
    if lib_end is not None:
        lib_raw = x[: lib_end + 1]
        sub = lib_raw[-lookback:] if len(lib_raw) >= lookback else lib_raw
        if len(sub) < off + h + k + 1:
            return np.nan
        states = embed(sub, m, tau)
        qstates = embed(qhist, m, tau)
        if not len(states) or not len(qstates):
            return np.nan
        cur = qstates[-1]
        base = lib_end + 1 - len(sub)
        max_j = len(sub) - off - h - 1
        if max_j < 0:
            return np.nan
        lib = states[: max_j + 1]
    else:
        hist = qhist
        if len(hist) < lookback + off + h + 1:
            return np.nan
        sub = hist[-lookback:]
        states = embed(sub, m, tau)
        if len(states) < k + h + 1:
            return np.nan
        cur = states[-1]
        base = len(hist) - lookback
        lib = states[: len(states) - h]
    D = np.sqrt(((lib - cur) ** 2).sum(axis=1))
    if len(D) < k:
        return np.nan
    nn = np.argpartition(D, k - 1)[:k]
    idx = base + nn + off + h
    if np.max(idx) > end:
        return np.nan
    succ = x[idx]
    return float(np.std(succ))


def rolling_acf1(x, W: int, end: int) -> float:
    win = np.asarray(x[max(0, end - W + 1) : end + 1], float)
    if len(win) < 3:
        return np.nan
    return float(np.corrcoef(win[:-1], win[1:])[0, 1])


def rolling_var(x, W: int, end: int) -> float:
    win = np.asarray(x[max(0, end - W + 1) : end + 1], float)
    return float(np.var(win))


def state_location(x, end: int, m: int, tau: int) -> np.ndarray:
    hist = np.asarray(x[: end + 1], float)
    states = embed(hist, m, tau)
    return states[-1].copy()
