"""Fixed forecast models (origin t uses data ≤ t only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from recplay.embedding import embed


class KNNAnalogue:
    def __init__(self, m, tau, k=10, horizons=(1,), *, expanding_library: bool = False):
        self.m, self.tau, self.k, self.horizons = m, tau, k, tuple(horizons)
        self.expanding_library = expanding_library
        self.library = None

    def fit(self, x):
        self.library = np.asarray(x, float)

    def predict(self, t: int, series=None):
        if self.library is None:
            raise RuntimeError("fit first")
        lib = self.library
        query = lib if series is None else np.asarray(series, float)
        same_path = series is None or series is lib or np.shares_memory(query, lib)
        if self.expanding_library:
            if series is not None and series is not lib and not np.shares_memory(query, lib):
                raise ValueError("expanding_library requires series is the fitted library")
            src = query
            lib_hist = query[: t + 1]
        else:
            src = lib
            lib_hist = lib
        off = (self.m - 1) * self.tau
        qhist = query[: t + 1]
        states = embed(qhist, self.m, self.tau)
        if not len(states):
            return {h: (np.nan, np.nan) for h in self.horizons}
        cur = states[-1]
        all_states = embed(lib_hist, self.m, self.tau) if self.expanding_library else embed(lib, self.m, self.tau)
        out = {}
        for h in self.horizons:
            if self.expanding_library:
                max_j = t - off - h
            else:
                max_j = len(lib) - off - h - 1
            if same_path and not self.expanding_library:
                max_j = min(max_j, t - off - h)
            if max_j < 0:
                out[h] = (np.nan, np.nan)
                continue
            lib_states = all_states[: max_j + 1]
            if len(lib_states) < self.k:
                out[h] = (np.nan, np.nan)
                continue
            D = np.sqrt(((lib_states - cur) ** 2).sum(axis=1))
            kth = min(self.k, len(D)) - 1
            nn = np.argpartition(D, kth)[: min(self.k, len(D))]
            succ = src[nn + off + h]
            out[h] = (float(succ.mean()), float(succ.std()))
        return out


class ARFixed:
    def __init__(self, p: int = 1):
        self.p = p
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x):
        x = np.asarray(x, float)
        X = np.column_stack([x[self.p - k - 1 : -k - 1] for k in range(self.p)])
        y = x[self.p :]
        reg = LinearRegression().fit(X, y)
        self.coef_ = reg.coef_
        self.intercept_ = float(reg.intercept_)

    def predict(self, t: int, h: int = 1, *, series):
        ser = np.asarray(series, float)
        hist = ser[: t + 1]
        if len(hist) <= self.p or self.coef_ is None:
            return np.nan
        yhat = list(hist[-self.p :])
        for _ in range(h):
            nxt = self.intercept_ + float(np.dot(self.coef_, yhat[-self.p :][::-1]))
            yhat.append(nxt)
        return float(yhat[-1])


class LogHAR:
    def __init__(self, refit_every=21, window=1000, horizons=(1, 5, 22)):
        self.refit_every = refit_every
        self.window = window
        self.horizons = horizons
        self.models = {}

    def fit(self, v):
        self.models = {}
        self._v_raw = np.asarray(v, float).clip(1e-10)
        self.v = np.log(self._v_raw)

    def predict(self, t: int, h: int):
        if not hasattr(self, "v") or t < 22:
            return np.nan
        ln_v = self.v
        v_raw = self._v_raw
        block_end = (t // self.refit_every) * self.refit_every
        X = np.array([ln_v[t], np.mean(ln_v[t - 4 : t + 1]), np.mean(ln_v[t - 21 : t + 1])])
        key = (block_end, h)
        if key not in self.models:
            end = block_end
            start = max(0, end - self.window)
            rows_x, rows_y = [], []
            for i in range(start + 22, end - h + 1):
                rows_x.append([ln_v[i], np.mean(ln_v[i - 4 : i + 1]), np.mean(ln_v[i - 21 : i + 1])])
                rows_y.append(float(np.log(np.mean(v_raw[i + 1 : i + h + 1]))))
            if len(rows_y) < 10:
                return np.nan
            reg = LinearRegression().fit(rows_x, rows_y)
            resid = np.asarray(rows_y) - reg.predict(rows_x)
            self.models[key] = (reg, float(resid.var()))
        reg, s2 = self.models[key]
        fit = reg.predict([X])[0]
        return float(np.exp(fit + s2 / 2))


class EWMAVariance:
    def __init__(self, lam=0.94):
        self.lam = lam
        self.var = None

    def fit(self, v):
        v = np.asarray(v, float)
        self.var = np.empty_like(v)
        self.var[0] = v[0]
        for t in range(1, len(v)):
            self.var[t] = self.lam * self.var[t - 1] + (1 - self.lam) * v[t]

    def predict(self, t: int, h: int = 1):
        return float(self.var[t])


class EWMACorrelation:
    def __init__(self, lam=0.97):
        self.lam = lam

    def fit(self, ra, rb):
        ra, rb = np.asarray(ra, float), np.asarray(rb, float)
        self.corr = np.zeros(len(ra))
        va, vb, cab = ra[0] ** 2, rb[0] ** 2, ra[0] * rb[0]
        for t in range(len(ra)):
            va = self.lam * va + (1 - self.lam) * ra[t] ** 2
            vb = self.lam * vb + (1 - self.lam) * rb[t] ** 2
            cab = self.lam * cab + (1 - self.lam) * ra[t] * rb[t]
            denom = max(va * vb, 1e-40)
            self.corr[t] = cab / np.sqrt(denom)

    def predict(self, t: int):
        return float(self.corr[t])


class RollingHedge:
    def __init__(self, window=120):
        self.window = window
        self.beta = None
        self.resid_var = None

    def fit(self, ya, yb):
        ya, yb = np.asarray(ya, float), np.asarray(yb, float)
        n = len(ya)
        self.beta = np.full(n, np.nan)
        self.resid_var = np.full(n, np.nan)
        for t in range(self.window, n):
            X = yb[t - self.window : t]
            Y = ya[t - self.window : t]
            b = np.linalg.lstsq(X[:, None], Y, rcond=None)[0][0]
            resid = Y - b * X
            self.beta[t] = b
            self.resid_var[t] = float(resid.var())

    def predict(self, t: int):
        return float(self.resid_var[t]), float(self.beta[t])


class TSMOM:
    def __init__(self, lookback=12, skip=1, vol_window=60):
        self.lookback, self.skip, self.vol_window = lookback, skip, vol_window

    def fit(self, prices):
        self.prices = np.asarray(prices, float)

    def predict(self, t: int):
        if t < self.lookback + self.skip:
            return np.nan
        ret = self.prices[t - self.skip] / self.prices[t - self.lookback] - 1
        vol = np.std(np.diff(np.log(self.prices[max(0, t - self.vol_window) : t + 1])))
        return float(ret / vol) if vol > 0 else np.nan
