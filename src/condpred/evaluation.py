"""Walk-forward splits, predictors, and OOS metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _as_dates(index):
    idx = pd.Index(index)
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    if idx.dtype == object or np.issubdtype(idx.dtype, np.datetime64):
        return pd.to_datetime(idx)
    return None


def _unique_dates(index):
    idx = pd.Index(index)
    d = _as_dates(index)
    if d is not None:
        return d.unique().sort_values()
    return None


def _target_end_for_row(index, i: int, h: int, uniq=None):
    if uniq is None:
        return i + h
    row_date = pd.Index(index)[i]
    pos = uniq.get_loc(row_date)
    if isinstance(pos, slice):
        pos = pos.start
    elif isinstance(pos, np.ndarray):
        pos = int(pos[0])
    if pos + h < len(uniq):
        return uniq[pos + h]
    return pd.NaT


def walk_forward(index, first_test, refit="yearly", h: int = 1, purge: bool = True):
    idx = pd.Index(index)
    dates = _as_dates(index)
    if refit == "yearly":
        years = idx.year if hasattr(idx, "year") else pd.to_datetime(idx).year
        uniq = sorted(set(years))
    else:
        uniq = list(range(int(refit), len(idx), int(refit)))
        years = np.arange(len(idx))
    started = False
    for i, y in enumerate(uniq):
        if not started:
            if (years >= y).any() and y >= (first_test if isinstance(first_test, int) else pd.Timestamp(first_test).year):
                started = True
            else:
                continue
        if refit == "yearly":
            train_mask = years < y
            test_mask = years == y
        else:
            train_mask = np.arange(len(idx)) < y
            test_mask = (np.arange(len(idx)) >= y) & (np.arange(len(idx)) < y + int(refit))
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if purge and len(train_idx) and len(test_idx):
            t0 = int(test_idx[0])
            uniq = _unique_dates(index)
            if uniq is not None:
                first_test_date = idx[t0]
                pos_first = uniq.get_loc(first_test_date)
                if isinstance(pos_first, slice):
                    pos_first = pos_first.start
                elif isinstance(pos_first, np.ndarray):
                    pos_first = int(pos_first[0])
                last_train_date = uniq[pos_first - 1] if pos_first > 0 else pd.NaT
                keep = []
                for ti in train_idx:
                    te = _target_end_for_row(index, int(ti), h, uniq)
                    if pd.notna(te) and (pos_first == 0 or (pd.notna(last_train_date) and te <= last_train_date)):
                        keep.append(ti)
                train_idx = np.asarray(keep, dtype=int)
            else:
                train_idx = train_idx[train_idx + h < t0]
        if len(test_idx):
            yield train_idx, test_idx


def fixed_split(n, fractions=(0.5, 0.25, 0.25)):
    f = np.asarray(fractions, float)
    f /= f.sum()
    cuts = (np.cumsum(f) * n).astype(int)
    a, b, c = 0, cuts[0], cuts[1]
    return np.arange(a, b), np.arange(b, c), np.arange(c, n)


class RidgePredictor:
    def __init__(self):
        self.pipe = None
        self.median_ = None

    def _augment(self, X):
        miss = np.isnan(X)
        X_imp = np.where(miss, self.median_, X)
        return np.hstack([X_imp, miss.astype(float)])

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.median_ = np.nanmedian(X, axis=0)
        X_aug = self._augment(X)
        alphas = 10.0 ** np.arange(-3, 4)
        best_a, best_s = 1.0, -np.inf
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(y) // 20)))
        for a in alphas:
            scores = []
            for tr, va in tscv.split(X_aug):
                subpipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge(alpha=a)),
                    ]
                )
                subpipe.fit(X_aug[tr], y[tr])
                pred = subpipe.predict(X_aug[va])
                scores.append(spearman(y[va], pred))
            m = np.nanmean(scores)
            if m > best_s:
                best_s, best_a = m, a
        self.pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=best_a)),
            ]
        )
        self.pipe.fit(X_aug, y)

    def predict(self, X):
        X = np.asarray(X, float)
        return self.pipe.predict(self._augment(X))


class LGBMPredictor:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=1.0,
            random_state=0,
            verbosity=-1,
            force_col_wise=True,
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def spearman(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 3:
        return np.nan
    return float(stats.spearmanr(y[ok], yhat[ok]).correlation)


def oos_r2(y, yhat, y_train_mean) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y_train_mean) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def auc_top(y_loss, score, q_train=0.9, *, y_train_loss=None, threshold=None) -> float:
    from sklearn.metrics import roc_auc_score

    y_loss, score = np.asarray(y_loss, float), np.asarray(score, float)
    if threshold is not None:
        thr = float(threshold)
    elif y_train_loss is not None:
        thr = float(np.nanquantile(np.asarray(y_train_loss, float), q_train))
    else:
        raise ValueError("auc_top requires y_train_loss or threshold (no test-set quantile)")
    event = (y_loss > thr).astype(int)
    ok = np.isfinite(score) & np.isfinite(event)
    if event[ok].sum() == 0 or event[ok].sum() == ok.sum():
        return np.nan
    return float(roc_auc_score(event[ok], score[ok]))


def rank_ic_by_date(dates, y, yhat) -> pd.Series:
    df = pd.DataFrame({"y": y, "yhat": yhat}, index=dates)
    return df.groupby(level=0).apply(lambda g: spearman(g["y"], g["yhat"]))


def _metric_fns(y_train_mean=None, y_loss=None, y_train_loss=None, auc_threshold=None, dates=None):
    fns = {
        "spearman": lambda p, y: spearman(p, y),
        "oos_r2": lambda p, y: oos_r2(y, p, y_train_mean) if y_train_mean is not None else np.nan,
    }
    if y_loss is not None:
        fns["auc_top"] = lambda p, y: auc_top(
            y_loss,
            p,
            y_train_loss=y_train_loss,
            threshold=auc_threshold,
        )
    if dates is not None:
        fns["rank_ic_median"] = lambda p, y: float(rank_ic_by_date(dates, y, p).median())
    return fns


def _panel_bootstrap_rows(dates, block, rng, n_rows: int) -> np.ndarray:
    from condpred.stats import panel_date_bootstrap

    date_arr = np.asarray(pd.Index(dates))
    picked = panel_date_bootstrap(dates, block, rng)
    rows: list[int] = []
    while len(rows) < n_rows:
        for d in picked:
            rows.extend(np.where(date_arr == d)[0].tolist())
            if len(rows) >= n_rows:
                break
        if len(rows) < n_rows:
            picked = panel_date_bootstrap(dates, block, rng)
    return np.asarray(rows[:n_rows], dtype=int)


def compare(M0_pred, M1_pred, y, bootstrap=None, *, y_train_mean=None, y_loss=None, y_train_loss=None, auc_threshold=None, dates=None):
    M0_pred = np.asarray(M0_pred, float)
    M1_pred = np.asarray(M1_pred, float)
    y = np.asarray(y, float)
    fns = _metric_fns(
        y_train_mean=y_train_mean,
        y_loss=y_loss,
        y_train_loss=y_train_loss,
        auc_threshold=auc_threshold,
        dates=dates,
    )
    out = {}
    deltas = {}
    for name, fn in fns.items():
        s0, s1 = fn(M0_pred, y), fn(M1_pred, y)
        delta = s1 - s0 if np.isfinite(s0) and np.isfinite(s1) else np.nan
        out[name] = {"M0": s0, "M1": s1, "delta": delta}
        deltas[name] = delta

    if bootstrap is not None:
        from condpred.stats import block_bootstrap_idx, panel_date_bootstrap

        rng = bootstrap.get("rng") or np.random.default_rng(0)
        block = int(bootstrap.get("block", max(5, len(y) // 20)))
        n_boot = int(bootstrap.get("n_boot", 500))
        date_arr = np.asarray(dates) if dates is not None else None
        n_uniq = len(pd.Index(dates).unique()) if dates is not None else len(y)
        use_panel = bool(bootstrap.get("panel")) or (
            dates is not None and n_uniq < len(y) // 2
        )
        for name in fns:
            boots = []
            for _ in range(n_boot):
                if use_panel and date_arr is not None:
                    bi = _panel_bootstrap_rows(date_arr, block, rng, len(y))
                else:
                    bi = block_bootstrap_idx(len(y), block, rng)
                y_b = y[bi]
                m0_b = M0_pred[bi]
                m1_b = M1_pred[bi]
                y_loss_b = np.asarray(y_loss, float)[bi] if y_loss is not None else None
                dates_b = date_arr[bi] if date_arr is not None else None
                draw_fns = _metric_fns(
                    y_train_mean=y_train_mean,
                    y_loss=y_loss_b,
                    y_train_loss=y_train_loss,
                    auc_threshold=auc_threshold,
                    dates=dates_b,
                )
                fn_b = draw_fns.get(name)
                if fn_b is None:
                    continue
                d0 = fn_b(m0_b, y_b)
                d1 = fn_b(m1_b, y_b)
                if np.isfinite(d0) and np.isfinite(d1):
                    boots.append(d1 - d0)
            if boots:
                lo, hi = np.percentile(boots, [2.5, 97.5])
                out[name]["ci_low"] = float(lo)
                out[name]["ci_high"] = float(hi)
            else:
                out[name]["ci_low"] = np.nan
                out[name]["ci_high"] = np.nan
    return out
