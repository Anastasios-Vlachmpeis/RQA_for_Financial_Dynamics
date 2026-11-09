"""E7 — When does a trend rule fail? RQA as a reliability signal for TSMOM.

Time alignment: a feature row is dated by its month-end t and uses only daily
data <= t. The target is the TSMOM outcome over month t+1. Training rows for a
test year are purged so their target month ends before that year starts.
Primary: pooled delta-AUC (M1 = B + WRQA on daily returns, m=1) for the hit
event, date-block bootstrap (block = 6 months). Full run only if G2 passed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from condpred import data as cdata
from condpred import report
from condpred.cache import cached
from condpred.config import smoke
from condpred.evaluation import RidgePredictor, spearman
from condpred.features.local_rqa import local_rqa
from condpred.features.window_rqa import window_rqa, wrqa_lrqa_frames
from condpred.geometry import Geometry
from condpred.selfcheck import leakage_test
from condpred.stats import holm, panel_date_bootstrap
from condpred.experiments import e4

E7_MODULES = [
    "condpred.experiments.e7",
    "condpred.experiments.e4",
    "condpred.data",
    "condpred.cache",
    "condpred.config",
    "condpred.evaluation",
    "condpred.features.window_rqa",
    "condpred.features.local_rqa",
    "condpred.geometry",
    "condpred.report",
    "condpred.selfcheck",
    "condpred.stats",
]

_REPO = Path(__file__).resolve().parents[3]
W, TAU, RR = 250, 1, 0.10
WRQA_KEYS = ["DET", "L", "ENTR", "LAM", "TT", "Lmax", "Vmax", "DET3_DET2", "K2_lines", "DET_LAM"]
LRQA_KEYS = ["RR_last", "Lb_mean", "Lb_max", "DET_last", "Lf_nb", "K2_local", "LAM_last", "TT_last"]
B_KEYS = ["signal_strength", "vol60", "drawdown12", "mkt_ret24", "mkt_bear", "mkt_vol60",
          "recent_hit12", "xs_dispersion"]
SMOKE_ASSETS = ["^GSPC", "SPY", "GLD"]
COMPLETION_KEYS = frozenset({"snapshot", "universes", "g2_status", "device"})


def build_initial_spec(smoke_mode: bool) -> dict:
    return {
        "experiment": "E7",
        "version": 3,
        "changelog": ["v3 (REVIEW_09 round 1, smoke-only so far): 8-year training span measured in calendar years (first test year no longer dropped); rows whose target month is incomplete at the snapshot end are dropped and the full panel_end is set explicitly; strategy-return IC stated as pooled Spearman. Primary test unchanged.", "v2: smoke panel_end corrected to 2024-12-31 (was 2024-12, which excluded December); primary test unchanged."],
        "smoke": smoke_mode,
        "data": "E4 Tier A Yahoo snapshot (name='e4_universe', daily Adj Close) via e4.load_tier_a + condpred.data.yahoo_ohlc_snapshot/quality_filter; no new download unless snapshot missing",
        "snapshot": None,
        "universes": {},
        "assets_full": e4.TIER_A_TICKERS,
        "assets_smoke": SMOKE_ASSETS,
        "market": "^GSPC",
        "month_end": "last trading day per calendar month per asset; panel date = calendar month-end, grouped by Period[M]",
        "tsmom": {"lookback_months": 12, "skip_months": 1, "vol_window_days": 60,
                  "rule": "signal_ret = P(month-end k-1)/P(month-end k-13)-1; position = sign(signal_ret)*min(0.10/vol60_ann, 2); vol60_ann = std(last 60 daily log returns)*sqrt(252); held over month k+1",
                  "note": "condpred.forecasters.TSMOM.predict returns ret/vol on daily prices and does not implement this monthly sign rule, so the frozen rule above is implemented in e7.tsmom_monthly (deviation recorded in BUILD_NOTES_09)"},
        "targets": {"primary": "hit = 1 if sign(position_t)==sign(return_{t+1}), NaN if either is 0/undefined",
                    "secondary": "strat_ret = position_t * return_{t+1}; IC = pooled Spearman across all asset-months (not a per-date rank IC), Ridge"},
        "features": {
            "B": "signal_strength=|signal_ret|/vol60_ann; vol60; drawdown12 = P/max(P, last 252d)-1; mkt_ret24 = ^GSPC 24-month month-end return at t; mkt_bear = sign(mkt_ret24) (Daniel & Moskowitz 2016 bear-market indicator; citation as named in plan, not re-verified); mkt_vol60; recent_hit12 = mean of last 12 hits already realised at t (origins k-12..k-1); xs_dispersion = cross-sectional sd of signal_ret across assets in that month",
            "WRQA_primary": "daily log returns, W=250, m=1, tau=1, wz/euclid/rr=0.10, DET_LAM included",
            "secondaries": "B+LRQA(daily returns, m=1); B+WRQA(daily returns, m=3); B+LRQA(daily log price detrended by its 250-day linear fit, m=1)",
        },
        "geometry": {"scaling": "wz", "distance": "euclid", "threshold": "rr", "rr": RR, "W": W, "tau": TAU},
        "predictor": {"event": "Logistic (median-impute + indicator, standardised, LogisticRegression C=1, lbfgs)",
                      "return": "RidgePredictor (condpred.evaluation)",
                      "pooling": "panel pooled, yearly refits, training span >= 8 years, target-month purge"},
        "panel_start": smoke("1992-01", "2010-01"),
        "panel_end": smoke("2026-09-30", "2024-12-31"),
        "panel_end_note": "full: E4 snapshot ends 2026-11-09, so Nov 2026 is an incomplete month; last retained row is Sep 2026, whose target month (Oct 2026) is complete. build_panel additionally drops any row whose target_date is after the last complete month-end (dynamic rule, snapshot-independent).",
        "first_test_year": smoke(2000, 2018),
        "primary": {"comparison": "M1 (B+WRQA returns m=1) vs M0 (B), pooled delta-AUC for hit",
                    "bootstrap": "date-block bootstrap over calendar months, block=6 months (panel_date_bootstrap)",
                    "bootstrap_n": smoke(2000, 80), "bootstrap_seed": 0,
                    "rule": "E7 positive iff CI lower bound > 0"},
        "secondaries_holm": "Holm within the secondary family: LRQA(ret), WRQA(m=3), LRQA(detrended log price), strategy-return pooled-Spearman IC (M1 vs M0)",
        "off_switch": "exploratory: flat when M1 predicted hit prob < 0.5 (fixed); hit rate / mean strat_ret vs always-on; no trading claims",
        "backend": "torch", "device_full": "cuda", "device_smoke": "cpu", "n_jobs": 4,
        "gate": "Full run only if G2 passed (read mechanically from non-smoke notebooks/tables/condpred/E4/decision.csv); smoke computes the primary mechanically and flags it",
        "g2_status": None,
    }


def verify_spec_immutable(out_dir: Path, spec: dict, modules: list[str]):
    path = Path(out_dir) / "spec.json"
    if not path.exists():
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    from condpred.config import spec_hash
    skip = {"spec_hash", "code_fingerprint", "written_at"} | set(COMPLETION_KEYS)
    a = {k: v for k, v in old.items() if k not in skip}
    b = {k: v for k, v in spec.items() if k not in skip}
    if spec_hash(a) != spec_hash(b):
        raise RuntimeError("SPEC changed after write_spec; bump version")


# ---------------------------------------------------------------- data (E4 loaders reused)
def load_prices(spec: dict) -> dict:
    """Universe/prices from the E4 snapshot. Reuses e4.load_tier_a for the
    kept universe and condpred.data loaders for prices (no copied loader)."""
    smoke_mode = spec["smoke"]
    assets_wanted = SMOKE_ASSETS if smoke_mode else list(e4.TIER_A_TICKERS)
    ta = e4.load_tier_a({"smoke": smoke_mode, "smoke_assets": {"A": assets_wanted}})
    snap = cdata.yahoo_ohlc_snapshot(list(e4.TIER_A_TICKERS) + ["^VIX"], "1990-01-01", "2026-11-10",
                                     name="e4_universe")
    clean, _rep = cdata.quality_filter(snap)
    tickers = sorted(ta["assets"]) if smoke_mode else [t for t in ta["kept"] if t != "^VIX"]
    prices = {}
    for tkr in tickers + (["^GSPC"] if "^GSPC" not in tickers else []):
        if not isinstance(clean.columns, pd.MultiIndex) or tkr not in clean.columns.get_level_values(0):
            continue
        sub = clean[tkr].copy()
        sub.columns = [str(c) for c in sub.columns]
        col = "Adj Close" if "Adj Close" in sub.columns else "Close"
        p = sub[col].astype(float).dropna()
        p = p[p > 0].sort_index()
        if len(p) >= 300:
            prices[tkr] = p
    if "^GSPC" not in prices:
        raise RuntimeError("^GSPC market series missing from E4 snapshot")
    return {"prices": {t: p for t, p in prices.items() if t in tickers},
            "market": prices["^GSPC"], "report": ta["report"], "kept": ta["kept"],
            "snapshot": ta["snapshot"]}


# ---------------------------------------------------------------- TSMOM + B features
def _daily_helpers(price: pd.Series):
    price = price.sort_index()
    logp = np.log(price)
    ret = np.log(price / price.shift(1))
    vol60 = ret.rolling(60).std() * np.sqrt(252.0)
    drawdown = price / price.rolling(252, min_periods=60).max() - 1.0
    return logp, ret, vol60, drawdown


def tsmom_monthly(price: pd.Series) -> pd.DataFrame:
    """One row per month-end k. Features use data <= k; target is month k+1."""
    price = price.sort_index()
    logp, ret, vol60, drawdown = _daily_helpers(price)
    me_pos, me_date, me_price = [], [], []
    period = price.index.to_period("M")
    for per, idx in pd.Series(np.arange(len(price)), index=price.index).groupby(period):
        pos = int(idx.iloc[-1])
        me_pos.append(pos); me_date.append(price.index[pos]); me_price.append(float(price.iloc[pos]))
    mp = pd.Series(me_price, dtype=float)  # month-end prices, ordinal k
    signal_ret = mp.shift(1) / mp.shift(13) - 1.0
    next_ret = mp.shift(-1) / mp - 1.0
    v60 = vol60.iloc[me_pos].to_numpy()
    df = pd.DataFrame({
        "daily_end": me_pos,
        "date_actual": pd.DatetimeIndex(me_date),
        "period": pd.PeriodIndex(period.unique(), freq="M") if len(me_pos) == len(period.unique()) else None,
        "price": mp.to_numpy(),
        "signal_ret": signal_ret.to_numpy(),
        "next_ret": next_ret.to_numpy(),
        "vol60": v60,
        "drawdown12": drawdown.iloc[me_pos].to_numpy(),
    })
    df["period"] = pd.PeriodIndex([d.to_period("M") for d in df["date_actual"]], freq="M")
    df["date"] = df["period"].dt.to_timestamp("M").dt.normalize()
    sign = np.sign(df["signal_ret"].to_numpy())
    scale = np.minimum(0.10 / np.where(df["vol60"] > 0, df["vol60"], np.nan), 2.0)
    df["position"] = sign * scale
    df["strat_ret"] = df["position"] * df["next_ret"]
    hit = np.where((df["position"] == 0) | (df["next_ret"] == 0) | ~np.isfinite(df["strat_ret"]),
                   np.nan, (np.sign(df["position"]) == np.sign(df["next_ret"])).astype(float))
    df["hit"] = hit
    # hits realised by t: origin s hit is known at s+1, so shift before rolling
    df["recent_hit12"] = pd.Series(hit).shift(1).rolling(12, min_periods=6).mean().to_numpy()
    df["signal_strength"] = np.abs(df["signal_ret"]) / df["vol60"]
    df["target_date"] = df["date"] + pd.offsets.MonthEnd(1)
    return df


def market_frame(market_price: pd.Series) -> pd.DataFrame:
    m = tsmom_monthly(market_price)[["period", "price", "vol60"]].copy()
    m["mkt_ret24"] = m["price"] / m["price"].shift(24) - 1.0
    m["mkt_bear"] = np.sign(m["mkt_ret24"])
    return m.rename(columns={"vol60": "mkt_vol60"})[["period", "mkt_ret24", "mkt_bear", "mkt_vol60"]]


# ---------------------------------------------------------------- RQA features
def _geom(m: int) -> Geometry:
    return Geometry(scaling="wz", distance="euclid", threshold="rr", rr=RR, m=m, tau=TAU, theiler=None)


def lrqa_detrended_row(logp: np.ndarray, end: int, W_: int = W) -> dict:
    """LRQA of the 250-day window of log price detrended by its linear fit (<= end)."""
    if end - W_ + 1 < 0:
        return {k: np.nan for k in LRQA_KEYS}
    win = np.asarray(logp[end - W_ + 1:end + 1], float)
    if not np.all(np.isfinite(win)):
        win = np.nan_to_num(win, nan=np.nanmean(win) if np.isfinite(win).any() else 0.0)
    tt = np.arange(W_, dtype=float)
    coef = np.polyfit(tt, win, 1)
    detr = win - np.polyval(coef, tt)
    row = local_rqa(detr, W_ - 1, W_, 1, TAU, _geom(1))
    return {k: float(row.get(k, np.nan)) for k in LRQA_KEYS}


def _rqa_asset(asset: str, price: pd.Series, ends: np.ndarray, spec: dict, cache_exp: str) -> pd.DataFrame:
    logp, ret, _v, _d = _daily_helpers(price)
    ret_arr = np.nan_to_num(ret.to_numpy(), nan=0.0)
    logp_arr = logp.to_numpy()
    backend, device, n_jobs = "torch", spec.get("device") or (spec["device_smoke"] if spec.get("smoke") else spec["device_full"]), spec["n_jobs"]
    key = {"asset": asset, "snapshot": spec.get("snapshot"), "n": int(len(price)),
           "last": str(price.index[-1]), "ends": [int(e) for e in ends],
           "W": W, "backend": backend, "device": device}

    def _build():
        wr1, lr1 = wrqa_lrqa_frames(ret_arr, W, 1, TAU, _geom(1), ends=ends, n_jobs=n_jobs,
                                    backend=backend, device=device)
        wr3, _lr3 = wrqa_lrqa_frames(ret_arr, W, 3, TAU, _geom(3), ends=ends, n_jobs=n_jobs,
                                     backend=backend, device=device)
        out = pd.concat([
            wr1[[c for c in WRQA_KEYS if c in wr1.columns]].add_prefix("wrqa_"),
            lr1[[c for c in LRQA_KEYS if c in lr1.columns]].add_prefix("lrqa_"),
            wr3[[c for c in WRQA_KEYS if c in wr3.columns]].add_prefix("wrqa3_"),
        ], axis=1)
        det = pd.DataFrame([lrqa_detrended_row(logp_arr, int(e)) for e in ends],
                           index=ends).add_prefix("lrqadet_")
        out = pd.concat([out, det], axis=1)
        out.index = [price.index[int(e)] for e in out.index]
        return out

    safe = re.sub(r"[^A-Za-z0-9]+", "_", asset)
    return cached(cache_exp, f"rqa_{safe}", key, _build, modules=E7_MODULES)


def last_complete_month_end(data: dict) -> pd.Timestamp:
    """Last month-end whose month is fully observed in the snapshot.

    The final period is incomplete when the snapshot's last daily date falls
    before that month's last business day.
    """
    last_daily = max([p.index.max() for p in data["prices"].values()] + [data["market"].index.max()])
    per = last_daily.to_period("M")
    month_end = per.to_timestamp("M").normalize()
    bday = month_end
    while bday.weekday() >= 5:
        bday -= pd.Timedelta(days=1)
    if last_daily.normalize() >= bday:
        return month_end
    return (per - 1).to_timestamp("M").normalize()


def build_panel(spec: dict, data: dict, cache_exp: str) -> pd.DataFrame:
    mk = market_frame(data["market"])
    frames = []
    for asset, price in sorted(data["prices"].items()):
        df = tsmom_monthly(price)
        df["asset"] = asset
        df = df.merge(mk, on="period", how="left")
        ends = df.loc[df["daily_end"] >= W - 1, "daily_end"].to_numpy(dtype=int)
        if len(ends):
            rqa = _rqa_asset(asset, price, ends, spec, cache_exp)
            rqa = rqa.reset_index(names="date_actual")
            df = df.merge(rqa, on="date_actual", how="left")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["xs_dispersion"] = panel.groupby("period")["signal_ret"].transform("std")
    start = pd.Timestamp(spec["panel_start"])
    panel = panel[panel["date"] >= start]
    if spec.get("panel_end"):
        panel = panel[panel["date"] <= pd.Timestamp(spec["panel_end"])]
    panel = panel[panel["target_date"] <= last_complete_month_end(data)]
    panel = panel.dropna(subset=["hit", "position"]).copy()
    panel["year"] = panel["date"].dt.year
    return panel.sort_values(["date", "asset"]).reset_index(drop=True)


def feature_groups(panel_cols) -> dict[str, list[str]]:
    wr = [f"wrqa_{k}" for k in WRQA_KEYS if f"wrqa_{k}" in panel_cols]
    lr = [f"lrqa_{k}" for k in LRQA_KEYS if f"lrqa_{k}" in panel_cols]
    wr3 = [f"wrqa3_{k}" for k in WRQA_KEYS if f"wrqa3_{k}" in panel_cols]
    lrd = [f"lrqadet_{k}" for k in LRQA_KEYS if f"lrqadet_{k}" in panel_cols]
    b = [c for c in B_KEYS if c in panel_cols]
    return {"M0": b, "M1": b + wr, "M1_LRQA": b + lr, "M1_WRQA_m3": b + wr3, "M1_LRQA_det": b + lrd}


# ---------------------------------------------------------------- predictors
class LogisticPredictor:
    """Median-impute + indicator, standardise, plain LogisticRegression (C=1)."""
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = np.asarray(X, float); y = np.asarray(y, float)
        ok = np.isfinite(y)
        X, y = X[ok], y[ok].astype(int)
        self.median_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        Xa = self._augment(X)
        self.scaler_ = StandardScaler().fit(Xa)
        self.model_ = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        self.model_.fit(self.scaler_.transform(Xa), y)
        return self

    def _augment(self, X):
        miss = ~np.isfinite(X)
        return np.hstack([np.where(miss, self.median_, X), miss.astype(float)])

    def predict_proba(self, X):
        return self.model_.predict_proba(self.scaler_.transform(self._augment(np.asarray(X, float))))[:, 1]


def oos_predictions(panel: pd.DataFrame, groups: dict, models: list[str], target: str,
                    kind: str, first_test_year: int, min_train_rows: int) -> pd.DataFrame:
    out = panel.copy()
    pred_col = {m: ("prob_" if kind == "logistic" else "pred_") + m for m in models}
    for c in pred_col.values():
        out[c] = np.nan
    dates = out["date"]
    for yr in sorted(out.loc[out["year"] >= first_test_year, "year"].unique()):
        cut = pd.Timestamp(int(yr), 1, 1)
        train = out[(out["year"] < yr) & (out["target_date"] < cut) & np.isfinite(out[target])]
        test_mask = (out["year"] == yr).to_numpy()
        if not test_mask.any() or len(train) < min_train_rows:
            continue
        if int(yr) - int(train["year"].min()) < 8:
            continue
        train = train.sort_values(["date", "asset"])
        for m in models:
            cols = groups[m]
            if kind == "logistic":
                pred = LogisticPredictor().fit(train[cols].to_numpy(), train[target].to_numpy())
                out.loc[test_mask, pred_col[m]] = pred.predict_proba(out.loc[test_mask, cols].to_numpy())
            else:
                pred = RidgePredictor()
                pred.fit(train[cols].to_numpy(), train[target].to_numpy())
                out.loc[test_mask, pred_col[m]] = pred.predict(out.loc[test_mask, cols].to_numpy())
    return out


# ---------------------------------------------------------------- bootstrap / metrics
def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok].astype(int), s[ok]))


def _boot_delta(df: pd.DataFrame, col0: str, col1: str, target: str, metric: str,
                n_boot: int, rng) -> dict:
    sub = df.dropna(subset=[col0, col1, target]).copy()
    if metric == "auc":
        base0, base1 = _auc(sub[target], sub[col0]), _auc(sub[target], sub[col1])
    else:
        base0, base1 = spearman(sub[target], sub[col0]), spearman(sub[target], sub[col1])
    delta = base1 - base0
    by_date = {d: g.index.to_numpy() for d, g in sub.groupby("date")}
    uniq_dates = np.array(sorted(by_date))
    pos_of = {d: i for i, d in enumerate(uniq_dates)}
    deltas = []
    for _ in range(int(n_boot)):
        picked = panel_date_bootstrap(sub["date"], 6, rng)
        idx = np.concatenate([by_date[d] for d in picked if d in by_date]) if len(picked) else np.array([], int)
        if len(idx) < 20:
            continue
        s = sub.loc[idx]
        if metric == "auc":
            d = _auc(s[target], s[col1]) - _auc(s[target], s[col0])
        else:
            d = spearman(s[target], s[col1]) - spearman(s[target], s[col0])
        if np.isfinite(d):
            deltas.append(d)
    deltas = np.asarray(deltas, float)
    if len(deltas):
        lo, hi = float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
        p = float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
        p = min(max(p, 1.0 / (len(deltas) + 1)), 1.0)
    else:
        lo = hi = np.nan; p = np.nan
    return {"n": int(len(sub)), "base0": base0, "base1": base1, "delta": delta,
            "ci_low": lo, "ci_high": hi, "p": p, "n_boot_ok": int(len(deltas))}


# ---------------------------------------------------------------- gate
def g2_status() -> tuple[str, str]:
    path = _REPO / "notebooks/tables/condpred/E4/decision.csv"
    if not path.exists():
        return "pending", "non-smoke E4 decision.csv missing; full E7 run only if G2 passed"
    try:
        d = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        return "pending", f"E4 decision.csv unreadable: {exc}"
    rows = d[d["gate"].astype(str) == "G2"] if "gate" in d.columns else d.iloc[0:0]
    if len(rows) and "verdict" in d.columns:
        v = str(rows.iloc[0]["verdict"]).lower()
        if v == "pass":
            return "pass", "E4 G2 pass in notebooks/tables/condpred/E4/decision.csv"
        return v, "E4 G2 not pass in notebooks/tables/condpred/E4/decision.csv"
    return "pending", "no G2 row in E4 decision.csv"


# ---------------------------------------------------------------- leakage
def _b_daily_at(price: np.ndarray, t: int) -> dict:
    p = pd.Series(np.asarray(price, float))
    _l, _r, vol60, dd = _daily_helpers(p)
    return {"vol60": float(vol60.iloc[t]) if t < len(p) else np.nan,
            "drawdown12": float(dd.iloc[t]) if t < len(p) else np.nan}


def run_leakage_e7() -> list[tuple[str, bool]]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(900).cumsum() * 0.01 + np.linspace(0, 1, 900)
    ret = np.diff(x, prepend=x[0])
    logp = x + 5.0
    t_list = [300, 500, 700]
    checks = [
        ("wrqa_ret_m1", leakage_test(lambda v, t: window_rqa(v, t, W, 1, TAU, _geom(1)), ret, t_list, rng)),
        ("lrqa_ret_m1", leakage_test(lambda v, t: local_rqa(v, t, W, 1, TAU, _geom(1)), ret, t_list, rng)),
        ("lrqa_detrended_price", leakage_test(lambda v, t: lrqa_detrended_row(v, t), logp, t_list, rng)),
        ("b_daily_vol_drawdown", leakage_test(lambda v, t: _b_daily_at(np.exp(v), t), logp, t_list, rng)),
    ]
    return [(k, bool(v)) for k, v in checks]


# ---------------------------------------------------------------- pipeline
def run_pipeline(spec: dict, out_dir: Path, fig_dir: Path, cache_exp: str,
                 data: dict | None = None) -> dict:
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    smoke_mode = spec["smoke"]
    spec["device"] = spec["device_smoke"] if smoke_mode else spec["device_full"]
    g2, g2_note = g2_status()
    spec["g2_status"] = g2

    if not smoke_mode and g2 != "pass":
        report.write_decision([{"gate": "G2", "statistic": "E4_G2_status", "value": np.nan,
                                "ci_low": np.nan, "ci_high": np.nan,
                                "rule": "full E7 run only if G2 passed", "verdict": g2,
                                "note": g2_note}], out_dir)
        return {"decision": [{"gate": "G2", "verdict": g2}], "g2": g2, "skipped": True}

    data = data if data is not None else load_prices(spec)
    spec["snapshot"] = data["snapshot"]
    spec["universes"] = {"assets": sorted(data["prices"]), "kept_tier_a": data["kept"]}
    report.save_table(data["report"], out_dir, "quality_report_A")
    # spec was written by the notebook before outcomes; refresh completion keys only
    report.write_spec(spec, out_dir, modules=E7_MODULES, completion_keys=COMPLETION_KEYS)
    verify_spec_immutable(out_dir, spec, E7_MODULES)

    panel = build_panel(spec, data, cache_exp)
    groups = feature_groups(panel.columns)
    min_train = smoke(200, 50)
    panel = oos_predictions(panel, groups, ["M0", "M1"], "hit", "logistic",
                            spec["first_test_year"], min_train)
    rng = np.random.default_rng(spec["primary"]["bootstrap_seed"])
    n_boot = int(spec["primary"]["bootstrap_n"])
    primary = _boot_delta(panel, "prob_M0", "prob_M1", "hit", "auc", n_boot, rng)
    primary_pass = bool(np.isfinite(primary["ci_low"]) and primary["ci_low"] > 0)

    # secondaries: event models + strategy-return IC (Ridge)
    for m in ["M1_LRQA", "M1_WRQA_m3", "M1_LRQA_det"]:
        panel = oos_predictions(panel, groups, [m], "hit", "logistic",
                                spec["first_test_year"], min_train)
    panel_r = oos_predictions(panel, groups, ["M0", "M1"], "strat_ret", "ridge",
                              spec["first_test_year"], min_train)
    panel["pred_M0"] = panel_r["pred_M0"]; panel["pred_M1"] = panel_r["pred_M1"]

    sec_rows, pvals, labels = [], [], []
    for m, label in [("M1_LRQA", "LRQA_ret_m1"), ("M1_WRQA_m3", "WRQA_ret_m3"),
                     ("M1_LRQA_det", "LRQA_detrended_logprice_m1")]:
        r = _boot_delta(panel, "prob_M0", f"prob_{m}", "hit", "auc", n_boot,
                        np.random.default_rng(1))
        sec_rows.append({"secondary": label, "metric": "delta_auc", "n": r["n"],
                         "M0": r["base0"], "M1": r["base1"], "delta": r["delta"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"]})
        pvals.append(r["p"]); labels.append(label)
    ic = _boot_delta(panel, "pred_M0", "pred_M1", "strat_ret", "spearman", n_boot,
                     np.random.default_rng(2))
    sec_rows.append({"secondary": "WRQA_ret_m1_stratret_IC", "metric": "delta_spearman_IC",
                     "n": ic["n"], "M0": ic["base0"], "M1": ic["base1"], "delta": ic["delta"],
                     "ci_low": ic["ci_low"], "ci_high": ic["ci_high"], "p": ic["p"]})
    pvals.append(ic["p"]); labels.append("WRQA_ret_m1_stratret_IC")
    sec = pd.DataFrame(sec_rows)
    if len(sec):
        sec["p_holm"] = holm(np.where(np.isfinite(sec["p"]), sec["p"], 1.0))
    report.save_table(sec, out_dir, "secondary_results")

    # off-switch (exploratory, uses primary M1 probabilities)
    osub = panel.dropna(subset=["prob_M1", "strat_ret", "hit"]).copy()
    if len(osub):
        on = osub["prob_M1"] >= 0.5
        off = pd.DataFrame([{
            "n": len(osub), "coverage_on": float(on.mean()),
            "hit_rate_always": float(osub["hit"].mean()),
            "hit_rate_on": float(osub.loc[on, "hit"].mean()) if on.any() else np.nan,
            "mean_strat_ret_always": float(osub["strat_ret"].mean()),
            "mean_strat_ret_offswitch": float((osub["strat_ret"] * on).mean()),
            "sum_strat_ret_always": float(osub["strat_ret"].sum()),
            "sum_strat_ret_offswitch": float((osub["strat_ret"] * on).sum()),
            "note": "exploratory; flat when prob_M1 < 0.5; no trading claims"}])
    else:
        off = pd.DataFrame()
    report.save_table(off, out_dir, "secondary_off_switch")

    # per-asset + panel outputs
    pa = []
    for asset, g in panel.groupby("asset"):
        pa.append({"asset": asset, "n": int(g["hit"].notna().sum()),
                   "n_oos": int(g["prob_M1"].notna().sum()), "hit_rate": float(g["hit"].mean()),
                   "auc_M0": _auc(g["hit"], g["prob_M0"]), "auc_M1": _auc(g["hit"], g["prob_M1"]),
                   "mean_strat_ret": float(g["strat_ret"].mean())})
    pa = pd.DataFrame(pa)
    if len(pa):
        pa["delta_auc"] = pa["auc_M1"] - pa["auc_M0"]
    report.save_table(pa, out_dir, "per_asset")
    keep = ["asset", "date", "period", "position", "next_ret", "strat_ret", "hit", "target_date"] + \
           [c for c in B_KEYS if c in panel.columns] + \
           [c for c in panel.columns if c.startswith(("wrqa_", "lrqa_", "wrqa3_", "lrqadet_"))] + \
           [c for c in panel.columns if c.startswith(("prob_", "pred_"))]
    report.save_table(panel[[c for c in keep if c in panel.columns]], out_dir, "panel_results")

    # figure: calibration by predicted-probability decile, M0 vs M1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    for col, lab in [("prob_M0", "M0 (B)"), ("prob_M1", "M1 (B+WRQA m=1)")]:
        s = panel.dropna(subset=[col, "hit"]).copy()
        if len(s) >= 50:
            s["bin"] = pd.qcut(s[col], 10, duplicates="drop")
            cal = s.groupby("bin", observed=True).agg(p=(col, "mean"), h=("hit", "mean"))
            ax.plot(cal["p"], cal["h"], marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("mean predicted hit probability (decile)")
    ax.set_ylabel("realised hit rate")
    ax.legend()
    report.save_fig(fig, fig_dir, "calibration_deciles")
    plt.close(fig)

    verdict = "pass" if primary_pass else "fail"
    decision = [
        {"gate": "G2", "statistic": "E4_G2_status", "value": np.nan, "ci_low": np.nan,
         "ci_high": np.nan, "rule": "full E7 run only if G2 passed", "verdict": g2, "note": g2_note},
        {"gate": "E7_primary", "statistic": "delta_auc_hit_M1_minus_M0", "value": primary["delta"],
         "ci_low": primary["ci_low"], "ci_high": primary["ci_high"],
         "rule": "CI lower bound > 0", "verdict": verdict,
         "note": ("smoke verdict is mechanical on smoke subset, not the full-run gate" if smoke_mode else "")},
    ]
    report.write_decision(decision, out_dir)
    return {"decision": decision, "primary": primary, "panel": panel, "secondary": sec,
            "off_switch": off, "per_asset": pa, "g2": g2}
