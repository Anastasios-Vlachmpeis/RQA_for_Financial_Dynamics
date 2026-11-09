"""E10 — Hedge failure: joint recurrence predicting correlation/hedge forecast failure.

Fixed models F1 (EWMA correlation) and F2 (rolling 120d OLS hedge), h=22.
Primary: F2 hedge-failure event, M1=B+JRQA vs M0=B, pooled ΔAUC,
date-block bootstrap (66 days, resampled jointly across pairs).
Full run only if G2 passed (see decision.csv); build+smoke regardless.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from condpred import data as cdata
from condpred import report
from condpred.cache import cached
from condpred.config import smoke
from condpred.evaluation import RidgePredictor, spearman
from condpred.features import baselines
from condpred.features.joint_rqa import joint_local_rqa, joint_rqa_frame
from condpred.features.window_rqa import wrqa_lrqa_frames
from condpred.forecasters import EWMACorrelation
from condpred.geometry import Geometry
from condpred.losses import fisher_abs
from condpred.selfcheck import leakage_test
from condpred.stats import holm, panel_date_bootstrap

E10_MODULES = [
    "condpred.experiments.e10",
    "condpred.data",
    "condpred.forecasters",
    "condpred.features.joint_rqa",
    "condpred.features.window_rqa",
    "condpred.features.local_rqa",
    "condpred.features.baselines",
    "condpred.evaluation",
    "condpred.geometry",
    "condpred.stats",
    "condpred.rp_batch",
]

_REPO = Path(__file__).resolve().parents[3]
W, M, TAU, RR = 250, 3, 1, 0.10
H = 22
TICKERS = ["CPER", "DBA", "DIA", "EEM", "EFA", "EWJ", "EWZ", "FXE", "FXF",
           "FXI", "FXY", "GLD", "HYG", "IEF", "IWM", "LQD", "QQQ", "SHY",
           "SLV", "SPY", "TIP", "TLT", "UNG", "USO", "UUP", "XLE", "XLF",
           "XLK", "XLP", "XLU"]
EXCLUDE_PAIRS = {tuple(sorted(p)) for p in [("SPY", "DIA"), ("QQQ", "XLK"), ("FXE", "UUP")]}
SMOKE_PAIRS = [("SPY", "QQQ"), ("SPY", "IWM"), ("IEF", "TLT"), ("GLD", "SLV"), ("EEM", "EFA")]
TRIPLETS = {
    "none": ("TIP", "UNG", "XLP"), "mid +": ("FXY", "GLD", "TIP"),
    "mid + (risk)": ("EWZ", "FXI", "HYG"), "high +": ("EFA", "IWM", "SPY"),
    "one-out +": ("GLD", "SLV", "UNG"), "mid mixed-sign": ("XLE", "XLP", "^VIX"),
    "high mixed-sign": ("IWM", "SPY", "^VIX"), "one-out -": ("EFA", "UNG", "^VIX"),
}
WRQA_KEYS = ["DET", "L", "ENTR", "LAM", "TT", "Lmax", "Vmax", "DET3_DET2", "K2_lines", "DET_LAM"]
LRQA_KEYS = ["RR_last", "Lb_mean", "Lb_max", "DET_last", "Lf_nb", "K2_local", "LAM_last", "TT_last"]
B_KEYS = ["corr_change60_250", "corr60", "corr250", "vol22_a", "vol22_b", "vol_ratio", "spy_vol22",
          "spy_ret22", "recent_loss_f1", "recent_loss_f2"]
JRQA_KEYS = ([f"joint_{k}" for k in ("RR", "DET", "LAM", "L")]
             + [f"excess_{k}" for k in ("RR", "DET", "LAM", "L")]
             + ["CRR_diag", "CRR_diag_chg22"])
JLRQA_KEYS = [f"jloc_{k}" for k in LRQA_KEYS]
IRQA_KEYS = ([f"a_wrqa_{k}" for k in WRQA_KEYS] + [f"a_lrqa_{k}" for k in LRQA_KEYS]
             + [f"b_wrqa_{k}" for k in WRQA_KEYS] + [f"b_lrqa_{k}" for k in LRQA_KEYS])


def build_initial_spec(smoke_mode: bool) -> dict:
    return {
        "experiment": "E10", "version": 2,
        "changelog": ["v1: initial build; joint_rqa torch null fix (REVIEW_01b_round2 fix 1) applied before build.",
                      "v2: REVIEW_11 fixes 1-10: magnitude closing-time both lags; global origin calendar + joint_rqa_frame ends=; beta arms on returns with pos_fn power control, no CRR_diag, beta models fit/evaluated only on beta rows; G2 enforced in full mode unless CONDPRED_E10_OVERRIDE_G2=1; TORP guarded per-triplet + sha256; ^VIX pairs with nontradable flag; F2 realised uncentered mean square; B adds corr60, JRQA trimmed to plan list; primary dropna on [M0,M1] only; window-slice joblib tasks, dead code removed."],
        "smoke": smoke_mode,
        "tickers": TICKERS, "pair_tickers": TICKERS + ["^VIX"], "auxiliary": "^VIX (daily log change; not a tradable return; nontradable flag in pairs.csv)",
        "snapshot": "e10_pairs, 2007-01-01 to latest, via condpred.data.yahoo_ohlc_snapshot",
        "pair_rule": "|full-sample corr of daily returns|>=0.3; closing-time check: lag=max(|corr(a_t,b_{t-1})|,|corr(a_{t-1},b_t)|), drop pair if lag-|corr|>0.2 (both lag columns in pairs.csv); near-duplicate pairs SPY/DIA, QQQ/XLK, FXE/UUP excluded; ^VIX pairs included with nontradable flag",
        "pairs": [], "snapshot_file": None,
        "W": W, "m": M, "tau": TAU, "rr": RR, "h": H,
        "geometry": {"scaling": "wz", "distance": "euclid", "threshold": "rr", "rr": RR, "m": M, "tau": TAU},
        "fixed_models": {"F1": "EWMACorrelation lam=0.97", "F2": "rolling 120d OLS hedge (no intercept, uncentered sums, window incl. t), forecast=in-sample resid mean square"},
        "targets": {"F1": "fisher_abs", "F2": "ln(realised/forecast resid var), realised=uncentered mean(resid^2); event ratio>2 (primary)"},
        "features": {"B": B_KEYS, "JRQA": JRQA_KEYS, "JLRQA": JLRQA_KEYS, "IRQA": IRQA_KEYS,
                     "TORP": "triplet sync_ratio, descriptive n=8 only"},
        "primary": {"target": "F2 hedge-failure event", "comparison": "M1=B+JRQA vs M0=B (logistic)",
                    "metric": "pooled ΔAUC", "bootstrap": "date-block 66 days across pairs",
                    "bootstrap_n": smoke(2000, 80), "bootstrap_seed": 0,
                    "rule": "E10 positive if CI lower bound > 0"},
        "first_test_year": smoke(2012, 2021), "min_train_years": smoke(4, 2),
        "step": smoke(5, 10), "backend": "torch", "device_smoke": "cpu", "device_full": "cuda",
        "origin_calendar": "global: every step-th date of market returns index from its first date; mapped into each pair index",
        "beta_secondary": "beta_exp b in {0,1,3} on returns (same input as JRQA); matched power_ctrl via joint_rqa pos_fn: positive_raw=exp(wz window), so control is Euclidean on exp(b*z/2) (log at b=0); CRR_diag excluded; beta arms on every 5th origin, M0 and M_BETA fit/evaluated only on rows with beta features",
        "gate_G2": "full run only if G2 passed (E4 decision.csv); enforced in full mode unless CONDPRED_E10_OVERRIDE_G2=1; read at run time, never redefined",
        "gate_G2_override": os.environ.get("CONDPRED_E10_OVERRIDE_G2") == "1",
        "smoke_pairs": SMOKE_PAIRS if smoke_mode else None,
        "smoke_period": ["2015-01-01", "2024-12-31"] if smoke_mode else None,
    }



# ---------------------------------------------------------------- data
def load_market(spec: dict) -> dict:
    """Reuse condpred.data loaders (as E4 does); new snapshot e10_pairs."""
    snap = cdata.yahoo_ohlc_snapshot(TICKERS + ["^VIX"], "2007-01-01", "2026-11-10", name="e10_pairs")
    files = sorted((_REPO / "data/raw/yahoo").glob("condpred_e10_pairs_*.csv"))
    clean, qrep = cdata.quality_filter(snap)
    rets, prices = {}, {}
    for tkr in TICKERS + ["^VIX"]:
        if tkr not in snap.columns.get_level_values(0):
            continue
        sub = snap[tkr].copy()
        if isinstance(clean.columns, pd.MultiIndex) and tkr in clean.columns.get_level_values(0):
            sub = clean[tkr].copy(); sub.columns = [str(c) for c in sub.columns]
        r = cdata.log_returns(sub)
        if tkr == "^VIX":
            close = sub["Close"].astype(float)
            r = np.log(close / close.shift(1))
        rets[tkr] = r
        prices[tkr] = (sub["Adj Close"] if "Adj Close" in sub.columns else sub["Close"]).astype(float)
    returns = pd.DataFrame(rets).sort_index()
    price_df = pd.DataFrame(prices).sort_index()
    if spec["smoke"]:
        lo, hi = spec["smoke_period"]
        returns = returns.loc[lo:hi]; price_df = price_df.loc[lo:hi]
    return {"returns": returns, "prices": price_df, "quality": qrep,
            "snapshot_file": files[-1].name if files else None}


def select_pairs(returns: pd.DataFrame, smoke_mode: bool) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    cand = [tuple(sorted(p)) for p in SMOKE_PAIRS] if smoke_mode else None
    rows, pairs = [], []
    tickers = [t for t in TICKERS + ["^VIX"] if t in returns.columns]
    all_pairs = cand if cand else [(tickers[i], tickers[j]) for i in range(len(tickers)) for j in range(i + 1, len(tickers))]
    for a, b in all_pairs:
        nontradable = bool(a == "^VIX" or b == "^VIX")
        df = returns[[a, b]].dropna()
        if len(df) < 300:
            rows.append({"a": a, "b": b, "n": len(df), "corr": np.nan, "corr_lag_b": np.nan, "corr_lag_a": np.nan, "lag_abs_max": np.nan, "nontradable": nontradable, "kept": False, "reason": "short"})
            continue
        corr = float(df[a].corr(df[b]))
        corr_lag_b = float(df[a].corr(df[b].shift(1)))
        corr_lag_a = float(df[a].shift(1).corr(df[b]))
        lag_abs_max = float(max(abs(corr_lag_b), abs(corr_lag_a)))
        keep, reason = True, ""
        if abs(corr) < 0.3:
            keep, reason = False, "|corr|<0.3"
        elif lag_abs_max - abs(corr) > 0.2:
            keep, reason = False, "closing-time: lag_abs_max-|corr|>0.2"
        elif tuple(sorted((a, b))) in EXCLUDE_PAIRS:
            keep, reason = False, "near-duplicate excluded"
        if keep:
            pairs.append((a, b))
        rows.append({"a": a, "b": b, "n": len(df), "corr": corr, "corr_lag_b": corr_lag_b, "corr_lag_a": corr_lag_a, "lag_abs_max": lag_abs_max, "nontradable": nontradable, "kept": keep, "reason": reason})
    return pairs, pd.DataFrame(rows)


# ---------------------------------------------------------------- pair panel
def _ewma_recent_loss(loss: np.ndarray, h: int, halflife: float, origins: np.ndarray) -> np.ndarray:
    """recent_loss (halflife EWMA of realised losses) evaluated at origins; causal."""
    decay = 0.5 ** (1.0 / halflife)
    out = np.full(len(loss), np.nan)
    num = den = 0.0
    realised_at = {}
    for s in range(len(loss)):
        if np.isfinite(loss[s]):
            realised_at[s + h] = loss[s]
    for t in range(len(loss)):
        num *= decay; den *= decay
        if t in realised_at:
            num += realised_at[t]; den += 1.0
        if den > 0:
            out[t] = num / den
    return out[origins]


def _pair_targets(ra: np.ndarray, rb: np.ndarray, origins: np.ndarray) -> pd.DataFrame:
    ew = EWMACorrelation(lam=0.97); ew.fit(ra, rb)
    n = len(ra)
    # rolling 120d hedge (no intercept), cumulative sums, window incl. t
    def _roll(x, w=120):
        cs = np.concatenate([[0.0], np.cumsum(x)])
        out = np.full(n, np.nan); out[w - 1:] = cs[w:] - cs[:-w]
        return out
    saa, sbb, sab = _roll(ra * ra), _roll(rb * rb), _roll(ra * rb)
    beta = np.where(sbb > 0, sab / np.where(sbb == 0, np.nan, sbb), np.nan)
    resid_var_fc = saa - 2 * beta * sab + beta ** 2 * sbb
    resid_var_fc = np.where(np.arange(n) >= 119, resid_var_fc / 120.0, np.nan)
    rows = []
    for t in origins:
        fut_a, fut_b = ra[t + 1: t + 1 + H], rb[t + 1: t + 1 + H]
        if len(fut_a) < H or not np.isfinite(beta[t]) or resid_var_fc[t] <= 0:
            rows.append((np.nan, np.nan, np.nan, np.nan)); continue
        rho_real = float(np.corrcoef(fut_a, fut_b)[0, 1]) if np.std(fut_a) > 0 and np.std(fut_b) > 0 else np.nan
        f1 = fisher_abs(rho_real, float(ew.predict(int(t)))) if np.isfinite(rho_real) else np.nan
        resid = fut_a - beta[t] * fut_b
        ratio = float(np.mean(resid ** 2) / resid_var_fc[t])
        rows.append((f1, np.log(ratio) if ratio > 0 else np.nan, float(ratio > 2), ratio))
    df = pd.DataFrame(rows, columns=["f1_loss", "f2_logratio", "f2_event", "f2_ratio"], index=origins)
    df["beta"] = beta[origins]
    return df


def _pair_bases(ra_s: pd.Series, rb_s: pd.Series, spy: pd.Series, targets_all: pd.DataFrame, origins: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(index=ra_s.index)
    corr60 = ra_s.rolling(60).corr(rb_s); corr250 = ra_s.rolling(250).corr(rb_s)
    df["corr_change60_250"] = corr60 - corr250; df["corr60"] = corr60; df["corr250"] = corr250
    va = ra_s.rolling(22).std(); vb = rb_s.rolling(22).std()
    df["vol22_a"] = va; df["vol22_b"] = vb; df["vol_ratio"] = va / vb.replace(0, np.nan)
    spy_al = spy.reindex(df.index)
    df["spy_vol22"] = spy_al.rolling(22).std(); df["spy_ret22"] = spy_al.rolling(22).sum()
    f1_all = np.full(len(df), np.nan); f2_all = np.full(len(df), np.nan)
    pos = {int(o): i for i, o in enumerate(targets_all.index)}
    for o, i in pos.items():
        f1_all[o] = targets_all["f1_loss"].iloc[i]; f2_all[o] = targets_all["f2_logratio"].iloc[i]
    out = df.iloc[origins].copy()
    out["recent_loss_f1"] = _ewma_recent_loss(f1_all, H, 66.0, origins)
    out["recent_loss_f2"] = _ewma_recent_loss(f2_all, H, 66.0, origins)
    return out[B_KEYS]


def _beta_pair_features(ra: np.ndarray, rb: np.ndarray, origins: np.ndarray, device, n_jobs) -> pd.DataFrame:
    """beta_exp on returns; matched power_ctrl via pos_fn=exp(wz window). Every 5th origin."""
    cols = {}
    use = origins[::5]
    wz = Geometry(scaling="wz", m=M, tau=TAU, rr=RR)
    def _pos_fn(w):
        return np.exp(wz.scale_window(np.asarray(w, float)))
    for b in (0.0, 1.0, 3.0):
        for dist in ("beta_exp", "power_ctrl"):
            g = Geometry(distance=dist, beta=b, m=M, tau=TAU, rr=RR)
            from condpred.features.joint_rqa import joint_rqa as _jr
            pos_fn = _pos_fn if dist == "power_ctrl" else None
            rows = Parallel(n_jobs=n_jobs)(delayed(_jr)(ra[int(t) - W + 1: int(t) + 1], rb[int(t) - W + 1: int(t) + 1], W - 1, W, M, TAU, g, pos_fn=pos_fn) for t in use)
            sub = pd.DataFrame(rows, index=list(use))
            tag = f"{dist}{int(b)}"
            for k in ("joint_DET", "joint_LAM", "excess_DET"):
                if k in sub.columns:
                    cols[f"beta_{tag}_{k}"] = pd.Series(sub[k].to_numpy(), index=use)
    if not cols:
        return pd.DataFrame(index=origins)
    return pd.DataFrame(cols).reindex(origins)


def build_pair_panel(a: str, b: str, market: dict, spec: dict, device) -> pd.DataFrame:
    returns, prices = market["returns"], market["prices"]
    pair_df = returns[[a, b]].dropna()
    dates = pair_df.index
    ra_s, rb_s = pair_df[a], pair_df[b]
    ra, rb = ra_s.to_numpy(), rb_s.to_numpy()
    step = int(spec["step"]); n_jobs = 4
    geom = Geometry(m=M, tau=TAU, rr=RR)
    min_end = W - 1 + (M - 1) * TAU
    global_dates = returns.index[::step]
    pos_by_date = {d: i for i, d in enumerate(dates)}
    origins = np.array([pos_by_date[d] for d in global_dates if d in pos_by_date and min_end <= pos_by_date[d] < len(dates) - H], dtype=int)
    grid = origins.copy()
    if len(origins) < 20:
        return pd.DataFrame()
    targets = _pair_targets(ra, rb, origins)
    spy = returns["SPY"] if "SPY" in returns.columns else pd.Series(dtype=float)
    bases = _pair_bases(ra_s, rb_s, spy, targets, origins)
    grid_t = _pair_targets(ra, rb, grid[grid + H < len(dates)])
    f1_all = np.full(len(dates), np.nan); f2_all = np.full(len(dates), np.nan)
    for o, row in grid_t.iterrows():
        f1_all[int(o)] = row["f1_loss"]; f2_all[int(o)] = row["f2_logratio"]
    bases["recent_loss_f1"] = _ewma_recent_loss(f1_all, H, 66.0, origins)
    bases["recent_loss_f2"] = _ewma_recent_loss(f2_all, H, 66.0, origins)

    def _compute():
        jr = joint_rqa_frame(ra, rb, W, M, TAU, geom, step=step, ends=origins, n_jobs=n_jobs, backend="torch", device=device)
        jr = jr.reindex(origins)
        jl_rows = Parallel(n_jobs=n_jobs)(delayed(joint_local_rqa)(ra[int(t) - W + 1: int(t) + 1], rb[int(t) - W + 1: int(t) + 1], W - 1, W, M, TAU, geom) for t in origins)
        jl = pd.DataFrame(jl_rows, index=list(origins))
        ir = {}
        for tag, x in (("a", ra), ("b", rb)):
            wr, lr = wrqa_lrqa_frames(x, W, M, TAU, geom, ends=origins, n_jobs=n_jobs, backend="torch", device=device)
            for k in WRQA_KEYS:
                if k in wr.columns: ir[f"{tag}_wrqa_{k}"] = wr[k].to_numpy()
            for k in LRQA_KEYS:
                if k in lr.columns: ir[f"{tag}_lrqa_{k}"] = lr[k].to_numpy()
        irf = pd.DataFrame(ir, index=list(origins))
        beta_f = _beta_pair_features(ra, rb, origins, device, n_jobs)
        bases_r = bases.reset_index(drop=True); bases_r.index = origins
        panel = pd.concat([targets, bases_r], axis=1)
        panel = panel.join(jr, how="left").join(jl, how="left").join(irf, how="left").join(beta_f, how="left")
        panel["date"] = dates[origins]; panel["pair"] = f"{a}/{b}"; panel["a"] = a; panel["b"] = b
        panel["nontradable"] = bool(a == "^VIX" or b == "^VIX")
        panel["target_end"] = dates[np.minimum(origins + H, len(dates) - 1)]
        return panel.reset_index(drop=True)

    key = {"pair": f"{a}/{b}", "start": str(dates[0]), "end": str(dates[-1]), "step": step,
           "origin_calendar": "global_returns_index[::step]", "backend": "torch", "device": str(device), "smoke": bool(spec["smoke"])}
    return cached("E10", f"pair_{a}_{b}".replace("/", "_"), key, _compute, modules=E10_MODULES)


# ---------------------------------------------------------------- TORP (descriptive)
def torp_table(market: dict, panel: pd.DataFrame, spec: dict) -> pd.DataFrame:
    try:
        torp_path = _REPO / "src/recplay/torp.py"
        torp_sha = hashlib.sha256(torp_path.read_bytes()).hexdigest() if torp_path.exists() else ""
    except Exception:
        torp_sha = ""
    if isinstance(spec, dict):
        spec["torp_sha256"] = torp_sha
    try:
        from recplay.embedding import embed
        from recplay.torp import perimeter_tensor
    except Exception as exc:  # interface changed: skip with notice, per plan
        print(f"[E10] TORP skipped: {exc}")
        return pd.DataFrame()
    returns = market["returns"]
    rows = []
    for name, tr in TRIPLETS.items():
        try:
            if not all(t in returns.columns for t in tr):
                continue
            df = returns[list(tr)].dropna()
            if len(df) < W + H:
                continue
            ends = np.arange(W - 1, len(df) - H, smoke(50, 250), dtype=int)
            ratios = []
            for t in ends[: smoke(40, 8)]:
                states = []
                for tkr in tr:
                    win = df[tkr].to_numpy()[t - W + 1: t + 1]
                    z = (win - win.mean()) / (win.std() or 1.0)
                    states.append(embed(z, M, TAU))
                T = perimeter_tensor(*states)
                idx = np.arange(len(T))
                ratios.append(float(T[idx, idx, idx].mean() / T.mean(dtype=np.float64)))
            ev = np.nan
            if len(panel):
                prs = [tuple(sorted(p)) for p in [(tr[0], tr[1]), (tr[0], tr[2]), (tr[1], tr[2])]]
                sub = panel[panel.apply(lambda r: tuple(sorted((r["a"], r["b"]))) in prs, axis=1)]
                if len(sub):
                    ev = float(sub["f2_event"].mean())
            rows.append({"triplet": name, "a": tr[0], "b": tr[1], "c": tr[2], "torp_sync_ratio": float(np.mean(ratios)),
                         "mean_pair_hedge_failure_rate": ev, "n_windows": len(ratios), "torp_sha256": torp_sha})
        except Exception as exc:
            print(f"[E10] TORP triplet {name} skipped: {exc}")
            continue
    out = pd.DataFrame(rows)
    if len(out):
        ok = out.dropna(subset=["mean_pair_hedge_failure_rate"])
        if len(ok) >= 3:
            print(f"[E10] TORP descriptive (n={len(ok)} triplets with panel pairs): spearman(torp, failure)={spearman(ok['torp_sync_ratio'], ok['mean_pair_hedge_failure_rate']):.3f}")
    return out


# ---------------------------------------------------------------- prediction
class _Logistic:
    def __init__(self):
        self.pipe = None
    def fit(self, X, y):
        self.pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                              ("lr", LogisticRegression(C=1.0, max_iter=500))])
        self.pipe.fit(np.asarray(X, float), np.asarray(y, float)); return self
    def predict(self, X):
        return self.pipe.predict_proba(np.asarray(X, float))[:, 1]


def _expanding_standardise(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = panel.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        def _tr(s):
            mu = s.expanding().mean().shift(1); sd = s.expanding().std().shift(1)
            return (s - mu) / sd.replace(0, np.nan)
        out[c] = out.groupby("pair", sort=False)[c].transform(_tr)
    return out


def predict_panel(panel: pd.DataFrame, cols: list[str], target: str, kind: str, spec: dict, label: str) -> pd.DataFrame:
    data = _expanding_standardise(panel, cols)
    data = data.sort_values(["date", "pair"]).reset_index(drop=True)
    years = pd.to_datetime(data["date"]).dt.year
    rows = []
    for yr in sorted(years.unique()):
        if yr < int(spec["first_test_year"]):
            continue
        tr = data[(years < yr) & (pd.to_datetime(data["target_end"]) < pd.Timestamp(year=yr, month=1, day=1))]
        te = data[years == yr]
        if len(te) == 0 or len(tr) < 50:
            continue
        if pd.to_datetime(tr["date"]).dt.year.nunique() < int(spec["min_train_years"]):
            continue
        ytr = tr[target].to_numpy(float); ok = np.isfinite(ytr)
        if kind == "logistic" and len(np.unique(ytr[ok])) < 2:
            continue
        model = _Logistic() if kind == "logistic" else RidgePredictor()
        model.fit(tr.loc[ok, cols].to_numpy(float), ytr[ok])
        pred = model.predict(te[cols].to_numpy(float))
        for (_, r), p in zip(te.iterrows(), pred):
            rows.append({"pair": r["pair"], "a": r["a"], "b": r["b"], "date": r["date"],
                         "f2_event": r["f2_event"], "f2_logratio": r["f2_logratio"], "f1_loss": r["f1_loss"],
                         "model": label, "prediction": float(p)})
    return pd.DataFrame(rows)


def _metric_delta(preds: pd.DataFrame, m0: str, m1: str, target: str, metric: str, n_boot: int, rng, block_dates: int) -> dict:
    _piv = preds.pivot_table(index=["pair", "date"], columns="model", values="prediction", aggfunc="first")
    p = _piv[[c for c in (m0, m1) if c in _piv.columns]].dropna() if len(_piv) else _piv
    ydf = preds.drop_duplicates(["pair", "date"]).set_index(["pair", "date"])
    if m0 not in p or m1 not in p or len(p) < 20:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": len(p)}
    y = ydf.loc[p.index, target].to_numpy(float)
    a0, a1 = p[m0].to_numpy(), p[m1].to_numpy()
    def _m(yy, pp):
        if metric == "auc":
            return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else np.nan
        return spearman(yy, pp)
    base = _m(y, a1) - _m(y, a0)
    date_arr = pd.to_datetime([d for _, d in p.index])
    pos_by_date = {d: np.where(date_arr == d)[0] for d in pd.Index(date_arr).unique()}
    deltas = []
    for _ in range(n_boot):
        picked = panel_date_bootstrap(date_arr, block_dates, rng)
        idx = np.concatenate([pos_by_date[d] for d in picked]) if len(picked) else np.array([], dtype=int)
        if len(idx) < 20:
            continue
        d = _m(y[idx], a1[idx]) - _m(y[idx], a0[idx])
        if np.isfinite(d):
            deltas.append(d)
    deltas = np.asarray(deltas)
    if len(deltas) == 0:
        return {"delta": float(base), "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": len(p)}
    pval = float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))) if base != 0 else 1.0
    return {"delta": float(base), "ci_low": float(np.nanpercentile(deltas, 2.5)),
            "ci_high": float(np.nanpercentile(deltas, 97.5)),
            "p": min(max(pval, 1.0 / (len(deltas) + 1)), 1.0), "n": len(p)}


def _gate_g2() -> str:
    p = _REPO / "notebooks/tables/condpred/E4/decision.csv"
    if not p.exists():
        return "pending (E4 decision.csv missing; full E10 run gated on G2)"
    try:
        d = pd.read_csv(p)
        row = d[d["gate"].astype(str).str.contains("G2", na=False)]
        if len(row) and str(row.iloc[0].get("verdict", "")).lower() == "pass":
            return "pass"
    except Exception:
        pass
    return "not passed (see E4 decision.csv; full E10 run gated on G2)"


# ---------------------------------------------------------------- pipeline
def run_pipeline(spec: dict, out_dir: Path, fig_dir: Path) -> dict:
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    override_g2 = os.environ.get("CONDPRED_E10_OVERRIDE_G2") == "1"
    spec["gate_G2_override"] = bool(override_g2)
    if not spec["smoke"]:
        _g2 = _gate_g2()
        if _g2 != "pass" and not override_g2:
            raise RuntimeError(f"E10 full run blocked: G2 not passed ({_g2}); set CONDPRED_E10_OVERRIDE_G2=1 to override (recorded in spec)")
    device = spec["device_smoke"] if spec["smoke"] else spec["device_full"]
    market = load_market(spec)
    pairs, pair_report = select_pairs(market["returns"], spec["smoke"])
    spec["pairs"] = [list(p) for p in pairs]; spec["snapshot_file"] = market["snapshot_file"]
    report.write_spec(spec, out_dir, modules=E10_MODULES,
                      completion_keys=frozenset({"pairs", "snapshot_file"}))
    report.save_table(pair_report, out_dir, "pairs")
    report.save_table(market["quality"], out_dir, "quality_report")
    print(f"[E10] pairs kept: {len(pairs)} device={device}")
    panels = []
    for a, b in pairs:
        try:
            p = build_pair_panel(a, b, market, spec, device)
        except Exception as exc:
            print(f"[E10] pair {a}/{b} failed: {exc}")
            continue
        if len(p):
            panels.append(p)
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    if len(panel) == 0:
        raise RuntimeError("E10: empty panel")
    panel.to_parquet(out_dir / "pair_panel.parquet", index=False)
    rng = np.random.default_rng(spec["primary"]["bootstrap_seed"])
    n_boot = int(spec["primary"]["bootstrap_n"])
    block_dates = max(1, int(np.ceil(66 / int(spec["step"]))))
    beta_cols = [c for c in panel.columns if c.startswith("beta_")]
    beta_tags = sorted({"_".join(c.split("_")[1:3]) for c in beta_cols if c.startswith("beta_")})
    model_sets_event = {
        "M0": B_KEYS, "M1": B_KEYS + JRQA_KEYS,
        "M1_JLRQA": B_KEYS + JRQA_KEYS + JLRQA_KEYS,
        "M_IRQA": B_KEYS + IRQA_KEYS, "M_IRQA_JRQA": B_KEYS + IRQA_KEYS + JRQA_KEYS,
    }
    preds = []
    for label, cols in model_sets_event.items():
        preds.append(predict_panel(panel, cols, "f2_event", "logistic", spec, label))
    if beta_cols:
        beta_panel = panel[panel[beta_cols].notna().any(axis=1)].copy()
        print(f"[E10] beta rows (not imputed): {len(beta_panel)} / {len(panel)}")
        preds.append(predict_panel(beta_panel, B_KEYS, "f2_event", "logistic", spec, "M0_BETA"))
        for tag in beta_tags:
            cols = [c for c in beta_cols if c.startswith(f"beta_{tag}_")]
            if cols:
                preds.append(predict_panel(beta_panel, B_KEYS + cols, "f2_event", "logistic", spec, f"M_BETA_{tag}"))
    for label, cols, tgt in [("R0", B_KEYS, "f2_logratio"), ("R1", B_KEYS + JRQA_KEYS, "f2_logratio"),
                             ("F10", B_KEYS, "f1_loss"), ("F11", B_KEYS + JRQA_KEYS, "f1_loss")]:
        pr = predict_panel(panel, cols, tgt, "ridge", spec, label)
        pr["model"] = label; preds.append(pr)
    preds = pd.concat([p for p in preds if len(p)], ignore_index=True) if any(len(p) for p in preds) else pd.DataFrame()
    report.save_table(preds, out_dir, "pair_panel_results")

    primary = _metric_delta(preds, "M0", "M1", "f2_event", "auc", n_boot, rng, block_dates)
    verdict = "pass" if np.isfinite(primary.get("ci_low", np.nan)) and primary["ci_low"] > 0 else "fail"
    if spec["smoke"]:
        verdict = verdict + " (smoke, mechanical)"
    decision = [
        {"gate": "G2_gate_for_full_run", "statistic": "E4 G2 status", "value": np.nan, "ci_low": np.nan,
         "ci_high": np.nan, "rule": "full E10 only if G2 passed", "verdict": _gate_g2()},
        {"gate": "E10_primary", "statistic": "ΔAUC F2 event M1(B+JRQA)-M0(B)", "value": primary.get("delta", np.nan),
         "ci_low": primary.get("ci_low", np.nan), "ci_high": primary.get("ci_high", np.nan),
         "rule": "CI lower bound > 0", "verdict": verdict},
    ]
    # secondaries with Holm within family
    sec_specs = [
        ("F1_continuous_M1_vs_M0", "F10", "F11", "f1_loss", "spearman"),
        ("F2_continuous_M1_vs_M0", "R0", "R1", "f2_logratio", "spearman"),
        ("JLRQA_event", "M1", "M1_JLRQA", "f2_event", "auc"),
        ("IRQA_control_event_BIRQA_vs_BIRQA_JRQA", "M_IRQA", "M_IRQA_JRQA", "f2_event", "auc"),
    ]
    for tag in beta_tags:
        sec_specs.append((f"BETA_{tag}_event", "M0_BETA", f"M_BETA_{tag}", "f2_event", "auc"))
    sec_rows = []
    for name, m0, m1, tgt, met in sec_specs:
        r = _metric_delta(preds, m0, m1, tgt, met, n_boot, rng, block_dates)
        sec_rows.append({"test": name, **r})
    if sec_rows:
        ps = [r["p"] if np.isfinite(r.get("p", np.nan)) else 1.0 for r in sec_rows]
        for r, ph in zip(sec_rows, holm(ps)):
            r["p_holm"] = float(ph)
    sec = pd.DataFrame(sec_rows)
    report.save_table(sec, out_dir, "secondary")
    for _, r in sec.iterrows():
        decision.append({"gate": f"secondary_{r['test']}", "statistic": "delta", "value": r.get("delta", np.nan),
                         "ci_low": r.get("ci_low", np.nan), "ci_high": r.get("ci_high", np.nan),
                         "rule": "exploratory, Holm within secondary family", "verdict": "see secondary.csv"})
    report.write_decision(decision, out_dir)

    per_pair = []
    for pair, sub in preds.groupby("pair"):
        r = _metric_delta(sub.assign(pair=pair), "M0", "M1", "f2_event", "auc", min(n_boot, 200), rng, block_dates) if set(["M0", "M1"]).issubset(set(sub["model"])) else {}
        per_pair.append({"pair": pair, "n": len(sub[sub["model"] == "M0"]), "event_rate": float(sub.drop_duplicates(["pair", "date"])["f2_event"].mean()) if len(sub) else np.nan,
                         "delta_auc": r.get("delta", np.nan), "ci_low": r.get("ci_low", np.nan), "ci_high": r.get("ci_high", np.nan)})
    report.save_table(pd.DataFrame(per_pair), out_dir, "per_pair")
    make_figures(preds, panel, fig_dir)
    torp = torp_table(market, panel, spec)
    if len(torp):
        report.save_table(torp, out_dir, "torp_triplets")
    print(f"[E10] primary ΔAUC={primary.get('delta')} CI=[{primary.get('ci_low')},{primary.get('ci_high')}] n={primary.get('n')}")
    return {"primary": primary, "panel": panel, "preds": preds, "pairs": pairs}


def make_figures(preds: pd.DataFrame, panel: pd.DataFrame, fig_dir: Path):
    if len(preds) == 0:
        return
    p = preds[preds["model"].isin(["M0", "M1"])]
    if len(p):
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, style in [("M0", "o-"), ("M1", "s-")]:
            sub = p[p["model"] == model].copy()
            if len(sub) < 20:
                continue
            sub["decile"] = pd.qcut(sub["prediction"], 10, labels=False, duplicates="drop")
            g = sub.groupby("decile").agg(rate=("f2_event", "mean"), pred=("prediction", "mean"))
            ax.plot(g.index + 1, g["rate"], style, label=model)
        ax.set_xlabel("predicted-probability decile"); ax.set_ylabel("hedge-failure rate")
        ax.legend(); ax.set_title("E10 hedge-failure rate by decile")
        report.save_fig(fig, fig_dir, "decile_hedge_failure"); plt.close(fig)
    # case studies: 3 pairs with most events, 2020 & 2022 windows
    ev = panel.groupby("pair")["f2_event"].sum().sort_values(ascending=False)
    top = list(ev.head(3).index)
    sub = preds[(preds["model"] == "M1") & (preds["pair"].isin(top))]
    for year in (2020, 2022):
        s2 = sub[pd.to_datetime(sub["date"]).dt.year == year]
        if len(s2) == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for pair, g in s2.groupby("pair"):
            g = g.sort_values("date")
            ax.plot(pd.to_datetime(g["date"]), g["prediction"], label=f"{pair} pred", lw=1)
            evd = g[g["f2_event"] > 0]
            ax.scatter(pd.to_datetime(evd["date"]), evd["prediction"], marker="x", s=25)
        ax.set_title(f"E10 case study {year} (x = hedge failure)"); ax.legend(fontsize=7)
        report.save_fig(fig, fig_dir, f"case_study_{year}"); plt.close(fig)


def run_leakage_e10() -> list[tuple[str, bool]]:
    rng = np.random.default_rng(0)
    n = 700
    xa = rng.standard_normal(n); xb = 0.5 * xa + rng.standard_normal(n)
    geom = Geometry(m=M, tau=TAU)
    from condpred.features.joint_rqa import joint_rqa
    ok = True
    for t in (300, 450, 600):
        base = joint_rqa(xa, xb, t, W, M, TAU, geom)
        xa2, xb2 = xa.copy(), xb.copy()
        xa2[t + 1:] = rng.standard_normal(n - t - 1); xb2[t + 1:] = rng.standard_normal(n - t - 1)
        pert = joint_rqa(xa2, xb2, t, W, M, TAU, geom)
        for k in base:
            a_, b_ = base[k], pert[k]
            if not ((np.isnan(a_) and np.isnan(b_)) or a_ == b_):
                ok = False
    return [("joint_rqa_pair", bool(ok))]
