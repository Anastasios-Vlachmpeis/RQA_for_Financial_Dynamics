"""E4 — Predicting the loss of a HAR volatility forecast.

Tier OM (Oxford-Man, primary), Tier A (Yahoo GK proxy), Tier BN (Binance).
Fixed model LogHAR; loss predictor pooled Ridge (LightGBM secondary).
PERC excluded: definition not approved by Tasos (01 §4.13).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from condpred import data as cdata
from condpred import report
from condpred.cache import cached
from condpred.config import smoke
from condpred.evaluation import LGBMPredictor, RidgePredictor, spearman, walk_forward
from condpred.features import baselines
from condpred.features.window_rqa import wrqa_lrqa_frames
from condpred.forecasters import EWMAVariance, LogHAR
from condpred.geometry import Geometry
from condpred.losses import log_ratio as _lr
from condpred.losses import qlike as _ql
from condpred.selfcheck import leakage_test
from condpred.stats import holm

E4_MODULES = [
    "condpred.experiments.e4",
    "condpred.data",
    "condpred.forecasters",
    "condpred.features.window_rqa",
    "condpred.features.local_rqa",
    "condpred.features.baselines",
    "condpred.evaluation",
    "condpred.geometry",
    "condpred.stats",
]

_REPO = Path(__file__).resolve().parents[3]
W, M, TAU, THEILER, RR = 250, 3, 1, 5, 0.10
HORIZONS = (1, 5, 22)
PRIMARY_H = 5
WRQA_KEYS = ["DET", "L", "ENTR", "LAM", "TT", "Lmax", "Vmax", "DET3_DET2", "K2_lines", "DET_LAM"]
LRQA_KEYS = ["RR_last", "Lb_mean", "Lb_max", "DET_last", "Lf_nb", "K2_local", "LAM_last", "TT_last"]
OM_SMOKE = [".SPX", ".FTSE"]
TIER_A_TICKERS = [
    "^GSPC", "^NDX", "^RUT", "^DJI", "^FTSE", "^GDAXI", "^N225", "^HSI",
    "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "USO", "XLE", "XLF",
]
BN_PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "TRXUSDT"]
US_OM = [".SPX", ".DJI", ".IXIC", ".RUT"]
COMPLETION_KEYS = frozenset({"universes", "om_sha256", "tier_a_snapshot", "binance_pairs", "smoke_assets"})


def build_initial_spec(smoke_mode: bool) -> dict:
    return {
        "experiment": "E4",
        "version": 4,
        "changelog": ["v4: REVIEW_07_E4_round1 A/B fixes: GW companion = OOS M0 residual (y_log-M0_pred) on per-asset-standardised LRQA with Driscoll-Kraay HAC (lags>=h) + Wald on LRQA coefs; B logv_mean5/22 = log of rolling mean v; log_bv = 22-day mean of log BV; vectorised/threaded CSR date-block bootstrap; vectorised loss_predictions assembly; geometry arms reuse primary M0. Primary test unchanged.", "v3: Oxford-Man dates fixed in condpred.data (label's first 10 chars = exchange-local date; duplicate (Symbol,date) assert); E4 reverted to om_quality cleaned output; variant cache keys, B_US features, per-asset target_end purge, >=250 training-row fold rule, loss_predictions keys, secondary CIs and Tier A/OM overlap fixed per REVIEW_07_E4_round1. Primary test unchanged.", "v2: E4-local OM universe (superseded by v3)."],
        "smoke": smoke_mode,
        "tiers": {"primary": "OM", "secondary": ["A", "BN"]},
        "om_source": "Oxford-Man archive via condpred.data.oxford_man (SHA-256 in manifest)",
        "om_quality_rule": (
            "E4 universe = cleaned output of condpred.data.om_quality on exchange-local dates "
            "(label's first 10 chars; see BUILD_NOTES_01 Oxford-Man date fix). om_quality rule "
            "(missing_frac_outage>0.05 / missing_frac_raw>0.10 / years>=12) still AWAITS Tasos sign-off "
            "per BUILD_NOTES_01. Rows: drop rv5<=0/NaN; low_nobs flag kept."
        ),
        "tier_a_universe_rule": "20 tickers listed; quality_filter start=first year <1% degenerate; drop <12y; VIX auxiliary only",
        "tier_a_tickers": TIER_A_TICKERS,
        "binance_pairs_rule": "BTC, ETH + 6 largest other USDT pairs with history since 2019 (list frozen here)",
        "binance_pairs": BN_PAIRS,
        "horizons": list(HORIZONS),
        "primary": {
            "tier": "OM", "h": PRIMARY_H, "fixed_model": "LogHAR(refit_every=21, window=1000)",
            "target": "rank of QLIKE; regression target ln(QLIKE+1e-4)",
            "learner": "RidgePredictor", "comparison": "M1 (B+LRQA logv+ret) vs M0 (B)",
            "bootstrap": "date-block 22 trading days, pooled panel",
            "bootstrap_n": smoke(2000, 80), "bootstrap_seed": 0,
            "gate_G2": "delta_spearman>0 and CI low>0 and delta>0 in >=60% of assets",
        },
        "gate_G1_status": "see decision.csv (read from E1/E2 decisions at run time; full E4 worth only if G1 passed)",
        "geometry": {"scaling": "wz", "distance": "euclid", "threshold": "rr", "rr": RR, "m": M, "tau": TAU, "W": W, "theiler": THEILER},
        "features": {"B": "recent_loss(hl22), logv levels, vol surprise, skew/kurt66, jump, downside(OM), perm_entropy(order4), knn_spread(m3,k20,lookback1000)",
                     "M1": "B+LRQA(logv)+LRQA(ret)", "M1w": "B+WRQA(logv)+WRQA(ret)", "M1all": "B+WRQA+LRQA both inputs"},
        "perc": "excluded - rho_perc definition not approved",
        "first_test_year": {"OM": smoke(2006, 2016), "A": smoke(2006, 2019), "BN": smoke(2021, 2020)},
        "backend": "torch", "device_full": "cuda", "device_smoke": "cpu",
        "universes": {},
        "om_sha256": None,
        "tier_a_snapshot": None,
        "smoke_assets": {"OM": OM_SMOKE, "A": ["^GSPC"], "BN": ["BTCUSDT"]} if smoke_mode else None,
        "note_period": "Tier OM ends 2020-02-21 (pre-COVID); Tier A covers 2020+ with noisier proxy. v excludes overnight (OM).",
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


# ---------------------------------------------------------------- data tiers
def _series_dict(asset, tier, dates, v, ret, bv=None, jump_share=None, downside=None, rq=None, overnight=None):
    idx = pd.DatetimeIndex(dates)
    def s(x):
        return pd.Series(np.asarray(x, float), index=idx) if x is not None else pd.Series(np.nan, index=idx)
    return {"asset": asset, "tier": tier, "dates": idx, "v": s(v), "ret": s(ret), "bv": s(bv),
            "jump_share": s(jump_share), "downside": s(downside), "rq": s(rq), "overnight": s(overnight)}


def load_om(spec: dict) -> dict:
    df = cdata.oxford_man()
    manifest = _REPO / "data/raw/oxford_man/manifest.json"
    sha = json.loads(manifest.read_text(encoding="utf-8"))["sha256"] if manifest.exists() else None
    clean, rep = cdata.om_quality(df)
    out = {"report": rep, "sha256": sha, "assets": {}, "variants": {}}
    smoke_mode = spec["smoke"]
    symbols = sorted(clean["Symbol"].unique().tolist())
    for vcol, label in [("rv5", "rv5"), ("rv5_ss", "rv5_ss"), ("rk_parzen", "rk_parzen")]:
        assets = {}
        for sym in symbols:
            if smoke_mode and label == "rv5" and sym not in OM_SMOKE:
                continue
            if smoke_mode and label != "rv5" and sym not in OM_SMOKE:
                continue
            g = clean[clean["Symbol"] == sym].sort_values("date")
            if vcol not in g.columns:
                continue
            g = g[g[vcol] > 0]
            if len(g) < 300:
                continue
            dates = pd.DatetimeIndex(g["date"])
            close = g["close_price"].astype(float).to_numpy() if "close_price" in g.columns else np.full(len(g), np.nan)
            ret = np.log(close[1:] / close[:-1]) if len(close) > 1 else np.array([])
            ret = np.concatenate([[np.nan], ret]) if len(close) else ret
            gap_days = np.concatenate([[0], np.diff(dates.to_numpy()) // np.timedelta64(1, "D")]) if len(dates) else np.array([])
            ret[gap_days > 5] = np.nan
            bv = g["bv"].astype(float) if "bv" in g.columns else None
            jump = (np.maximum(g["rv5"].astype(float) - g["bv"].astype(float), 0) / g["rv5"].astype(float)).to_numpy() if bv is not None else None
            down = (g["rsv"].astype(float) / g["rv5"].astype(float)).to_numpy() if "rsv" in g.columns else None
            assets[sym] = _series_dict(sym, "OM", dates, g[vcol].astype(float).to_numpy(), ret,
                                       bv=bv.to_numpy() if bv is not None else None, jump_share=jump, downside=down)
        if label == "rv5":
            out["assets"] = assets
        else:
            out["variants"][label] = assets
    return out


def load_tier_a(spec: dict) -> dict:
    tickers = TIER_A_TICKERS + ["^VIX"]
    snap = cdata.yahoo_ohlc_snapshot(tickers, "1990-01-01", "2026-11-10", name="e4_universe")
    files = sorted((_REPO / "data/raw/yahoo").glob("condpred_e4_universe_*.csv"))
    snapshot_name = files[-1].name if files else None
    clean, rep = cdata.quality_filter(snap)
    vix = None
    if isinstance(snap.columns, pd.MultiIndex) and "^VIX" in snap.columns.get_level_values(0):
        vix_sub = snap["^VIX"]
        if "Close" in vix_sub.columns:
            vix = vix_sub["Close"].astype(float)
    assets = {}
    kept = [t for t in rep.loc[~rep["dropped"], "ticker"].tolist() if t != "^VIX"] if len(rep) else []
    for tkr in kept:
        if spec["smoke"] and tkr not in spec["smoke_assets"]["A"]:
            continue
        sub = snap[tkr].copy() if isinstance(snap.columns, pd.MultiIndex) else snap.copy()
        # restrict to quality-filtered span: re-derive from clean columns
        if isinstance(clean.columns, pd.MultiIndex) and tkr in clean.columns.get_level_values(0):
            sub = clean[tkr].copy()
            sub.columns = [str(c) for c in sub.columns]
        v = cdata.variance_proxy(sub, kind="gk_overnight")
        ret = cdata.log_returns(sub)
        o, c = sub["Open"].astype(float), sub["Close"].astype(float)
        co = np.log(o / c.shift(1))
        overnight = (co ** 2 / v.replace(0, np.nan)).clip(0, 1)
        dfv = pd.DataFrame({"v": v, "ret": ret}).dropna(subset=["v"])
        dfv = dfv[dfv["v"] > 0]
        if len(dfv) < 300:
            continue
        ov = overnight.reindex(dfv.index)
        assets[tkr] = _series_dict(tkr, "A", dfv.index, dfv["v"].to_numpy(), dfv["ret"].to_numpy(), overnight=ov.to_numpy())
    return {"report": rep, "assets": assets, "vix": vix, "snapshot": snapshot_name, "kept": kept}


def load_bn(spec: dict) -> dict:
    smoke_mode = spec["smoke"]
    symbols = ["BTCUSDT"] if smoke_mode else BN_PAIRS
    start, end = ("2018-01-01", "2021-01-01") if smoke_mode else ("2018-01-01", "2026-11-01")
    raw = cdata.binance_klines(symbols, start, end, interval="1m")
    bars = cdata.to_5min(raw)
    assets = {}
    for sym in symbols:
        try:
            sub = bars.loc[sym]
        except Exception:
            continue
        close = sub["close"].astype(float).sort_index()
        r5 = np.log(close / close.shift(1))
        day = close.index.normalize()
        rows = []
        for d, idx in pd.Series(np.arange(len(close)), index=close.index).groupby(day):
            rr = r5.iloc[idx.to_numpy()]
            rr = rr[np.isfinite(rr)]
            if len(rr) < 200:
                continue
            rv, bv, rq = cdata.realized_from_intraday(rr)
            rows.append((d, rv, bv, rq, float(close.iloc[idx.to_numpy()[-1]])))
        if len(rows) < 300:
            continue
        dfr = pd.DataFrame(rows, columns=["date", "rv", "bv", "rq", "close"]).set_index("date")
        ret = np.log(dfr["close"] / dfr["close"].shift(1))
        jump = (np.maximum(dfr["rv"] - dfr["bv"], 0) / dfr["rv"]).to_numpy()
        assets[sym] = _series_dict(sym, "BN", dfr.index, dfr["rv"].to_numpy(), ret.to_numpy(),
                                   bv=dfr["bv"].to_numpy(), jump_share=jump, rq=dfr["rq"].to_numpy())
    return {"assets": assets, "report": pd.DataFrame({"symbol": list(assets), "n_days": [len(a["dates"]) for a in assets.values()]})}


def load_all(spec: dict, out_dir: Path) -> dict:
    om = load_om(spec)
    ta = load_tier_a(spec)
    bn = load_bn(spec)
    report.save_table(om["report"], out_dir, "quality_report_OM")
    report.save_table(ta["report"], out_dir, "quality_report_A")
    report.save_table(bn["report"], out_dir, "quality_report_BN")
    universes = {
        "OM": sorted(om["assets"]) if not spec["smoke"] else sorted(om["assets"]),
        "OM_quality_kept": sorted(om["assets"]),
        "A": ta["kept"], "BN": BN_PAIRS if not spec["smoke"] else ["BTCUSDT"],
    }
    for tier, assets in [("OM", om["assets"]), ("A", ta["assets"]), ("BN", bn["assets"])]:
        report.save_table(pd.DataFrame({"asset": sorted(assets)}), out_dir, f"universe_{tier}")
    spec["universes"] = universes
    spec["om_sha256"] = om["sha256"]
    spec["tier_a_snapshot"] = ta["snapshot"]
    spec["binance_pairs"] = universes["BN"]
    report.write_spec(spec, out_dir, modules=E4_MODULES, completion_keys=COMPLETION_KEYS)
    return {"OM": om, "A": ta, "BN": bn}


# ---------------------------------------------------------------- fixed models
class _LogHARQ:
    """Log-HAR + log RQ term (Tier BN secondary). Causal block refits like LogHAR."""

    def __init__(self, refit_every=21, window=1000):
        self.refit_every, self.window, self.models = refit_every, window, {}

    def fit(self, v, rq):
        self._v = np.asarray(v, float).clip(1e-12)
        self.v = np.log(self._v)
        self.rq = np.log(np.asarray(rq, float).clip(1e-18))

    def predict(self, t, h):
        if t < 22:
            return np.nan
        from sklearn.linear_model import LinearRegression
        ln_v, v_raw = self.v, self._v
        def feats(i):
            return [ln_v[i], np.mean(ln_v[i-4:i+1]), np.mean(ln_v[i-21:i+1]), self.rq[i]]
        block_end = (t // self.refit_every) * self.refit_every
        key = (block_end, h)
        if key not in self.models:
            rows_x, rows_y = [], []
            for i in range(max(0, block_end - self.window) + 22, block_end - h + 1):
                rows_x.append(feats(i)); rows_y.append(float(np.log(np.mean(v_raw[i+1:i+h+1]))))
            if len(rows_y) < 10:
                return np.nan
            reg = LinearRegression().fit(rows_x, rows_y)
            self.models[key] = (reg, float(np.var(np.asarray(rows_y) - reg.predict(rows_x))))
        reg, s2 = self.models[key]
        return float(np.exp(reg.predict([feats(t)])[0] + s2 / 2))


def forecasts_for_asset(s: dict, model_name: str, horizons, origins) -> pd.DataFrame:
    v = s["v"].to_numpy(); dates = s["dates"]
    if model_name == "LogHAR":
        mdl = LogHAR(refit_every=21, window=1000, horizons=horizons); mdl.fit(v)
        pred = lambda t, h: mdl.predict(t, h)
    elif model_name == "EWMA":
        mdl = EWMAVariance(0.94); mdl.fit(v); pred = lambda t, h: mdl.predict(t, h)
    elif model_name == "HARQ":
        mdl = _LogHARQ(); mdl.fit(v, s["rq"].to_numpy()); pred = lambda t, h: mdl.predict(t, h)
    else:
        raise ValueError(model_name)
    rows = []
    for t in origins:
        for h in horizons:
            if t + h >= len(v):
                continue
            f = pred(int(t), int(h))
            y = float(np.mean(v[t+1:t+h+1]))
            if not np.isfinite(f) or f <= 0 or y <= 0:
                continue
            rows.append({"asset": s["asset"], "tier": s["tier"], "date": dates[t], "target_end": dates[t + h], "origin_idx": int(t),
                         "model": model_name, "h": int(h), "forecast": float(f), "realised": y,
                         "qlike": float(_ql(y, f)), "log_ratio": float(_lr(y, f))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- features
def _perm_entropy_series(x: np.ndarray, W: int, order=4) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < W:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(x, order)  # (n-order+1, order)
    ranks = np.argsort(np.argsort(sw, axis=1), axis=1)
    pid = ranks[:, 0] * 64 + ranks[:, 1] * 16 + ranks[:, 2] * 4 + ranks[:, 3]
    npat = 256
    onehot = np.zeros((len(pid), npat)); onehot[np.arange(len(pid)), pid] = 1.0
    cum = np.vstack([np.zeros((1, npat)), np.cumsum(onehot, axis=0)])
    # window ending at end: tuples starting in [end-W+1, end-order+1]
    for end in range(W - 1, n):
        a, b = end - W + 1, end - order + 2
        if b <= a:
            continue
        cnt = cum[b] - cum[a]
        p = cnt[cnt > 0] / cnt.sum()
        out[end] = float(-(p * np.log(p)).sum() / np.log(24))
    return out


def b_static(s: dict) -> pd.DataFrame:
    """h-independent B features (vectorised rolling), indexed by date."""
    v, ret = s["v"], s["ret"]
    logv = np.log(v.clip(lower=1e-12))
    df = pd.DataFrame(index=s["dates"])
    df["logv"] = logv
    df["logv_mean5"] = np.log(v.rolling(5).mean().clip(lower=1e-12))
    df["logv_mean22"] = np.log(v.rolling(22).mean().clip(lower=1e-12))
    df["vol_surprise"] = v / v.rolling(W).median()
    df["skew66"] = ret.rolling(66).skew()
    df["kurt66"] = ret.rolling(66).kurt()
    if s["tier"] in ("OM", "BN"):
        df["jump_share22"] = s["jump_share"].rolling(22).mean()
        df["log_bv"] = np.log(s["bv"].clip(lower=1e-12)).rolling(22).mean()
    else:
        sd66 = ret.rolling(66).std()
        df["jump_share22"] = (ret.abs() > 3 * sd66).astype(float).rolling(22).mean()
        df["log_bv"] = np.nan
    df["downside_share"] = s["downside"]
    df["overnight_share22"] = s["overnight"].rolling(22).mean()
    logv_a = logv.to_numpy(); ret_a = np.nan_to_num(ret.to_numpy(), nan=0.0)
    df["perm_ret"] = _perm_entropy_series(ret_a, W)
    df["perm_logv"] = _perm_entropy_series(logv_a, W)
    if s["tier"] == "BN":
        df["log_rq"] = np.log(s["rq"].clip(lower=1e-18))
        df["rq_rv2"] = s["rq"] / (v ** 2).clip(lower=1e-18)
    return df


def b_for_h(s: dict, static: pd.DataFrame, fc_h: pd.DataFrame, h: int) -> pd.DataFrame:
    """Add recent_loss + knn_spread for horizon h (fc_h = forecasts of one fixed model at h)."""
    out = static.copy()
    v = s["v"].to_numpy(); dates = s["dates"]
    loss_by_origin = np.full(len(v), np.nan)
    for r in fc_h.itertuples():
        loss_by_origin[int(r.origin_idx)] = float(r.qlike)
    logv = np.log(np.clip(v, 1e-12, None))
    rl = np.full(len(v), np.nan); kn = np.full(len(v), np.nan)
    ends = sorted(set(int(t) for t in fc_h["origin_idx"])) if len(fc_h) else []
    for t in ends:
        rl[t] = baselines.recent_loss(loss_by_origin, h, 22.0, t)
        kn[t] = baselines.knn_spread(logv, t, M, TAU, 20, h, 1000)
    out["recent_loss"] = rl
    out["knn_spread"] = kn
    return out


def _geom_default():
    return Geometry(scaling="wz", distance="euclid", threshold="rr", rr=RR, m=M, tau=TAU, theiler=THEILER)


def rqa_frames_for_asset(s: dict, ends, backend, device, n_jobs=4):
    geom = _geom_default()
    logv = np.log(s["v"].clip(lower=1e-12).to_numpy())
    ret = np.nan_to_num(s["ret"].to_numpy(), nan=0.0)
    res = {}
    for name, x in [("logv", logv), ("ret", ret)]:
        wr, lr = wrqa_lrqa_frames(x, W, M, TAU, geom, ends=np.asarray(ends, dtype=int),
                                  n_jobs=n_jobs, backend=backend, device=device)
        wr = wr[[c for c in WRQA_KEYS if c in wr.columns]].add_prefix(f"wrqa_{name}_")
        lr = lr[[c for c in LRQA_KEYS if c in lr.columns]].add_prefix(f"lrqa_{name}_")
        res[name] = (wr, lr)
    return res


def geom_frames_for_asset(s: dict, ends, train_cut_pos: int, backend, device, n_jobs=4):
    """GEOM secondary arms on log v (gz, eps_train, beta_pos b=0/1 + power_ctrl controls)."""
    logv = np.log(s["v"].clip(lower=1e-12).to_numpy())
    v = s["v"].to_numpy()
    arms = {}
    g = _geom_default(); g.scaling = "gz"
    g.train_mean, g.train_sd = float(np.mean(logv[:train_cut_pos])), float(np.std(logv[:train_cut_pos]) or 1.0)
    arms["gz"] = (g, None)
    g2 = _geom_default(); g2.threshold = "eps_train"
    wins = [logv[max(0, t - W + 1):t + 1] for t in range(W - 1, max(W, train_cut_pos), 25)][:50]
    g2.fit_eps_train([w for w in wins if len(w) == W])
    arms["eps_train"] = (g2, None)
    med = float(np.median(v[:train_cut_pos])) if train_cut_pos else float(np.median(v))
    for b in (0.0, 1.0):
        for dist in ("beta_pos", "power_ctrl"):
            gb = _geom_default(); gb.distance = dist; gb.beta = b; gb.train_median = med
            arms[f"{dist}{int(b)}"] = (gb, v)
    out = {}
    for name, (geom, pos) in arms.items():
        wr, lr = wrqa_lrqa_frames(logv, W, M, TAU, geom, ends=np.asarray(ends, dtype=int), n_jobs=n_jobs,
                                  backend=backend, device=device, positive_raw=pos)
        both = pd.concat([wr[[c for c in WRQA_KEYS if c in wr.columns]].add_prefix("wrqa_logv_"),
                          lr[[c for c in LRQA_KEYS if c in lr.columns]].add_prefix("lrqa_logv_")], axis=1)
        both.index = [s["dates"][int(i)] for i in both.index]
        out[name] = both.add_prefix(f"geom_{name}_")
    return out


def feature_groups(panel_cols, tier="OM", us=False):
    b = ["recent_loss", "knn_spread", "logv", "logv_mean5", "logv_mean22", "vol_surprise", "skew66", "kurt66",
         "jump_share22", "log_bv", "perm_ret", "perm_logv"]
    if tier == "OM":
        b += ["downside_share"]
    if tier == "A":
        b = [c if c != "log_bv" else "overnight_share22" for c in b]
    if tier == "BN":
        b += ["log_rq", "rq_rv2"]
    b = [c for c in b if c in panel_cols]
    wr = [c for c in panel_cols if c.startswith("wrqa_")]
    lr = [c for c in panel_cols if c.startswith("lrqa_")]
    groups = {"M0": b, "M1": b + lr, "M1w": b + wr, "M1all": b + wr + lr}
    if us:
        extra = [c for c in ["vix_log", "vrp"] if c in panel_cols]
        groups.update({"M0_US": b + extra, "M1_US": b + extra + lr})
    return groups


# ---------------------------------------------------------------- loss predictor
def _standardise(train_X: pd.DataFrame, test_X: pd.DataFrame, train_assets, test_assets, cols):
    Xtr = train_X.copy(); Xte = test_X.copy()
    for a in pd.unique(train_assets):
        mtr = train_assets == a
        mu = Xtr.loc[mtr, cols].mean(); sd = Xtr.loc[mtr, cols].std().replace(0, np.nan)
        Xtr.loc[mtr, cols] = (Xtr.loc[mtr, cols] - mu) / sd
        mte = test_assets == a
        if mte.any():
            Xte.loc[mte, cols] = (Xte.loc[mte, cols] - mu) / sd
    return Xtr, Xte


def loss_predictions(panel: pd.DataFrame, groups: dict, h: int, first_test_year: int,
                     target="qlike", learner="ridge", fixed_model="LogHAR", variant="rv5",
                     table="", item="") -> pd.DataFrame:
    sub = panel[panel["h"] == h].copy().sort_values("date").reset_index(drop=True)
    if target == "qlike":
        sub["_y"] = np.log(sub["qlike"].to_numpy() + 1e-4); eval_col = "qlike"
    else:
        sub["_y"] = sub["log_ratio"].to_numpy(); eval_col = "log_ratio"
    sub["_eval"] = sub[eval_col].to_numpy()
    sub["_date"] = pd.to_datetime(sub["date"])
    sub["_target_end"] = pd.to_datetime(sub["target_end"]) if "target_end" in sub.columns else pd.Series(pd.NaT, index=sub.index)
    all_cols = sorted(set(c for v in groups.values() for c in v))
    parts = []
    coefs = []
    dropped_fix4 = []
    for fold_year in sorted(y for y in sub["_date"].dt.year.unique() if y >= first_test_year):
        test_mask = (sub["_date"].dt.year == fold_year).to_numpy()
        test_idx_all = np.where(test_mask)[0]
        if len(test_idx_all) == 0:
            continue
        first_test_date = sub["_date"].iloc[test_idx_all].min()
        # Fix 5: purge on each asset's own target_end, not the pooled calendar.
        train_mask = (sub["_date"].dt.year < fold_year).to_numpy()
        if sub["_target_end"].notna().any():
            train_mask &= (sub["_target_end"] < first_test_date).to_numpy()
        train_idx = np.where(train_mask)[0]
        train, test = sub.iloc[train_idx], sub.iloc[test_idx_all]
        ok = np.isfinite(train["_y"].to_numpy())
        train = train.loc[ok]
        # Fix 4: drop an asset's test rows unless >=250 training rows in this fold.
        keep_assets = []
        for a, cnt in train.groupby("asset").size().items():
            if cnt >= smoke(250, 40):
                keep_assets.append(a)
        dropped = test[~test["asset"].isin(keep_assets)]
        if len(dropped):
            for a, cnt in dropped.groupby("asset").size().items():
                n_train = int((train["asset"] == a).sum())
                dropped_fix4.append({"asset": a, "fold": int(fold_year), "n_train": n_train, "n_test_dropped": int(cnt)})
                print(f"[fix4] drop test asset={a} fold={fold_year} n_train={n_train} (<{smoke(250, 40)}) n_test_dropped={int(cnt)}")
            test = test[test["asset"].isin(keep_assets)]
        if len(train) < 50 or len(test) == 0:
            continue
        Xtr_all, Xte_all = _standardise(train[all_cols], test[all_cols], train["asset"].to_numpy(), test["asset"].to_numpy(), all_cols)
        _allnan = Xtr_all.columns[Xtr_all.isna().all()].tolist()
        if _allnan:
            Xtr_all[_allnan] = 0.0; Xte_all[_allnan] = 0.0
        for name, cols in groups.items():
            mdl = RidgePredictor() if learner == "ridge" else LGBMPredictor()
            mdl.fit(Xtr_all[cols].to_numpy(), train["_y"].to_numpy())
            pred = mdl.predict(Xte_all[cols].to_numpy())
            if name == "M1" and learner == "ridge" and getattr(mdl, "pipe", None) is not None:
                coef = mdl.pipe.named_steps["ridge"].coef_[:len(cols)]
                coefs.append(pd.DataFrame({"feature": cols, "coef": coef, "fold": fold_year}))
            # B4: vectorised assembly (no per-row iterrows).
            n_te = len(test)
            parts.append(pd.DataFrame({
                "asset": test["asset"].to_numpy(), "tier": test["tier"].to_numpy(), "date": test["date"].to_numpy(),
                "h": np.full(n_te, h), "fold": np.full(n_te, int(fold_year)), "model": np.full(n_te, name),
                "learner": np.full(n_te, learner), "target": np.full(n_te, target), "prediction": np.asarray(pred, float),
                "fixed_model": np.full(n_te, fixed_model), "variant": np.full(n_te, variant),
                "table": np.full(n_te, table), "item": np.full(n_te, item),
                "qlike": test["qlike"].to_numpy(dtype=float), "log_ratio": test["log_ratio"].to_numpy(dtype=float),
                "y_log": test["_y"].to_numpy(dtype=float), "y_eval": test["_eval"].to_numpy(dtype=float),
                "train_y_mean": np.full(n_te, float(train["_y"].mean())),
            }))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["asset", "tier", "date", "h", "fold", "model", "learner", "target", "prediction", "fixed_model", "variant", "table", "item", "qlike", "log_ratio", "y_log", "y_eval", "train_y_mean"])
    out.attrs["coefs"] = pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame(columns=["feature", "coef", "fold"])
    out.attrs["dropped_fix4"] = pd.DataFrame(dropped_fix4, columns=["asset", "fold", "n_train", "n_test_dropped"])
    print(f"[fix4] total test rows dropped (<{smoke(250, 40)} train): {int(out.attrs['dropped_fix4']['n_test_dropped'].sum()) if dropped_fix4 else 0} across {len(dropped_fix4)} asset-folds")
    return out


def _delta_from_preds(preds: pd.DataFrame, m0="M0", m1="M1") -> dict:
    p = preds.pivot_table(index=["asset", "date"], columns="model", values="prediction", aggfunc="first").dropna()
    ydf = preds.drop_duplicates(["asset", "date"]).set_index(["asset", "date"])
    if m0 not in p or m1 not in p or len(p) < 20:
        return {"delta": np.nan, "n": len(p)}
    y = ydf.loc[p.index, "y_eval"].to_numpy()
    return {"delta": spearman(y, p[m1].to_numpy()) - spearman(y, p[m0].to_numpy()), "n": len(p),
            "spearman_M0": spearman(y, p[m0].to_numpy()), "spearman_M1": spearman(y, p[m1].to_numpy())}


def _csr_date_groups(date_arr):
    """CSR offsets for rows grouped by unique date (sorted); rows reordered by date."""
    codes, uniq = pd.factorize(pd.to_datetime(date_arr), sort=True)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(uniq)).astype(np.int64)
    offsets = np.zeros(len(uniq) + 1, dtype=np.int64); np.cumsum(counts, out=offsets[1:])
    return np.asarray(uniq), order, counts, offsets


def _all_draw_date_idx(n_dates: int, block: int, rng, n_boot: int) -> np.ndarray:
    """All date-block bootstrap draws at once: (n_boot, n_dates) unique-date indices.
    Same sampling law as panel_date_bootstrap (block starts uniform, truncated to n_dates)."""
    if n_dates == 0 or n_boot <= 0:
        return np.empty((0, 0), dtype=np.int64)
    block = max(1, int(block))
    n_blocks = int(np.ceil(n_dates / block))
    starts = rng.integers(0, max(1, n_dates - block + 1), size=(n_boot, n_blocks))
    picked = (starts[:, :, None] + np.arange(block, dtype=np.int64)).reshape(n_boot, -1)[:, :n_dates]
    return np.clip(picked, 0, n_dates - 1)


def _row_idx_for_draw(offsets, counts, picked) -> np.ndarray:
    """Expand picked unique-date indices to row indices via CSR offsets (vectorised)."""
    reps = counts[picked]
    total = int(reps.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(reps) - reps, reps)
    return np.repeat(offsets[picked], reps) + within


def _threaded_deltas(fn, n_boot: int) -> np.ndarray:
    workers = max(1, min(4, (os.cpu_count() or 1)))
    if n_boot <= 0:
        return np.empty(0)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        vals = list(ex.map(fn, range(n_boot)))
    out = np.asarray(vals, dtype=float)
    return out[np.isfinite(out)]


def _boot_ci(base_delta: float, deltas: np.ndarray) -> dict:
    if len(deltas) == 0 or not np.isfinite(base_delta):
        return {"ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    pval = float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))) if base_delta != 0 else 1.0
    return {"ci_low": float(np.nanpercentile(deltas, 2.5)), "ci_high": float(np.nanpercentile(deltas, 97.5)),
            "p": min(max(pval, 1.0 / (len(deltas) + 1)), 1.0)}


def gw_companion(preds: pd.DataFrame, panel: pd.DataFrame, groups: dict, h: int = PRIMARY_H) -> dict:
    """Plan GW companion: OOS M0 residual y_log-M0_pred on per-asset-standardised LRQA,
    Driscoll-Kraay HAC (lags>=h, scores summed by date) + Wald test on LRQA coefficients."""
    out = {"gw_stat": np.nan, "gw_p": np.nan, "gw_lags": np.nan, "gw_n": 0, "gw_n_dates": 0, "gw_k_lrqa": 0}
    if not len(preds) or not len(panel):
        return out
    piv = preds.pivot_table(index=["asset", "date"], columns="model", values="prediction", aggfunc="first").dropna()
    lrqa_cols = [c for c in groups.get("M1", []) if c.startswith("lrqa_")]
    if "M0" not in piv or not lrqa_cols:
        return out
    base = panel[panel["h"] == h].drop_duplicates(["asset", "date"]).set_index(["asset", "date"])
    common = piv.index.intersection(base.index)
    if len(common) <= 50:
        return out
    yd = preds.drop_duplicates(["asset", "date"]).set_index(["asset", "date"]).loc[common, "y_log"].to_numpy(dtype=float)
    resid = yd - piv.loc[common, "M0"].to_numpy(dtype=float)
    zdf = base.loc[common, lrqa_cols].apply(pd.to_numeric, errors="coerce")
    assets = np.array([a for a, _ in common]); dates = pd.to_datetime([d for _, d in common])
    # Per-asset standardisation (mirrors loss-predictor scaling), NaN -> 0 after scaling.
    zdf = zdf.copy(); zdf["_asset"] = assets
    grp = zdf.groupby("_asset")[lrqa_cols]
    z = ((zdf[lrqa_cols] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)).to_numpy(dtype=float)
    z = np.where(np.isfinite(z), z, 0.0)
    keep = np.array([np.std(z[:, j]) > 0 for j in range(z.shape[1])])
    if not keep.any():
        return out
    z = z[:, keep]
    ok = np.isfinite(resid)
    if ok.sum() <= z.shape[1] + 10:
        return out
    resid, z, dates, assets = resid[ok], z[ok], dates[ok], assets[ok]
    X = np.column_stack([np.ones(len(resid)), z])
    beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
    e = resid - X @ beta
    scores = X * e[:, None]
    codes, uniq_dates = pd.factorize(dates, sort=True)
    S = np.zeros((len(uniq_dates), X.shape[1])); np.add.at(S, codes, scores)
    T = len(uniq_dates)
    lags = int(max(int(h), np.floor(4 * (T / 100) ** (2 / 9)) if T > 0 else int(h)))
    meat = S.T @ S
    for j in range(1, lags + 1):
        if j >= T:
            break
        w = 1 - j / (lags + 1)
        gj = S[j:].T @ S[:-j]
        meat += w * (gj + gj.T)
    try:
        bread = np.linalg.pinv(X.T @ X)
        vcov = bread @ meat @ bread; vcov = 0.5 * (vcov + vcov.T)
        sub = vcov[1:, 1:]; bsub = beta[1:]
        wald = float(bsub @ np.linalg.pinv(sub) @ bsub)
        pval = float(stats.chi2.sf(wald, df=len(bsub))) if np.isfinite(wald) else np.nan
    except Exception:
        return out
    return {"gw_stat": wald, "gw_p": pval, "gw_lags": lags, "gw_n": int(len(resid)), "gw_n_dates": int(T), "gw_k_lrqa": int(z.shape[1])}


def delta_bootstrap(preds: pd.DataFrame, n_boot: int, rng, m0="M0", m1="M1", block=22) -> dict:
    base = _delta_from_preds(preds, m0, m1)
    if not np.isfinite(base.get("delta", np.nan)):
        return {**base, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    dates = preds.drop_duplicates(["asset", "date"]).set_index(["asset", "date"])
    p = preds.pivot_table(index=["asset", "date"], columns="model", values="prediction", aggfunc="first").dropna()
    y = dates.loc[p.index, "y_eval"].to_numpy()
    a0, a1 = p[m0].to_numpy(), p[m1].to_numpy()
    date_arr = pd.to_datetime([d for _, d in p.index])
    uniq, order, counts, offsets = _csr_date_groups(date_arr)
    y, a0, a1 = y[order], a0[order], a1[order]
    picked_all = _all_draw_date_idx(len(uniq), block, rng, int(n_boot))
    def _one(d):
        idx = _row_idx_for_draw(offsets, counts, picked_all[d])
        return spearman(y[idx], a1[idx]) - spearman(y[idx], a0[idx]) if len(idx) >= 20 else np.nan
    deltas = _threaded_deltas(_one, int(n_boot))
    if len(deltas) == 0:
        return {**base, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    return {**base, **_boot_ci(base["delta"], deltas), "boot_deltas": deltas}


def auc_delta_bootstrap(preds: pd.DataFrame, thr_by_asset, n_boot: int, rng, block=22) -> dict:
    """Date-block bootstrap for M1-M0 event AUC (event = qlike > train 90th pct). CSR + threaded."""
    from sklearn.metrics import roc_auc_score
    p = preds.pivot_table(index=["asset", "date"], columns="model", values="prediction", aggfunc="first").dropna()
    qdf = preds.drop_duplicates(["asset", "date"]).set_index(["asset", "date"])
    if "M0" not in p or "M1" not in p or len(p) < 20:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": len(p)}
    q = qdf.loc[p.index, "qlike"].to_numpy()
    assets_arr = np.array([a for a, _ in p.index])
    thr = np.array([thr_by_asset.get(a, np.nan) for a in assets_arr], dtype=float)
    ev = (q > thr).astype(float)
    ok = np.isfinite(ev) & np.isfinite(thr)
    p, ev = p[ok], ev[ok]
    if len(np.unique(ev)) < 2:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": len(p)}
    a0, a1 = p["M0"].to_numpy(), p["M1"].to_numpy()
    base = float(roc_auc_score(ev, a1) - roc_auc_score(ev, a0))
    date_arr = pd.to_datetime([d for _, d in p.index])
    uniq, order, counts, offsets = _csr_date_groups(date_arr)
    ev, a0, a1 = ev[order], a0[order], a1[order]
    picked_all = _all_draw_date_idx(len(uniq), block, rng, int(n_boot))
    def _one(d):
        idx = _row_idx_for_draw(offsets, counts, picked_all[d])
        if len(idx) < 20 or len(np.unique(ev[idx])) < 2:
            return np.nan
        return float(roc_auc_score(ev[idx], a1[idx]) - roc_auc_score(ev[idx], a0[idx]))
    deltas = _threaded_deltas(_one, int(n_boot))
    if len(deltas) == 0:
        return {"delta": base, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": len(p)}
    return {"delta": base, **_boot_ci(base, deltas), "n": len(p)}


# ---------------------------------------------------------------- panel building
def build_panel(assets: dict, tier: str, fixed_model: str, horizons, spec: dict, cache_exp: str,
                vix: pd.Series | None = None, want_geom=False, want_rqa=True, variant="rv5") -> tuple[pd.DataFrame, dict]:
    smoke_mode = spec["smoke"]
    backend = "torch"; device = "cpu" if smoke_mode else "cuda"
    step = smoke(1, 5)
    first_year = spec["first_test_year"][tier]
    panels, feat_frames = [], {"B": [], "WRQA": [], "LRQA": [], "GEOM": []}
    for asset, s in assets.items():
        n = len(s["dates"])
        maxh = max(horizons)
        origins = np.arange(W - 1, n - maxh, step, dtype=int)
        if smoke_mode and tier == "OM":
            yrs = s["dates"].year
            origins = np.array([t for t in origins if 2010 <= yrs[t] <= 2019], dtype=int)
        if smoke_mode and tier == "A":
            yrs = s["dates"].year
            origins = np.array([t for t in origins if 2015 <= yrs[t] <= 2019], dtype=int)
        if smoke_mode and tier == "BN":
            yrs = s["dates"].year
            origins = np.array([t for t in origins if yrs[t] in (2019, 2020)], dtype=int)
        if len(origins) == 0:
            continue
        key = {"tier": tier, "asset": asset, "model": fixed_model, "variant": variant, "horizons": list(horizons), "origins": [int(origins[0]), int(origins[-1]), int(step)]}
        fc = cached(cache_exp, f"fc_{tier}_{asset}_{fixed_model}_{variant}".replace("/", "_").replace("^", "").replace(".", ""),
                    key, lambda s=s, o=origins: forecasts_for_asset(s, fixed_model, horizons, o), modules=E4_MODULES)
        if len(fc) == 0:
            continue
        static = b_static(s)
        if vix is not None and asset in US_OM:
            # Fix 3 (B_US): same-day US VIX close is causal; dates are exchange-local (fix 1).
            vix_s = pd.Series(np.asarray(vix, dtype=float), index=pd.DatetimeIndex(vix.index)).sort_index()
            vix_al = vix_s.reindex(s["dates"])
            static["vix_log"] = np.log(vix_al.clip(lower=1e-12))
            static["vrp"] = (vix_al / 100.0) ** 2 - 252.0 * s["v"].rolling(22).mean()
        rqa_by_input = {}
        if want_rqa:
            rkey = {"tier": tier, "asset": asset, "variant": variant, "ends": [int(origins[0]), int(origins[-1]), int(step)], "backend": backend, "device": device}
            rqa_by_input = cached(cache_exp, f"rqa_{tier}_{asset}_{variant}".replace("/", "_").replace("^", "").replace(".", ""),
                                  rkey, lambda s=s, o=origins: rqa_frames_for_asset(s, o, backend, device, n_jobs=4), modules=E4_MODULES)
        base_frames = []
        for name in ("logv", "ret"):
            if name in rqa_by_input:
                wr, lr = rqa_by_input[name]
                for fr, fam in [(wr, "WRQA"), (lr, "LRQA")]:
                    f2 = fr.copy(); f2.index = [s["dates"][int(i)] for i in f2.index]
                    f2 = f2.reset_index(names="date"); f2["asset"] = asset; f2["tier"] = tier
                    feat_frames[fam].append(f2)
                base_frames.append(rqa_by_input[name][0]); base_frames.append(rqa_by_input[name][1])
        rqa_all = None
        if base_frames:
            rqa_all = base_frames[0]
            for fr in base_frames[1:]:
                rqa_all = rqa_all.join(fr, how="outer")
            rqa_all.index = [s["dates"][int(i)] for i in rqa_all.index]
        geom_all = None
        if want_geom and tier == "OM":
            cut_dates = s["dates"][s["dates"].year < first_year]
            cut_pos = len(cut_dates) if len(cut_dates) else W
            gkey = {"tier": tier, "asset": asset, "variant": variant, "ends": [int(origins[0]), int(origins[-1]), int(step)], "cut": int(cut_pos)}
            gdict = cached(cache_exp, f"geom_{asset}_{variant}".replace(".", ""), gkey,
                           lambda s=s, o=origins, c=cut_pos: geom_frames_for_asset(s, o, c, backend, device, n_jobs=4), modules=E4_MODULES)
            geom_all = pd.concat(gdict.values(), axis=1) if gdict else None
            if geom_all is not None:
                f2 = geom_all.reset_index(names="date"); f2["asset"] = asset; f2["tier"] = tier
                feat_frames["GEOM"].append(f2)
        for h in horizons:
            fc_h = fc[fc["h"] == h]
            if len(fc_h) == 0:
                continue
            bh = b_for_h(s, static, fc_h, h)
            bf = bh.reset_index(names="date"); bf["asset"] = asset; bf["tier"] = tier; bf["h"] = h
            feat_frames["B"].append(bf)
            panel = fc_h.copy()
            panel = panel.merge(bh.reset_index(names="date"), on="date", how="left")
            if rqa_all is not None:
                panel = panel.merge(rqa_all.reset_index(names="date"), on="date", how="left")
            if geom_all is not None:
                panel = panel.merge(geom_all.reset_index(names="date"), on="date", how="left")
            panels.append(panel)
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    return panel, feat_frames


def run_pipeline(spec: dict, out_dir: Path, fig_dir: Path, cache_exp: str, data: dict | None = None) -> dict:
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec["primary"]["bootstrap_seed"])
    n_boot = int(spec["primary"]["bootstrap_n"])
    smoke_mode = spec["smoke"]
    data = data if data is not None else load_all(spec, out_dir)
    om_assets, ta_assets, bn_assets = data["OM"]["assets"], data["A"]["assets"], data["BN"]["assets"]
    vix = data["A"].get("vix")
    horizons = (PRIMARY_H,) if smoke_mode else HORIZONS

    # --- panels (LogHAR primary; EWMA / variants / BN secondary) ---
    panel_om, feats_om = build_panel(om_assets, "OM", "LogHAR", horizons, spec, cache_exp, vix=vix, want_geom=True)
    panel_a, feats_a = build_panel(ta_assets, "A", "LogHAR", horizons, spec, cache_exp, vix=vix)
    panel_bn, feats_bn = build_panel(bn_assets, "BN", "LogHAR", horizons, spec, cache_exp)
    panel_om_ewma, _ = build_panel(om_assets, "OM", "EWMA", (PRIMARY_H,), spec, cache_exp, vix=vix)
    panel_bn_harq, _ = build_panel(bn_assets, "BN", "HARQ", (PRIMARY_H,), spec, cache_exp) if bn_assets else (pd.DataFrame(), {})
    variant_panels = {}
    for label, assets in data["OM"].get("variants", {}).items():
        if assets:
            variant_panels[label], _ = build_panel(assets, "OM", "LogHAR", (PRIMARY_H,), spec, cache_exp, vix=vix, variant=label)

    # --- reusable artefacts ---
    fc_all = pd.concat([p for p in [panel_om, panel_a, panel_bn, panel_om_ewma, panel_bn_harq] if len(p)], ignore_index=True) if any(len(p) for p in [panel_om, panel_a, panel_bn]) else pd.DataFrame()
    if len(fc_all):
        fc_all[[c for c in ["asset", "tier", "date", "model", "h", "forecast", "realised", "qlike", "log_ratio"] if c in fc_all]].to_parquet(out_dir / "forecasts.parquet", index=False)
    for fam in ("B", "WRQA", "LRQA", "GEOM"):
        frames = []
        for d in (feats_om, feats_a, feats_bn):
            frames += d.get(fam, [])
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(out_dir / f"features_{fam}.parquet", index=False)

    groups_om = feature_groups(panel_om.columns, "OM", us=True) if len(panel_om) else {}
    groups_a = feature_groups(panel_a.columns, "A") if len(panel_a) else {}
    groups_bn = feature_groups(panel_bn.columns, "BN") if len(panel_bn) else {}
    fy = spec["first_test_year"]

    # --- primary ---
    preds_primary = loss_predictions(panel_om, groups_om, PRIMARY_H, fy["OM"], target="qlike", learner="ridge", fixed_model="LogHAR", variant="rv5", table="primary", item="M1_vs_M0") if len(panel_om) else pd.DataFrame()
    boot = delta_bootstrap(preds_primary, n_boot, rng, "M0", "M1") if len(preds_primary) else {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": 0}
    per_asset_rows = []
    if len(preds_primary):
        for asset, gdf in preds_primary.groupby("asset"):
            b = delta_bootstrap(gdf, n_boot, rng, "M0", "M1")
            per_asset_rows.append({"asset": asset, "n": b.get("n", 0), "delta_spearman": b.get("delta", np.nan),
                                   "ci_low": b.get("ci_low", np.nan), "ci_high": b.get("ci_high", np.nan), "p_raw": b.get("p", np.nan),
                                   "spearman_M0": b.get("spearman_M0", np.nan), "spearman_M1": b.get("spearman_M1", np.nan),
                                   "flag_2020_partial": "2020 test year is Jan-Feb only (OM ends 2020-02-21)"})
    per_asset = pd.DataFrame(per_asset_rows)
    share_pos = float((per_asset["delta_spearman"] > 0).mean()) if len(per_asset) else np.nan
    # pooled OOS R2 of ln(QLIKE)
    r2 = {}
    if len(preds_primary):
        for mname in ("M0", "M1"):
            pp = preds_primary[preds_primary["model"] == mname]
            ss_res = float(((pp["y_log"] - pp["prediction"]) ** 2).sum())
            ss_tot = float(((pp["y_log"] - pp["train_y_mean"]) ** 2).sum())
            r2[mname] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # A1 GW companion (plan): OOS M0 residual on standardised LRQA, Driscoll-Kraay HAC lags>=h, Wald.
    gw = gw_companion(preds_primary, panel_om, groups_om, h=PRIMARY_H) if len(preds_primary) else {"gw_stat": np.nan, "gw_p": np.nan, "gw_lags": np.nan, "gw_n": 0, "gw_n_dates": 0, "gw_k_lrqa": 0}
    gw_stat, gw_p = gw["gw_stat"], gw["gw_p"]
    g2_pass = bool(np.isfinite(boot.get("delta", np.nan)) and boot["delta"] > 0 and np.isfinite(boot.get("ci_low", np.nan)) and boot["ci_low"] > 0 and np.isfinite(share_pos) and share_pos >= 0.60)
    primary_row = {"tier": "OM", "h": PRIMARY_H, "fixed_model": "LogHAR", "learner": "ridge", "comparison": "M1_vs_M0",
                   "delta_spearman": boot.get("delta"), "ci_low": boot.get("ci_low"), "ci_high": boot.get("ci_high"),
                   "p_bootstrap": boot.get("p"), "n": boot.get("n"), "share_assets_positive": share_pos,
                   "oos_r2_M0": r2.get("M0"), "oos_r2_M1": r2.get("M1"),
                   "oos_r2_delta": (r2.get("M1", np.nan) - r2.get("M0", np.nan)) if r2 else np.nan,
                   "gw_stat": gw_stat, "gw_p": gw_p, "gw_method": "M0-residual on per-asset-standardised LRQA; Driscoll-Kraay HAC lags>=h; Wald on LRQA", "gw_lags": gw.get("gw_lags"), "gw_n": gw.get("gw_n"), "gw_n_dates": gw.get("gw_n_dates"), "gw_k_lrqa": gw.get("gw_k_lrqa"), "gate_G2": "pass" if g2_pass else "fail", "smoke": smoke_mode,
                   "flag_2020_partial": "2020 test year is Jan-Feb only (OM ends 2020-02-21)"}
    report.save_table(pd.DataFrame([primary_row]), out_dir, "results_primary")
    report.save_table(per_asset, out_dir, "per_asset")
    coef_tbl = preds_primary.attrs.get("coefs", pd.DataFrame())
    if len(coef_tbl):
        coef_tbl = coef_tbl.groupby("feature", as_index=False)["coef"].mean().sort_values("coef", key=lambda s: s.abs(), ascending=False)
    report.save_table(coef_tbl, out_dir, "ridge_coef_M1")

    # --- secondaries (each row tagged with holm table) ---
    sec_rows = []
    all_preds = [preds_primary] if len(preds_primary) else []
    def add_sec(table, label, preds, m0="M0", m1="M1", tier="OM", h=PRIMARY_H, fixed="LogHAR", learner="ridge", target="qlike", variant="rv5", append=True):
        if preds is None or len(preds) == 0:
            sec_rows.append({"table": table, "item": label, "tier": tier, "h": h, "fixed_model": fixed, "learner": learner,
                             "target": target, "comparison": f"{m1}_vs_{m0}", "delta_spearman": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_raw": np.nan, "n": 0})
            return
        b = delta_bootstrap(preds, smoke(n_boot, 40) if not smoke_mode else 40, rng, m0, m1)
        sec_rows.append({"table": table, "item": label, "tier": tier, "h": h, "fixed_model": fixed, "learner": learner, "target": target,
                         "comparison": f"{m1}_vs_{m0}", "delta_spearman": b.get("delta"), "ci_low": b.get("ci_low"), "ci_high": b.get("ci_high"),
                         "p_raw": b.get("p"), "n": b.get("n")})
        if append:
            tagged = preds.copy(); tagged.attrs = {}
            tagged["fixed_model"] = fixed; tagged["variant"] = variant; tagged["table"] = table; tagged["item"] = label
            all_preds.append(tagged)
    if not smoke_mode:
        for h in (1, 22):
            add_sec("horizons", f"M1_h{h}", loss_predictions(panel_om, groups_om, h, fy["OM"], table="horizons", item=f"M1_h{h}") if len(panel_om) else None, h=h)
    # Fix 6: M1w/M1all already in preds_primary (groups_om); do not re-run or re-append.
    add_sec("models", "M1w", preds_primary if len(preds_primary) else None, m1="M1w", append=False)
    add_sec("models", "M1all", preds_primary if len(preds_primary) else None, m1="M1all", append=False)
    add_sec("models", "M1_lgbm", loss_predictions(panel_om, groups_om, PRIMARY_H, fy["OM"], learner="lgbm", table="models", item="M1_lgbm") if len(panel_om) else None, learner="lgbm")
    preds_signed = loss_predictions(panel_om, groups_om, PRIMARY_H, fy["OM"], target="log_ratio", table="targets", item="signed_log_ratio") if len(panel_om) else pd.DataFrame()
    add_sec("targets", "signed_log_ratio", preds_signed, target="log_ratio", append=False)
    if len(preds_signed):
        _ts = preds_signed.copy(); _ts.attrs = {}
        all_preds.append(_ts)
    # event AUC with date-block bootstrap for Delta AUC (fix 7)
    auc_rows = []
    auc_boot = {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": 0}
    if len(preds_primary) and len(panel_om):
        from sklearn.metrics import roc_auc_score
        pp = preds_primary.copy()
        train_panel = panel_om[(panel_om["h"] == PRIMARY_H) & (pd.to_datetime(panel_om["date"]).dt.year < fy["OM"])]
        thr = train_panel.groupby("asset")["qlike"].quantile(0.90)
        for mname in ("M0", "M1"):
            gdf = pp[pp["model"] == mname]
            ev = (gdf["qlike"] > gdf["asset"].map(thr)).astype(float).to_numpy()
            ok = np.isfinite(ev)
            if len(np.unique(ev[ok])) == 2:
                auc_rows.append({"model": mname, "auc_event_trainq90": float(roc_auc_score(ev[ok], gdf["prediction"].to_numpy()[ok]))})
        auc_boot = auc_delta_bootstrap(preds_primary, thr.to_dict(), smoke(n_boot, 40) if not smoke_mode else 40, rng)
    report.save_table(pd.DataFrame(auc_rows), out_dir, "secondary_event_auc")
    sec_rows.append({"table": "targets", "item": "event_auc_M1_minus_M0", "tier": "OM", "h": PRIMARY_H, "fixed_model": "LogHAR", "learner": "ridge",
                     "target": "event", "comparison": "M1_vs_M0", "delta_spearman": auc_boot.get("delta"),
                     "ci_low": auc_boot.get("ci_low"), "ci_high": auc_boot.get("ci_high"), "p_raw": auc_boot.get("p"), "n": auc_boot.get("n", 0)})
    # B_US (US assets only)
    if len(panel_om) and ".SPX" in set(panel_om["asset"]):
        us_panel = panel_om[panel_om["asset"].isin(US_OM)]
        gus = feature_groups(us_panel.columns, "OM", us=True)
        add_sec("bus", "B_US_M1_US_vs_M0_US", loss_predictions(us_panel, gus, PRIMARY_H, fy["OM"], table="bus", item="B_US_M1_US_vs_M0_US"), m0="M0_US", m1="M1_US")
    for label, vp in variant_panels.items():
        if len(vp):
            gv = feature_groups(vp.columns, "OM", us=True)
            add_sec("variants", f"v_{label}", loss_predictions(vp, gv, PRIMARY_H, fy["OM"], variant=label, table="variants", item=f"v_{label}"), variant=label)
    if len(panel_om_ewma):
        ge = feature_groups(panel_om_ewma.columns, "OM", us=True)
        add_sec("fixed_models", "EWMA_M1", loss_predictions(panel_om_ewma, ge, PRIMARY_H, fy["OM"], fixed_model="EWMA", table="fixed_models", item="EWMA_M1"), fixed="EWMA")
    # geometry arms
    geom_cols = {}
    if len(panel_om):
        arms = [a for a in ("gz", "eps_train", "beta_pos0", "beta_pos1", "power_ctrl0", "power_ctrl1") if any(c.startswith(f"geom_{a}_") for c in panel_om.columns)]
        for arm in arms:
            cols = [c for c in panel_om.columns if c.startswith(f"geom_{arm}_")]
            if cols:
                geom_cols[arm] = cols
                # B4: reuse identical M0 from preds_primary; fit only the arm's M1.
                garm = {"M1": groups_om["M0"] + cols}
                arm_m1 = loss_predictions(panel_om, garm, PRIMARY_H, fy["OM"], table="geometry", item=f"GEOM_{arm}")
                m0_reuse = preds_primary[preds_primary["model"] == "M0"].copy() if len(preds_primary) else pd.DataFrame()
                if len(m0_reuse) and len(arm_m1):
                    m0_reuse.attrs = {}; arm_m1.attrs = {}
                    arm_preds = pd.concat([m0_reuse, arm_m1], ignore_index=True)
                elif len(arm_m1):
                    arm_preds = loss_predictions(panel_om, {"M0": groups_om["M0"], **garm}, PRIMARY_H, fy["OM"], table="geometry", item=f"GEOM_{arm}")
                else:
                    arm_preds = arm_m1
                add_sec("geometry", f"GEOM_{arm}", arm_preds)
    # Tier A + overlap + out-of-period; Tier BN
    preds_a = loss_predictions(panel_a, groups_a, PRIMARY_H, fy["A"], table="tiers", item="TierA_M1") if len(panel_a) else pd.DataFrame()
    add_sec("tiers", "TierA_M1", preds_a, tier="A", append=False)
    if len(preds_a):
        _ta = preds_a.copy(); _ta.attrs = {}
        all_preds.append(_ta)
        oop = preds_a[pd.to_datetime(preds_a["date"]) >= "2020-03-01"]
        b_oop = delta_bootstrap(oop, smoke(n_boot, 40) if not smoke_mode else 40, rng, "M0", "M1") if len(oop) else {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": 0}
        sec_rows.append({"table": "tiers", "item": "TierA_outofperiod_2020plus", "tier": "A", "h": PRIMARY_H, "fixed_model": "LogHAR", "learner": "ridge",
                         "target": "qlike", "comparison": "M1_vs_M0", "delta_spearman": b_oop.get("delta"), "ci_low": b_oop.get("ci_low"), "ci_high": b_oop.get("ci_high"), "p_raw": b_oop.get("p"), "n": b_oop.get("n", 0)})
    # Fix 8: Tier A vs OM overlap on paired assets, same asset-dates, 2006-01..2020-02.
    _pairs = [("^GSPC", ".SPX"), ("^DJI", ".DJI"), ("^RUT", ".RUT"), ("^FTSE", ".FTSE"), ("^GDAXI", ".GDAXI")]
    for _ta_tkr, _om_sym in _pairs:
        _have = len(panel_a) and len(panel_om) and _ta_tkr in set(panel_a.get("asset", [])) and _om_sym in set(panel_om.get("asset", []))
        if not _have:
            continue
        _om_p = preds_primary[(preds_primary["asset"] == _om_sym) & (pd.to_datetime(preds_primary["date"]) >= "2006-01-01") & (pd.to_datetime(preds_primary["date"]) < "2020-03-01")] if len(preds_primary) else pd.DataFrame()
        _a_p = preds_a[(preds_a["asset"] == _ta_tkr) & (pd.to_datetime(preds_a["date"]) >= "2006-01-01") & (pd.to_datetime(preds_a["date"]) < "2020-03-01")] if len(preds_a) else pd.DataFrame()
        _common = sorted(set(pd.to_datetime(_om_p.get("date", pd.Series(dtype="datetime64[ns]")))) & set(pd.to_datetime(_a_p.get("date", pd.Series(dtype="datetime64[ns]")))))
        for _tier_lbl, _pp in (("A", _a_p), ("OM", _om_p)):
            _sub = _pp[pd.to_datetime(_pp["date"]).isin(_common)] if len(_pp) and _common else _pp.iloc[:0] if len(_pp) else pd.DataFrame()
            _b = delta_bootstrap(_sub, smoke(n_boot, 40) if not smoke_mode else 40, rng, "M0", "M1") if len(_sub) else {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": 0}
            sec_rows.append({"table": "overlap", "item": f"{_ta_tkr}~{_om_sym}", "tier": _tier_lbl, "h": PRIMARY_H, "fixed_model": "LogHAR", "learner": "ridge",
                             "target": "qlike", "comparison": "M1_vs_M0", "delta_spearman": _b.get("delta"), "ci_low": _b.get("ci_low"), "ci_high": _b.get("ci_high"), "p_raw": _b.get("p"), "n": _b.get("n", 0)})
    preds_bn = loss_predictions(panel_bn, groups_bn, PRIMARY_H, fy["BN"], table="tiers", item="TierBN_M1") if len(panel_bn) else pd.DataFrame()
    add_sec("tiers", "TierBN_M1", preds_bn, tier="BN", append=False)
    if len(preds_bn):
        _tb = preds_bn.copy(); _tb.attrs = {}
        all_preds.append(_tb)
    if len(panel_bn_harq):
        gh = feature_groups(panel_bn_harq.columns, "BN")
        add_sec("tiers", "TierBN_HARQ_M1", loss_predictions(panel_bn_harq, gh, PRIMARY_H, fy["BN"], fixed_model="HARQ", table="tiers", item="TierBN_HARQ_M1"), tier="BN", fixed="HARQ")
    # per-period / crisis from primary predictions
    if len(preds_primary):
        pp = preds_primary.copy(); yrs = pd.to_datetime(pp["date"]).dt.year
        for label, mask in [("period_2006_2012", (yrs >= 2006) & (yrs <= 2012)), ("period_2013_2020", yrs >= 2013),
                            ("crisis_2008_2009", yrs.isin([2008, 2009])), ("calm_other", ~yrs.isin([2008, 2009]))]:
            _sub = pp[mask]
            d = delta_bootstrap(_sub, smoke(n_boot, 40) if not smoke_mode else 40, rng, "M0", "M1") if len(_sub) else {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": 0}
            sec_rows.append({"table": "periods", "item": label, "tier": "OM", "h": PRIMARY_H, "fixed_model": "LogHAR", "learner": "ridge",
                             "target": "qlike", "comparison": "M1_vs_M0", "delta_spearman": d.get("delta"), "ci_low": d.get("ci_low"), "ci_high": d.get("ci_high"), "p_raw": d.get("p"), "n": d.get("n", 0),
                             "flag_2020_partial": "2020 test year is Jan-Feb only" if label == "period_2013_2020" else ""})
    sec = pd.DataFrame(sec_rows)
    if len(sec):
        sec["p_holm"] = np.nan
        for table, idx in sec.groupby("table").groups.items():
            p = sec.loc[idx, "p_raw"].to_numpy(dtype=float)
            finite = np.isfinite(p)
            adj = np.full(len(p), np.nan)
            if finite.any():
                adj[finite] = holm(np.where(finite, p, 1.0)[finite]) if False else holm(p[finite])
            sec.loc[idx, "p_holm"] = adj
    report.save_table(sec, out_dir, "secondary_all")
    for table in (sec["table"].unique().tolist() if len(sec) else []):
        report.save_table(sec[sec["table"] == table], out_dir, f"secondary_{table}")

    if all_preds:
        for _p in all_preds:
            _p.attrs = {}
        lp = pd.concat(all_preds, ignore_index=True)
        _lp_cols = [c for c in ["asset", "tier", "date", "h", "model", "learner", "target", "prediction", "fold", "fixed_model", "variant", "table", "item"] if c in lp]
        _key = [c for c in ["asset", "tier", "date", "h", "model", "learner", "target", "fixed_model", "variant", "table", "item"] if c in lp]
        _dup = int(lp.duplicated(subset=_key).sum()) if _key else 0
        print(f"[fix6] loss_predictions rows={len(lp)} duplicate_keys={_dup} key={_key}")
        assert _dup == 0, f"loss_predictions duplicate keys: {_dup}"
        lp[_lp_cols].to_parquet(out_dir / "loss_predictions.parquet", index=False)

    # --- figures ---
    if len(per_asset):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(per_asset))))
        pa = per_asset.sort_values("delta_spearman")
        ax.barh(pa["asset"], pa["delta_spearman"]); ax.axvline(0, color="k", lw=0.8)
        if {"ci_low", "ci_high"}.issubset(pa.columns):
            _xerr = np.vstack([pa["delta_spearman"] - pa["ci_low"], pa["ci_high"] - pa["delta_spearman"]])
            ax.errorbar(pa["delta_spearman"], pa["asset"], xerr=_xerr, fmt="none", ecolor="k", capsize=2, lw=0.8)
        ax.set_xlabel("Delta Spearman M1-M0 (OM, h=5) with 95% CI")
        report.save_fig(fig, fig_dir, "per_asset_delta"); plt.close(fig)
    if len(preds_primary) and ".SPX" in set(preds_primary["asset"]):
        spx = preds_primary[preds_primary["asset"] == ".SPX"].copy().sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(spx["date"], spx["qlike"], label="realised QLIKE", lw=0.7)
        for mname in ("M0", "M1"):
            gdf = spx[spx["model"] == mname]
            ax.plot(gdf["date"], np.exp(gdf["prediction"]) - 1e-4, label=f"{mname} pred QLIKE", lw=0.7)
        ax.legend(); ax.set_title(".SPX QLIKE vs loss predictions (OM ends Feb 2020, pre-COVID)")
        report.save_fig(fig, fig_dir, "spx_timeseries"); plt.close(fig)
    if len(coef_tbl):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(coef_tbl))))
        ct = coef_tbl.head(20).iloc[::-1]
        ax.barh(ct["feature"], ct["coef"]); ax.axvline(0, color="k", lw=0.8)
        report.save_fig(fig, fig_dir, "ridge_coef_M1"); plt.close(fig)

    # --- decision (G1 read from earlier experiments; G2 from primary) ---
    def _g1_status():
        notes, statuses = [], {}
        for exp in ("E1", "E2"):
            pth = _REPO / f"notebooks/tables/condpred/{exp}/decision.csv"
            spec_p = _REPO / f"notebooks/tables/condpred/{exp}/spec.json"
            if not pth.exists():
                statuses[exp] = "missing"; notes.append(f"{exp}: decision.csv missing"); continue
            try:
                dfr = pd.read_csv(pth)
                spec_j = json.loads(spec_p.read_text(encoding="utf-8")) if spec_p.exists() else {}
                written = pd.Timestamp(spec_j.get("written_at")) if spec_j.get("written_at") else None
                dec_time = pd.Timestamp(pth.stat().st_mtime, unit="s", tz="UTC")
                if written is not None and dec_time < written:
                    statuses[exp] = "stale"; notes.append(f"{exp}: decision.csv stale (before its spec.json written_at {spec_j.get('written_at')})"); continue
                is_smoke = bool(spec_j.get("smoke")) or ("smoke" in dfr.columns and dfr["smoke"].astype(str).str.lower().isin(["true", "1"]).any())
                if is_smoke:
                    statuses[exp] = "smoke"; notes.append(f"{exp}: decision.csv is smoke, not full-run"); continue
                passed = None
                for _, r in dfr.iterrows():
                    gate_v = str(r.get("gate", "")); stat_v = str(r.get("statistic", ""))
                    if gate_v == "G1" or gate_v == exp or stat_v == "primary_test":
                        verdict = str(r.get("verdict", "")).lower()
                        val = r.get("value", None)
                        row_pass = verdict == "pass" or (isinstance(val, (bool, np.bool_)) and bool(val)) or str(val).lower() in ("true", "1", "pass")
                        row_fail = verdict == "fail" or str(val).lower() in ("false", "0", "fail")
                        if row_pass:
                            passed = True
                        elif row_fail and passed is not True:
                            passed = False
                if passed is True:
                    statuses[exp] = "pass"
                elif passed is False:
                    statuses[exp] = "fail"
                else:
                    statuses[exp] = "unknown"; notes.append(f"{exp}: no G1/primary verdict in current decision.csv")
            except Exception as exc:
                statuses[exp] = "unparseable"; notes.append(f"{exp}: could not parse decision ({exc})")
        if any(v == "pass" for v in statuses.values()):
            return "pass", "; ".join(notes + [f"{k}={v}" for k, v in statuses.items()])
        if all(statuses.get(e) == "fail" for e in ("E1", "E2")):
            return "fail", "; ".join(notes + [f"{k}={v}" for k, v in statuses.items()])
        return "pending", "G1 pending: " + "; ".join(notes + [f"{k}={v}" for k, v in statuses.items() if v != "fail"])
    g1_verdict, g1_note = _g1_status()
    decision = [
        {"gate": "G1", "verdict": g1_verdict, "note": g1_note + "; full E4 run only worth if G1 passed (build+smoke regardless)"},
        {"gate": "G2", "tier": "OM", "h": PRIMARY_H, "delta_spearman": boot.get("delta"), "ci_low": boot.get("ci_low"), "ci_high": boot.get("ci_high"),
         "share_assets_positive": share_pos, "verdict": "pass" if g2_pass else "fail", "smoke": smoke_mode,
         "note": "smoke verdict is mechanical on smoke subset, not the full-run gate" if smoke_mode else "primary gate"},
    ]
    report.write_decision(decision, out_dir)
    return {"decision": decision, "primary": primary_row, "per_asset": per_asset, "secondary": sec,
            "preds_primary": preds_primary, "panels": {"OM": panel_om, "A": panel_a, "BN": panel_bn}}


# ---------------------------------------------------------------- leakage
def run_leakage_e4() -> list[tuple[str, bool]]:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(800)) + 0.05 * np.arange(800)
    v = np.exp(0.3 * np.sin(np.arange(800) / 20) + rng.standard_normal(800) * 0.2)
    t_list = [300, 500, 700]
    geom = _geom_default()
    checks = []
    from condpred.features.window_rqa import window_rqa
    from condpred.features.local_rqa import local_rqa
    checks.append(("WRQA", leakage_test(lambda xx, t: window_rqa(xx, t, W, M, TAU, _geom_default()), x, t_list, rng)))
    checks.append(("LRQA", leakage_test(lambda xx, t: local_rqa(xx, t, W, M, TAU, _geom_default()), x, t_list, rng)))
    checks.append(("B_perm", leakage_test(lambda xx, t: baselines.perm_entropy(xx, W, t), x, t_list, rng)))
    checks.append(("B_knn", leakage_test(lambda xx, t: baselines.knn_spread(xx, t, M, TAU, 5, 5, 200), x, [600], rng)))
    def har_fn(vv, t):
        mdl = LogHAR(refit_every=21, window=1000, horizons=(5,)); mdl.fit(vv); return mdl.predict(t, 5)
    checks.append(("LogHAR", leakage_test(har_fn, v, [400, 600], rng)))
    # recent_loss alignment: losses realised after t must not affect value at t
    loss = rng.random(800)
    checks.append(("recent_loss", leakage_test(lambda ll, t: baselines.recent_loss(ll, 5, 22.0, t), loss, t_list, rng)))
    return checks


