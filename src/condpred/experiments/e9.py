"""E9 — CPI reference-model error predictability (monthly panel)."""
from __future__ import annotations

import json
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

from condpred import report
from condpred.cache import cached
from condpred.config import smoke
from condpred.evaluation import RidgePredictor, _panel_bootstrap_rows, spearman
from condpred.stats import holm
from condpred.features import baselines
from condpred.features.local_rqa import local_rqa
from condpred.features.window_rqa import window_rqa
from condpred.geometry import Geometry

E9_MODULES = [
    "condpred.experiments.e9",
    "condpred.features.window_rqa",
    "condpred.features.local_rqa",
    "condpred.features.baselines",
    "condpred.evaluation",
    "condpred.geometry",
]

_REPO = Path(__file__).resolve().parents[3]
_FRED = _REPO / "data" / "raw" / "fred"
_OECD = _REPO / "data" / "raw" / "oecd_mei"

COUNTRIES = [
    "CAN", "GBR", "JPN", "DEU", "FRA", "ITA", "CHE", "SWE", "NOR", "DNK", "NLD", "BEL",
    "ESP", "AUT", "FIN", "PRT", "GRC", "KOR", "MEX", "ISL", "LUX", "TUR", "ISR", "ZAF",
]
ALL_COUNTRIES = ["USA"] + COUNTRIES

HORIZONS = [6, 12]
PRIMARY_H = 12
FIRST_REF_YEAR = 1990
LOSS_PRED_FIRST_YEAR = 2000
W, M, TAU, RR = 120, 2, 1, 0.05
THEILER_PRIMARY = 1
REF_CHECK_AUC = {
    12: {"USA": 0.690, "CAN": 0.669, "GBR": 0.678, "DEU": 0.720},
    6: {"USA": 0.685, "CAN": 0.707, "GBR": 0.706, "DEU": 0.674},
}
AUC_TOL = 0.01

WRQA_KEYS = ["DET", "L", "ENTR", "LAM", "TT", "Lmax", "Vmax", "DET3_DET2", "K2_lines", "DET_LAM"]
LRQA_KEYS = ["RR_last", "Lb_mean", "Lb_max", "DET_last", "Lf_nb", "K2_local", "LAM_last", "TT_last"]
B_KEYS = [
    "recent_loss",
    "model_confidence",
    "rolling_pi_std24",
    "abs_base_effect",
    "yoy_level",
    "months_since_yoy_turn",
]
RQA_Z_SUFFIX = "_ez"


def country_universe(smoke_mode: bool) -> list[str]:
    if smoke_mode:
        return ["USA", "CAN", "GBR"]
    return ALL_COUNTRIES


def ref_check_isos(smoke_mode: bool) -> list[str]:
    if smoke_mode:
        return ["USA", "CAN", "GBR"]
    return ["USA", "CAN", "GBR", "DEU"]


def build_initial_spec(smoke_mode: bool) -> dict:
    return {
        "experiment": "E9",
        "version": 2,
        "changelog": [
            "v2 records bootstrap_n, bootstrap_seed, secondary_families and holm_secondary. "
            "The primary pass rule is unchanged. v1 loss-predictor numbers are invalid "
            "(no h-month purge, horizons pooled in one fit, cross-horizon z-scores, "
            "M1 trained on fewer rows than M0, month bootstrap, full-sample rank target)."
        ],
        "smoke": smoke_mode,
        "low_power_warning": (
            "Monthly macro panel: few large-error episodes; a positive primary gate is a lead, not a finding."
        ),
        "countries": country_universe(smoke_mode),
        "horizons": HORIZONS,
        "primary_h": PRIMARY_H,
        "geometry": {"scaling": "wz", "distance": "euclid", "threshold": "rr", "rr": RR, "m": M, "tau": TAU, "W": W},
        "theiler_primary": THEILER_PRIMARY,
        "theiler_secondary": 3,
        "reference_model": "LightGBM lags+base (02_macro hyperparameters)",
        "loss_target": "rank of monthly log loss of reference direction probability",
        "event_rule": "log loss above training-period 90th percentile",
        "predictor_primary": "RidgePredictor",
        "predictor_event": "LogisticRegression",
        "loss_predictor_first_test_year": LOSS_PRED_FIRST_YEAR,
        "models": {"M0": "B", "M1": "B+WRQA+LRQA"},
        "primary_test": {
            "h": PRIMARY_H,
            "metric": "delta_spearman",
            "pool": "24 OECD + USA",
            "bootstrap": "year-block panel (calendar years jointly across countries)",
            "bootstrap_n": smoke(2000, 80),
            "bootstrap_seed": 0,
            "pass_rule": "95% CI lower bound of ΔSpearman(M1−M0) > 0",
        },
        "gate_E9": "primary_test pass_rule",
        "reference_repro_tolerance": AUC_TOL,
        "secondary_families": ["B+WRQA", "B+LRQA", "theiler_3"],
        "holm_secondary": True,
    }


def verify_spec_immutable(out_dir: Path, spec: dict, modules: list[str]):
    path = out_dir / "spec.json"
    if not path.exists():
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    from condpred.config import spec_hash

    skip = {"spec_hash", "code_fingerprint", "written_at", "horizons", "per_system"}
    a = {k: v for k, v in old.items() if k not in skip}
    b = {k: v for k, v in spec.items() if k not in skip}
    if spec_hash(a) != spec_hash(b):
        raise RuntimeError("SPEC changed after write_spec; bump version")


def _load_usa_cpi() -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(
        _FRED / "CPIAUCSL.csv",
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    s = df["CPIAUCSL"].astype(float).asfreq("MS")
    imputed = s.isna()
    s = s.interpolate(method="time").bfill().ffill()
    return s.rename("USA"), imputed


def _load_oecd_cpi(iso: str) -> pd.Series:
    path = _OECD / f"{iso}CPIALLMINMEI.csv"
    df = pd.read_csv(path)
    s = pd.Series(
        pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
        index=pd.to_datetime(df.iloc[:, 0]),
    )
    s = s.loc[s.first_valid_index() : s.last_valid_index()].asfreq("MS")
    if s.isna().any():
        raise ValueError(f"{iso}: gap in monthly CPI")
    return s.rename(iso)


def load_cpi_series(iso: str) -> tuple[pd.Series, pd.Series | None]:
    if iso == "USA":
        return _load_usa_cpi()
    return _load_oecd_cpi(iso), None


def seasonal_factor(pi: pd.Series, k: int) -> pd.Series:
    dev = pi - pi.rolling(12).mean()
    return pd.concat([dev.shift(12 * j - k) for j in range(1, 11)], axis=1).mean(axis=1, skipna=False)


def base_effect(pi: pd.Series, h: int) -> pd.Series:
    return pi.rolling(h).sum().shift(12 - h)


def country_frame(level: pd.Series, *, usa: bool = False) -> pd.DataFrame:
    logp = np.log(level)
    pi = 100 * logp.diff()
    out = pd.DataFrame({"PI": pi, "YOY": 100 * (logp - logp.shift(12))})
    if usa:
        out["PI_SA"] = pi
    else:
        out["PI_SA"] = pi - seasonal_factor(pi, 0)
    for h in HORIZONS:
        future_season = sum(seasonal_factor(pi, k) for k in range(1, h + 1))
        out[f"rule_h{h}"] = h * out.PI_SA.rolling(3).mean() + future_season - base_effect(pi, h)
    return out


def targets(frame: pd.DataFrame, h: int, *, label_ok: pd.Series | None = None) -> pd.Series:
    fut = frame.YOY.shift(-h)
    ok = fut.notna() & frame.YOY.notna()
    if label_ok is not None:
        ok &= label_ok.reindex(frame.index, fill_value=False)
    return (fut > frame.YOY)[ok].astype(int)


def reference_features(frame: pd.DataFrame, h: int, rule: bool = True) -> pd.DataFrame:
    pi, yoy = frame.PI, frame.YOY
    cols = {f"pi_lag{k}": pi.shift(k) for k in range(13)}
    cols |= {f"dyoy_lag{k}": yoy.diff().shift(k) for k in range(12)}
    cols |= {"yoy": yoy, "base_effect": base_effect(pi, h)}
    if rule:
        cols["rule"] = frame[f"rule_h{h}"]
    return pd.DataFrame(cols)


def make_classifier():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=1,
        verbose=-1,
    )


def reference_walk_forward(
    X: pd.DataFrame, y: pd.Series, h: int, first_year: int
) -> pd.Series:
    X = X.reindex(y.index)
    pred = pd.Series(np.nan, index=y.index)
    for year in range(first_year, y.index.year.max() + 1):
        start = pd.Timestamp(year, 1, 1)
        train = y.index.shift(h, freq="MS") <= start
        test = y.index.year == year
        if not test.any():
            continue
        if train.any() and test.any():
            assert (
                y.index[train].max() + pd.DateOffset(months=h) <= start <= y.index[test].min()
            ), (year, h)
        pred[test] = make_classifier().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return pred.dropna()


def direction_log_loss(p: pd.Series, y: pd.Series) -> pd.Series:
    p = p.clip(1e-6, 1 - 1e-6)
    yy = y.reindex(p.index).astype(float)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def first_test_year(pi_sa: pd.Series, base: int = FIRST_REF_YEAR) -> int:
    """Cross-country: first test = max(1990, year of month index W−1 of PI_SA + 10)."""
    idx = pi_sa.dropna().index
    if len(idx) < W:
        return base
    first_rp = idx[W - 1]
    return max(base, first_rp.year + 10)


def _wf_masks(dates: pd.Series, h: int, test_year: int) -> tuple[np.ndarray, np.ndarray]:
    start = pd.Timestamp(test_year, 1, 1)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    train = np.asarray(idx.shift(h, freq="MS") <= start)
    test = np.asarray(idx.year == test_year)
    return train, test


def _assert_wf_purge(dates: pd.Series, h: int, test_year: int) -> bool:
    train, test = _wf_masks(dates, h, test_year)
    if not test.any() or not train.any():
        return True
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    start = pd.Timestamp(test_year, 1, 1)
    ok = idx[train].max() + pd.DateOffset(months=h) <= start <= idx[test].min()
    return bool(ok)


def _rank_train_target(log_loss: np.ndarray) -> np.ndarray:
    s = pd.Series(log_loss, dtype=float)
    r = s.rank(method="average").to_numpy()
    n = np.sum(np.isfinite(log_loss))
    return r / n if n else r


def _geom(theiler: int) -> Geometry:
    g = Geometry(scaling="wz", distance="euclid", threshold="rr", rr=RR, m=M, tau=TAU, theiler=theiler)
    return g


def _rqa_row(x: np.ndarray, t: int, geom: Geometry) -> dict:
    row = {}
    wr = window_rqa(x, t, W, M, TAU, geom)
    for k in WRQA_KEYS:
        row[f"wrqa_{k}"] = wr.get(k, np.nan)
    lr = local_rqa(x, t, W, M, TAU, geom)
    for k in LRQA_KEYS:
        row[f"lrqa_{k}"] = lr.get(k, np.nan)
    return row


def rqa_feature_frame(pi_sa: pd.Series, theiler: int, *, cache_exp: str, iso: str) -> pd.DataFrame:
    key = {"iso": iso, "W": W, "m": M, "tau": TAU, "rr": RR, "theiler": theiler}

    def _build():
        x = pi_sa.dropna().to_numpy(float)
        idx = pi_sa.dropna().index
        geom = _geom(theiler)
        min_t = W - 1 + (M - 1) * TAU
        ts = list(range(min_t, len(x)))
        n_jobs = smoke(4, 1)
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_rqa_row)(x, t, geom) for t in ts
        )
        return pd.DataFrame(rows, index=idx[ts])

    return cached(cache_exp, f"rqa_{iso}", key, _build, modules=E9_MODULES)


def months_since_direction_change(yoy: pd.Series) -> pd.Series:
    up = yoy.diff() > 0
    chg = up.ne(up.shift(1)).cumsum()
    out = pd.Series(np.nan, index=yoy.index)
    for _, g in chg.groupby(chg):
        idx = g.index
        out.loc[idx] = np.arange(len(idx))
    return out


def build_b_features(
    frame: pd.DataFrame,
    h: int,
    probs: pd.Series,
    log_loss: pd.Series,
    base_eff: pd.Series,
) -> pd.DataFrame:
    pi = frame.PI
    loss_on_idx = pd.Series(np.nan, index=frame.index, dtype=float)
    loss_on_idx.loc[log_loss.index] = log_loss
    loss_arr = loss_on_idx.to_numpy(float)
    pos_map = {d: i for i, d in enumerate(frame.index)}
    recent = []
    for d in probs.index:
        t = pos_map[d]
        recent.append(baselines.recent_loss(loss_arr, h, 12.0, t))
    return pd.DataFrame(
        {
            "recent_loss": recent,
            "model_confidence": (probs - 0.5).abs(),
            "rolling_pi_std24": frame.PI.rolling(24).std().reindex(probs.index),
            "abs_base_effect": base_eff.reindex(probs.index).abs(),
            "yoy_level": frame.YOY.reindex(probs.index),
            "months_since_yoy_turn": months_since_direction_change(frame.YOY).reindex(probs.index),
        },
        index=probs.index,
    )


def causal_expanding_z(
    panel: pd.DataFrame,
    cols: list[str],
    *,
    group_cols: tuple[str, ...] = ("iso", "h"),
) -> pd.DataFrame:
    out = panel.copy()
    for c in cols:
        zcol = c + RQA_Z_SUFFIX
        out[zcol] = np.nan
        for _key, g in panel.groupby(list(group_cols)):
            g = g.sort_values("date")
            mu = g[c].expanding(min_periods=5).mean().shift(1)
            sd = g[c].expanding(min_periods=5).std().shift(1).replace(0, np.nan)
            out.loc[g.index, zcol] = (g[c] - mu) / sd
    return out


def feature_groups() -> dict[str, list[str]]:
    b = [k + RQA_Z_SUFFIX for k in B_KEYS]
    w = [f"wrqa_{k}{RQA_Z_SUFFIX}" for k in WRQA_KEYS]
    l = [f"lrqa_{k}{RQA_Z_SUFFIX}" for k in LRQA_KEYS]
    return {"B": b, "WRQA": w, "LRQA": l, "RQA": w + l}


def feature_groups_t3() -> dict[str, list[str]]:
    b = [k + RQA_Z_SUFFIX for k in B_KEYS]
    w = [f"wrqa_{k}_t3{RQA_Z_SUFFIX}" for k in WRQA_KEYS]
    l = [f"lrqa_{k}_t3{RQA_Z_SUFFIX}" for k in LRQA_KEYS]
    return {"B": b, "WRQA": w, "LRQA": l, "RQA": w + l, "RQA_T3": w + l}


def _label_imputed_usa(imputed: pd.Series, h: int, index: pd.DatetimeIndex) -> pd.Series:
    inflation_imputed = imputed | imputed.shift(12, fill_value=False)
    bad = inflation_imputed | inflation_imputed.shift(-h, fill_value=False)
    return ~bad.reindex(index, fill_value=True)


def load_panel(spec: dict, cache_exp: str) -> dict:
    isos = spec["countries"]
    cpi = {}
    frames = {}
    label_ok = {}
    for iso in isos:
        s, imp = load_cpi_series(iso)
        cpi[iso] = s
        frames[iso] = country_frame(s, usa=(iso == "USA"))
        if imp is not None:
            for h in HORIZONS:
                label_ok[(iso, h)] = _label_imputed_usa(imp, h, frames[iso].index)
        else:
            for h in HORIZONS:
                label_ok[(iso, h)] = pd.Series(True, index=frames[iso].index)
    rp = {}
    first_test = {}
    rp_t3 = {}
    for iso in isos:
        rp[iso] = rqa_feature_frame(
            frames[iso].PI_SA, spec["theiler_primary"], cache_exp=cache_exp, iso=iso
        )
        rp_t3[iso] = rqa_feature_frame(
            frames[iso].PI_SA, spec["theiler_secondary"], cache_exp=cache_exp, iso=f"{iso}_t3"
        )
        first_test[iso] = first_test_year(frames[iso].PI_SA)
    return {
        "cpi": cpi,
        "frames": frames,
        "label_ok": label_ok,
        "rp": rp,
        "rp_t3": rp_t3,
        "first_test": first_test,
    }


def run_reference_reproduction(
    data: dict, h: int, check_isos: list[str]
) -> tuple[pd.DataFrame, bool, bool]:
    rows = []
    oecd_fail = False
    usa_warn = False
    for iso in check_isos:
        frame = data["frames"][iso]
        y = targets(frame, h, label_ok=data["label_ok"][(iso, h)])
        X = reference_features(frame, h)
        ft = data["first_test"][iso]
        pred = reference_walk_forward(X, y, h, ft)
        idx = pred.index.intersection(y.index)
        auc = roc_auc_score(y.loc[idx], pred.loc[idx])
        exp = REF_CHECK_AUC[h].get(iso, np.nan)
        diff = auc - exp if np.isfinite(exp) else np.nan
        ok = np.isfinite(diff) and abs(diff) <= AUC_TOL
        rows.append(
            {
                "iso": iso,
                "h": h,
                "expected_auc": exp,
                "reproduced_auc": auc,
                "difference": diff,
                "within_tolerance": ok,
                "first_test_year": ft,
            }
        )
        if not ok:
            if iso == "USA":
                usa_warn = True
            else:
                oecd_fail = True
    return pd.DataFrame(rows), not oecd_fail, usa_warn


def run_leakage_e9(data: dict, iso: str = "CAN") -> list[tuple[str, bool]]:
    frame = data["frames"][iso]
    cpi = data["cpi"][iso]
    month = "2010-06-01"
    trunc = country_frame(cpi.loc[:month], usa=(iso == "USA"))
    trunc_rp = rqa_feature_frame(trunc.PI_SA, THEILER_PRIMARY, cache_exp="E9/leak", iso=f"{iso}_trunc")
    full_rp = data["rp"][iso]
    checks = []
    m = pd.Timestamp(month)
    if m in full_rp.index and m in trunc_rp.index:
        checks.append(
            (
                "rqa_trunc",
                bool(np.allclose(full_rp.loc[m], trunc_rp.loc[m], equal_nan=True)),
            )
        )
    for h in [PRIMARY_H]:
        a = reference_features(frame, h).loc[m]
        b = reference_features(trunc, h).loc[m]
        checks.append(("reference_features", bool(np.allclose(a, b, equal_nan=True))))
    if iso != "USA":
        raw_sf = seasonal_factor(frame.PI, 0).loc["1990":].groupby(frame.PI.loc["1990":].index.month).mean().std()
        adj_sf = frame.PI_SA.loc["1990":].groupby(frame.PI_SA.loc["1990":].index.month).mean().std()
        checks.append(("seasonal_adjustment", bool(adj_sf < raw_sf)))
    h = PRIMARY_H
    y = targets(frame, h, label_ok=data["label_ok"][(iso, h)])
    X = reference_features(frame, h)
    pred = reference_walk_forward(X, y, h, data["first_test"][iso])
    if len(pred) > 24:
        for yr in (2005, 2015):
            checks.append((f"reference_wf_purge_{yr}", _assert_wf_purge(pred.index, h, yr)))
    half = frame.index[120:320]
    n_half = len(half)
    probe = pd.concat(
        [
            pd.DataFrame({"iso": iso, "h": 6, "date": half, "yoy_level": np.linspace(0.0, 1.0, n_half)}),
            pd.DataFrame(
                {"iso": iso, "h": 12, "date": half, "yoy_level": np.sin(np.linspace(0.0, 8.0, n_half))}
            ),
        ],
        ignore_index=True,
    )
    z = causal_expanding_z(probe, ["yoy_level"])
    z6 = z.loc[z["h"] == 6, "yoy_level_ez"].to_numpy(float)
    z12 = z.loc[z["h"] == 12, "yoy_level_ez"].to_numpy(float)
    leaked = probe.copy()
    leaked.loc[leaked["h"] == 6, "yoy_level"] = 0.0
    z_after = causal_expanding_z(leaked, ["yoy_level"])
    z6_after = z_after.loc[z_after["h"] == 6, "yoy_level_ez"].to_numpy(float)
    z12_after = z_after.loc[z_after["h"] == 12, "yoy_level_ez"].to_numpy(float)
    checks.append(
        (
            "zscore_per_iso_h",
            bool(
                np.isfinite(z6).sum() > 10
                and np.isfinite(z12).sum() > 10
                and np.allclose(z12, z12_after, equal_nan=True)
                and not np.allclose(z6, z6_after, equal_nan=True)
            ),
        )
    )
    toy = pd.DataFrame(
        {
            "iso": ["USA", "CAN", "USA"],
            "date": pd.to_datetime(["2000-02-01", "2000-01-01", "2000-01-01"]),
        }
    )
    toy_s, toy_y = _ordered_train(toy, np.array([2.0, 0.0, 1.0]))
    checks.append(
        (
            "ridge_row_order",
            bool(list(toy_s["iso"]) == ["CAN", "USA", "USA"] and np.allclose(toy_y, [0.0, 1.0, 2.0])),
        )
    )
    pan_one = assemble_country_rows(data, iso, PRIMARY_H, "E9/leak")
    yr = LOSS_PRED_FIRST_YEAR + 1
    tr_naive = (pan_one["year"] < yr).to_numpy()
    tr_purge, te_purge = _wf_masks(pan_one["date"], PRIMARY_H, yr)
    checks.append(
        (
            "loss_predictor_purge",
            bool(
                tr_purge.sum() < tr_naive.sum()
                and te_purge.any()
                and _assert_wf_purge(pan_one["date"], PRIMARY_H, yr)
            ),
        )
    )
    return checks


def assemble_country_rows(
    data: dict,
    iso: str,
    h: int,
    cache_exp: str,
) -> pd.DataFrame:
    frame = data["frames"][iso]
    y = targets(frame, h, label_ok=data["label_ok"][(iso, h)])
    X = reference_features(frame, h)
    ft = data["first_test"][iso]
    probs = reference_walk_forward(X, y, h, ft)
    idx = probs.index
    ll = direction_log_loss(probs, y)
    rank_eval = ll.rank(method="average") / ll.count()
    be = base_effect(frame.PI, h)
    b = build_b_features(frame, h, probs, ll, be)
    rqa = data["rp"][iso].reindex(idx)
    rqa3 = data.get("rp_t3", {}).get(iso)
    parts = [
        b,
        rqa,
        pd.DataFrame(
            {
                "log_loss": ll,
                "loss_rank_eval": rank_eval,
                "direction": y,
                "prob": probs,
            },
            index=idx,
        ),
    ]
    if rqa3 is not None:
        r3 = rqa3.reindex(idx).rename(columns=lambda c: f"{c}_t3")
        parts.insert(2, r3)
    df = parts[0]
    for p in parts[1:]:
        df = df.join(p, how="inner")
    df["iso"] = iso
    df["date"] = df.index
    df["year"] = df.index.year
    df["h"] = h
    return df.reset_index(drop=True)


def _ordered_train(frame: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Date, then iso. RidgePredictor TimeSeriesSplit follows row order."""
    iso = frame["iso"].astype(str).to_numpy()
    dates = pd.to_datetime(frame["date"]).to_numpy()
    order = np.lexsort((iso, dates))
    return frame.iloc[order], np.asarray(y, float)[order]


def _loss_predictor_oos(
    panel: pd.DataFrame,
    groups: dict[str, list[str]],
    h: int,
    *,
    m1_cols: list[str] | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Yearly refit OOS; one horizon; purge h months; train rank within that fit only."""
    sub = panel.loc[panel["h"] == h].sort_values(["date", "iso"])
    m0 = groups["B"]
    m1 = m1_cols if m1_cols is not None else groups["B"] + groups["RQA"]
    p0 = pd.Series(np.nan, index=sub.index, dtype=float)
    p1 = pd.Series(np.nan, index=sub.index, dtype=float)
    meta = []
    for y in sorted(sub["year"].unique()):
        if y < LOSS_PRED_FIRST_YEAR:
            continue
        tr, te = _wf_masks(sub["date"], h, int(y))
        if te.sum() == 0 or tr.sum() < smoke(50, 15):
            continue
        assert _assert_wf_purge(sub["date"], h, int(y)), (y, h)
        cal = sub.iloc[np.flatnonzero(tr)]
        ev = sub.iloc[np.flatnonzero(te)]
        y_train = _rank_train_target(cal["log_loss"].to_numpy(float))
        ok = np.isfinite(y_train)
        min_fit = smoke(30, 10)
        if ok.sum() < min_fit:
            continue
        cal_s, y_tr = _ordered_train(cal.iloc[np.flatnonzero(ok)], y_train[ok])
        dates_tr = pd.to_datetime(cal_s["date"])
        if len(dates_tr) > 1:
            assert bool(dates_tr.is_monotonic_increasing)
        Xc0 = cal_s[m0].to_numpy(float)
        Xe0 = ev[m0].to_numpy(float)
        Xc1 = cal_s[m1].to_numpy(float)
        Xe1 = ev[m1].to_numpy(float)
        m0_model = RidgePredictor()
        m0_model.fit(Xc0, y_tr)
        m1_model = RidgePredictor()
        m1_model.fit(Xc1, y_tr)
        p0.loc[ev.index] = m0_model.predict(Xe0)
        p1.loc[ev.index] = m1_model.predict(Xe1)
        meta.append({"test_year": int(y), "h": h, "n_train": int(len(y_tr)), "n_test": int(te.sum())})
    return p0, p1, pd.DataFrame(meta)


def _delta_spearman_ci(
    p0: np.ndarray,
    p1: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    keep = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(y)
    p0, p1, y, years = p0[keep], p1[keep], y[keep], years[keep]
    if len(y) < 5:
        return {
            "M0": np.nan,
            "M1": np.nan,
            "delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_boot": np.nan,
            "n_obs": int(len(y)),
        }
    d0, d1 = spearman(p0, y), spearman(p1, y)
    delta = d1 - d0
    boots: list[float] = []
    for _ in range(n_boot):
        bi = _panel_bootstrap_rows(years, 1, rng, len(y))
        b0 = spearman(p0[bi], y[bi])
        b1 = spearman(p1[bi], y[bi])
        if np.isfinite(b0) and np.isfinite(b1):
            boots.append(b1 - b0)
    if boots:
        arr = np.asarray(boots, float)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        # Equal-tailed percentile bootstrap p for H0: delta = 0 (same draws as the CI).
        p_boot = min(1.0, (2 * min(int(np.sum(arr <= 0)), int(np.sum(arr >= 0))) + 1) / (len(arr) + 1))
    else:
        lo = hi = p_boot = np.nan
    return {
        "M0": d0,
        "M1": d1,
        "delta": delta,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_boot": float(p_boot),
        "n_obs": len(y),
    }


def run_primary(
    panel_h: pd.DataFrame,
    p0: pd.Series,
    p1: pd.Series,
    spec: dict,
    rng: np.random.Generator,
) -> dict:
    sub = panel_h.loc[panel_h["year"] >= LOSS_PRED_FIRST_YEAR]
    p0s = p0.reindex(sub.index).to_numpy(float)
    p1s = p1.reindex(sub.index).to_numpy(float)
    y = sub["loss_rank_eval"].to_numpy(float)
    years = sub["year"].to_numpy()
    n_boot = int(spec["primary_test"]["bootstrap_n"])
    out = _delta_spearman_ci(p0s, p1s, y, years, n_boot=n_boot, rng=rng)
    verdict = bool(
        np.isfinite(out["delta"])
        and np.isfinite(out.get("ci_low"))
        and out["ci_low"] > 0
    )
    return {**out, "verdict": verdict}


def _fit_event_scores(
    sub: pd.DataFrame,
    h: int,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.full(len(sub), np.nan)
    y_ev = np.full(len(sub), np.nan)
    for y in sorted(sub["year"].unique()):
        if y < LOSS_PRED_FIRST_YEAR:
            continue
        tr, te = _wf_masks(sub["date"], h, int(y))
        if te.sum() == 0 or tr.sum() < smoke(30, 10):
            continue
        assert _assert_wf_purge(sub["date"], h, int(y)), (y, h)
        cal = sub.iloc[np.flatnonzero(tr)]
        ev = sub.iloc[np.flatnonzero(te)]
        ytr = cal["log_loss"].to_numpy(float)
        thr = np.nanquantile(ytr, 0.9)
        event_tr = (ytr > thr).astype(int)
        finite = np.isfinite(ytr)
        if finite.sum() < smoke(30, 10) or len(np.unique(event_tr[finite])) < 2:
            continue
        cal_s, event_s = _ordered_train(cal.iloc[np.flatnonzero(finite)], event_tr[finite])
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        pipe.fit(cal_s[feat_cols].to_numpy(float), event_s)
        try:
            sc = pipe.decision_function(ev[feat_cols].to_numpy(float))
        except ValueError:
            continue
        scores[np.flatnonzero(te)] = sc
        y_ev[np.flatnonzero(te)] = (ev["log_loss"].to_numpy(float) > thr).astype(float)
    return scores, y_ev, np.isfinite(scores) & np.isfinite(y_ev)


def fit_event_compare(panel: pd.DataFrame, groups: dict[str, list[str]], h: int) -> dict:
    sub = panel.loc[panel["h"] == h].sort_values(["date", "iso"])
    s0, y0, ok0 = _fit_event_scores(sub, h, groups["B"])
    s1, y1, ok1 = _fit_event_scores(sub, h, groups["B"] + groups["RQA"])
    ok = ok0 & ok1
    if ok.sum() < 10 or len(np.unique(y0[ok])) < 2:
        return {"auc_M0": np.nan, "auc_M1": np.nan, "delta_auc": np.nan}
    auc0 = float(roc_auc_score(y0[ok], s0[ok]))
    auc1 = float(roc_auc_score(y1[ok], s1[ok]))
    return {"auc_M0": auc0, "auc_M1": auc1, "delta_auc": auc1 - auc0}


def run_secondaries(
    panel: pd.DataFrame,
    groups: dict[str, list[str]],
    groups_t3: dict[str, list[str]],
    spec: dict,
    rng: np.random.Generator,
    out_dir: Path,
) -> None:
    n_boot = int(spec["primary_test"]["bootstrap_n"])
    pooled_rows = []
    family_tests = []
    pooled_h12 = None
    for h in HORIZONS:
        sub_h = panel.loc[panel["h"] == h]
        p0, p1, _ = _loss_predictor_oos(sub_h, groups, h)
        eval_mask = sub_h["year"] >= LOSS_PRED_FIRST_YEAR
        eval_sub = sub_h.loc[eval_mask]
        res = _delta_spearman_ci(
            p0.reindex(eval_sub.index).to_numpy(float),
            p1.reindex(eval_sub.index).to_numpy(float),
            eval_sub["loss_rank_eval"].to_numpy(float),
            eval_sub["year"].to_numpy(),
            n_boot=n_boot,
            rng=rng,
        )
        pooled_rows.append({"scope": "pooled", "h": h, "model": "B+WRQA+LRQA", **res})
        if h == PRIMARY_H:
            pooled_h12 = (p0, p1)
            for name, m1c in (
                ("B+WRQA", groups["B"] + groups["WRQA"]),
                ("B+LRQA", groups["B"] + groups["LRQA"]),
            ):
                _, p1f, _ = _loss_predictor_oos(sub_h, groups, h, m1_cols=m1c)
                rf = _delta_spearman_ci(
                    p0.reindex(eval_sub.index).to_numpy(float),
                    p1f.reindex(eval_sub.index).to_numpy(float),
                    eval_sub["loss_rank_eval"].to_numpy(float),
                    eval_sub["year"].to_numpy(),
                    n_boot=n_boot,
                    rng=rng,
                )
                family_tests.append({"family": name, "h": h, **rf})
            _, p1t3, _ = _loss_predictor_oos(
                sub_h, groups_t3, h, m1_cols=groups_t3["B"] + groups_t3["RQA_T3"]
            )
            rt3 = _delta_spearman_ci(
                p0.reindex(eval_sub.index).to_numpy(float),
                p1t3.reindex(eval_sub.index).to_numpy(float),
                eval_sub["loss_rank_eval"].to_numpy(float),
                eval_sub["year"].to_numpy(),
                n_boot=n_boot,
                rng=rng,
            )
            family_tests.append({"family": "theiler_3", "h": h, **rt3})
    if pooled_rows:
        report.save_table(pd.DataFrame(pooled_rows), out_dir, "secondary_pooled")
    if family_tests:
        fdf = pd.DataFrame(family_tests)
        if spec.get("holm_secondary"):
            p_raw = fdf["p_boot"].to_numpy(float) if "p_boot" in fdf.columns else np.ones(len(fdf))
            p_raw = np.where(np.isfinite(p_raw), p_raw, 1.0)
            fdf["p_raw"] = p_raw
            fdf["p_holm"] = holm(p_raw)
        report.save_table(fdf, out_dir, "secondary_families_holm")
    us_rows = []
    for h in HORIZONS:
        sub_us = panel.loc[(panel["iso"] == "USA") & (panel["h"] == h)]
        if sub_us.empty:
            continue
        p0, p1, _ = _loss_predictor_oos(sub_us, groups, h)
        ev = sub_us.loc[sub_us["year"] >= LOSS_PRED_FIRST_YEAR]
        ru = _delta_spearman_ci(
            p0.reindex(ev.index).to_numpy(float),
            p1.reindex(ev.index).to_numpy(float),
            ev["loss_rank_eval"].to_numpy(float),
            ev["year"].to_numpy(),
            n_boot=n_boot,
            rng=rng,
        )
        us_rows.append({"scope": "USA", "h": h, **ru})
    if us_rows:
        report.save_table(pd.DataFrame(us_rows), out_dir, "secondary_us")
    ev = fit_event_compare(panel, groups, PRIMARY_H)
    report.save_table(pd.DataFrame([{"h": PRIMARY_H, **ev}]), out_dir, "secondary_event")
    case_parts = []
    if pooled_h12 is not None:
        p0_case, p1_case = pooled_h12
        for iso in ["USA", "GBR", "DEU"]:
            sub = panel.loc[(panel["iso"] == iso) & (panel["h"] == PRIMARY_H)]
            window = sub[(sub["date"] >= "2021-01-01") & (sub["date"] <= "2023-12-31")]
            if window.empty:
                continue
            case_parts.append(
                window.assign(
                    pred_M0=p0_case.reindex(window.index).to_numpy(),
                    pred_M1=p1_case.reindex(window.index).to_numpy(),
                )[["iso", "date", "log_loss", "loss_rank_eval", "pred_M0", "pred_M1"]]
            )
    if case_parts:
        report.save_table(pd.concat(case_parts, ignore_index=True), out_dir, "secondary_case_2021_2023")


def figure_us_loss(panel: pd.DataFrame, p0: pd.Series, p1: pd.Series, fig_dir: Path):
    sub = panel[(panel["iso"] == "USA") & (panel["h"] == PRIMARY_H)].copy()
    sub = sub[(sub["date"] >= "2015-01-01") & (sub["date"] <= "2026-12-31")]
    if sub.empty:
        return
    sub = sub.sort_values("date")
    p0a = p0.loc[sub.index]
    p1a = p1.loc[sub.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub["date"], sub["log_loss"], "k-", alpha=0.5, label="realised log loss")
    ax.plot(sub["date"], p0a, "b--", label="M0 pred rank")
    ax.plot(sub["date"], p1a, "r--", label="M1 pred rank")
    ax.set_title("USA: realised log loss vs M0/M1 predicted loss rank (h=12)")
    ax.legend(loc="upper right")
    report.save_fig(fig, fig_dir, "us_loss_2015_2026")
    plt.close(fig)


def run_full_pipeline(
    spec: dict,
    out_dir: Path,
    fig_dir: Path,
    cache_exp: str,
    *,
    data: dict | None = None,
) -> dict:
    rng = np.random.default_rng(0)
    smoke_mode = spec["smoke"]
    data = data if data is not None else load_panel(spec, cache_exp)
    check_isos = ref_check_isos(smoke_mode)
    ref_rows = []
    oecd_ok = True
    usa_note = ""
    for h in [PRIMARY_H]:
        ref_df, ok, usa_warn = run_reference_reproduction(data, h, check_isos)
        ref_rows.append(ref_df)
        oecd_ok &= ok
        if usa_warn:
            usa_note = "USA AUC differs from crosscountry (OECD USA); expected with CPIAUCSL"
    reference_check = pd.concat(ref_rows, ignore_index=True)
    report.save_table(reference_check, out_dir, "reference_check")
    if not oecd_ok:
        raise RuntimeError(f"OECD reference AUC reproduction failed tolerance {AUC_TOL}")

    country_panels = []
    for iso in spec["countries"]:
        for h in HORIZONS:
            country_panels.append(assemble_country_rows(data, iso, h, cache_exp))
    panel = pd.concat(country_panels, ignore_index=True)
    panel = panel.sort_values(["date", "iso", "h"]).reset_index(drop=True)

    feat_cols = (
        B_KEYS
        + [f"wrqa_{k}" for k in WRQA_KEYS]
        + [f"lrqa_{k}" for k in LRQA_KEYS]
        + [f"wrqa_{k}_t3" for k in WRQA_KEYS]
        + [f"lrqa_{k}_t3" for k in LRQA_KEYS]
    )
    panel = causal_expanding_z(panel, feat_cols)
    groups = feature_groups()
    groups_t3 = feature_groups_t3()

    panel_h12 = panel.loc[(panel["h"] == PRIMARY_H) & (panel["year"] >= LOSS_PRED_FIRST_YEAR)]
    p0, p1, wf_meta = _loss_predictor_oos(panel, groups, PRIMARY_H)
    primary = run_primary(panel.loc[panel["h"] == PRIMARY_H], p0, p1, spec, rng)
    report.save_table(wf_meta, out_dir, "loss_predictor_walkforward")
    panel_out = panel_h12.copy()
    panel_out["pred_M0"] = p0.reindex(panel_out.index).to_numpy()
    panel_out["pred_M1"] = p1.reindex(panel_out.index).to_numpy()
    report.save_table(panel_out, out_dir, "loss_prediction_panel")
    per_country = (
        panel_h12.groupby("iso")
        .agg(
            n=("log_loss", "count"),
            mean_loss=("log_loss", "mean"),
            mean_rank_eval=("loss_rank_eval", "mean"),
        )
        .reset_index()
    )
    report.save_table(per_country, out_dir, "per_country")

    decision = [
        {"gate": "reference_check_oecd", "verdict": "pass" if oecd_ok else "fail", "note": usa_note},
        {
            "gate": "E9_primary",
            "h": PRIMARY_H,
            "delta_spearman": primary["delta"],
            "ci_low": primary.get("ci_low"),
            "ci_high": primary.get("ci_high"),
            "verdict": "pass" if primary["verdict"] else "fail",
        },
    ]
    if smoke_mode:
        decision.append({"gate": "smoke", "verdict": "pass"})
    report.write_decision(decision, out_dir)

    figure_us_loss(panel, p0, p1, fig_dir)
    run_secondaries(panel, groups, groups_t3, spec, rng, out_dir)

    return {
        "decision": decision,
        "primary": primary,
        "reference_check": reference_check,
        "usa_note": usa_note,
    }

