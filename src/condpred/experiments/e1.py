"""E1 — local forecast error in chaotic systems."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from condpred import contaminate, systems
from condpred.cache import cached
from condpred.config import smoke
from condpred.evaluation import LGBMPredictor, RidgePredictor, compare, spearman
from condpred.features import baselines
from condpred.features.local_rqa import local_rqa
from condpred.features.window_rqa import window_rqa
from condpred.forecasters import ARFixed, KNNAnalogue
from condpred.ftle import ftle_flow, ftle_logistic, ftle_map, lorenz_jacobian, lorenz_rhs
from condpred.geometry import Geometry
from condpred.losses import loss_transform, sq_norm
from condpred.selfcheck import leakage_test
from condpred.stats import holm
from recplay.embedding import average_mutual_information, embed, false_nearest_fraction, first_local_minimum

E1_MODULES = [
    "condpred.experiments.e1",
    "condpred.features.window_rqa",
    "condpred.features.local_rqa",
    "condpred.features.baselines",
    "condpred.forecasters",
]

LYAP_TARGETS = (0.1, 0.5, 1.0)
B_KEYS = ["recent_loss", "knn_spread", "state_loc"]
B_SCALAR = ["level_cur", "level_mean", "level_vol_surprise", "perm_entropy", "rolling_var"]
WRQA_KEYS = ["DET", "L", "ENTR", "LAM", "TT", "Lmax", "Vmax", "DET3_DET2", "K2_lines", "DET_LAM"]
LRQA_KEYS = ["RR_last", "Lb_mean", "Lb_max", "DET_last", "Lf_nb", "K2_local", "LAM_last", "TT_last"]

_SYSTEM_META = {
    "logistic": {"kind": "map", "r": 3.9, "delta": 1.0, "gen": lambda n, rng: systems.logistic(3.9, n, rng)},
    "henon": {
        "kind": "map",
        "delta": 1.0,
        "gen": lambda n, rng: systems.henon(n, rng, a=1.4, b=0.3),
    },
    "lorenz63": {
        "kind": "flow",
        "delta": 0.05,
        "gen": lambda n, rng: systems.lorenz63(n, rng, dt=0.01, sample_every=5),
    },
    "rossler": {
        "kind": "flow",
        "delta": 0.2,
        "gen": lambda n, rng: systems.rossler(n, rng, dt=0.05, sample_every=4),
    },
}


def system_list(smoke_mode: bool) -> list[str]:
    full = ["logistic", "henon", "lorenz63", "rossler"]
    return full[:2] if smoke_mode else full


def segment_lengths(smoke_mode: bool) -> dict[str, int]:
    if smoke_mode:
        return {"library": 2_000, "calibration": 1_500, "evaluation": 1_500}
    return {"library": 20_000, "calibration": 10_000, "evaluation": 10_000}


def seed_list(smoke_mode: bool) -> list[int]:
    return list(range(2 if smoke_mode else 5))


def build_initial_spec(smoke_mode: bool) -> dict:
    segs = segment_lengths(smoke_mode)
    return {
        "experiment": "E1",
        "version": 1,
        "changelog": [],
        "smoke": smoke_mode,
        "systems": system_list(smoke_mode),
        "seeds": seed_list(smoke_mode),
        "segments": segs,
        "sigma_obs_primary": 0.0,
        "sigma_obs_levels": [0.0, 0.05],
        "W_primary": 500,
        "W_secondary": 200,
        "geometry": {"scaling": "wz", "distance": "euclid", "threshold": "rr", "rr": 0.10},
        "theiler_rule": "(m - 1) * tau",
        "step": 20 if smoke_mode else 5,
        "horizons": {
            "rule": "h * lambda_bar * Delta ≈ [0.1, 0.5, 1.0]; integers filled before losses",
            "lyapunov_units": list(LYAP_TARGETS),
            "per_system": None,
        },
        "forecaster_primary": "KNNAnalogue(k=10)",
        "predictor_primary": "RidgePredictor",
        "models": {"M0": "B", "M1a": "B+WRQA", "M1b": "B+LRQA", "M1c": "B+WRQA+LRQA"},
        "primary_test": {
            "metric": "delta_spearman",
            "M0": "B",
            "M1": "M1b",
            "predictor": "Ridge",
            "W": 500,
            "sigma_obs": 0.0,
            "forecaster": "KNN",
            "delta_threshold": 0.02,
            "ci_rule": "95% CI lower > 0",
            "pass_rule": "≥2 of 4 systems with ≥2 of 3 horizons each passing",
        },
        "gate_E1": "primary_test pass_rule",
        "gate_G1": "E1_pass OR E2_pass (E2 not run here → G1 = E1_pass)",
    }


@dataclass
class Trajectory:
    x: np.ndarray
    state: np.ndarray | None


def _total_n(segs: dict[str, int]) -> int:
    return segs["library"] + segs["calibration"] + segs["evaluation"]


def generate_trajectory(system: str, seed: int, segs: dict[str, int], *, cache_exp: str) -> Trajectory:
    n = _total_n(segs)

    def _gen():
        rng = np.random.default_rng(seed + 17_001)
        meta = _SYSTEM_META[system]
        if system == "lorenz63":
            raw, x = meta["gen"](n, rng)
            return Trajectory(np.asarray(x, float), np.asarray(raw, float))
        if system == "rossler":
            raw = meta["gen"](n, rng)
            x = raw[:, 0]
            return Trajectory(np.asarray(x, float), np.asarray(raw, float))
        if system == "henon":
            x, st = meta["gen"](n, rng)
            return Trajectory(np.asarray(x, float), np.asarray(st, float))
        x = meta["gen"](n, rng)
        return Trajectory(np.asarray(x, float), None)

    key = {"system": system, "seed": seed, "n": n}
    return cached(cache_exp, f"traj_{system}_{seed}", key, _gen, modules=E1_MODULES)


def choose_embedding(
    x_lib: np.ndarray, is_map: bool, *, max_n: int = 5000
) -> tuple[int, int, np.ndarray]:
    x_lib = np.asarray(x_lib, float)
    if len(x_lib) > max_n:
        step = max(1, len(x_lib) // max_n)
        x_fit = x_lib[::step][:max_n]
    else:
        x_fit = x_lib
    max_lag = min(40, max(5, len(x_fit) // 20))
    ami = average_mutual_information(x_fit, max_lag)
    tau = first_local_minimum(ami)
    if tau is None:
        tau = 1 if is_map else 1
    fnn = false_nearest_fraction(x_fit, tau, max_m=8)
    m = 8
    for i, f in enumerate(fnn):
        if f < 0.05:
            m = i + 1
            break
    return m, tau, fnn


def _henon_jac(st):
    a, b = 1.4, 0.3
    x = st[0] if st.ndim else st
    return np.array([[-2 * a * x, 1.0], [b, 0.0]])


def estimate_lambda_bar(system: str, traj: Trajectory, h_ref: int = 50) -> float:
    meta = _SYSTEM_META[system]
    x = traj.x
    n = len(x)
    h_ref = min(h_ref, max(5, n // 100))
    if system == "logistic":
        ftle = ftle_logistic(x[: min(n, 5000)], r=3.9, h=h_ref)
    elif system == "henon" and traj.state is not None:
        jac_fn = lambda s: _henon_jac(s)
        ftle = ftle_map(jac_fn, traj.state[: min(n, 5000)], h_ref)
    elif system == "lorenz63" and traj.state is not None:
        ftle, _ = ftle_flow(
            lorenz_rhs,
            lorenz_jacobian,
            traj.state[0],
            0.01,
            5,
            min(n, 3000),
            h_ref,
            sub=5,
            burn=0,
        )
    elif system == "rossler" and traj.state is not None:

        def rossler_rhs(s, a=0.2, b=0.2, c=5.7):
            return np.array([-s[1] - s[2], s[0] + a * s[1], b + s[2] * (s[0] - c)])

        def rossler_jac(s, a=0.2, b=0.2, c=5.7):
            return np.array([[0, -1, -1], [1, a, 0], [0, 1, s[0] - c]])

        ftle, _ = ftle_flow(
            rossler_rhs,
            rossler_jac,
            traj.state[0],
            0.05,
            4,
            min(n, 3000),
            h_ref,
            sub=1,
            burn=0,
        )
    else:
        ftle = ftle_logistic(x[: min(n, 5000)], r=3.9, h=h_ref)
    ok = np.isfinite(ftle)
    if not ok.any():
        return 0.5
    return float(np.nanmean(ftle[ok]))


def integer_horizons(lambda_bar: float, delta: float) -> list[int]:
    out = []
    for target in LYAP_TARGETS:
        denom = max(lambda_bar * delta, 1e-6)
        h = max(1, int(round(target / denom)))
        if out and h <= out[-1]:
            h = out[-1] + 1
        out.append(h)
    return out


def complete_horizons(spec: dict, horizons: dict[str, list[int]], out_dir, modules) -> dict:
    from condpred.report import write_spec

    spec = dict(spec)
    spec["horizons"] = {**spec["horizons"], "per_system": horizons}
    write_spec(spec, out_dir, modules=modules, completion_keys=frozenset({"horizons"}))
    return spec


def bounds(segs: dict[str, int]) -> dict[str, slice]:
    lib = slice(0, segs["library"])
    cal = slice(segs["library"], segs["library"] + segs["calibration"])
    ev = slice(segs["library"] + segs["calibration"], _total_n(segs))
    return {"library": lib, "calibration": cal, "evaluation": ev}


def apply_obs_noise(x: np.ndarray, level: float, seed: int) -> np.ndarray:
    if level <= 0:
        return x.copy()
    rng = np.random.default_rng(seed + 99_001)
    omega = level * float(np.std(x))
    return contaminate.add_obs_noise(x, omega, rng)


def fit_knn(x_lib: np.ndarray, m: int, tau: int, horizons: list[int]) -> KNNAnalogue:
    knn = KNNAnalogue(m, tau, k=10, horizons=tuple(horizons))
    knn.fit(x_lib)
    return knn


def squared_loss_series(
    x: np.ndarray,
    forecaster: KNNAnalogue | ARFixed,
    horizons: list[int],
    t_start: int,
    t_end: int,
    var_lib: float,
    *,
    ar: bool = False,
    cal_start: int = 0,
) -> dict[int, np.ndarray]:
    """One ``predict`` per origin ``t``; sparse library grid (step 5), dense from ``cal_start``."""
    n = len(x)
    max_h = max(horizons)
    if not ar:
        t_start = max(t_start, (forecaster.m - 1) * forecaster.tau)
    else:
        t_start = max(t_start, forecaster.p)
    out = {h: np.full(n, np.nan) for h in horizons}
    t_last = t_end - max_h
    lib_sparse = 10 if os.environ.get("CONDPRED_SMOKE") != "1" else 5
    n_done = 0
    for t in range(t_start, t_last):
        if t < cal_start and (t - t_start) % lib_sparse != 0:
            continue
        n_done += 1
        if n_done % 5000 == 0:
            print(f"E1 losses t={t}/{t_last}", flush=True)
        if ar:
            by_h = {h: forecaster.predict(t, h, series=x) for h in horizons}
        else:
            pred = forecaster.predict(t, series=x)
            by_h = {h: pred[h][0] for h in horizons}
        for h in horizons:
            if t + h >= t_end:
                continue
            yhat = by_h[h]
            if not np.isfinite(yhat):
                continue
            out[h][t] = sq_norm(x[t + h], yhat, var_lib)
    return out


def ftle_at_t(system: str, traj: Trajectory, t: int, h: int) -> float:
    meta = _SYSTEM_META[system]
    if system == "logistic":
        arr = ftle_logistic(traj.x[max(0, t - h) : t + h + 1], r=3.9, h=h)
        return float(arr[0]) if len(arr) else np.nan
    if system == "henon" and traj.state is not None:
        st = traj.state[max(0, t - h) : t + h + 1]
        arr = ftle_map(lambda s: _henon_jac(s), st, h)
        idx = min(len(arr) - 1, h)
        return float(arr[idx]) if idx >= 0 else np.nan
    return np.nan


def _rossler_rhs(s, a=0.2, b=0.2, c=5.7):
    return np.array([-s[1] - s[2], s[0] + a * s[1], b + s[2] * (s[0] - c)])


def _rossler_jac(s, a=0.2, b=0.2, c=5.7):
    return np.array([[0, -1, -1], [1, a, 0], [0, 1, s[0] - c]])


def ftle_at_rows(system: str, traj: Trajectory, ts, h: int) -> np.ndarray:
    """True FTLE over [t, t+h] at each t in ``ts`` (clean trajectory, per-sample units)."""
    ts = np.asarray(list(ts), int)
    out = np.full(len(ts), np.nan)
    if len(ts) == 0 or traj.state is None and system in ("henon", "lorenz63", "rossler"):
        return out
    if system == "logistic":
        arr = ftle_logistic(traj.x, r=3.9, h=h)
    elif system == "henon":
        arr = ftle_map(lambda s: _henon_jac(s), traj.state, h)
    elif system in ("lorenz63", "rossler"):
        # Re-integrate only from the first requested t (true state there) to bound cost.
        start = int(ts.min())
        n = int(ts.max()) - start + h + 1
        if system == "lorenz63":
            expon, _ = ftle_flow(lorenz_rhs, lorenz_jacobian, traj.state[start], 0.01, 5, n, h, sub=5, burn=0)
            arr_full = expon * 0.05  # per time unit -> per sample (scale-invariant for Spearman)
        else:
            expon, _ = ftle_flow(_rossler_rhs, _rossler_jac, traj.state[start], 0.05, 4, n, h, sub=1, burn=0)
            arr_full = expon * 0.2
        idx = ts - start
        ok = (idx >= 0) & (idx < len(arr_full))
        out[ok] = arr_full[idx[ok]]
        return out
    else:
        return out
    idx = ts
    ok = (idx >= 0) & (idx < len(arr))
    out[ok] = arr[idx[ok]]
    return out


def _geom(m: int, tau: int, spec_geom: dict) -> Geometry:
    g = Geometry(
        scaling=spec_geom.get("scaling", "wz"),
        distance=spec_geom.get("distance", "euclid"),
        threshold=spec_geom.get("threshold", "rr"),
        rr=spec_geom.get("rr", 0.10),
        theiler=None,
        m=m,
        tau=tau,
    )
    return g


def _rqa_feature_row(
    x: np.ndarray,
    t: int,
    W: int,
    m: int,
    tau: int,
    geom: Geometry,
) -> dict:
    row: dict = {}
    lev = baselines.level_features(x, t, W)
    for k, v in lev.items():
        row[k] = v
    row["perm_entropy"] = baselines.perm_entropy(x, W, t, order=4)
    row["rolling_var"] = baselines.rolling_var(x, W, t)
    sl = baselines.state_location(x, t, m, tau)
    for i, v in enumerate(sl):
        row[f"state_loc_{i}"] = float(v)
    try:
        wr = window_rqa(x, t, W, m, tau, geom)
    except (IndexError, ValueError):
        wr = {k: np.nan for k in WRQA_KEYS}
    for k in WRQA_KEYS:
        row[f"wrqa_{k}"] = wr.get(k, np.nan)
    try:
        lr = local_rqa(x, t, W, m, tau, geom)
    except (IndexError, ValueError):
        lr = {k: np.nan for k in LRQA_KEYS}
    for k in LRQA_KEYS:
        row[f"lrqa_{k}"] = lr.get(k, np.nan)
    return row


def _lrqa_feature_row(x, t, W, m, tau, geom) -> dict:
    """B scalars + LRQA only (secondaries 3/4/8 test M1b; WRQA not needed)."""
    row: dict = {}
    for k, v in baselines.level_features(x, t, W).items():
        row[k] = v
    row["perm_entropy"] = baselines.perm_entropy(x, W, t, order=4)
    row["rolling_var"] = baselines.rolling_var(x, W, t)
    for i, v in enumerate(baselines.state_location(x, t, m, tau)):
        row[f"state_loc_{i}"] = float(v)
    try:
        lr = local_rqa(x, t, W, m, tau, geom)
    except (IndexError, ValueError):
        lr = {k: np.nan for k in LRQA_KEYS}
    for k in LRQA_KEYS:
        row[f"lrqa_{k}"] = lr.get(k, np.nan)
    for k in WRQA_KEYS:
        row[f"wrqa_{k}"] = np.nan
    return row


def _h_baselines(
    x: np.ndarray,
    t: int,
    h: int,
    m: int,
    tau: int,
    loss_h: np.ndarray,
    lookback_lib: int,
    lib_end: int,
) -> dict:
    return {
        "recent_loss": baselines.recent_loss(loss_h, h, 20.0, t),
        "knn_spread": baselines.knn_spread(
            x, t, m, tau, 10, h, lookback_lib, lib_end=lib_end
        ),
    }


def _one_feature_row(
    x: np.ndarray,
    t: int,
    W: int,
    m: int,
    tau: int,
    h: int,
    geom: Geometry,
    loss_by_h: dict[int, np.ndarray],
    lookback_lib: int,
    lib_end: int,
) -> dict:
    row = _rqa_feature_row(x, t, W, m, tau, geom)
    row.update(_h_baselines(x, t, h, m, tau, loss_by_h[h], lookback_lib, lib_end))
    return row


def build_rqa_frame(
    x: np.ndarray,
    seg: slice,
    W: int,
    m: int,
    tau: int,
    geom: Geometry,
    step: int,
    n_jobs: int = -1,
    lrqa_only: bool = False,
) -> pd.DataFrame:
    min_t = seg.start + max(W, (m - 1) * tau)
    ts = list(range(min_t, seg.stop, step))
    fn = _lrqa_feature_row if lrqa_only else _rqa_feature_row
    rows = Parallel(n_jobs=n_jobs)(delayed(fn)(x, t, W, m, tau, geom) for t in ts)
    df = pd.DataFrame(rows, index=ts)
    df.index.name = "t"
    return df


def attach_horizon(
    df: pd.DataFrame, x, h, loss_h, m, tau, lookback_lib, lib_end, seg_stop
) -> pd.DataFrame:
    out = df.copy()
    for t in out.index:
        for k, v in _h_baselines(x, t, h, m, tau, loss_h, lookback_lib, lib_end).items():
            out.at[t, k] = v
        out.at[t, "L"] = loss_h[t]
        out.at[t, "y"] = loss_transform(loss_h[t])
    return purge_rows(out, h, seg_stop - 1)


def build_rows(
    x: np.ndarray,
    seg: slice,
    h: int,
    loss_h: np.ndarray,
    W: int,
    m: int,
    tau: int,
    geom: Geometry,
    step: int,
    lookback_lib: int,
    lib_end: int,
    n_jobs: int = -1,
) -> pd.DataFrame:
    loss_by_h = {h: loss_h}
    min_t = seg.start + max(W, (m - 1) * tau)
    ts = list(range(min_t, seg.stop, step))
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_one_feature_row)(
            x, t, W, m, tau, h, geom, loss_by_h, lookback_lib, lib_end
        )
        for t in ts
    )
    df = pd.DataFrame(rows, index=ts)
    df.index.name = "t"
    df["L"] = [loss_h[t] for t in ts]
    df["y"] = [loss_transform(loss_h[t]) for t in ts]
    return df


def column_groups(m: int) -> dict[str, list[str]]:
    state_cols = [f"state_loc_{i}" for i in range(m)]
    b_cols = (
        ["recent_loss", "knn_spread"]
        + state_cols
        + ["level_cur", "level_mean", "level_vol_surprise", "perm_entropy", "rolling_var"]
    )
    w_cols = [f"wrqa_{k}" for k in WRQA_KEYS]
    l_cols = [f"lrqa_{k}" for k in LRQA_KEYS]
    return {"B": b_cols, "WRQA": w_cols, "LRQA": l_cols}


def purge_rows(df: pd.DataFrame, h: int, seg_stop: int) -> pd.DataFrame:
    return df.loc[df.index + h <= seg_stop - 1]


def fit_predict(
    df_cal: pd.DataFrame,
    df_ev: pd.DataFrame,
    cols: list[str],
    y_col: str,
    predictor: str = "ridge",
) -> tuple[np.ndarray, np.ndarray, object]:
    X_cal = df_cal[cols].to_numpy(float)
    y_cal = df_cal[y_col].to_numpy(float)
    X_ev = df_ev[cols].to_numpy(float)
    ok = np.isfinite(y_cal)
    if ok.sum() < 10:
        nan = np.full(len(df_ev), np.nan)
        return nan, nan, None
    col_ok = np.isfinite(X_cal[ok]).any(axis=0)
    if not col_ok.any():
        nan = np.full(len(df_ev), np.nan)
        return nan, nan, None
    X_cal = X_cal[:, col_ok]
    X_ev = X_ev[:, col_ok]
    if predictor == "lgbm":
        model = LGBMPredictor()
    else:
        model = RidgePredictor()
    model.fit(X_cal[ok], y_cal[ok])
    pred = model.predict(X_ev)
    return pred, y_cal[ok], model


def primary_system_horizon(
    cal_frames: list[pd.DataFrame],
    ev_frames: list[pd.DataFrame],
    group_list: list[dict[str, list[str]]],
    y_col: str = "y",
    h: int = 1,
    step: int = 5,
    rng: np.random.Generator | None = None,
    n_boot: int = 200,
) -> dict:
    p0_all, p1_all, y_all = [], [], []
    for cal, ev, groups in zip(cal_frames, ev_frames, group_list):
        cols_m0 = groups["B"]
        cols_m1 = groups["B"] + groups["LRQA"]
        p0, _, _ = fit_predict(cal, ev, cols_m0, y_col)
        p1, _, _ = fit_predict(cal, ev, cols_m1, y_col)
        p0_all.append(p0)
        p1_all.append(p1)
        y_all.append(ev[y_col].to_numpy(float))
    p0 = np.concatenate(p0_all)
    p1 = np.concatenate(p1_all)
    y = np.concatenate(y_all)
    ok = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(y)
    delta = spearman(p1[ok], y[ok]) - spearman(p0[ok], y[ok]) if ok.sum() >= 5 else np.nan
    from condpred.stats import block_bootstrap_idx

    block = max(1, int(5 * h / step))
    boots = []
    for _ in range(n_boot):
        parts_d = []
        for p0s, p1s, ys in zip(p0_all, p1_all, y_all):
            n = len(ys)
            idx = block_bootstrap_idx(n, block, rng)
            okb = np.isfinite(p0s[idx]) & np.isfinite(p1s[idx]) & np.isfinite(ys[idx])
            if okb.sum() < 5:
                continue
            parts_d.append(spearman(p1s[idx][okb], ys[idx][okb]) - spearman(p0s[idx][okb], ys[idx][okb]))
        if parts_d:
            boots.append(float(np.mean(parts_d)))
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo, hi = np.nan, np.nan
    verdict = (
        bool(np.isfinite(delta) and delta > 0.02 and np.isfinite(lo) and lo > 0)
        if not np.isnan(delta)
        else False
    )
    return {"delta": delta, "ci_low": float(lo), "ci_high": float(hi), "verdict": verdict}


def e1_overall_pass(decision_rows: list[dict]) -> bool:
    by_sys: dict[str, list[bool]] = {}
    for r in decision_rows:
        if r.get("gate") != "E1_cell":
            continue
        by_sys.setdefault(r["system"], []).append(r["verdict"] == "pass")
    n_sys_pass = sum(sum(v) >= 2 for v in by_sys.values())
    return n_sys_pass >= 2


def _read_e2_pass(out_dir) -> bool | None:
    """E2 full verdict from its decision.csv; None when the file/row is missing."""
    out_dir = Path(out_dir)
    candidates = [out_dir.parent / "E2" / "decision.csv", out_dir.parent.parent / "E2" / "decision.csv"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "smoke" in df.columns and df["smoke"].fillna(False).astype(bool).all():
            continue
        rows = df[df["gate"].astype(str) == "E2"] if "gate" in df.columns else df.iloc[0:0]
        if len(rows) and "verdict" in df.columns:
            return str(rows.iloc[0]["verdict"]).strip().lower() == "pass"
    return None


def build_decision_table(primary_results: list[dict], smoke_mode: bool, e2_pass: bool | None = None) -> list[dict]:
    rows = []
    for r in primary_results:
        rows.append(
            {
                "gate": "E1_cell",
                "system": r["system"],
                "horizon": r["h"],
                "statistic": "delta_spearman_M1b_M0",
                "value": r["delta"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "rule": "delta>0.02 and ci_low>0",
                "verdict": "pass" if r["verdict"] else "fail",
            }
        )
    e1_pass = e1_overall_pass(rows)
    rows.append(
        {
            "gate": "E1",
            "system": "all",
            "horizon": "all",
            "statistic": "primary_test",
            "value": e1_pass,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "rule": "≥2 systems each with ≥2 horizons passing",
            "verdict": "pass" if e1_pass else "fail",
        }
    )
    if e1_pass or e2_pass:
        g1_verdict, g1_value = "pass", True
    elif e2_pass is False:
        g1_verdict, g1_value = "fail", False
    else:
        g1_verdict, g1_value = "pending", np.nan
    rows.append(
        {
            "gate": "G1",
            "system": "all",
            "horizon": "all",
            "statistic": "E1_pass OR E2_pass",
            "value": g1_value,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "rule": "G1 = E1 pass OR E2 pass (E2 read from its full decision.csv); pending only if that file is missing",
            "verdict": g1_verdict,
        }
    )
    if smoke_mode:
        for r in rows:
            r["smoke"] = True
    return rows


def run_leakage_e1(x: np.ndarray, rng: np.random.Generator) -> list[tuple[str, bool]]:
    W, m, tau, h = 80, 2, 1, 5
    geom = _geom(m, tau, {"scaling": "wz", "distance": "euclid", "threshold": "rr", "rr": 0.1})
    n = len(x)
    t_list = list(rng.integers(W + 20, n - 50, size=8))
    loss = np.abs(rng.standard_normal(n)) ** 2

    def pipe(arr, t):
        return _one_feature_row(arr, t, W, m, tau, h, geom, {h: loss}, 200, lib_end=len(arr) // 2)

    checks = [("E1_features", leakage_test(pipe, x, t_list, rng))]

    knn = KNNAnalogue(m, tau, k=5, horizons=(h,))
    knn.fit(x[: max(t_list)])
    checks.append(("E1_KNN", leakage_test(lambda arr, t: knn.predict(t, series=arr)[h], x, t_list, rng)))
    return checks


def run_embedding_and_horizons(
    spec: dict,
    trajectories: dict[tuple[str, int], Trajectory],
    out_dir,
) -> tuple[dict, pd.DataFrame]:
    segs = spec["segments"]
    horizons_map: dict[str, list[int]] = {}
    emb_rows = []
    for system in spec["systems"]:
        # pool lambda over seeds
        labs = [estimate_lambda_bar(system, trajectories[(system, s)]) for s in spec["seeds"]]
        lam = float(np.mean(labs))
        hs = integer_horizons(lam, _SYSTEM_META[system]["delta"])
        horizons_map[system] = hs
        for seed in spec["seeds"]:
            x_lib = trajectories[(system, seed)].x[bounds(segs)["library"]]
            m, tau, fnn = choose_embedding(x_lib, _SYSTEM_META[system]["kind"] == "map")
            emb_rows.append({"system": system, "seed": seed, "m": m, "tau": tau, "fnn_curve": json.dumps(fnn.tolist())})
    spec = complete_horizons(spec, horizons_map, out_dir, E1_MODULES)
    emb_df = pd.DataFrame(emb_rows)
    return spec, emb_df, horizons_map, { (r["system"], r["seed"]): (r["m"], r["tau"]) for r in emb_rows }


def load_trajectories(spec: dict, cache_prefix: str) -> dict[tuple[str, int], Trajectory]:
    segs = spec["segments"]
    out: dict[tuple[str, int], Trajectory] = {}
    for system in spec["systems"]:
        for seed in spec["seeds"]:
            out[(system, seed)] = generate_trajectory(system, seed, segs, cache_exp=cache_prefix)
    return out


def _build_packs(spec, trajectories, horizons_map, emb_choice, *, sigma, W, forecaster_kind="knn", scaling=None, lrqa_only=False, loss_by_h_map=None, label="packs"):
    """Packs per (system, h, seed): losses + RQA frames attached per horizon."""
    bounds_map = bounds(spec["segments"])
    lookback_lib = spec["segments"]["library"]
    lib_end = bounds_map["library"].stop - 1
    cal_start = bounds_map["calibration"].start
    step = spec["step"]
    n_jobs_feat = 1 if spec["smoke"] else 12
    packs: list[dict] = []
    for system in spec["systems"]:
        horizons = horizons_map[system]
        for seed in spec["seeds"]:
            traj = trajectories[(system, seed)]
            m, tau = emb_choice[(system, seed)]
            x = apply_obs_noise(traj.x, sigma, seed)
            var_lib = float(np.var(x[bounds_map["library"]]))
            geom = _geom(m, tau, spec["geometry"])
            geom.theiler = max(1, (m - 1) * tau)
            if scaling is not None:
                geom.scaling = scaling
                if scaling == "gz":
                    lib = x[bounds_map["library"]]
                    geom.train_mean = float(np.mean(lib))
                    geom.train_sd = float(np.std(lib))
            print(f"E1 {label} {system} seed {seed} losses+features", flush=True)
            if loss_by_h_map is not None:
                loss_by_h = {h: loss_by_h_map[(system, seed, h)] for h in horizons}
            elif forecaster_kind == "ar":
                arf = ARFixed(p=5)
                arf.fit(x[bounds_map["library"]])
                loss_by_h = squared_loss_series(x, arf, horizons, 0, bounds_map["evaluation"].stop, var_lib, ar=True, cal_start=cal_start)
            else:
                knn = fit_knn(x[bounds_map["library"]], m, tau, horizons)
                loss_by_h = squared_loss_series(x, knn, horizons, 0, bounds_map["evaluation"].stop, var_lib, cal_start=cal_start)
            cal_rqa = build_rqa_frame(x, bounds_map["calibration"], W, m, tau, geom, step, n_jobs=n_jobs_feat, lrqa_only=lrqa_only)
            ev_rqa = build_rqa_frame(x, bounds_map["evaluation"], W, m, tau, geom, step, n_jobs=n_jobs_feat, lrqa_only=lrqa_only)
            for h in horizons:
                loss_h = loss_by_h[h]
                df_cal = attach_horizon(cal_rqa, x, h, loss_h, m, tau, lookback_lib, lib_end, bounds_map["calibration"].stop)
                df_ev = attach_horizon(ev_rqa, x, h, loss_h, m, tau, lookback_lib, lib_end, bounds_map["evaluation"].stop)
                packs.append({"system": system, "seed": seed, "h": h, "traj": traj, "x": x, "m": m, "tau": tau, "loss_h": loss_h, "cal": df_cal, "ev": df_ev, "cal_rqa": cal_rqa, "ev_rqa": ev_rqa})
    return packs


def run_primary_and_outputs(
    spec: dict,
    out_dir: Path,
    fig_dir: Path,
    trajectories: dict[tuple[str, int], Trajectory],
    horizons_map: dict[str, list[int]],
    emb_choice: dict[tuple[str, int], tuple[int, int]],
    *,
    n_boot: int | None = None,
) -> dict:
    from condpred import report

    segs = spec["segments"]
    step = spec["step"]
    rng = np.random.default_rng(2026)
    n_boot = n_boot if n_boot is not None else smoke(500, 80)
    bounds_map = bounds(segs)
    lookback_lib = segs["library"]
    lib_end = bounds_map["library"].stop - 1

    primary_rows = []
    seed_rows: list[dict] = []
    packs = _build_packs(spec, trajectories, horizons_map, emb_choice, sigma=spec["sigma_obs_primary"], W=spec["W_primary"], label="primary")
    for system in spec["systems"]:
        for h in horizons_map[system]:
            sel = [p for p in packs if p["system"] == system and p["h"] == h]
            sel.sort(key=lambda p: p["seed"])
            cal_frames = [p["cal"] for p in sel]
            ev_frames = [p["ev"] for p in sel]
            group_list = [column_groups(p["m"]) for p in sel]
            res = primary_system_horizon(cal_frames, ev_frames, group_list, "y", h, step, rng, n_boot)
            primary_rows.append({"system": system, "h": h, **res})
            d = res["delta"]
            print(f"E1 primary {system} h={h} delta={d:.4f}" if np.isfinite(d) else f"E1 primary {system} h={h} delta=nan", flush=True)
            for p, cal, ev, groups in zip(sel, cal_frames, ev_frames, group_list):
                p0, _, _ = fit_predict(cal, ev, groups["B"], "y")
                p1, _, _ = fit_predict(cal, ev, groups["B"] + groups["LRQA"], "y")
                y = ev["y"].to_numpy(float)
                ok = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(y)
                d = spearman(p1[ok], y[ok]) - spearman(p0[ok], y[ok]) if ok.sum() >= 5 else np.nan
                seed_rows.append({"system": system, "horizon": h, "seed": p["seed"], "model": "M1b_vs_M0", "metric": "delta_spearman", "value": d, "ci_low": np.nan, "ci_high": np.nan})
    decision = build_decision_table(primary_rows, spec["smoke"], e2_pass=_read_e2_pass(out_dir))
    report.write_decision(decision, out_dir)
    prim_df = pd.DataFrame(seed_rows)
    report.save_table(prim_df, out_dir, "results_primary")
    report.save_table(pd.DataFrame(primary_rows), out_dir, "results_primary_pooled")
    secondary_all = _run_secondary_analyses(spec, packs, horizons_map, bounds_map, lookback_lib, lib_end, rng, n_boot, trajectories=trajectories, emb_choice=emb_choice)
    for k, df in secondary_all.items():
        if df is not None and len(df):
            report.save_table(df, out_dir, f"results_secondary_{k}")

    ftle_df = _feature_vs_ftle(packs, horizons_map)
    if len(ftle_df):
        report.save_table(ftle_df, out_dir, "feature_vs_ftle")

    _make_figures(primary_rows, horizons_map, spec, packs, fig_dir)
    return {"decision": decision, "primary": prim_df, "spec": spec, "packs": packs}


def run_full_pipeline(
    spec: dict,
    out_dir: Path,
    fig_dir: Path,
    *,
    cache_prefix: str = "E1",
    n_boot: int | None = None,
) -> dict:
    trajectories = load_trajectories(spec, cache_prefix)
    spec, emb_df, horizons_map, emb_choice = run_embedding_and_horizons(spec, trajectories, out_dir)
    from condpred import report

    report.save_table(emb_df, out_dir, "embedding_choice")
    return run_primary_and_outputs(
        spec, out_dir, fig_dir, trajectories, horizons_map, emb_choice, n_boot=n_boot
    )


def _summarize_per_seed(values, *, fisher=False) -> dict:
    """Mean per-seed effect with t CI / one-sample t p-value (exploratory secondaries)."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return {"value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": 1.0, "n_seed": 0}
    z = np.arctanh(np.clip(v, -0.999, 0.999)) if fisher else v
    point = float(np.mean(v))
    if len(v) < 3 or np.std(z, ddof=1) == 0:
        return {"value": point, "ci_low": np.nan, "ci_high": np.nan, "p_value": 1.0 if point == 0 else 0.5, "n_seed": len(v)}
    res = stats.ttest_1samp(z, 0.0)
    se = float(np.std(z, ddof=1) / np.sqrt(len(z)))
    crit = float(stats.t.ppf(0.975, len(z) - 1))
    if fisher:
        lo, hi = np.tanh(np.mean(z) - crit * se), np.tanh(np.mean(z) + crit * se)
    else:
        lo, hi = np.mean(z) - crit * se, np.mean(z) + crit * se
    return {"value": point, "ci_low": float(lo), "ci_high": float(hi), "p_value": float(res.pvalue), "n_seed": len(v)}


def _single_spearman(cal_frames, ev_frames, cols_list) -> list[float]:
    vals = []
    for cal, ev, cols in zip(cal_frames, ev_frames, cols_list):
        pred, _, _ = fit_predict(cal, ev, cols, "y")
        y = ev["y"].to_numpy(float)
        ok = np.isfinite(pred) & np.isfinite(y)
        vals.append(spearman(pred[ok], y[ok]) if ok.sum() >= 5 else np.nan)
    return vals


def _delta_rows_for(pack_list, spec, horizons_map, *, model_label, extra_fn, predictor="ridge", metric="delta_spearman"):
    rows = []
    for system in spec["systems"]:
        for h in horizons_map[system]:
            sel = sorted([p for p in pack_list if p["system"] == system and p["h"] == h], key=lambda p: p["seed"])
            if not sel:
                continue
            cals = [p["cal"] for p in sel]
            evs = [p["ev"] for p in sel]
            gl = [column_groups(p["m"]) for p in sel]
            if metric == "delta_spearman":
                extras = [extra_fn(g) for g in gl]
                deltas = []
                for cal, ev, g, extra in zip(cals, evs, gl, extras):
                    p0, _, _ = fit_predict(cal, ev, g["B"], "y", predictor)
                    p1, _, _ = fit_predict(cal, ev, g["B"] + list(extra), "y", predictor)
                    y = ev["y"].to_numpy(float)
                    ok = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(y)
                    deltas.append(spearman(p1[ok], y[ok]) - spearman(p0[ok], y[ok]) if ok.sum() >= 5 else np.nan)
                rows.append({"system": system, "horizon": h, "model": model_label, "metric": metric, **_summarize_per_seed(deltas)})
            elif metric == "delta_auc_top":
                from sklearn.metrics import roc_auc_score
                deltas = []
                for cal, ev, g in zip(cals, evs, gl):
                    p0, _, _ = fit_predict(cal, ev, g["B"], "y", predictor)
                    p1, _, _ = fit_predict(cal, ev, g["B"] + g["LRQA"], "y", predictor)
                    loss_ev = ev["L"].to_numpy(float)
                    thresh = float(np.nanquantile(cal["L"].to_numpy(float), 0.9))
                    event = (loss_ev > thresh).astype(int)
                    ok = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(loss_ev)
                    if ok.sum() < 10 or event[ok].sum() in (0, ok.sum()):
                        deltas.append(np.nan)
                        continue
                    deltas.append(float(roc_auc_score(event[ok], p1[ok]) - roc_auc_score(event[ok], p0[ok])))
                rows.append({"system": system, "horizon": h, "model": model_label, "metric": metric, **_summarize_per_seed(deltas)})
    return rows


def _holm_table(rows: list[dict], family: str) -> pd.DataFrame:
    pvals = [r.get("p_value", np.nan) for r in rows]
    adj = holm([p if np.isfinite(p) else 1.0 for p in pvals])
    df = pd.DataFrame(rows)
    df["family"] = family
    df["holm_p"] = adj
    df["exploratory"] = True
    return df


def _run_secondary_analyses(spec, packs, horizons_map, bounds_map, lookback_lib, lib_end, rng, n_boot, trajectories=None, emb_choice=None):
    """Exploratory secondaries 1-5, 7, 8 (6 is feature_vs_ftle); Holm within each table.

    Effects are per-seed deltas summarised across seeds (mean, t CI, t p-value).
    """
    out: dict[int, pd.DataFrame] = {}
    if not packs:
        return out
    loss_map = {(p["system"], p["seed"], p["h"]): p["loss_h"] for p in packs}
    # 1 — M1a / M1c at all horizons + single-family / single-B-member models
    rows = []
    rows += _delta_rows_for(packs, spec, horizons_map, model_label="M1a", extra_fn=lambda g: g["WRQA"])
    rows += _delta_rows_for(packs, spec, horizons_map, model_label="M1c", extra_fn=lambda g: g["WRQA"] + g["LRQA"])
    for system in spec["systems"]:
        for h in horizons_map[system]:
            sel = sorted([p for p in packs if p["system"] == system and p["h"] == h], key=lambda p: p["seed"])
            if not sel:
                continue
            cals = [p["cal"] for p in sel]
            evs = [p["ev"] for p in sel]
            gl = [column_groups(p["m"]) for p in sel]
            g0 = gl[0]
            singles = [("WRQA_only", g0["WRQA"]), ("LRQA_only", g0["LRQA"]), ("B_recent_loss", ["recent_loss"]), ("B_knn_spread", ["knn_spread"]), ("B_state_location", [c for c in g0["B"] if c.startswith("state_loc")]), ("B_level_features", ["level_cur", "level_mean", "level_vol_surprise"]), ("B_perm_entropy", ["perm_entropy"]), ("B_rolling_var", ["rolling_var"])]
            for label, cols in singles:
                vals = _single_spearman(cals, evs, [list(cols) for _ in gl])
                rows.append({"system": system, "horizon": h, "model": label, "metric": "spearman_single", **_summarize_per_seed(vals, fisher=True)})
    if rows:
        out[1] = _holm_table(rows, "secondary_1_models")
    # 2 — LGBM predictor, M1b
    rows = _delta_rows_for(packs, spec, horizons_map, model_label="M1b_lgbm", extra_fn=lambda g: g["LRQA"], predictor="lgbm")
    if rows:
        out[2] = _holm_table(rows, "secondary_2_lgbm")
    if trajectories is not None and emb_choice is not None:
        # 3 — observation noise 0.05*sd, KNN, W=500, M1b (LRQA-only frames suffice)
        noise_packs = _build_packs(spec, trajectories, horizons_map, emb_choice, sigma=0.05, W=spec["W_primary"], lrqa_only=True, label="secondary3_noise")
        rows = _delta_rows_for(noise_packs, spec, horizons_map, model_label="M1b_noise0.05", extra_fn=lambda g: g["LRQA"])
        if rows:
            out[3] = _holm_table(rows, "secondary_3_noise")
        del noise_packs
        # 4 — W=200, M1b (primary losses reused)
        w200_packs = _build_packs(spec, trajectories, horizons_map, emb_choice, sigma=spec["sigma_obs_primary"], W=spec["W_secondary"], lrqa_only=True, loss_by_h_map=loss_map, label="secondary4_W200")
        rows = _delta_rows_for(w200_packs, spec, horizons_map, model_label="M1b_W200", extra_fn=lambda g: g["LRQA"])
        if rows:
            out[4] = _holm_table(rows, "secondary_4_W200")
        del w200_packs
        # 8 — geometry arms gz / rank for LRQA (primary losses reused)
        rows8 = []
        for scaling in ("gz", "rank"):
            geom_packs = _build_packs(spec, trajectories, horizons_map, emb_choice, sigma=spec["sigma_obs_primary"], W=spec["W_primary"], scaling=scaling, lrqa_only=True, loss_by_h_map=loss_map, label=f"secondary8_{scaling}")
            rows8 += _delta_rows_for(geom_packs, spec, horizons_map, model_label=f"M1b_geom_{scaling}", extra_fn=lambda g: g["LRQA"])
            del geom_packs
        if rows8:
            out[8] = _holm_table(rows8, "secondary_8_geometry")
    # 5 — AR(5) forecaster, primary RQA frames, M1b
    ar_packs: list[dict] = []
    ar_loss_cache: dict[tuple[str, int], dict] = {}
    segs = spec["segments"]
    for p in packs:
        key = (p["system"], p["seed"])
        if key not in ar_loss_cache:
            x = p["x"]
            arf = ARFixed(p=5)
            arf.fit(x[bounds_map["library"]])
            var_lib = float(np.var(x[bounds_map["library"]]))
            ar_loss_cache[key] = squared_loss_series(x, arf, horizons_map[p["system"]], 0, bounds_map["evaluation"].stop, var_lib, ar=True, cal_start=bounds_map["calibration"].start)
        loss_h = ar_loss_cache[key][p["h"]]
        df_cal = attach_horizon(p["cal_rqa"], p["x"], p["h"], loss_h, p["m"], p["tau"], lookback_lib, lib_end, bounds_map["calibration"].stop)
        df_ev = attach_horizon(p["ev_rqa"], p["x"], p["h"], loss_h, p["m"], p["tau"], lookback_lib, lib_end, bounds_map["evaluation"].stop)
        ar_packs.append({**p, "cal": df_cal, "ev": df_ev, "loss_h": loss_h})
    rows = _delta_rows_for(ar_packs, spec, horizons_map, model_label="M1b_AR5", extra_fn=lambda g: g["LRQA"])
    if rows:
        out[5] = _holm_table(rows, "secondary_5_AR5")
    # 7 — event AUC (top decile, calibration threshold) M0 vs M1b, all horizons
    rows = _delta_rows_for(packs, spec, horizons_map, model_label="M1b", extra_fn=lambda g: g["LRQA"], metric="delta_auc_top")
    if rows:
        out[7] = _holm_table(rows, "secondary_7_auc")
    return out


def _partial_spearman(v, y, ctrl) -> float:
    ok = np.isfinite(v) & np.isfinite(y) & np.isfinite(ctrl)
    if ok.sum() < 10:
        return np.nan
    rv = stats.rankdata(v[ok])
    ry = stats.rankdata(y[ok])
    rc = stats.rankdata(ctrl[ok])
    X = np.column_stack([np.ones(ok.sum()), rc])
    bv, *_ = np.linalg.lstsq(X, rv, rcond=None)
    by, *_ = np.linalg.lstsq(X, ry, rcond=None)
    ev_ = rv - X @ bv
    ey_ = ry - X @ by
    if np.std(ev_) == 0 or np.std(ey_) == 0:
        return np.nan
    return float(np.corrcoef(ev_, ey_)[0, 1])


def _feature_vs_ftle(packs, horizons_map) -> pd.DataFrame:
    """Secondary 6: every WRQA/LRQA feature vs true FTLE, raw + partial on knn_spread."""
    rows = []
    ftle_cache: dict[tuple[str, int, int], np.ndarray] = {}
    for p in packs:
        if p["h"] != horizons_map[p["system"]][1]:
            continue
        ev = p["ev"]
        key = (p["system"], p["seed"], p["h"])
        if key not in ftle_cache:
            ftle_cache[key] = ftle_at_rows(p["system"], p["traj"], ev.index.to_numpy(), p["h"])
        ftle = ftle_cache[key]
        ctrl = ev["knn_spread"].to_numpy(float) if "knn_spread" in ev.columns else np.full(len(ev), np.nan)
        feats = [f"wrqa_{k}" for k in WRQA_KEYS] + [f"lrqa_{k}" for k in LRQA_KEYS]
        for feat in feats:
            if feat not in ev.columns:
                continue
            v = ev[feat].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(ftle)
            if ok.sum() < 10:
                rho, pval = np.nan, 1.0
            else:
                r = stats.spearmanr(v[ok], ftle[ok])
                rho, pval = float(r[0]), float(r[1])
            rows.append({"system": p["system"], "seed": p["seed"], "horizon": p["h"], "feature": feat, "spearman_ftle": rho, "p_value": pval, "partial_spearman_ftle_given_knn_spread": _partial_spearman(v, ftle, ctrl)})
    if not rows:
        return pd.DataFrame(rows)
    df = pd.DataFrame(rows)
    df["holm_p"] = holm([p if np.isfinite(p) else 1.0 for p in df["p_value"]])
    df["family"] = "secondary_6_feature_vs_ftle"
    df["exploratory"] = True
    return df


def _make_figures(primary_rows, horizons_map, spec, packs, fig_dir):
    from condpred import report

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for system in spec["systems"]:
        sub = [r for r in primary_rows if r["system"] == system]
        hs = [r["h"] for r in sub]
        deltas = np.asarray([r["delta"] for r in sub], float)
        los = np.asarray([r["ci_low"] for r in sub], float)
        his = np.asarray([r["ci_high"] for r in sub], float)
        lo_err = np.maximum(deltas - los, 0.0)
        hi_err = np.maximum(his - deltas, 0.0)
        lo_err[~np.isfinite(lo_err)] = 0.0
        hi_err[~np.isfinite(hi_err)] = 0.0
        ax.errorbar(hs, deltas, yerr=[lo_err, hi_err], label=system, capsize=3)
    ax.axhline(0.02, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("horizon h (samples)")
    ax.set_ylabel("ΔSpearman (M1b − M0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    report.save_fig(fig, fig_dir, "delta_spearman_by_horizon")
    plt.close(fig)

    if packs:
        p0 = packs[0]
        ev = p0["ev"]
        if len(ev) > 5:
            g = column_groups(p0["m"])
            p0pred, _, _ = fit_predict(p0["cal"], ev, g["B"], "y")
            p1pred, _, _ = fit_predict(p0["cal"], ev, g["B"] + g["LRQA"], "y")
            y = ev["y"].to_numpy(float)
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            for ax, pred, title in zip(axes, [p0pred, p1pred], ["M0", "M1b"]):
                ok = np.isfinite(pred) & np.isfinite(y)
                ax.scatter(pred[ok], y[ok], s=6, alpha=0.4)
                ax.set_title(title)
                ax.set_xlabel("predicted ln(L)")
                ax.set_ylabel("realised ln(L)")
            fig.suptitle(f"{p0['system']} seed {p0['seed']} h={p0['h']}")
            fig.tight_layout()
            report.save_fig(fig, fig_dir, "pred_vs_loss_scatter")
            plt.close(fig)
        if "lrqa_K2_local" in ev.columns:
            try:
                ftle_vals = ftle_at_rows(p0["system"], p0["traj"], ev.index.to_numpy(), p0["h"])
            except Exception:
                ftle_vals = np.full(len(ev), np.nan)
            fig, ax = plt.subplots(figsize=(4, 3))
            k2 = ev["lrqa_K2_local"].to_numpy(float)
            ok = np.isfinite(k2) & np.isfinite(ftle_vals)
            ax.scatter(k2[ok], ftle_vals[ok], s=6, alpha=0.4)
            ax.set_xlabel("K2_local")
            ax.set_ylabel("FTLE over [t, t+h]")
            fig.tight_layout()
            report.save_fig(fig, fig_dir, "ftle_vs_k2local")
            plt.close(fig)


def verify_spec_immutable(out_dir: Path, spec: dict, modules: list[str]):
    from condpred.report import write_spec

    bad = dict(spec)
    bad["step"] = spec["step"] + 1
    try:
        write_spec(bad, out_dir, modules=modules)
    except RuntimeError:
        return True
    raise AssertionError("spec should reject silent overwrite")
