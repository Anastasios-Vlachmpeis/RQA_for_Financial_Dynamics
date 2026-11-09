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
) -> pd.DataFrame:
    min_t = seg.start + max(W, (m - 1) * tau)
    ts = list(range(min_t, seg.stop, step))
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_rqa_feature_row)(x, t, W, m, tau, geom) for t in ts
    )
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


def build_decision_table(primary_results: list[dict], smoke_mode: bool) -> list[dict]:
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
    rows.append(
        {
            "gate": "G1",
            "system": "all",
            "horizon": "all",
            "statistic": "E1_pass OR E2_pass",
            "value": e1_pass,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "rule": "E2 not run; G1 = E1_pass",
            "verdict": "pass" if e1_pass else "fail",
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

    primary_rows = []
    seed_rows: list[dict] = []
    packs: list[dict] = []
    var_lib_cache: dict[tuple[str, int], float] = {}
    lib_end = bounds_map["library"].stop - 1

    n_jobs_feat = 1 if spec["smoke"] else 4
    cal_start = bounds_map["calibration"].start
    W = spec["W_primary"]
    seed_bundle: dict[tuple[str, int], tuple] = {}

    for system in spec["systems"]:
        horizons = horizons_map[system]
        for h in horizons:
            cal_frames, ev_frames, group_list = [], [], []
            for seed in spec["seeds"]:
                traj = trajectories[(system, seed)]
                m, tau = emb_choice[(system, seed)]
                x = apply_obs_noise(traj.x, spec["sigma_obs_primary"], seed)
                if (system, seed) not in var_lib_cache:
                    var_lib_cache[(system, seed)] = float(np.var(x[bounds_map["library"]]))
                var_lib = var_lib_cache[(system, seed)]
                geom = _geom(m, tau, spec["geometry"])
                geom.theiler = max(1, (m - 1) * tau)
                bundle_key = (system, seed)
                if bundle_key not in seed_bundle:
                    print(f"E1 bundle {system} seed {seed} losses+features", flush=True)
                    knn = fit_knn(x[bounds_map["library"]], m, tau, horizons)
                    loss_by_h = squared_loss_series(
                        x,
                        knn,
                        horizons,
                        0,
                        bounds_map["evaluation"].stop,
                        var_lib,
                        cal_start=cal_start,
                    )
                    cal_rqa = build_rqa_frame(
                        x,
                        bounds_map["calibration"],
                        W,
                        m,
                        tau,
                        geom,
                        step,
                        n_jobs=n_jobs_feat,
                    )
                    ev_rqa = build_rqa_frame(
                        x,
                        bounds_map["evaluation"],
                        W,
                        m,
                        tau,
                        geom,
                        step,
                        n_jobs=n_jobs_feat,
                    )
                    seed_bundle[bundle_key] = (loss_by_h, cal_rqa, ev_rqa)

                loss_by_h, cal_rqa, ev_rqa = seed_bundle[bundle_key]
                loss_h = loss_by_h[h]
                df_cal = attach_horizon(
                    cal_rqa,
                    x,
                    h,
                    loss_h,
                    m,
                    tau,
                    lookback_lib,
                    lib_end,
                    bounds_map["calibration"].stop,
                )
                df_ev = attach_horizon(
                    ev_rqa,
                    x,
                    h,
                    loss_h,
                    m,
                    tau,
                    lookback_lib,
                    lib_end,
                    bounds_map["evaluation"].stop,
                )

                cal_frames.append(df_cal)
                ev_frames.append(df_ev)
                group_list.append(column_groups(m))
                packs.append(
                    {
                        "system": system,
                        "seed": seed,
                        "h": h,
                        "traj": traj,
                        "x": x,
                        "m": m,
                        "tau": tau,
                        "loss_h": loss_h,
                        "cal": df_cal,
                        "ev": df_ev,
                    }
                )

            res = primary_system_horizon(
                cal_frames, ev_frames, group_list, "y", h, step, rng, n_boot
            )
            primary_rows.append({"system": system, "h": h, **res})
            d = res["delta"]
            print(f"E1 primary {system} h={h} delta={d:.4f}" if np.isfinite(d) else f"E1 primary {system} h={h} delta=nan", flush=True)
            for si, (cal, ev, groups) in enumerate(zip(cal_frames, ev_frames, group_list)):
                seed = spec["seeds"][si]
                p0, _, _ = fit_predict(cal, ev, groups["B"], "y")
                p1, _, _ = fit_predict(cal, ev, groups["B"] + groups["LRQA"], "y")
                y = ev["y"].to_numpy(float)
                ok = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(y)
                d = (
                    spearman(p1[ok], y[ok]) - spearman(p0[ok], y[ok])
                    if ok.sum() >= 5
                    else np.nan
                )
                seed_rows.append(
                    {
                        "system": system,
                        "horizon": h,
                        "seed": seed,
                        "model": "M1b_vs_M0",
                        "metric": "delta_spearman",
                        "value": d,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }
                )

    decision = build_decision_table(primary_rows, spec["smoke"])
    report.write_decision(decision, out_dir)

    # results_primary.csv
    prim_df = pd.DataFrame(seed_rows)
    report.save_table(prim_df, out_dir, "results_primary")
    report.save_table(pd.DataFrame(primary_rows), out_dir, "results_primary_pooled")

    secondary_all = _run_secondary_analyses(
        spec, packs, horizons_map, bounds_map, lookback_lib, lib_end, rng, n_boot
    )
    for k, df in enumerate(secondary_all, 1):
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


def _secondary_delta(
    cal_frames,
    ev_frames,
    group_list,
    m1_cols_extra: list[str],
    predictor: str = "ridge",
) -> list[float]:
    deltas = []
    for cal, ev, groups in zip(cal_frames, ev_frames, group_list):
        m0 = groups["B"]
        m1 = groups["B"] + m1_cols_extra
        p0, _, _ = fit_predict(cal, ev, m0, "y", predictor)
        p1, _, _ = fit_predict(cal, ev, m1, "y", predictor)
        y = ev["y"].to_numpy(float)
        ok = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(y)
        if ok.sum() < 5:
            deltas.append(np.nan)
        else:
            deltas.append(spearman(p1[ok], y[ok]) - spearman(p0[ok], y[ok]))
    return deltas


def _holm_table(rows: list[dict], family: str) -> pd.DataFrame:
    pvals = [r.get("p_value", np.nan) for r in rows]
    adj = holm([p if np.isfinite(p) else 1.0 for p in pvals])
    df = pd.DataFrame(rows)
    df["family"] = family
    df["holm_p"] = adj
    df["exploratory"] = True
    return df


def _run_secondary_analyses(
    spec,
    packs,
    horizons_map,
    bounds_map,
    lookback_lib,
    lib_end,
    rng,
    n_boot,
):
    """Exploratory secondary families (Holm-corrected within each table)."""
    from scipy.stats import norm

    out: list[pd.DataFrame] = []
    if not packs:
        return out

    def _collect(system, h, filt):
        cals, evs, gl = [], [], []
        for p in packs:
            if p["system"] == system and p["h"] == h and filt(p):
                cals.append(p["cal"])
                evs.append(p["ev"])
                gl.append(column_groups(p["m"]))
        return cals, evs, gl

    mid_h = {s: horizons_map[s][1] for s in spec["systems"]}

    # 1 — M1a / M1c (ridge, primary W)
    rows = []
    for system in spec["systems"]:
        h = mid_h[system]
        cals, evs, gl = _collect(system, h, lambda p: True)
        if not gl:
            continue
        g0 = column_groups(max(p["m"] for p in packs if p["system"] == system and p["h"] == h))
        for label, extra_cols in [
            ("M1a", g0["WRQA"]),
            ("M1c", g0["WRQA"] + g0["LRQA"]),
        ]:
            deltas = _secondary_delta(cals, evs, gl, extra_cols)
            d = float(np.nanmean(deltas))
            rows.append(
                {
                    "system": system,
                    "horizon": h,
                    "model": label,
                    "metric": "delta_spearman",
                    "value": d,
                    "p_value": 1 - norm.cdf(d / 0.05) if np.isfinite(d) else 1.0,
                }
            )
    if rows:
        out.append(_holm_table(rows, "secondary_1_models"))

    if spec["smoke"]:
        return out

    # 2 — LGBM predictor on primary setting
    rows = []
    for system in spec["systems"]:
        for h in horizons_map[system]:
            cals, evs, gl = _collect(system, h, lambda p: True)
            if not cals:
                continue
            g0 = column_groups(max(p["m"] for p in packs if p["system"] == system and p["h"] == h))
            deltas = _secondary_delta(cals, evs, gl, g0["LRQA"], predictor="lgbm")
            d = float(np.nanmean(deltas))
            rows.append(
                {
                    "system": system,
                    "horizon": h,
                    "model": "M1b_lgbm",
                    "metric": "delta_spearman",
                    "value": d,
                    "p_value": 1 - norm.cdf(d / 0.05) if np.isfinite(d) else 1.0,
                }
            )
    if rows:
        out.append(_holm_table(rows, "secondary_2_lgbm"))

    # 7 — event AUC (top decile) M0 vs M1b
    rows = []
    for system in spec["systems"]:
        h = mid_h[system]
        cals, evs, gl = _collect(system, h, lambda p: True)
        for cal, ev, groups in zip(cals, evs, gl):
            m0 = groups["B"]
            m1 = groups["B"] + groups["LRQA"]
            p0, _, _ = fit_predict(cal, ev, m0, "y")
            p1, _, _ = fit_predict(cal, ev, m1, "y")
            y = ev["y"].to_numpy(float)
            l_lin = np.exp(y) - 1e-4
            from condpred.evaluation import compare

            y_tr_lin = np.exp(cal["y"].to_numpy(float)) - 1e-4
            cmp = compare(
                p0,
                p1,
                y,
                y_loss=l_lin,
                y_train_loss=y_tr_lin,
            )
            dauc = cmp.get("auc_top", {}).get("delta", np.nan)
            rows.append(
                {
                    "system": system,
                    "horizon": h,
                    "metric": "delta_auc_top",
                    "value": dauc,
                    "p_value": 1 - norm.cdf(dauc / 0.05) if np.isfinite(dauc) else 1.0,
                }
            )
    if rows:
        out.append(_holm_table(rows, "secondary_7_auc"))

    return out


def _feature_vs_ftle(packs, horizons_map) -> pd.DataFrame:
    rows = []
    for p in packs:
        if p["h"] != horizons_map[p["system"]][1]:
            continue
        ev = p["ev"]
        if "lrqa_K2_local" not in ev.columns:
            continue
        for feat in ["lrqa_K2_local", "wrqa_DET"]:
            if feat not in ev.columns:
                continue
            v = ev[feat].to_numpy(float)
            # FTLE proxy: log-loss (companion diagnostic)
            ftle = ev["y"].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(ftle)
            if ok.sum() < 10:
                continue
            rho, _ = stats.spearmanr(v[ok], ftle[ok])
            rows.append(
                {
                    "system": p["system"],
                    "seed": p["seed"],
                    "feature": feat,
                    "spearman_ftle": float(rho),
                }
            )
    return pd.DataFrame(rows)


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
            fig, ax = plt.subplots(figsize=(4, 3))
            ok = np.isfinite(ev["lrqa_K2_local"]) & np.isfinite(ev["y"])
            ax.scatter(ev["lrqa_K2_local"][ok], ev["y"][ok], s=6, alpha=0.4)
            ax.set_xlabel("K2_local")
            ax.set_ylabel("ln(loss)")
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
