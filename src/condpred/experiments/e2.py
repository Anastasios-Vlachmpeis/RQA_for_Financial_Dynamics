"""E2 — regime-switching stochastic systems: loss prediction and detection lag."""
from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from condpred import systems
from condpred.cache import cached
from condpred.config import smoke
from condpred.evaluation import LGBMPredictor, RidgePredictor, spearman
from condpred.features import baselines
from condpred.features.local_rqa import local_rqa, local_rqa_frame
from condpred.features.window_rqa import window_rqa, window_rqa_frame
from condpred.forecasters import ARFixed, LogHAR
from condpred.geometry import Geometry
from condpred.losses import loss_transform, qlike, sq_norm
from condpred.selfcheck import leakage_test
from condpred.stats import block_bootstrap_idx, holm

E2_MODULES = [
    "condpred.experiments.e2",
    "condpred.features.window_rqa",
    "condpred.features.local_rqa",
    "condpred.features.baselines",
    "condpred.forecasters",
    "condpred.rp_batch",
]

WRQA_KEYS = ["DET", "L", "ENTR", "LAM", "TT", "Lmax", "Vmax", "DET3_DET2", "K2_lines", "DET_LAM"]
LRQA_KEYS = ["RR_last", "Lb_mean", "Lb_max", "DET_last", "Lf_nb", "K2_local", "LAM_last", "TT_last"]

LAG_BIN_EDGES = [0, 10, 25, 50, 100, 200, np.inf]
LAG_BIN_LABELS = ["[0,10)", "[10,25)", "[25,50)", "[50,100)", "[100,200)", ">=200"]
# Classifier families at a row: each family alone, plus B+WRQA and B+LRQA.
DETECT_MODELS = ("M0", "WRQA", "LRQA", "M1a", "M1b")
# ms_sv_intraday defaults, pinned so the S3 oracle uses the simulated transition.
S3_CALM = 0.008
S3_STRESS = 0.02
S3_PHI = 0.9
S3_LOGVOL_SD = 0.15
S3_P = ((0.99, 0.01), (0.03, 0.97))

_SYSTEM_META = {
    "S1": {
        "kind": "ms_ar1",
        "phis": (0.2, 0.95),
        "p_stay": (0.99, 0.99),
        "horizons": [1, 5, 10],
        "m": 2,
        "tau": 1,
        "forecaster": "ar",
        "loss": "sq_norm",
        "secondary": False,
    },
    "S2": {
        "kind": "ms_ar1",
        "phis": (0.2, 0.95),
        "p_stay": (0.995, 0.995),
        "horizons": [1, 5, 10],
        "m": 2,
        "tau": 1,
        "forecaster": "ar",
        "loss": "sq_norm",
        "secondary": False,
    },
    "S3": {
        "kind": "ms_sv",
        "horizons": [1, 5, 22],
        "m": 3,
        "tau": 1,
        "forecaster": "loghar",
        "loss": "qlike",
        "secondary": False,
    },
    "S4": {
        "kind": "ar_henon",
        "p_stay": 0.99,
        "horizons": [1, 5, 10],
        "m": 2,
        "tau": 1,
        "forecaster": "ar",
        "loss": "sq_norm",
        "secondary": True,
    },
}


@dataclass
class RegimeTrajectory:
    x: np.ndarray
    labels: np.ndarray
    truth_var: np.ndarray | None = None
    rv: np.ndarray | None = None
    component_ar: np.ndarray | None = None
    henon_state: np.ndarray | None = None
    henon_affine: tuple | None = None


def system_list(smoke_mode: bool) -> list[str]:
    full = ["S1", "S2", "S3", "S4"]
    return smoke(full, ["S1", "S4"])


def segment_lengths(smoke_mode: bool, system: str) -> dict[str, int]:
    if system == "S3":
        n = smoke(6_000, 500)
        return {"library": n, "calibration": n, "evaluation": n}
    n = smoke(20_000, 2_000)
    return {"library": n, "calibration": n, "evaluation": n}


def seed_list(smoke_mode: bool) -> list[int]:
    return list(range(smoke(10, 2)))


def w_primary_list(smoke_mode: bool) -> list[int]:
    return smoke([50, 100, 200], [100])


def build_initial_spec(smoke_mode: bool) -> dict:
    segs_s1 = segment_lengths(smoke_mode, "S1")
    spec = {
        "experiment": "E2",
        "version": 5,
        "changelog": [
            "feature backend numpy → torch; RPs identical by parity test",
            "record torch dtype float64; persist rp_fallback_arms",
            "M1 trained on same rows as M0 (NaN-imputed, not dropped); bootstrap stat aligned",
            "gz secondary arm: global z-score with eps_train fit on the library segment (keeps level)",
            "v5: secondary p-values two-sided from bootstrap SE (z=delta/std(boots, ddof=1), p=2*norm.sf(|z|)); percentile CI kept; primary gate rule unchanged; per-cell predictions saved as .npz",
        ],
        "backend": "torch",
        "torch_dtype": "float64",
        "smoke": smoke_mode,
        "systems": system_list(smoke_mode),
        "seeds": seed_list(smoke_mode),
        "segments": {"default_ms_ar": segs_s1, "S3": segment_lengths(smoke_mode, "S3")},
        "W_primary": 100,
        "W_grid": w_primary_list(smoke_mode),
        "geometry": {"scaling": "wz", "distance": "euclid", "threshold": "rr", "rr": 0.10},
        "geometry_secondary_gz": {
            "scaling": "gz",
            "distance": "euclid",
            "threshold": "eps_train",
            "rr": 0.10,
        },
        "auc_bootstrap": smoke(200, 60),
        "theiler_rule": "(m - 1) * tau",
        "step": smoke(2, 10),
        "embedding": {"S1_S2_S4": "m=2,tau=1", "S3": "m=3,tau=1"},
        "horizons": {
            "S1_S2_S4": [1, 5, 10],
            "S3": [1, 5, 22],
        },
        "forecaster": {"S1_S2_S4": "ARFixed(p=2)", "S3": "LogHAR(library fit)"},
        "predictor_primary": "RidgePredictor",
        "models": {"M0": "B", "M1a": "B+WRQA", "M1b": "B+LRQA"},
        "lag_bins": LAG_BIN_LABELS,
        "primary_test": {
            "system": "S1",
            "W": 100,
            "h": 5,
            "M0": "B",
            "M1": "M1b",
            "predictor": "Ridge",
            "delta_threshold": 0.02,
            "ci_rule": "95% block bootstrap CI lower > 0",
            "lag_bin": "[25,50)",
            "lag_rule": "M1b AUC not below M0 by >0.02 in lag bin",
            "block_steps": 50,
        },
        "bootstrap": {
            "n_boot": smoke(400, 60),
            "secondary_p_value": "two-sided normal approximation from bootstrap SE: z=delta/std(boots, ddof=1), p=2*norm.sf(|z|)",
            "ci": "percentile [2.5, 97.5] of bootstrap deltas",
            "primary_gate_rule": "unchanged: delta>0.02 and ci_low>0 (p-value not used)",
        },
        "gate_E2": "primary_test pass_rule",
        "gate_G1": "E1_pass OR E2_pass",
    }
    from condpred import rp_batch

    return rp_batch.persist_fallbacks_into_spec(spec)


def _total_n(segs: dict[str, int]) -> int:
    return segs["library"] + segs["calibration"] + segs["evaluation"]


def _rescale_to_moments(x: np.ndarray, target_mean: float, target_var: float) -> np.ndarray:
    x = np.asarray(x, float)
    mu, var = float(x.mean()), float(x.var())
    if var < 1e-12:
        return np.full_like(x, target_mean)
    z = (x - mu) / np.sqrt(var)
    return z * np.sqrt(target_var) + target_mean


def _markov_labels(n: int, rng: np.random.Generator, p_stay: float) -> np.ndarray:
    labels = np.zeros(n, dtype=int)
    state = 0
    for t in range(1, n):
        if rng.random() >= p_stay:
            state = 1 - state
        labels[t] = state
    return labels


def _generate_s4(n: int, rng: np.random.Generator, p_stay: float = 0.99, burn: int = 500):
    """Observation switches between AR(1) φ=0.9 and Hénon-x.

    Both clocks always advance. Hénon-x is affine-mapped onto the AR series'
    mean and variance, so a switch changes determinism rather than level.
    """
    phi = 0.9
    child = np.random.SeedSequence(int(rng.integers(0, 2**31 - 1)))
    rng_ar, rng_h, rng_reg = (np.random.default_rng(s) for s in child.spawn(3))
    ar = np.asarray(systems.ar1(phi, n, rng_ar, burn=burn), float)
    hx, hstate = systems.henon(n, rng_h, burn=burn)
    hx = np.asarray(hx, float)
    hstate = np.asarray(hstate, float)
    mu_a, var_a = float(ar.mean()), float(ar.var())
    hx_s = _rescale_to_moments(hx, mu_a, var_a if var_a > 0 else 1.0)
    mu_h, sd_h = float(hx.mean()), float(hx.std())
    sd_a = float(np.sqrt(var_a if var_a > 0 else 1.0))
    labels = _markov_labels(n, rng_reg, p_stay)
    x = np.where(labels == 0, ar, hx_s).astype(float)
    return x, labels, ar, hstate, (mu_h, sd_h if sd_h > 0 else 1.0, mu_a, sd_a)


def generate_trajectory(system: str, seed: int, segs: dict[str, int], *, cache_exp: str) -> RegimeTrajectory:
    meta = _SYSTEM_META[system]
    n = _total_n(segs)

    def _gen():
        rng = np.random.default_rng(seed + 17_002)
        kind = meta["kind"]
        if kind == "ms_ar1":
            x, lab = systems.ms_ar1(
                phis=meta["phis"],
                p_stay=meta["p_stay"],
                n=n,
                rng=rng,
                equal_variance=True,
            )
            return RegimeTrajectory(np.asarray(x, float), np.asarray(lab, int), None)
        if kind == "ms_sv":
            out = systems.ms_sv_intraday(
                n,
                rng,
                bars_per_day=smoke(78, 24),
                burn_days=smoke(50, 10),
                calm=S3_CALM,
                stress=S3_STRESS,
                P=np.array(S3_P, float),
                logvol_phi=S3_PHI,
                logvol_sd=S3_LOGVOL_SD,
            )
            daily_iv = np.asarray(out["daily_iv"], float)
            rv = np.sum(np.asarray(out["intraday"], float) ** 2, axis=1)
            log_rv = np.log(np.clip(rv, 1e-12, None))
            return RegimeTrajectory(log_rv, np.asarray(out["regime"], int), daily_iv, rv)
        if kind == "ar_henon":
            x, lab, ar, hstate, affine = _generate_s4(n, rng, p_stay=meta["p_stay"])
            return RegimeTrajectory(x, lab, None, None, ar, hstate, affine)
        raise ValueError(system)

    key = {"system": system, "seed": seed, "n": n, "segments": segs, "s3": "noisy_rv", "s4": "matched_clocks"}
    return cached(cache_exp, f"traj_{system}_{seed}", key, _gen, modules=E2_MODULES)


def bounds(segs: dict[str, int]) -> dict[str, slice]:
    lib = slice(0, segs["library"])
    cal = slice(segs["library"], segs["library"] + segs["calibration"])
    ev = slice(segs["library"] + segs["calibration"], _total_n(segs))
    return {"library": lib, "calibration": cal, "evaluation": ev}


def _geom(m: int, tau: int, spec_geom: dict) -> Geometry:
    return Geometry(
        scaling=spec_geom.get("scaling", "wz"),
        distance=spec_geom.get("distance", "euclid"),
        threshold=spec_geom.get("threshold", "rr"),
        rr=spec_geom.get("rr", 0.10),
        theiler=max(1, (m - 1) * tau),
        m=m,
        tau=tau,
    )


def time_since_switch(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, int)
    out = np.zeros(len(labels), dtype=int)
    last = 0
    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            last = t
        out[t] = t - last
    return out


def lag_bin_label(tss: int) -> str:
    for lo, hi, lab in zip(LAG_BIN_EDGES[:-1], LAG_BIN_EDGES[1:], LAG_BIN_LABELS):
        if lo <= tss < hi:
            return lab
    return LAG_BIN_LABELS[-1]


def fit_ar_library(x_lib: np.ndarray) -> ARFixed:
    ar = ARFixed(p=2)
    ar.fit(x_lib)
    return ar


def fit_loghar_library(v_lib: np.ndarray, horizons: list[int]) -> LogHAR:
    """Fit HAR on library RV levels and freeze the coefficients (no later refit)."""
    v_lib = np.asarray(v_lib, float)
    har = LogHAR(refit_every=max(len(v_lib), 1), window=len(v_lib), horizons=tuple(horizons))
    har.fit(v_lib)
    frozen: dict[int, tuple | None] = {}
    ln = np.asarray(har.v, float)
    raw = np.asarray(har._v_raw, float)
    for h in horizons:
        rows_x, rows_y = [], []
        for i in range(22, len(ln) - h):
            rows_x.append([ln[i], float(np.mean(ln[i - 4 : i + 1])), float(np.mean(ln[i - 21 : i + 1]))])
            rows_y.append(float(np.log(max(float(np.mean(raw[i + 1 : i + h + 1])), 1e-18))))
        if len(rows_y) < 10:
            frozen[int(h)] = None
            continue
        x_mat = np.asarray(rows_x, float)
        y_vec = np.asarray(rows_y, float)
        reg = LinearRegression().fit(x_mat, y_vec)
        resid = y_vec - reg.predict(x_mat)
        frozen[int(h)] = (np.asarray(reg.coef_, float).copy(), float(reg.intercept_), float(np.var(resid)))
    har._e2_frozen = frozen
    return har


def predict_loghar_frozen(har: LogHAR, rv_full: np.ndarray, t: int, h: int) -> float:
    """HAR forecast at t from the full RV path, using library-only coefficients."""
    frozen = getattr(har, "_e2_frozen", None)
    if not frozen or frozen.get(int(h)) is None or t < 22:
        return np.nan
    coef, intercept, s2 = frozen[int(h)]
    ln = np.log(np.clip(np.asarray(rv_full, float), 1e-12, None))
    if t >= len(ln):
        return np.nan
    xrow = np.array(
        [ln[t], float(np.mean(ln[t - 4 : t + 1])), float(np.mean(ln[t - 21 : t + 1]))],
        float,
    )
    return float(np.exp(float(intercept + np.dot(coef, xrow)) + 0.5 * s2))


def _recent_loss_series(loss: np.ndarray, h: int, halflife: float) -> np.ndarray:
    """EWMA of losses realised by each t. Same weights as ``baselines.recent_loss``."""
    loss = np.asarray(loss, float)
    n = len(loss)
    alpha = 0.5 ** (1.0 / float(halflife))
    wh = alpha**h
    out = np.full(n, np.nan)
    sw = 0.0
    ww = 0.0
    for t in range(h, n):
        sw *= alpha
        ww *= alpha
        v = loss[t - h]
        if np.isfinite(v):
            sw += float(v) * wh
            ww += wh
        if ww > 0.0:
            out[t] = sw / ww
    return out


def loss_series_ar(
    x: np.ndarray,
    ar: ARFixed,
    horizons: list[int],
    t_start: int,
    t_end: int,
    var_lib: float,
) -> dict[int, np.ndarray]:
    n = len(x)
    t_start = max(t_start, ar.p)
    out = {h: np.full(n, np.nan) for h in horizons}
    for h in horizons:
        for t in range(t_start, t_end - h):
            yhat = ar.predict(t, h, series=x)
            if np.isfinite(yhat):
                out[h][t] = sq_norm(x[t + h], yhat, var_lib)
    return out


def loss_series_qlike(
    truth_iv: np.ndarray,
    rv_proxy: np.ndarray,
    har: LogHAR,
    horizons: list[int],
    t_start: int,
    t_end: int,
) -> dict[int, np.ndarray]:
    """QLIKE of a frozen HAR. The forecast sees RV; the target is integrated variance."""
    n = len(rv_proxy)
    truth_iv = np.asarray(truth_iv, float)
    out = {h: np.full(n, np.nan) for h in horizons}
    for h in horizons:
        for t in range(max(t_start, 22), t_end - h):
            yhat = predict_loghar_frozen(har, rv_proxy, t, h)
            y = float(np.mean(truth_iv[t + 1 : t + h + 1]))
            if not np.isfinite(yhat) or yhat <= 0 or not np.isfinite(y) or y <= 0:
                continue
            out[h][t] = float(qlike(np.array([y]), np.array([yhat]))[0])
    return out


def _baseline_row(
    x: np.ndarray,
    t: int,
    W: int,
    m: int,
    tau: int,
    h: int,
    loss_by_h: dict[int, np.ndarray],
    lookback_lib: int,
    lib_end: int,
    *,
    is_s3: bool,
    shared: dict | None = None,
) -> dict:
    row: dict = {}
    loss_h = loss_by_h[h]
    for hl in (10.0, 50.0):
        series_key = ("recent_series", id(loss_h), h, hl)
        if shared is not None and series_key in shared:
            row[f"recent_loss_hl{int(hl)}"] = float(shared[series_key][t])
        else:
            row[f"recent_loss_hl{int(hl)}"] = baselines.recent_loss(loss_h, h, hl, t)
    row["rolling_acf1"] = baselines.rolling_acf1(x, W, t)
    row["rolling_var"] = baselines.rolling_var(x, W, t)
    row["perm_entropy"] = baselines.perm_entropy(x, W, t, order=4)
    knn_key = ("knn", id(x), t, m, tau, h, lookback_lib, lib_end)
    if shared is not None and knn_key in shared:
        row["knn_spread"] = shared[knn_key]
    else:
        row["knn_spread"] = baselines.knn_spread(x, t, m, tau, 10, h, lookback_lib, lib_end=lib_end)
    if is_s3:
        row.update(baselines.level_features(x, t, W))
    return row


def _knn_chunk(x, ts_chunk, m, tau, h, lookback, lib_end):
    return [
        baselines.knn_spread(x, int(t), m, tau, 10, h, lookback, lib_end=lib_end) for t in ts_chunk
    ]


def _ensure_shared_baselines(
    shared: dict,
    x: np.ndarray,
    ts: list[int],
    m: int,
    tau: int,
    h: int,
    loss_h: np.ndarray,
    lookback_lib: int,
    lib_end: int,
    n_jobs: int,
) -> None:
    """Recent-loss and kNN do not depend on W or geometry. Compute each pair once."""
    for hl in (10.0, 50.0):
        key = ("recent_series", id(loss_h), h, hl)
        if key not in shared:
            shared[key] = _recent_loss_series(loss_h, h, hl)
    missing = [
        t
        for t in ts
        if ("knn", id(x), t, m, tau, h, lookback_lib, lib_end) not in shared
    ]
    if not missing:
        return
    if n_jobs == 1 or len(missing) < 32:
        vals = [
            baselines.knn_spread(x, t, m, tau, 10, h, lookback_lib, lib_end=lib_end) for t in missing
        ]
    else:
        chunks = [c for c in np.array_split(np.asarray(missing, int), min(n_jobs, len(missing))) if len(c)]
        parts = Parallel(n_jobs=n_jobs)(
            delayed(_knn_chunk)(x, chunk, m, tau, h, lookback_lib, lib_end) for chunk in chunks
        )
        vals = [v for part in parts for v in part]
    for t, v in zip(missing, vals):
        shared[("knn", id(x), t, m, tau, h, lookback_lib, lib_end)] = v


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
    *,
    is_s3: bool,
) -> dict:
    row = _baseline_row(x, t, W, m, tau, h, loss_by_h, lookback_lib, lib_end, is_s3=is_s3)
    wr = window_rqa(x, t, W, m, tau, geom)
    for k in WRQA_KEYS:
        row[f"wrqa_{k}"] = wr.get(k, np.nan)
    lr = local_rqa(x, t, W, m, tau, geom)
    for k in LRQA_KEYS:
        row[f"lrqa_{k}"] = lr.get(k, np.nan)
    return row


def _geom_key(geom: Geometry) -> tuple:
    return (
        geom.scaling,
        geom.distance,
        geom.threshold,
        float(geom.rr),
        float(geom.beta),
        int(geom.theiler_band()),
        int(geom.m),
        int(geom.tau),
        None if geom.train_mean is None else round(float(geom.train_mean), 12),
        None if geom.train_sd is None else round(float(geom.train_sd), 12),
        None if geom.eps_train is None else round(float(geom.eps_train), 12),
    )


def _feature_ends(seg: slice, W: int, m: int, tau: int, step: int) -> list[int]:
    min_t = seg.start + max(W, (m - 1) * tau)
    return list(range(min_t, seg.stop, step))


def _rolling_var_all(x: np.ndarray, W: int) -> np.ndarray:
    """Population variance of each window, same divisor as ``np.var``."""
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan)
    if W < 1 or n < W:
        return out
    c1 = np.cumsum(x)
    c2 = np.cumsum(x * x)
    t = np.arange(W - 1, n)
    left = t - W
    s = c1[t] - np.where(left >= 0, c1[left], 0.0)
    ss = c2[t] - np.where(left >= 0, c2[left], 0.0)
    mean = s / W
    out[t] = ss / W - mean * mean
    return out


def _rolling_acf1_all(x: np.ndarray, W: int) -> np.ndarray:
    """Lag-1 correlation of each window. The /n terms cancel, matching ``np.corrcoef``."""
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan)
    if W < 3 or n < W:
        return out
    a = x[:-1]
    b = x[1:]
    pair_n = W - 1
    c1 = np.cumsum(a)
    c2 = np.cumsum(b)
    c11 = np.cumsum(a * a)
    c22 = np.cumsum(b * b)
    c12 = np.cumsum(a * b)
    t = np.arange(W - 1, n)
    # Pairs in the window ending at t are indices [t-W+1, t).
    right = t - 1
    left = t - W
    sa = c1[right] - np.where(left >= 0, c1[left], 0.0)
    sb = c2[right] - np.where(left >= 0, c2[left], 0.0)
    saa = c11[right] - np.where(left >= 0, c11[left], 0.0)
    sbb = c22[right] - np.where(left >= 0, c22[left], 0.0)
    sab = c12[right] - np.where(left >= 0, c12[left], 0.0)
    ma = sa / pair_n
    mb = sb / pair_n
    var_a = saa / pair_n - ma * ma
    var_b = sbb / pair_n - mb * mb
    cov = sab / pair_n - ma * mb
    denom = np.sqrt(np.maximum(var_a, 0.0) * np.maximum(var_b, 0.0))
    corr = np.full(len(t), np.nan)
    ok = denom > 1e-15
    corr[ok] = cov[ok] / denom[ok]
    out[t] = corr
    return out


def _perm_entropy_all(x: np.ndarray, W: int, order: int = 4) -> np.ndarray:
    """Rolling permutation entropy, normalised by log(order!), matching ``baselines.perm_entropy``."""
    from math import factorial

    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan)
    if W < order or n < W:
        return out
    n_pat_pos = n - order + 1
    mat = np.stack([x[k : k + n_pat_pos] for k in range(order)], axis=1)
    ranks = np.argsort(mat, axis=1)
    code = np.zeros(n_pat_pos, dtype=np.int32)
    for k in range(order):
        code = code * order + ranks[:, k].astype(np.int32)
    n_bins = order**order
    counts = np.zeros((n_pat_pos, n_bins), dtype=np.int16)
    counts[np.arange(n_pat_pos), code] = 1
    csum = np.cumsum(counts, axis=0, dtype=np.int32)
    t = np.arange(W - 1, n)
    n_pat = W - order + 1
    right = t - order + 1
    left = right - n_pat
    left_c = np.clip(left, 0, None)
    hist = csum[right] - np.where(left[:, None] >= 0, csum[left_c], 0)
    total = hist.sum(axis=1, keepdims=True).astype(float)
    p = np.divide(hist, total, out=np.zeros_like(hist, dtype=float), where=total > 0)
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    ent = -(p * logp).sum(axis=1) / np.log(factorial(order))
    out[t] = ent
    return out


def _level_all(x: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    s = pd.Series(x)
    mean = s.rolling(W, min_periods=W).mean().to_numpy()
    med = s.rolling(W, min_periods=W).median().to_numpy()
    surprise = np.divide(x, med, out=np.full(len(x), np.nan), where=np.isfinite(med) & (med != 0))
    return x.copy(), mean, surprise


def _vector_baselines(
    x: np.ndarray,
    ts: list[int],
    W: int,
    m: int,
    tau: int,
    h: int,
    loss_h: np.ndarray,
    lookback_lib: int,
    lib_end: int,
    *,
    is_s3: bool,
    shared: dict,
    n_jobs: int,
) -> pd.DataFrame:
    """Baselines that ignore geometry, computed once per series and window length."""
    idx = np.asarray(ts, dtype=int)
    key_w = (id(x), int(W))
    if ("acf", key_w) not in shared:
        shared[("acf", key_w)] = _rolling_acf1_all(x, W)
        shared[("var", key_w)] = _rolling_var_all(x, W)
        shared[("perm", key_w)] = _perm_entropy_all(x, W)
        if is_s3:
            shared[("level", key_w)] = _level_all(x, W)
    for hl in (10.0, 50.0):
        rkey = ("recent_series", id(loss_h), h, hl)
        if rkey not in shared:
            shared[rkey] = _recent_loss_series(loss_h, h, hl)
    knn_key = ("knn_arr", id(x), m, tau, h, lookback_lib, lib_end)
    arr = shared.get(knn_key)
    if arr is None:
        arr = np.full(len(x), np.nan)
        shared[knn_key] = arr
    missing_ts = [t for t in ts if ("knn", id(x), t, m, tau, h, lookback_lib, lib_end) not in shared]
    if missing_ts:
        _ensure_shared_baselines(shared, x, missing_ts, m, tau, h, loss_h, lookback_lib, lib_end, n_jobs)
        for t in missing_ts:
            arr[t] = shared[("knn", id(x), t, m, tau, h, lookback_lib, lib_end)]
    frame = pd.DataFrame(
        {
            "recent_loss_hl10": shared[("recent_series", id(loss_h), h, 10.0)][idx],
            "recent_loss_hl50": shared[("recent_series", id(loss_h), h, 50.0)][idx],
            "rolling_acf1": shared[("acf", key_w)][idx],
            "rolling_var": shared[("var", key_w)][idx],
            "perm_entropy": shared[("perm", key_w)][idx],
            "knn_spread": shared[knn_key][idx],
        },
        index=idx,
    )
    if is_s3:
        cur, mean, surprise = shared[("level", key_w)]
        frame["level_cur"] = cur[idx]
        frame["level_mean"] = mean[idx]
        frame["level_vol_surprise"] = surprise[idx]
    frame.index.name = "t"
    return frame


def _cached_rqa_frames(
    frame_cache: dict,
    key: tuple,
    x: np.ndarray,
    ts: list[int],
    W: int,
    m: int,
    tau: int,
    geom: Geometry,
    step: int,
    n_jobs: int,
    backend: str,
):
    if key in frame_cache:
        return frame_cache[key]
    from condpred.features.window_rqa import wrqa_lrqa_frames

    wrdf, lrdf = wrqa_lrqa_frames(
        x,
        W,
        m,
        tau,
        geom,
        ends=np.asarray(ts, dtype=int),
        step=step,
        n_jobs=n_jobs,
        backend=backend,
    )
    frame_cache[key] = (wrdf, lrdf)
    return wrdf, lrdf


def build_rows(
    x: np.ndarray,
    labels: np.ndarray,
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
    *,
    is_s3: bool,
    n_jobs: int = 1,
    backend: str = "numpy",
    frame_cache: dict | None = None,
    frame_key: tuple | None = None,
    shared: dict | None = None,
    tss: np.ndarray | None = None,
    rqa_frames: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """One feature table. RQA frames are reused when ``frame_key`` is already cached.

    Horizons share that frame: only recent loss, kNN spread, and the target change with h.
    """
    loss_by_h = {h: loss_h}
    ts = _feature_ends(seg, W, m, tau, step)
    if tss is None:
        tss = time_since_switch(labels)
    if frame_cache is None:
        frame_cache = {}
    if frame_key is None:
        frame_key = (id(x), seg.start, seg.stop, W, _geom_key(geom), step, backend)
    if rqa_frames is None:
        wrdf, lrdf = _cached_rqa_frames(
            frame_cache, frame_key, x, ts, W, m, tau, geom, step, n_jobs, backend
        )
    else:
        wrdf, lrdf = rqa_frames
    base = _vector_baselines(
        x,
        ts,
        W,
        m,
        tau,
        h,
        loss_h,
        lookback_lib,
        lib_end,
        is_s3=is_s3,
        shared=shared if shared is not None else {},
        n_jobs=n_jobs,
    )
    wr_al = wrdf.reindex(ts).reindex(columns=WRQA_KEYS).add_prefix("wrqa_")
    lr_al = lrdf.reindex(ts).reindex(columns=LRQA_KEYS).add_prefix("lrqa_")
    df = pd.concat([base, wr_al, lr_al], axis=1)
    df.index = pd.Index(ts, name="t")
    df["L"] = loss_h[ts]
    df["y"] = loss_transform(df["L"].to_numpy(float))
    df["regime"] = labels[ts].astype(int)
    df["tss"] = tss[ts].astype(int)
    df["lag_bin"] = [lag_bin_label(int(v)) for v in df["tss"].to_numpy()]
    return df


def column_groups(is_s3: bool) -> dict[str, list[str]]:
    b_cols = [
        "recent_loss_hl10",
        "recent_loss_hl50",
        "rolling_acf1",
        "rolling_var",
        "perm_entropy",
        "knn_spread",
    ]
    if is_s3:
        b_cols += ["level_cur", "level_mean", "level_vol_surprise"]
    w_cols = [f"wrqa_{k}" for k in WRQA_KEYS]
    l_cols = [f"lrqa_{k}" for k in LRQA_KEYS]
    return {"B": b_cols, "WRQA": w_cols, "LRQA": l_cols}


def purge_rows(df: pd.DataFrame, h: int, seg_stop: int) -> pd.DataFrame:
    return df.loc[df.index + h <= seg_stop - 1]


def _as_float_matrix(X) -> np.ndarray:
    X = np.asarray(X, float)
    return np.where(np.isfinite(X), X, np.nan)


def _median_indicator(X_cal: np.ndarray, X_ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training-median fill plus a missingness column, the same augmentation Ridge uses."""
    med = np.nanmedian(X_cal, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def _aug(X):
        miss = np.isnan(X)
        filled = np.where(miss, med, X)
        return np.hstack([filled, miss.astype(float)])

    return _aug(X_cal), _aug(X_ev)


def fit_predict(
    df_cal: pd.DataFrame,
    df_ev: pd.DataFrame,
    cols: list[str],
    y_col: str,
    predictor: str = "ridge",
) -> tuple[np.ndarray, np.ndarray, object]:
    """Fit on every calibration row with a finite target. Missing features are imputed, not dropped."""
    X_cal = _as_float_matrix(df_cal[cols].to_numpy(float))
    y_cal = _as_float_matrix(df_cal[y_col].to_numpy(float))
    X_ev = _as_float_matrix(df_ev[cols].to_numpy(float))
    ok = np.isfinite(y_cal)
    if int(ok.sum()) < 10:
        nan = np.full(len(df_ev), np.nan)
        return nan, nan, None
    try:
        if predictor == "lgbm":
            import lightgbm as lgb

            Xc, Xe = _median_indicator(X_cal[ok], X_ev)
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                num_leaves=7,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=1.0,
                random_state=0,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(Xc, y_cal[ok])
            pred = np.asarray(model.predict(Xe), float)
        else:
            model = RidgePredictor()
            model.fit(X_cal[ok], y_cal[ok])
            pred = np.asarray(model.predict(X_ev), float)
    except (ValueError, np.linalg.LinAlgError):
        nan = np.full(len(df_ev), np.nan)
        return nan, y_cal[ok], None
    return pred, y_cal[ok], model


def fit_predict_logistic(
    df_cal: pd.DataFrame,
    df_ev: pd.DataFrame,
    cols: list[str],
) -> np.ndarray:
    X_cal = _as_float_matrix(df_cal[cols].to_numpy(float))
    y_cal = df_cal["regime"].to_numpy(int)
    if len(y_cal) < 20 or len(np.unique(y_cal)) < 2:
        return np.full(len(df_ev), np.nan)
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    try:
        pipe.fit(X_cal, y_cal)
        prob = pipe.predict_proba(_as_float_matrix(df_ev[cols].to_numpy(float)))[:, 1]
    except (ValueError, np.linalg.LinAlgError):
        return np.full(len(df_ev), np.nan)
    return np.asarray(prob, float)


def _pooled_delta(p0: np.ndarray, p1: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(y)
    if int(ok.sum()) < 5:
        return float("nan")
    return float(spearman(p1[ok], y[ok]) - spearman(p0[ok], y[ok]))


def pooled_delta_ci(
    p0_all: list[np.ndarray],
    p1_all: list[np.ndarray],
    y_all: list[np.ndarray],
    *,
    step: int,
    block_steps: int,
    rng: np.random.Generator,
    n_boot: int,
    n_jobs: int = 1,
) -> dict:
    """Pooled Spearman difference, with a block bootstrap of that same pooled statistic.

    Bootstrap index sets are drawn serially from ``rng`` (stream unchanged);
    only the Spearman evaluations are threaded when ``n_jobs`` > 1.
    """
    if not p0_all:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "delta_pass": False}
    p0_cat = np.concatenate(p0_all)
    p1_cat = np.concatenate(p1_all)
    y_cat = np.concatenate(y_all)
    delta = _pooled_delta(p0_cat, p1_cat, y_cat)
    block = max(1, int(block_steps / step))
    offsets = np.cumsum([0] + [len(y) for y in y_all[:-1]])
    boot_idx: list[np.ndarray] = []
    for _ in range(n_boot):
        parts = [
            block_bootstrap_idx(len(ys), block, rng) + int(off)
            for ys, off in zip(y_all, offsets)
            if len(ys) >= block
        ]
        if parts:
            boot_idx.append(np.concatenate(parts))
    if n_jobs and int(n_jobs) > 1 and len(boot_idx) > 1:
        boots = Parallel(n_jobs=int(n_jobs), backend="threading")(
            delayed(lambda idx: _pooled_delta(p0_cat[idx], p1_cat[idx], y_cat[idx]))(idx)
            for idx in boot_idx
        )
    else:
        boots = [_pooled_delta(p0_cat[idx], p1_cat[idx], y_cat[idx]) for idx in boot_idx]
    boots_a = np.asarray(boots, float)
    boots_a = boots_a[np.isfinite(boots_a)]
    if len(boots_a):
        lo, hi = np.percentile(boots_a, [2.5, 97.5])
        # Two-sided p from the bootstrap SE (percentile CI above is kept as-is).
        # The primary gate does not use this p-value (delta_pass below is unchanged).
        if len(boots_a) > 1 and np.isfinite(delta):
            se = float(np.std(boots_a, ddof=1))
            if se > 0:
                p_value = float(2.0 * stats.norm.sf(abs(float(delta) / se)))
            else:
                p_value = 0.0 if float(delta) != 0.0 else 1.0
        else:
            p_value = np.nan
    else:
        lo = hi = p_value = np.nan
    return {
        "delta": float(delta) if np.isfinite(delta) else np.nan,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "delta_pass": bool(np.isfinite(delta) and delta > 0.02 and np.isfinite(lo) and lo > 0),
    }


def primary_delta_spearman(
    cal_frames: list[pd.DataFrame],
    ev_frames: list[pd.DataFrame],
    group_list: list[dict[str, list[str]]],
    *,
    h: int,
    step: int,
    block_steps: int,
    rng: np.random.Generator,
    n_boot: int,
) -> dict:
    p0_all, p1_all, y_all = [], [], []
    for cal, ev, groups in zip(cal_frames, ev_frames, group_list):
        cols_m0 = groups["B"]
        cols_m1 = groups["B"] + groups["LRQA"]
        p0, _, _ = fit_predict(cal, ev, cols_m0, "y")
        p1, _, _ = fit_predict(cal, ev, cols_m1, "y")
        p0_all.append(p0)
        p1_all.append(p1)
        y_all.append(ev["y"].to_numpy(float))
    return pooled_delta_ci(
        p0_all, p1_all, y_all, step=step, block_steps=block_steps, rng=rng, n_boot=n_boot
    )


def auc_binned(
    ev: pd.DataFrame,
    prob: np.ndarray,
    *,
    bin_label: str,
    rng: np.random.Generator,
    n_boot: int,
    block: int,
) -> dict:
    sub = ev.loc[ev["lag_bin"] == bin_label].copy()
    if len(sub) < 10:
        return {"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": len(sub)}
    y = sub["regime"].to_numpy(int)
    p = prob[ev.index.get_indexer(sub.index)]
    ok = np.isfinite(p)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return {"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": int(ok.sum())}
    auc = float(roc_auc_score(y[ok], p[ok]))
    boots = []
    n = int(ok.sum())
    for _ in range(n_boot):
        idx = block_bootstrap_idx(n, block, rng)
        yb, pb = y[ok][idx], p[ok][idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boots.append(float(roc_auc_score(yb, pb)))
        except ValueError:
            pass
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"auc": auc, "ci_low": float(lo), "ci_high": float(hi), "n": int(ok.sum())}


def detection_delays(
    ev: pd.DataFrame,
    prob: np.ndarray,
    labels: np.ndarray,
    *,
    step: int,
    hold: int = 5,
) -> tuple[list[float], int, int]:
    """Delays in time steps. The 5-step hold is in steps, and the search stops at the next switch."""
    t_idx = np.asarray(ev.index.to_numpy(), int)
    ev_labels = np.asarray(labels, int)[t_idx]
    switches = [i for i in range(1, len(ev_labels)) if ev_labels[i] != ev_labels[i - 1]]
    hold_rows = max(1, int(np.ceil(hold / max(int(step), 1))))
    delays: list[float] = []
    missed = 0
    for s, i in enumerate(switches):
        nxt = switches[s + 1] if s + 1 < len(switches) else len(ev_labels)
        target = int(ev_labels[i])
        found = None
        for j in range(i, nxt - hold_rows + 1):
            window = prob[j : j + hold_rows]
            if len(window) < hold_rows or not np.all(np.isfinite(window)):
                continue
            if target == 1 and np.all(window >= 0.5):
                found = float(t_idx[j] - t_idx[i])
                break
            if target == 0 and np.all(window <= 0.5):
                found = float(t_idx[j] - t_idx[i])
                break
        if found is None:
            missed += 1
        else:
            delays.append(found)
    return delays, missed, len(switches)


def _load_e1_pass() -> bool | None:
    """Full E1 decision only. A smoke file must not decide G1."""
    repo = Path(__file__).resolve().parents[3]
    path = repo / "notebooks" / "tables" / "condpred" / "E1" / "decision.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    gate = df.loc[df["gate"] == "E1", "verdict"]
    if not len(gate):
        return None
    return bool(gate.iloc[0] == "pass")


def build_decision_table(
    primary: dict,
    lag_compare: dict,
    *,
    smoke_mode: bool,
    e1_pass: bool | None,
) -> list[dict]:
    rows = [
        {
            "gate": "E2_delta",
            "system": "S1",
            "horizon": 5,
            "statistic": "delta_spearman_M1b_M0",
            "value": primary.get("delta"),
            "ci_low": primary.get("ci_low"),
            "ci_high": primary.get("ci_high"),
            "rule": "delta>0.02 and ci_low>0",
            "verdict": "pass" if primary.get("delta_pass") else "fail",
        },
        {
            "gate": "E2_lag",
            "system": "S1",
            "horizon": 5,
            "statistic": "lag_auc_M1b_vs_M0",
            "value": lag_compare.get("delta_auc"),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "rule": "M1b AUC >= M0 AUC - 0.02 in [25,50)",
            "verdict": "pass" if lag_compare.get("lag_pass") else "fail",
        },
    ]
    e2_pass = bool(primary.get("delta_pass") and lag_compare.get("lag_pass"))
    rows.append(
        {
            "gate": "E2",
            "system": "all",
            "horizon": "all",
            "statistic": "primary_test",
            "value": e2_pass,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "rule": "delta and lag rules",
            "verdict": "pass" if e2_pass else "fail",
        }
    )
    if e2_pass:
        g1_value, g1_verdict = True, "pass"
    elif e1_pass is None:
        g1_value, g1_verdict = np.nan, "pending"
    else:
        g1_value = bool(e1_pass)
        g1_verdict = "pass" if g1_value else "fail"
    rows.append(
        {
            "gate": "G1",
            "system": "all",
            "horizon": "all",
            "statistic": "E1_pass OR E2_pass",
            "value": g1_value,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "rule": "E1_pass OR E2_pass; pending only if E2 fails and full E1 decision is absent",
            "verdict": g1_verdict,
        }
    )
    if smoke_mode:
        for r in rows:
            r["smoke"] = True
    return rows


def run_leakage_e2(x: np.ndarray, rng: np.random.Generator) -> list[tuple[str, bool]]:
    W, m, tau, h = 80, 2, 1, 5
    geom = _geom(m, tau, {"scaling": "wz", "distance": "euclid", "threshold": "rr", "rr": 0.1})
    n = len(x)
    t_list = list(rng.integers(W + 20, n - 50, size=6))
    loss = np.abs(rng.standard_normal(n)) ** 2

    def pipe(arr, t):
        return _one_feature_row(
            arr, t, W, m, tau, h, geom, {h: loss}, 200, lib_end=len(arr) // 2, is_s3=False
        )

    checks = [("E2_features", leakage_test(pipe, x, t_list, rng))]

    feats = pipe(x, t_list[0])
    forbidden = {"regime", "label", "tss", "lag_bin"}
    label_ok = forbidden.isdisjoint(feats.keys())
    for t in t_list[:3]:
        row = pipe(x, t)
        for k, v in row.items():
            if k in forbidden:
                label_ok = False
            if isinstance(v, (int, float)) and not np.isfinite(v):
                continue
    checks.append(("E2_label_not_feature", label_ok))

    ar = ARFixed(p=2)
    ar.fit(x[: max(t_list)])
    checks.append(
        (
            "E2_AR",
            leakage_test(lambda arr, t: ar.predict(t, h=1, series=arr), x, t_list, rng),
        )
    )
    return checks


def verify_spec_immutable(out_dir: Path, spec: dict, modules: list[str]):
    from condpred.report import write_spec

    bad = dict(spec)
    bad["step"] = spec["step"] + 1
    try:
        write_spec(bad, out_dir, modules=modules)
    except RuntimeError:
        return True
    raise AssertionError("spec should reject silent overwrite")


def segments_for_system(spec: dict, system: str) -> dict[str, int]:
    if system == "S3":
        return spec["segments"]["S3"]
    return spec["segments"]["default_ms_ar"]


def load_trajectories(spec: dict, cache_prefix: str) -> dict[tuple[str, int], RegimeTrajectory]:
    out: dict[tuple[str, int], RegimeTrajectory] = {}
    for system in spec["systems"]:
        segs = segments_for_system(spec, system)
        for seed in spec["seeds"]:
            out[(system, seed)] = generate_trajectory(system, seed, segs, cache_exp=cache_prefix)
    return out


def secondary_horizons(system: str) -> list[int]:
    if system == "S3":
        return smoke([1, 22], [22])
    return smoke([1, 10], [1])


def secondary_w_values() -> list[int]:
    return smoke([50, 200], [50])


def _model_cols(groups: dict[str, list[str]], model: str) -> list[str]:
    if model in ("M0", "B"):
        return groups["B"]
    if model == "WRQA":
        return groups["WRQA"]
    if model == "LRQA":
        return groups["LRQA"]
    if model == "M1a":
        return groups["B"] + groups["WRQA"]
    if model == "M1b":
        return groups["B"] + groups["LRQA"]
    if model == "M1c":
        return groups["B"] + groups["WRQA"] + groups["LRQA"]
    raise KeyError(model)


def horizons_for_run(system: str, spec: dict) -> list[int]:
    """Primary horizon plus every horizon the plan scores for this system."""
    meta = _SYSTEM_META[system]
    primary_h = int(spec["primary_test"]["h"])
    if spec["smoke"]:
        hs = [primary_h] if primary_h in meta["horizons"] else []
        for h in secondary_horizons(system):
            if h in meta["horizons"] and h not in hs:
                hs.append(h)
        return hs
    return list(meta["horizons"])


def w_values_for_run(spec: dict) -> list[int]:
    ws = [int(spec["W_primary"])]
    for w in secondary_w_values():
        if int(w) not in ws:
            ws.append(int(w))
    return ws


def _fit_library_eps(x_lib: np.ndarray, geom: Geometry, W: int, step: int) -> float:
    """rr-quantile of distances on library windows, same linear rule as ``np.quantile``."""
    from condpred import rp_batch

    x_lib = np.asarray(x_lib, float)
    m, tau = int(geom.m), int(geom.tau)
    n_state = W - (m - 1) * tau
    ends = np.arange(W - 1 + (m - 1) * tau, len(x_lib), max(int(step), 1), dtype=int)
    if len(ends) == 0 or n_state < 2:
        return 0.0
    torch = rp_batch._torch()
    dev = rp_batch.device()
    dt = torch.float64
    th = geom.theiler_band()
    tri = np.triu_indices(n_state, th)
    parts: list[np.ndarray] = []
    budget = rp_batch.batch_budget(n_state, dt, torch_device=str(dev))
    if rp_batch.banded_ok(geom):
        band, band_n, band_m, band_tau = rp_batch._banded_series_states(
            x_lib, W, geom, dt, dev=dev, positive_raw=None
        )
        for i in range(0, len(ends), budget):
            eb = ends[i : i + budget]
            D, _ = rp_batch._banded_distances_from_band(band, eb, W, band_m, band_tau, band_n)
            parts.append(D[:, tri[0], tri[1]].detach().cpu().numpy().ravel())
            del D
        del band
    else:
        for i in range(0, len(ends), budget):
            eb = ends[i : i + budget]
            wins = np.stack([x_lib[int(e) - W + 1 : int(e) + 1] for e in eb])
            D, _ = rp_batch._direct_distances(wins, geom, None, dt, dev=dev)
            parts.append(D[:, tri[0], tri[1]].detach().cpu().numpy().ravel())
            del D
    pool = np.concatenate(parts) if parts else np.array([0.0])
    return float(np.quantile(pool, geom.rr, method="linear"))


def _prepare_geom(template: Geometry, lib: np.ndarray, *, W: int, step: int) -> Geometry:
    g = replace(template)
    if g.scaling == "gz":
        sd = float(np.std(lib))
        g = replace(g, train_mean=float(np.mean(lib)), train_sd=sd if sd > 0 else 1.0)
    if g.threshold == "eps_train":
        g = replace(g, eps_train=_fit_library_eps(lib, g, W, step))
    return g


def _fit_all_losses(
    traj: RegimeTrajectory, system: str, meta: dict, bounds_map: dict[str, slice]
) -> tuple[dict[int, np.ndarray], dict]:
    """Fixed-model losses on the whole path, plus the objects the oracle reuses."""
    horizons = list(meta["horizons"])
    x = traj.x
    lib = bounds_map["library"]
    if meta["forecaster"] == "ar":
        ar = fit_ar_library(x[lib])
        var_lib = float(np.var(x[lib]))
        losses = loss_series_ar(x, ar, horizons, 0, bounds_map["evaluation"].stop, var_lib)
        return losses, {"kind": "ar", "ar": ar, "var_lib": var_lib, "horizons": horizons}
    if traj.rv is None or traj.truth_var is None:
        raise RuntimeError(f"{system} needs noisy RV and integrated variance")
    har = fit_loghar_library(traj.rv[lib], horizons)
    losses = loss_series_qlike(
        traj.truth_var, traj.rv, har, horizons, 0, bounds_map["evaluation"].stop
    )
    return losses, {"kind": "har", "har": har, "horizons": horizons}


def _ms_mean_var(x0: np.ndarray, s0: np.ndarray, h: int, phis, p_stay, sigmas) -> tuple[np.ndarray, np.ndarray]:
    """Conditional mean and variance of an equal-variance MS-AR, h steps ahead."""
    phis = np.asarray(phis, float)
    sigmas = np.asarray(sigmas, float)
    p_stay = np.asarray(p_stay, float)
    trans = np.array(
        [[p_stay[0], 1 - p_stay[0]], [1 - p_stay[1], p_stay[1]]],
        float,
    )
    n = len(x0)
    s0 = np.asarray(s0, int)
    p = np.zeros((n, 2))
    m = np.zeros((n, 2))
    q = np.zeros((n, 2))
    idx = np.arange(n)
    p[idx, s0] = 1.0
    m[idx, s0] = x0
    q[idx, s0] = x0**2
    for _ in range(h):
        p2 = np.zeros_like(p)
        m2 = np.zeros_like(m)
        q2 = np.zeros_like(q)
        for s in range(2):
            for sp in range(2):
                tr = trans[s, sp]
                p2[:, sp] += tr * p[:, s]
                m2[:, sp] += phis[sp] * tr * m[:, s]
                q2[:, sp] += (phis[sp] ** 2) * tr * q[:, s] + (sigmas[sp] ** 2) * tr * p[:, s]
        p, m, q = p2, m2, q2
    mean = m.sum(axis=1)
    var = np.maximum(q.sum(axis=1) - mean**2, 0.0)
    return mean, var


def _oracle_rows_one(
    traj: RegimeTrajectory,
    system: str,
    meta: dict,
    seed: int,
    loss_by_h: dict[int, np.ndarray],
    ctx: dict,
    bounds_map: dict[str, slice],
    rng: np.random.Generator,
) -> list[dict]:
    """Ceiling: Spearman of regime-aware expected loss vs the fixed model's realised loss."""
    ev = bounds_map["evaluation"]
    rows = []
    if ctx["kind"] == "ar" and system in ("S1", "S2"):
        phis = np.asarray(meta["phis"], float)
        sigmas = np.sqrt(np.maximum(1.0 - phis**2, 0.0))
        definition = "MS-AR conditional expected sq_norm of AR(2), given regime and x_t"
        for h in ctx["horizons"]:
            t_idx = np.arange(ev.start, ev.stop - h)
            t_idx = t_idx[np.isfinite(loss_by_h[h][t_idx])]
            if not len(t_idx):
                continue
            mean, var = _ms_mean_var(traj.x[t_idx], traj.labels[t_idx], h, phis, meta["p_stay"], sigmas)
            yhat = np.array([ctx["ar"].predict(int(t), h, series=traj.x) for t in t_idx])
            expect = (var + (mean - yhat) ** 2) / ctx["var_lib"]
            realised = loss_by_h[h][t_idx]
            rho, n = _spear_pair(expect, realised)
            rows.append(_oracle_record(system, seed, h, n, rho, expect, realised, definition))
        return rows
    if ctx["kind"] == "ar" and system == "S4":
        definition = (
            "mixture of persistent AR(1) and deterministic Hénon-x, "
            "Markov regime at t+h; expected sq_norm of AR(2)"
        )
        phi = 0.9
        q = float(meta["p_stay"])
        mu_h, sd_h, mu_a, sd_a = traj.henon_affine
        for h in ctx["horizons"]:
            t_idx = np.arange(ev.start, min(ev.stop, len(traj.x)) - h)
            t_idx = t_idx[np.isfinite(loss_by_h[h][t_idx])]
            if not len(t_idx):
                continue
            p_same = 0.5 + 0.5 * (2 * q - 1) ** h
            s0 = traj.labels[t_idx]
            p_henon = np.where(s0 == 0, 1.0 - p_same, p_same)
            ar_mean = (phi**h) * traj.component_ar[t_idx]
            var_ar = 1.0 - phi ** (2 * h)
            raw = traj.henon_state[t_idx + h, 0]
            henon_obs = (raw - mu_h) / sd_h * sd_a + mu_a
            mean = (1.0 - p_henon) * ar_mean + p_henon * henon_obs
            ex2 = (1.0 - p_henon) * (var_ar + ar_mean**2) + p_henon * henon_obs**2
            var = np.maximum(ex2 - mean**2, 0.0)
            yhat = np.array([ctx["ar"].predict(int(t), h, series=traj.x) for t in t_idx])
            expect = (var + (mean - yhat) ** 2) / ctx["var_lib"]
            realised = loss_by_h[h][t_idx]
            rho, n = _spear_pair(expect, realised)
            rows.append(_oracle_record(system, seed, h, n, rho, expect, realised, definition))
        return rows
    if ctx["kind"] == "har":
        definition = "MC expected QLIKE of frozen HAR given latent regime and log-volatility (256 paths)"
        for h in ctx["horizons"]:
            t_idx = np.arange(max(ev.start, 22), ev.stop - h)
            t_idx = t_idx[np.isfinite(loss_by_h[h][t_idx])]
            if not len(t_idx):
                continue
            expect = _s3_expected_qlike(traj, ctx["har"], h, t_idx, rng)
            realised = loss_by_h[h][t_idx]
            rho, n = _spear_pair(expect, realised)
            rows.append(_oracle_record(system, seed, h, n, rho, expect, realised, definition))
        return rows
    return rows


def _spear_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 5:
        return float("nan"), int(ok.sum())
    return float(spearman(a[ok], b[ok])), int(ok.sum())


def _oracle_record(system, seed, h, n, rho, expect, realised, definition) -> dict:
    return {
        "system": system,
        "seed": int(seed),
        "h": int(h),
        "n": int(n),
        "spearman_oracle": rho,
        "mean_oracle_loss": float(np.nanmean(expect)) if len(expect) else np.nan,
        "mean_realized_loss": float(np.nanmean(realised)) if len(realised) else np.nan,
        "definition": definition,
    }


def _s3_expected_qlike(traj: RegimeTrajectory, har: LogHAR, h: int, t_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_paths = 256
    n = len(t_idx)
    level0 = np.where(traj.labels == 0, S3_CALM, S3_STRESS)
    log_h0 = np.log(np.sqrt(np.clip(traj.truth_var, 1e-18, None)) / level0)
    f = np.array([predict_loghar_frozen(har, traj.rv, int(t), h) for t in t_idx])
    log_h = np.repeat(log_h0[t_idx][None, :], n_paths, axis=0)
    reg = np.repeat(traj.labels[t_idx][None, :], n_paths, axis=0)
    stay = np.array([S3_P[0][0], S3_P[1][1]])
    levels = np.array([S3_CALM, S3_STRESS])
    acc = np.zeros((n_paths, n))
    for _k in range(h):
        draw = rng.random((n_paths, n))
        reg = np.where(draw < stay[reg], reg, 1 - reg)
        log_h = S3_PHI * log_h + S3_LOGVOL_SD * rng.standard_normal((n_paths, n))
        acc += (levels[reg] * np.exp(log_h)) ** 2
    y = np.clip(acc / h, 1e-12, None)
    f2 = np.clip(f, 1e-12, None)[None, :]
    return (y / f2 - np.log(y / f2) - 1.0).mean(axis=0)


def _delta_spearman_pair(
    cal_frames: list[pd.DataFrame],
    ev_frames: list[pd.DataFrame],
    group_list: list[dict],
    *,
    m1_model: str,
    predictor: str = "ridge",
) -> float:
    deltas = []
    for cal, ev, groups in zip(cal_frames, ev_frames, group_list):
        m0 = _model_cols(groups, "M0")
        m1 = _model_cols(groups, m1_model)
        p0, _, _ = fit_predict(cal, ev, m0, "y", predictor)
        p1, _, _ = fit_predict(cal, ev, m1, "y", predictor)
        y = ev["y"].to_numpy(float)
        ok = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(y)
        if ok.sum() < 5:
            deltas.append(np.nan)
        else:
            deltas.append(spearman(p1[ok], y[ok]) - spearman(p0[ok], y[ok]))
    return float(np.nanmean(deltas))


def _holm_secondary_table(rows: list[dict], family: str) -> pd.DataFrame:
    pvals = [r.get("p_value", np.nan) for r in rows]
    adj = holm([p if np.isfinite(p) else 1.0 for p in pvals])
    df = pd.DataFrame(rows)
    df["family"] = family
    df["holm_p"] = adj
    df["exploratory"] = True
    return df

def _save_cell_predictions(out_dir, name: str, seeds, p0_list, p1_list, y_list) -> Path:
    """Save one cell's per-seed predictions so statistics can be recomputed without refitting."""
    pred_dir = Path(out_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "seeds": np.asarray([int(s) for s in seeds], dtype=int),
        "p0": np.concatenate([np.asarray(a, float) for a in p0_list]) if p0_list else np.empty(0),
        "p1": np.concatenate([np.asarray(a, float) for a in p1_list]) if p1_list else np.empty(0),
        "y": np.concatenate([np.asarray(a, float) for a in y_list]) if y_list else np.empty(0),
    }
    for i, (p0, p1, y) in enumerate(zip(p0_list, p1_list, y_list)):
        arrays[f"p0_{i}"] = np.asarray(p0, float)
        arrays[f"p1_{i}"] = np.asarray(p1, float)
        arrays[f"y_{i}"] = np.asarray(y, float)
    path = pred_dir / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return path



def _geom_payload(g: Geometry) -> dict:
    return {
        "scaling": g.scaling,
        "distance": g.distance,
        "threshold": g.threshold,
        "rr": g.rr,
        "beta": g.beta,
        "theiler": g.theiler,
        "m": g.m,
        "tau": g.tau,
        "train_mean": g.train_mean,
        "train_sd": g.train_sd,
        "eps_train": g.eps_train,
    }


def _rqa_disk_key(system: str, seed: int, seg_name: str, W: int, step: int, geom: Geometry, backend: str) -> dict:
    geom_bits = []
    for v in _geom_key(geom):
        geom_bits.append(None if v is None else v)
    return {
        "artifact": "wrqa_lrqa",
        "system": system,
        "seed": int(seed),
        "segment": seg_name,
        "W": int(W),
        "step": int(step),
        "geom": geom_bits,
        "backend": backend,
        "torch_dtype": "float64",
    }


def _rqa_cache_name(system: str, seed: int, seg_name: str, W: int) -> str:
    return f"rqa_{system}_s{int(seed)}_{seg_name}_W{int(W)}"


def _store_rqa_frame(
    cache_exp: str,
    system: str,
    seed: int,
    seg_name: str,
    x: np.ndarray,
    seg: slice,
    W: int,
    m: int,
    tau: int,
    geom: Geometry,
    step: int,
    n_jobs: int,
    backend: str,
) -> None:
    """GPU recurrence for one segment, written to disk. Workers only read it."""
    ends = _feature_ends(seg, W, m, tau, step)
    key = _rqa_disk_key(system, seed, seg_name, W, step, geom, backend)
    name = _rqa_cache_name(system, seed, seg_name, W)

    def _fn():
        from condpred.features.window_rqa import wrqa_lrqa_frames

        wr, lr = wrqa_lrqa_frames(
            x,
            W,
            m,
            tau,
            geom,
            ends=np.asarray(ends, dtype=int),
            step=step,
            n_jobs=n_jobs,
            backend=backend,
        )
        return {"wr": wr, "lr": lr}

    cached(cache_exp, name, key, _fn, modules=E2_MODULES)


def _load_rqa_frame(
    cache_exp: str,
    system: str,
    seed: int,
    seg_name: str,
    W: int,
    step: int,
    geom: Geometry,
    backend: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = _rqa_disk_key(system, seed, seg_name, W, step, geom, backend)
    name = _rqa_cache_name(system, seed, seg_name, W)

    def _missing():
        raise RuntimeError(f"RQA cache missing for {name}")

    blob = cached(cache_exp, name, key, _missing, modules=E2_MODULES)
    return blob["wr"], blob["lr"]


def _fit_from_disk(task: dict) -> dict:
    """One seed, one horizon, one geometry. Loads the trajectory and the RQA frames from disk."""
    spec = task["spec"]
    system = task["system"]
    seed = int(task["seed"])
    W = int(task["W"])
    h = int(task["h"])
    segs = segments_for_system(spec, system)
    bounds_map = bounds(segs)
    meta = _SYSTEM_META[system]
    traj = generate_trajectory(system, seed, segs, cache_exp=task["cache_exp"])
    loss_h = _fit_all_losses(traj, system, meta, bounds_map)[0][h]
    geom = Geometry(**task["geom"])
    m, tau = meta["m"], meta["tau"]
    is_s3 = system == "S3"
    lookback = min(1000, segs["library"])
    lib_end = bounds_map["library"].stop - 1
    step = int(spec["step"])
    backend = spec.get("backend", "torch")
    labels = traj.labels
    tss = time_since_switch(labels)
    shared: dict = {}

    def _build(seg_name: str) -> pd.DataFrame:
        frames = _load_rqa_frame(
            task["cache_exp"], system, seed, seg_name, W, step, geom, backend
        )
        df = build_rows(
            traj.x,
            labels,
            bounds_map[seg_name],
            h,
            loss_h,
            W,
            m,
            tau,
            geom,
            step,
            lookback,
            lib_end,
            is_s3=is_s3,
            n_jobs=1,
            backend=backend,
            shared=shared,
            tss=tss,
            rqa_frames=frames,
        )
        return purge_rows(df, h, bounds_map[seg_name].stop - 1)

    cal = _build("calibration")
    ev = _build("evaluation")
    groups = column_groups(is_s3)
    y = ev["y"].to_numpy(float)
    needed = (
        ("M0", "ridge"),
        ("M1a", "ridge"),
        ("M1b", "ridge"),
        ("M1c", "ridge"),
        ("M0", "lgbm"),
        ("M1b", "lgbm"),
    )
    preds = {}
    for model, predictor in needed:
        preds[f"{model}|{predictor}"] = fit_predict(
            cal, ev, _model_cols(groups, model), "y", predictor
        )[0]
    loss_rows = []
    for model in ("M0", "M1a", "M1b"):
        p = preds[f"{model}|ridge"]
        ok = np.isfinite(p) & np.isfinite(y)
        rho = spearman(p[ok], y[ok]) if int(ok.sum()) >= 5 else np.nan
        loss_rows.append(
            {
                "system": system,
                "seed": seed,
                "W": W,
                "h": h,
                "model": model,
                "predictor": "ridge",
                "geometry": task["geometry"],
                "analysis": task["analysis"],
                "spearman": rho,
                "n": int(ok.sum()),
            }
        )
    lag_rows, delay_rows = [], []
    detect_probs: dict[str, np.ndarray] = {}
    if task["do_detection"]:
        rng = np.random.default_rng(int(task["auc_seed"]))
        block = max(1, int(task["block_steps"]) // int(task["step"]))
        for model_name in DETECT_MODELS:
            prob = fit_predict_logistic(cal, ev, _model_cols(groups, model_name))
            detect_probs[model_name] = prob
            for bin_label in LAG_BIN_LABELS:
                stats_bin = auc_binned(
                    ev, prob, bin_label=bin_label, rng=rng, n_boot=int(task["n_boot_auc"]), block=block
                )
                lag_rows.append(
                    {
                        "system": system,
                        "seed": seed,
                        "W": W,
                        "h": h,
                        "model": model_name,
                        "geometry": task["geometry"],
                        "bin": bin_label,
                        **stats_bin,
                    }
                )
            delays, missed, n_sw = detection_delays(ev, prob, labels, step=int(task["step"]))
            delay_rows.append(
                {
                    "system": system,
                    "seed": seed,
                    "W": W,
                    "h": h,
                    "model": model_name,
                    "geometry": task["geometry"],
                    "median_delay": float(np.median(delays)) if delays else np.nan,
                    "q25": float(np.percentile(delays, 25)) if delays else np.nan,
                    "q75": float(np.percentile(delays, 75)) if delays else np.nan,
                    "n_switches": int(n_sw),
                    "n_detected": len(delays),
                    "n_undetected": int(missed),
                }
            )
    return {
        "loss_rows": loss_rows,
        "lag_rows": lag_rows,
        "delay_rows": delay_rows,
        "y": y,
        "preds": preds,
        "seed": seed,
        "system": system,
        "W": W,
        "h": h,
        "geometry": task["geometry"],
        "detect_probs": detect_probs,
    }




def _secondary_pairs(system: str, W: int, h: int, geom_name: str, spec: dict) -> list[tuple[str, str]]:
    """Ridge WRQA / WRQA+LRQA, LGBM on the primary family, and ridge M1b off the pre-registered cell."""
    pairs = [("M1a", "ridge"), ("M1c", "ridge"), ("M1b", "lgbm")]
    primary_cell = (
        system == spec["primary_test"]["system"]
        and int(W) == int(spec["W_primary"])
        and int(h) == int(spec["primary_test"]["h"])
        and geom_name == spec["geometry"]["scaling"]
    )
    if not primary_cell:
        pairs.append(("M1b", "ridge"))
    return pairs


def _analysis_label(W: int, h: int, geom_name: str, spec: dict) -> str:
    if int(W) == int(spec["W_primary"]) and geom_name == spec["geometry"]["scaling"]:
        if int(h) == int(spec["primary_test"]["h"]):
            return "primary"
        return "primary_extra_horizon"
    return "secondary"


def _cpu_load() -> float:
    try:
        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).LoadPercentage",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        nums = [float(x) for x in r.stdout.split() if x.strip().replace(".", "", 1).isdigit()]
        return float(np.mean(nums)) if nums else float("nan")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return float("nan")


def _boot_seed(*parts) -> int:
    import zlib

    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def run_primary_and_outputs(
    spec: dict,
    out_dir: Path,
    fig_dir: Path,
    trajectories: dict[tuple[str, int], RegimeTrajectory],
    *,
    n_boot: int | None = None,
    cache_prefix: str = "E2",
) -> dict:
    from condpred import report

    rng = np.random.default_rng(2026)
    oracle_rng = np.random.default_rng(2026 + 17)
    n_boot = n_boot if n_boot is not None else smoke(400, 60)
    step = spec["step"]
    h_primary = spec["primary_test"]["h"]
    block_steps = spec["primary_test"]["block_steps"]
    lag_bin = spec["primary_test"]["lag_bin"]
    W_primary = int(spec["W_primary"])

    loss_rows: list[dict] = []
    lag_rows: list[dict] = []
    delay_rows: list[dict] = []
    packs: list[dict] = []
    secondary_rows: list[dict] = []
    oracle_rows: list[dict] = []
    loss_cache: dict = {}
    shared: dict = {}

    for system in spec["systems"]:
        meta = _SYSTEM_META[system]
        segs = segments_for_system(spec, system)
        bmap = bounds(segs)
        for seed in spec["seeds"]:
            traj = trajectories[(system, seed)]
            losses, ctx = _fit_all_losses(traj, system, meta, bmap)
            loss_cache[(system, seed)] = {"loss": losses, "ctx": ctx}
            oracle_rows.extend(
                _oracle_rows_one(traj, system, meta, seed, losses, ctx, bmap, oracle_rng)
            )

    backend = spec.get("backend", "torch")
    rqa_jobs = 1 if spec["smoke"] else 12
    fit_tasks: list[dict] = []
    auc_boot = int(spec.get("auc_bootstrap", 200))
    for system in spec["systems"]:
        meta = _SYSTEM_META[system]
        segs = segments_for_system(spec, system)
        bmap = bounds(segs)
        geoms = (
            ("wz", _geom(meta["m"], meta["tau"], spec["geometry"])),
            ("gz", _geom(meta["m"], meta["tau"], spec["geometry_secondary_gz"])),
        )
        for W in w_values_for_run(spec):
            for gname, geom in geoms:
                for seed in spec["seeds"]:
                    traj = trajectories[(system, seed)]
                    geom_s = _prepare_geom(geom, traj.x[bmap["library"]], W=int(W), step=step)
                    for seg_name in ("calibration", "evaluation"):
                        _store_rqa_frame(
                            cache_prefix,
                            system,
                            int(seed),
                            seg_name,
                            traj.x,
                            bmap[seg_name],
                            int(W),
                            meta["m"],
                            meta["tau"],
                            geom_s,
                            step,
                            rqa_jobs,
                            backend,
                        )
                    for h in horizons_for_run(system, spec):
                        fit_tasks.append(
                            {
                                "spec": spec,
                                "cache_exp": cache_prefix,
                                "system": system,
                                "seed": int(seed),
                                "W": int(W),
                                "h": int(h),
                                "geometry": gname,
                                "geom": _geom_payload(geom_s),
                                "analysis": _analysis_label(W, h, gname, spec),
                                "do_detection": gname == "wz"
                                and int(W) == W_primary
                                and int(h) == int(h_primary),
                                "step": int(step),
                                "block_steps": int(block_steps),
                                "n_boot_auc": auc_boot,
                                "auc_seed": 50_000 + int(seed) * 1009 + int(W) + int(h),
                            }
                        )

    fit_jobs = min(12, max(1, len(fit_tasks)))
    cpu_samples: list[float] = []
    stop_sample = threading.Event()

    def _sample_cpu():
        while not stop_sample.wait(2.0):
            cpu_samples.append(_cpu_load())

    sampler = threading.Thread(target=_sample_cpu, daemon=True)
    sampler.start()
    print(f"E2 fits: {len(fit_tasks)} tasks, n_jobs={fit_jobs}", flush=True)
    if fit_jobs == 1:
        scored = [_fit_from_disk(task) for task in fit_tasks]
    else:
        scored = Parallel(n_jobs=fit_jobs)(delayed(_fit_from_disk)(task) for task in fit_tasks)
    stop_sample.set()
    sampler.join(timeout=5)
    finite_cpu = [v for v in cpu_samples if np.isfinite(v)]
    if finite_cpu:
        print(
            f"E2 fit CPU load %: n={len(finite_cpu)} mean={float(np.mean(finite_cpu)):.1f} max={float(np.max(finite_cpu)):.1f}",
            flush=True,
        )

    by_cell: dict[tuple, list] = {}
    for task, score in zip(fit_tasks, scored):
        loss_rows.extend(score["loss_rows"])
        lag_rows.extend(score["lag_rows"])
        delay_rows.extend(score["delay_rows"])
        by_cell.setdefault((task["system"], task["W"], task["h"], task["geometry"]), []).append(score)
        if task["do_detection"] and task["system"] == "S1":
            packs.append(
                {
                    "system": "S1",
                    "seed": score["seed"],
                    "W": task["W"],
                    "h": task["h"],
                    "p0": score["preds"]["M0|ridge"],
                    "p1": score["preds"]["M1b|ridge"],
                    "y": score["y"],
                    "detect_probs": score.get("detect_probs", {}),
                }
            )

    ci_jobs = []
    for (system, W, h, gname), pred_packs in by_cell.items():
        pred_packs = sorted(pred_packs, key=lambda s: s["seed"])
        for model, predictor in _secondary_pairs(system, W, h, gname, spec):
            ci_jobs.append(
                {
                    "system": system,
                    "W": int(W),
                    "h": int(h),
                    "geometry": gname,
                    "model": model,
                    "predictor": predictor,
                    "p0": [pp["preds"][f"M0|{predictor}"] for pp in pred_packs],
                    "p1": [pp["preds"][f"{model}|{predictor}"] for pp in pred_packs],
                    "y": [pp["y"] for pp in pred_packs],
                    "seeds": [int(pp["seed"]) for pp in pred_packs],
                    "boot_seed": _boot_seed("sec", system, W, h, gname, model, predictor),
                }
            )

    for job in ci_jobs:
        _save_cell_predictions(
            out_dir,
            f"{job['system']}_W{job['W']}_h{job['h']}_{job['geometry']}_{job['model']}_{job['predictor']}",
            job["seeds"],
            job["p0"],
            job["p1"],
            job["y"],
        )

    def _ci_one(job: dict) -> dict:
        stats_d = pooled_delta_ci(
            job["p0"],
            job["p1"],
            job["y"],
            step=step,
            block_steps=block_steps,
            rng=np.random.default_rng(job["boot_seed"]),
            n_boot=n_boot,
        )
        return {
            "system": job["system"],
            "W": job["W"],
            "h": job["h"],
            "geometry": job["geometry"],
            "model": job["model"],
            "predictor": job["predictor"],
            "metric": "delta_spearman_pooled",
            "value": stats_d["delta"],
            "ci_low": stats_d["ci_low"],
            "ci_high": stats_d["ci_high"],
            "p_value": stats_d["p_value"],
        }

    ci_n = min(12, max(1, len(ci_jobs)))
    if ci_n == 1:
        secondary_rows = [_ci_one(job) for job in ci_jobs]
    else:
        secondary_rows = Parallel(n_jobs=ci_n, backend="threading")(
            delayed(_ci_one)(job) for job in ci_jobs
        )

    p0_all = [p["p0"] for p in packs]
    p1_all = [p["p1"] for p in packs]
    y_all = [p["y"] for p in packs]
    order = sorted(range(len(packs)), key=lambda i: packs[i]["seed"])
    p0_all = [p0_all[i] for i in order]
    p1_all = [p1_all[i] for i in order]
    y_all = [y_all[i] for i in order]
    primary = pooled_delta_ci(
        p0_all,
        p1_all,
        y_all,
        step=step,
        block_steps=block_steps,
        rng=np.random.default_rng(_boot_seed("primary", "S1", W_primary, h_primary)),
        n_boot=n_boot,
        n_jobs=12,
    )
    if packs:
        _save_cell_predictions(
            out_dir,
            f"primary_S1_W{W_primary}_h{h_primary}_wz_M1b_ridge",
            [int(packs[i]["seed"]) for i in order],
            p0_all,
            p1_all,
            y_all,
        )
    if lag_rows:
        lag_df = pd.DataFrame(lag_rows)
        sub_bin = lag_df.loc[
            (lag_df["system"] == "S1")
            & (lag_df["bin"] == lag_bin)
            & (lag_df["W"] == W_primary)
            & (lag_df["h"] == h_primary)
            & (lag_df["geometry"] == "wz")
        ]
        m0_vals = sub_bin.loc[sub_bin["model"] == "M0", "auc"]
        m1_vals = sub_bin.loc[sub_bin["model"] == "M1b", "auc"]
        m0_auc = {"auc": float(m0_vals.mean()) if len(m0_vals) else np.nan}
        m1_auc = {"auc": float(m1_vals.mean()) if len(m1_vals) else np.nan}
    else:
        m0_auc = m1_auc = {"auc": np.nan}
    delta_auc = (
        float(m1_auc["auc"] - m0_auc["auc"])
        if np.isfinite(m1_auc.get("auc")) and np.isfinite(m0_auc.get("auc"))
        else np.nan
    )
    lag_pass = bool(np.isfinite(delta_auc) and delta_auc >= -0.02)
    primary["lag_pass"] = lag_pass
    primary["delta_auc"] = delta_auc

    secondary_tables = [_holm_secondary_table(secondary_rows, "secondary_E2")] if secondary_rows else []

    e1_pass = _load_e1_pass()
    decision = build_decision_table(
        primary,
        {"delta_auc": delta_auc, "lag_pass": lag_pass},
        smoke_mode=spec["smoke"],
        e1_pass=e1_pass,
    )
    report.write_decision(decision, out_dir)
    report.save_table(pd.DataFrame(loss_rows), out_dir, "loss_prediction")
    report.save_table(pd.DataFrame(lag_rows), out_dir, "lag_curve")
    report.save_table(pd.DataFrame(delay_rows), out_dir, "detection_delay")
    report.save_table(pd.DataFrame(oracle_rows), out_dir, "oracle")
    if not secondary_tables:
        report.save_table(pd.DataFrame(columns=["holm_p"]), out_dir, "results_secondary_1")
    for i, sdf in enumerate(secondary_tables, 1):
        report.save_table(sdf, out_dir, f"results_secondary_{i}")

    _make_figures(packs, lag_rows, spec, fig_dir, h_primary)
    return {"decision": decision, "primary": primary, "spec": spec, "secondary": secondary_tables}


def run_full_pipeline(
    spec: dict,
    out_dir: Path,
    fig_dir: Path,
    *,
    cache_prefix: str = "E2",
    n_boot: int | None = None,
) -> dict:
    trajectories = load_trajectories(spec, cache_prefix)
    return run_primary_and_outputs(
        spec, out_dir, fig_dir, trajectories, n_boot=n_boot, cache_prefix=cache_prefix
    )


def _make_figures(packs, lag_rows, spec, fig_dir, h):
    from condpred import report

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not lag_rows:
        return
    lag_df = pd.DataFrame(lag_rows)
    for system in spec["systems"]:
        sub = lag_df.loc[lag_df["system"] == system]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for model in sub["model"].unique():
            msub = sub.loc[sub["model"] == model].groupby("bin", as_index=False)["auc"].mean()
            order = {b: i for i, b in enumerate(LAG_BIN_LABELS)}
            msub = msub.sort_values("bin", key=lambda s: s.map(order))
            ax.plot(msub["bin"], msub["auc"], marker="o", label=model)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUC (regime)")
        ax.set_xlabel("time since switch")
        ax.legend(fontsize=8)
        ax.set_title(f"{system} h={h}")
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        report.save_fig(fig, fig_dir, f"lag_curve_{system}")
        plt.close(fig)

    if packs:
        p0 = next((p for p in packs if p["system"] == "S1"), packs[0])
        probs = p0.get("detect_probs") or {}
        if "M0" in probs and "M1b" in probs:
            cache_prefix = "E2/smoke" if spec.get("smoke") else "E2"
            traj = load_trajectories(spec, cache_prefix)[(p0["system"], int(p0["seed"]))]
            x, labels = traj.x, traj.labels
            segs = segments_for_system(spec, p0["system"])
            bmap = bounds(segs)
            meta = _SYSTEM_META[p0["system"]]
            ends = _feature_ends(bmap["evaluation"], int(p0["W"]), meta["m"], meta["tau"], int(spec["step"]))
            ends = [t for t in ends if t + int(h) <= bmap["evaluation"].stop - 2]
            t_ev = np.asarray(ends[: len(probs["M0"])], dtype=int)
            loss_h = _fit_all_losses(traj, p0["system"], meta, bmap)[0][int(h)]
            prob_m0 = np.asarray(probs["M0"], float)[: len(t_ev)]
            prob_m1 = np.asarray(probs["M1b"], float)[: len(t_ev)]
            if len(t_ev) > 5:
                fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
                sl = slice(int(t_ev[0]), int(t_ev[-1]) + 1)
                for reg in np.unique(labels[sl]):
                    mask = labels[sl] == reg
                    axes[0].fill_between(
                        np.arange(sl.start, sl.stop)[mask],
                        0,
                        1,
                        alpha=0.15,
                        label=f"regime {reg}",
                    )
                axes[0].set_ylabel("regime")
                axes[0].legend(loc="upper right", fontsize=7)
                axes[1].plot(t_ev, prob_m0, label="M0", alpha=0.8)
                axes[1].plot(t_ev, prob_m1, label="M1b", alpha=0.8)
                axes[1].axhline(0.5, color="gray", lw=0.5)
                axes[1].set_ylabel("P(regime=1)")
                axes[1].legend(fontsize=7)
                axes[2].plot(t_ev, loss_h[t_ev], lw=0.8)
                axes[2].set_ylabel("loss")
                axes[2].set_xlabel("t")
                fig.suptitle(f"{p0['system']} seed {p0['seed']} example")
                fig.tight_layout()
                report.save_fig(fig, fig_dir, "example_trajectory")
                plt.close(fig)
