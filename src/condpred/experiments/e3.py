"""E3 — robustness of recurrence representations: E3a fidelity, E3b regime detection, E3d gap filling.

Indexing: a window row is dated by its end day t and uses samples ≤ t. Training statistics of the
geometry arms (gz mean/sd, eps_train, RV median) come from the first ``train_days`` of the series the
arm sees. E3c (reliability under contamination) is not built here.
"""
from __future__ import annotations

import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from condpred import completion, contaminate, rp_batch, systems
from condpred.cache import cached
from condpred.config import smoke
from condpred.features import baselines
from condpred.features.local_rqa import local_rqa, local_rqa_matrix
from condpred.features.window_rqa import window_rqa
from condpred.geometry import Geometry
from condpred.rp_rqa_stage2 import line_hist_torch, rqa_from_histograms
from condpred.selfcheck import leakage_test
from condpred.stats import holm

E3_MODULES = [
    "condpred.experiments.e3",
    "condpred.features.window_rqa",
    "condpred.features.local_rqa",
    "condpred.features.baselines",
    "condpred.rp_batch",
    "condpred.rp_rqa_stage2",
    "condpred.geometry",
    "condpred.contaminate",
    "condpred.completion",
    "condpred.systems",
]

WR_COLS = ["DET", "L", "ENTR", "LAM", "DET3_DET2", "K2_lines", "TT", "Lmax", "Vmax", "ENTR_v", "DET_LAM", "RR"]
LR_COLS = ["RR_last", "Lb_mean", "Lb_max", "DET_last", "Lf_nb", "K2_local", "LAM_last", "TT_last"]
FEAT_COLS = WR_COLS + LR_COLS
BETAS = [0.0, 0.5, 1.0, 3.0, 5.0]
BETAS_EXP = [0.0, 1.0, 3.0, 5.0]
SYSTEMS = ["ms_sv", "lorenz", "msar"]
_SYS_ID = {"ms_sv": 1, "lorenz": 2, "msar": 3}
FILL_METHODS = ["ffill", "linear", "hankel"]
E3D_SERIES = ["logRV", "lorenz"]


# ----------------------------------------------------------------------------- spec


def _cond(name, factor, level, *, jump=None, noise=0.0, shocks="normal", missing=None, stale=None):
    return {
        "name": name,
        "factor": factor,
        "level": level,
        "jump": jump,
        "noise": noise,
        "shocks": shocks,
        "missing": missing,
        "stale": stale,
    }


def _fmt(v) -> str:
    return f"{v:g}"


def condition_grid(smoke_mode: bool, system: str) -> list[dict]:
    """One factor at a time, then (ms_sv only) the combined cells. Clean comes first."""
    is_sv = system == "ms_sv"
    # ms_sv noise is ω in log-price units (0.05 % of price = 0.0005); synthetic noise is a fraction of sd.
    noise_levels = [0.0002, 0.0005, 0.001] if is_sv else [0.02, 0.05, 0.1]
    noise_label = (lambda w: _fmt(w * 100)) if is_sv else _fmt
    noise_primary = 0.0005 if is_sv else 0.05
    grid = [_cond("clean", "clean", 0.0)]
    if smoke_mode:
        grid.append(_cond("J_l0.2_k5", "jumps", 0.2, jump=[0.2, 5.0]))
        grid.append(_cond(f"N_w{noise_label(noise_primary)}", "noise", noise_primary, noise=noise_primary))
        if is_sv:
            grid.append(
                _cond("C_JN_normal", "combined", 1.0, jump=[0.2, 5.0], noise=noise_primary)
            )
        else:
            grid.append(_cond("M_random_0.4", "missing_random", 0.4, missing=[0.4, "random"]))
        return grid
    for lam in [0.05, 0.2, 0.5]:
        for k in [3.0, 5.0, 8.0]:
            grid.append(_cond(f"J_l{_fmt(lam)}_k{_fmt(k)}", "jumps", lam, jump=[lam, k]))
    for w in noise_levels:
        grid.append(_cond(f"N_w{noise_label(w)}", "noise", w, noise=w))
    if is_sv:
        grid.append(_cond("T_t5", "tails", 1.0, shocks="t5"))
    for mode in ["random", "block"]:
        for frac in [0.1, 0.2, 0.4, 0.6]:
            grid.append(_cond(f"M_{mode}_{_fmt(frac)}", f"missing_{mode}", frac, missing=[frac, mode]))
    for frac in [0.05, 0.1, 0.2]:
        grid.append(_cond(f"S_{_fmt(frac)}", "stale", frac, stale=frac))
    if is_sv:
        jp = [0.2, 5.0]
        grid.append(_cond("C_JN_normal", "combined", 1.0, jump=jp, noise=noise_primary))
        grid.append(_cond("C_JN_t5", "combined", 2.0, jump=jp, noise=noise_primary, shocks="t5"))
        grid.append(_cond("C_J_t5", "combined", 3.0, jump=jp, shocks="t5"))
        grid.append(_cond("C_N_t5", "combined", 4.0, noise=noise_primary, shocks="t5"))
    return grid


def _arm(name, inp, scaling, distance, threshold="rr", beta=2.0):
    return {"name": name, "input": inp, "scaling": scaling, "distance": distance, "threshold": threshold, "beta": beta}


def arm_defs(system: str) -> list[dict]:
    inp = "logrv" if system == "ms_sv" else "x"
    arms = [
        _arm("G-wz", inp, "wz", "euclid"),
        _arm("G-gz", inp, "gz", "euclid"),
        _arm("G-eps", inp, "gz", "euclid", "eps_train"),
        _arm("G-mad", inp, "wmad", "euclid"),
        _arm("G-rank", inp, "rank", "euclid"),
        _arm("G-ord", inp, "wz", "ordinal"),
    ]
    if system == "ms_sv":
        arms += [_arm(f"G-beta({_fmt(b)})", "rv", "wz", "beta_pos", beta=b) for b in BETAS]
        arms += [_arm(f"G-pow({_fmt(b)})", "rv", "wz", "power_ctrl", beta=b) for b in BETAS]
        arms += [_arm(f"G-betaexp({_fmt(b)})", "ret", "wz", "beta_exp", beta=b) for b in BETAS_EXP]
        arms.append(_arm("BV-wz", "logbv", "wz", "euclid"))
    else:
        arms += [_arm(f"G-betaexp({_fmt(b)})", "x", "wz", "beta_exp", beta=b) for b in BETAS_EXP]
    return arms


def build_initial_spec(smoke_mode: bool) -> dict:
    n_seeds = smoke(30, 2)
    seeds = list(range(n_seeds))
    n_train_seeds = smoke(20, 1)
    spec = {
        "experiment": "E3",
        "version": 1,
        "smoke": smoke_mode,
        "parts_built": ["E3a", "E3b", "E3d"],
        "E3c": "not built (outside this build's scope; plan gates it on G1, which is pending)",
        "backend": "torch",
        "torch_dtype": "float64",
        "simulator": {
            "name": "ms_sv (E3 re-implementation of systems.ms_sv_intraday, vectorised draws)",
            "n_days": smoke(2000, 300),
            "bars_per_day": 78,
            "calm": 0.008,
            "stress": 0.02,
            "P": [[0.99, 0.01], [0.03, 0.97]],
            "logvol_phi": 0.9,
            "logvol_sd": 0.15,
            "burn_days": 50,
            "rng_order": "per day: uniform (switch), normal (log-vol); then one (n_days+burn) x 78 normal block",
            "heavy_tails": "t5 shocks = t5 quantile of Phi(z), times sqrt(3/5), of the same normal draws z; "
            "regimes, log-vol and innovation ranks shared with the normal path",
            "daily_series": "rv = sum r^2; bv = (pi/2) sum |r_i||r_(i-1)| within the day; ret = sum r; logrv, logbv",
        },
        "synthetic": {
            "n": smoke(2000, 300),
            "lorenz": "systems.lorenz63 defaults (dt 0.01, sample_every 5), observed x",
            "msar": "systems.ms_ar1 phis (0.2, 0.95), p_stay (0.99, 0.99), equal_variance (E2 S1)",
            "contamination": "jumps: Poisson rate per sample, size N(0,(k sd)^2); noise: N(0,(omega sd)^2); "
            "sd = clean-series sd",
        },
        "seeds": seeds,
        "conditions": {s: condition_grid(smoke_mode, s) for s in SYSTEMS},
        "jump_scale_ms_sv": "sigma_bar = sqrt(mean clean daily IV / 78), per-bar; jumps added to 5-min returns "
        "(= intraday log price); rate per day",
        "noise_ms_sv": "iid N(0, omega^2) on every intraday log price (omega 0.05 % = 0.0005); returns re-differenced",
        "combined_cells": "2x2x2 factorial {jump l0.2 k5} x {noise 0.05 %} x {normal, t5} minus clean and single-factor cells",
        "missing_def": "same day indices on every daily input; block = non-overlapping runs of 10 at random offsets",
        "stale_def": "random days repeat the last fresh value, same indices on every daily input",
        "missing_handling_E3a": "causal forward fill of the observed series (E3d compares fills)",
        "arms": {s: arm_defs(s) for s in SYSTEMS},
        "W_primary": 120,
        "W_secondary": smoke({"60": ["G-wz", "G-gz", "G-beta(1)", "G-pow(1)"], "250": ["G-wz", "G-gz", "G-beta(1)", "G-pow(1)"]}, {}),
        "W_secondary_scope": "ms_sv only (plan: W 60/250 for G-wz, G-gz, G-beta(1), G-pow(1))",
        "m": 3,
        "tau": 1,
        "rr": 0.10,
        "theiler": 2,
        "step": 5,
        "train_days": smoke(500, 150),
        "train_stats": "gz mean/sd, eps_train (rr-quantile of distances pooled over training windows, step 5) and "
        "RV training median from the first train_days of the observed series the arm sees",
        "lrqa_H": 8,
        "fidelity_metric": "Jaccard over |i-j| >= theiler of RP(contaminated window) vs RP(clean window), same path, "
        "same arm, same window end; mean over windows per seed, then mean/sd over seeds",
        "drift_metric": "|f_cont - f_clean| / sd_clean(f); sd over windows of all clean paths (system, arm, W)",
        "bootstrap": {"B": smoke(2000, 200), "unit": "seed (percentile 95 % CI of the across-seed mean)"},
        "hypotheses": {
            "H1a": {
                "text": "under jumps no beta arm improves Jaccard over G-wz by more than 0.02",
                "cell": "ms_sv J_l0.2_k5, W 120",
                "rule": "supported iff mean(J_beta(b) - J_wz) <= 0.02 for every b",
            },
            "H1b": {
                "text": "log BV instead of log RV improves Jaccard by more than any arm",
                "cell": "ms_sv J_l0.2_k5, W 120; BV-wz = G-wz on log BV (clean log BV vs contaminated log BV)",
                "rule": "supported iff CI low of J_BV-wz - J_best > 0, best = highest-mean other ms_sv arm",
            },
            "H2": {
                "text": "under noise omega 0.05 %, G-beta(5) beats G-beta(0) by >= 0.05 Jaccard",
                "cell": "ms_sv N_w0.05, W 120",
                "rule": "supported iff mean diff >= 0.05 and CI low > 0",
            },
            "H3": {
                "text": "under isolated jumps G-eps degrades less than G-wz",
                "cell": "ms_sv J_l0.05_k5, W 120",
                "rule": "supported iff CI low of J_eps - J_wz > 0",
            },
            "H4": {
                "text": "G-beta(b) and G-pow(b) RPs have Jaccard >= 0.9 for b in [0, 3], dropping at b = 5",
                "cell": "ms_sv clean, W 120, Jaccard between the two arms' RPs of the same window",
                "rule": "supported iff mean >= 0.9 for b in {0, 0.5, 1, 3} and mean(b=5) < 0.9",
            },
            "H5": {
                "text": "best arm's AUC loss clean -> contaminated is smaller than the rolling log RV baseline's",
                "cell": "ms_sv C_JN_normal, W 120, trained on the same condition",
                "rule": "best arm = highest clean AUC; supported iff CI low of (loss_rollRV - loss_best) > 0",
            },
            "E3b_jump_vs_BV": {
                "text": "some arm on log RV / RV beats the rolling log BV baseline under jumps",
                "cell": "ms_sv J_l0.2_k5, W 120, trained on the same condition",
                "rule": "yes iff any arm has AUC_arm - AUC_rollBV > 0 with Holm-adjusted one-sided bootstrap p < 0.05",
            },
            "H6": {
                "text": "at >= 40 % random missing, Hankel completion beats linear interpolation on relative RP error",
                "cell": "E3d per series (logRV, lorenz), M_random_0.4 and M_random_0.6",
                "rule": "supported iff CI low of (err_linear - err_hankel) > 0 at every frac",
            },
            "H7": {
                "text": "Hankel completion preserves DET and LAM better than linear interpolation",
                "cell": "E3d per series, M_random_0.4 and M_random_0.6",
                "rule": "supported iff CI low of (drift_linear - drift_hankel) > 0 for DET and LAM at every frac",
            },
        },
        "e3b": {
            "train_seeds": seeds[:n_train_seeds],
            "test_seeds": seeds[n_train_seeds:],
            "target": "true stress regime at t",
            "classifier": "median impute + standardise + LogisticRegression(class_weight=balanced, max_iter=1000)",
            "features": "WRQA + LRQA of one arm",
            "baselines": {"B-rollRV": "rolling mean of log RV over W", "B-rollBV": "rolling mean of log BV over W",
                          "B-PE": "permutation entropy (order 4) of log RV over W"},
            "train_on": ["same", "clean"],
            "auc": "pooled over test seeds; seed mean/sd reported",
        },
        "e3d": {
            "series": E3D_SERIES,
            "patterns": smoke(
                [f"M_{mode}_{_fmt(f)}" for mode in ["random", "block"] for f in [0.1, 0.2, 0.4, 0.6]]
                + [f"S_{_fmt(f)}" for f in [0.05, 0.1, 0.2]],
                ["M_random_0.4", "M_block_0.4", "S_0.1"],
            ),
            "fill": "inside each window only (data <= t): ffill, linear, hankel_complete; stale has no fill",
            "hankel": "m = 3 (window embedding m), tau 1, rank = smallest rank with 99 % energy of the linear-fill "
            "Hankel matrix, capped at 10; n_iter 200, tol 1e-6; approximation of Bousse-Dreesen, not Tensorlab SDF",
            "geometry": "G-wz (default)",
            "rel_error": "||D_fill - D_clean||_F / ||D_clean||_F over the full window distance matrix",
        },
    }
    return rp_batch.persist_fallbacks_into_spec(spec)


def verify_spec_immutable(out_dir: Path, spec: dict, modules: list[str]):
    from condpred.report import write_spec

    bad = dict(spec)
    bad["W_primary"] = int(spec["W_primary"]) + 1
    try:
        write_spec(bad, out_dir, modules=modules)
    except RuntimeError:
        return
    raise AssertionError("write_spec accepted a changed spec with the same version")


# ----------------------------------------------------------------------------- simulation


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng([3, *[int(p) for p in parts]])


def _crc(s: str) -> int:
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def simulate_ms_sv(seed: int, sim: dict) -> dict:
    """Two-regime Markov SV, 5-minute shocks. Same law as ``systems.ms_sv_intraday``; draws are vectorised."""
    rng = _rng(_SYS_ID["ms_sv"], seed)
    P = np.asarray(sim["P"], float)
    burn, n_days, bars = int(sim["burn_days"]), int(sim["n_days"]), int(sim["bars_per_day"])
    n_all = burn + n_days
    reg = np.zeros(n_all, dtype=int)
    log_h = np.zeros(n_all)
    lh = 0.0
    for d in range(n_all):
        if d > 0 and rng.random() > P[reg[d - 1], reg[d - 1]]:
            reg[d] = 1 - reg[d - 1]
        elif d > 0:
            reg[d] = reg[d - 1]
        lh = sim["logvol_phi"] * lh + sim["logvol_sd"] * rng.standard_normal()
        log_h[d] = lh
    z = rng.standard_normal((n_all, bars))
    daily_vol = np.where(reg == 0, sim["calm"], sim["stress"]) * np.exp(log_h)
    return {"regime": reg[burn:], "daily_vol": daily_vol[burn:], "z": z[burn:], "daily_iv": daily_vol[burn:] ** 2}


def _t5_from_normal(z: np.ndarray) -> np.ndarray:
    pos = z > 0
    out = np.empty_like(z)
    out[pos] = -stats.t.ppf(stats.norm.cdf(-z[pos]), 5)
    out[~pos] = stats.t.ppf(stats.norm.cdf(z[~pos]), 5)
    return out * np.sqrt(3 / 5)


def intraday_returns(base: dict, shocks: str, bars: int) -> np.ndarray:
    z = base["z"] if shocks == "normal" else _t5_from_normal(base["z"])
    return (base["daily_vol"] / np.sqrt(bars))[:, None] * z


def daily_series(r: np.ndarray) -> dict:
    rv = (r**2).sum(axis=1)
    bv = (np.pi / 2) * (np.abs(r[:, 1:]) * np.abs(r[:, :-1])).sum(axis=1)
    return {"rv": rv, "logrv": np.log(rv), "logbv": np.log(bv), "ret": r.sum(axis=1)}


def simulate_base(spec: dict, system: str, seed: int, cache_exp: str) -> dict:
    def _fn():
        if system == "ms_sv":
            return simulate_ms_sv(seed, spec["simulator"])
        n = int(spec["synthetic"]["n"])
        rng = _rng(_SYS_ID[system], seed)
        if system == "lorenz":
            _, x = systems.lorenz63(n, rng)
            return {"x": np.asarray(x, float), "regime": None}
        x, lab = systems.ms_ar1(phis=(0.2, 0.95), p_stay=(0.99, 0.99), n=n, rng=rng, equal_variance=True)
        return {"x": np.asarray(x, float), "regime": np.asarray(lab, int)}

    key = {"artifact": "base", "system": system, "seed": int(seed), "sim": spec["simulator"], "syn": spec["synthetic"]}
    return cached(cache_exp, f"base_{system}_s{int(seed)}", key, _fn, modules=E3_MODULES)


def observe(spec: dict, system: str, seed: int, base: dict, cond: dict) -> dict:
    """Observed series under one condition. Inputs are never modified."""
    crng = _rng(_SYS_ID[system], seed, _crc(cond["name"]))
    if system == "ms_sv":
        bars = int(spec["simulator"]["bars_per_day"])
        r = intraday_returns(base, cond["shocks"], bars)
        if cond["jump"]:
            lam, k = cond["jump"]
            sigma_bar = float(np.sqrt(np.mean(base["daily_iv"]) / bars))
            r = contaminate.add_jumps(r.ravel(), lam, k, sigma_bar, crng, bars_per_day=bars).reshape(r.shape)
        if cond["noise"]:
            p = np.concatenate([[0.0], np.cumsum(r.ravel())])
            p = contaminate.add_obs_noise(p, cond["noise"], crng)
            r = np.diff(p).reshape(r.shape)
        series = daily_series(r)
    else:
        x = np.asarray(base["x"], float)
        sd = float(np.std(x))
        if cond["jump"]:
            lam, k = cond["jump"]
            x = contaminate.add_jumps(x, lam, k, sd, crng, bars_per_day=1)
        if cond["noise"]:
            x = x + crng.normal(0.0, cond["noise"] * sd, size=len(x))
        series = {"x": x}
    if cond["missing"]:
        frac, mode = cond["missing"]
        for k in list(series):
            xm, _ = contaminate.mask_missing(series[k], frac, mode, 10, rng=_rng(_SYS_ID[system], seed, _crc(cond["name"]), 7))
            series[k] = completion.ffill(xm)
    if cond["stale"]:
        for k in list(series):
            series[k] = contaminate.make_stale(series[k], cond["stale"], "random", _rng(_SYS_ID[system], seed, _crc(cond["name"]), 8))
    return series


def observed_all(spec: dict, system: str, seed: int, cache_exp: str) -> dict:
    def _fn():
        base = simulate_base(spec, system, seed, cache_exp)
        return {c["name"]: observe(spec, system, seed, base, c) for c in spec["conditions"][system]}

    key = {"artifact": "observed", "system": system, "seed": int(seed), "conds": spec["conditions"][system],
           "sim": spec["simulator"], "syn": spec["synthetic"]}
    return cached(cache_exp, f"obs_{system}_s{int(seed)}", key, _fn, modules=E3_MODULES)


def simulate_all(spec: dict, cache_exp: str) -> None:
    for system in SYSTEMS:
        for seed in spec["seeds"]:
            simulate_base(spec, system, seed, cache_exp)


def contaminate_all(spec: dict, cache_exp: str) -> None:
    for system in SYSTEMS:
        for seed in spec["seeds"]:
            observed_all(spec, system, seed, cache_exp)


# ----------------------------------------------------------------------------- geometry / RPs


def feature_ends(spec: dict, n: int) -> np.ndarray:
    return np.arange(int(spec["train_days"]), n, int(spec["step"]), dtype=int)


def prepare_geom(arm: dict, x: np.ndarray, W: int, spec: dict) -> Geometry:
    g = Geometry(
        scaling=arm["scaling"],
        distance=arm["distance"],
        threshold=arm["threshold"],
        rr=float(spec["rr"]),
        beta=float(arm["beta"]),
        theiler=int(spec["theiler"]),
        m=int(spec["m"]),
        tau=int(spec["tau"]),
    )
    tr = np.asarray(x[: int(spec["train_days"])], float)
    if g.scaling == "gz":
        sd = float(np.std(tr))
        g.train_mean, g.train_sd = float(np.mean(tr)), sd if sd > 0 else 1.0
    if g.distance in ("beta_pos", "power_ctrl"):
        g.train_median = float(np.median(tr))
    if g.threshold == "eps_train":
        wins = [tr[e - W + 1 : e + 1] for e in range(W - 1, len(tr), int(spec["step"]))]
        g.eps_train = rp_batch.pooled_quantile(wins, g, g.rr)
    return g


def _is_positive(g: Geometry) -> bool:
    return g.distance in ("beta_pos", "power_ctrl")


def window_rp_batch(x: np.ndarray, ends: np.ndarray, W: int, g: Geometry) -> np.ndarray:
    """[B, n, n] bool RPs for windows ending at ``ends`` (identical to ``g.recurrence``)."""
    pos = x if _is_positive(g) else None
    if rp_batch.arm_falls_back(g, W):
        return np.stack([g.recurrence(x[e - W + 1 : e + 1], positive_raw=None if pos is None else pos[e - W + 1 : e + 1])[1] for e in ends])
    parts = [Rb for _, Rb, _ in rp_batch.window_rps(x, ends, W, g, positive_raw=pos)]
    return np.concatenate(parts, axis=0)


def jaccard_np(Ra: np.ndarray, Rb: np.ndarray, theiler: int) -> np.ndarray:
    torch = rp_batch._torch()
    dev = rp_batch.device()
    return rp_batch.jaccard(torch.as_tensor(Ra, device=dev), torch.as_tensor(Rb, device=dev), theiler)


def _host_features(packed: np.ndarray, dh: np.ndarray, vh: np.ndarray, n: int, threshold: str, distance: str,
                   theiler: int, H: int) -> np.ndarray:
    """Worker: WRQA (from GPU line histograms) + LRQA for each packed RP. Rows follow ``FEAT_COLS``."""
    geom = Geometry(threshold=threshold, distance=distance)
    out = np.full((len(packed), len(FEAT_COLS)), np.nan)
    for j in range(len(packed)):
        R = np.unpackbits(packed[j])[: n * n].reshape(n, n).astype(bool)
        wr = rqa_from_histograms(dh[j], vh[j], n, geom, rr_from_R=float(R.mean()))
        lr = local_rqa_matrix(R, theiler=theiler, H=H)
        out[j] = [wr.get(c, np.nan) for c in WR_COLS] + [lr[c] for c in LR_COLS]
    return out


def submit_features(pool, R: np.ndarray, g: Geometry, H: int):
    dh, vh = line_hist_torch(R, device=rp_batch.device())
    packed = np.packbits(R.reshape(len(R), -1), axis=1)
    return pool.submit(_host_features, packed, dh, vh, R.shape[1], g.threshold, g.distance, g.theiler_band(), H)


def make_pool(max_workers: int | None = None) -> ProcessPoolExecutor:
    n = max_workers or max(2, min(16, (os.cpu_count() or 4) - 4))
    print(f"E3 pool: {n} workers", flush=True)
    return ProcessPoolExecutor(max_workers=n)


# ----------------------------------------------------------------------------- E3a (+ features for E3b)


def armW_list(spec: dict, system: str) -> list[tuple[dict, int]]:
    out = [(a, int(spec["W_primary"])) for a in spec["arms"][system]]
    if system == "ms_sv":
        for W, names in spec["W_secondary"].items():
            out += [(a, int(W)) for a in spec["arms"][system] if a["name"] in names]
    return out


def _baseline_frame(obs: dict, ends: np.ndarray, Ws: list[int]) -> pd.DataFrame:
    rows = []
    for W in Ws:
        rv_roll = pd.Series(obs["logrv"]).rolling(W, min_periods=W).mean().to_numpy()
        bv_roll = pd.Series(obs["logbv"]).rolling(W, min_periods=W).mean().to_numpy()
        pe = np.array([baselines.perm_entropy(obs["logrv"], W, int(e), order=4) for e in ends])
        rows.append(pd.DataFrame({"W": W, "end": ends, "B-rollRV": rv_roll[ends], "B-rollBV": bv_roll[ends], "B-PE": pe}))
    return pd.concat(rows, ignore_index=True)


def _compute_e3a_seed(spec: dict, system: str, seed: int, cache_exp: str, pool) -> dict:
    t0 = time.time()
    base = simulate_base(spec, system, seed, cache_exp)
    obs_all = observed_all(spec, system, seed, cache_exp)
    conds = spec["conditions"][system]
    n = len(next(iter(obs_all["clean"].values())))
    ends = feature_ends(spec, n)
    regime = None if base.get("regime") is None else np.asarray(base["regime"])[ends]
    H = int(spec["lrqa_H"])
    W0 = int(spec["W_primary"])
    armWs = armW_list(spec, system)
    R_clean: dict[tuple[str, int], np.ndarray] = {}
    jobs, pair_rows, base_frames = [], [], []
    for cond in conds:
        obs = obs_all[cond["name"]]
        R_keep: dict[str, np.ndarray] = {}
        for arm, W in armWs:
            x = obs[arm["input"]]
            g = prepare_geom(arm, x, W, spec)
            R = window_rp_batch(x, ends, W, g)
            key = (arm["name"], W)
            if cond["name"] == "clean":
                R_clean[key] = R
            jac = jaccard_np(R_clean[key], R, g.theiler_band())
            jobs.append((cond["name"], arm["name"], W, jac, submit_features(pool, R, g, H)))
            if W == W0:
                R_keep[arm["name"]] = R
        th = int(spec["theiler"])
        if system == "ms_sv":
            for b in BETAS:
                jb = jaccard_np(R_keep[f"G-beta({_fmt(b)})"], R_keep[f"G-pow({_fmt(b)})"], th)
                pair_rows.append({"cond": cond["name"], "pair": f"beta~pow({_fmt(b)})", "W": W0, "jaccard": float(np.mean(jb))})
            jx = jaccard_np(R_clean[("G-wz", W0)], R_keep["BV-wz"], th)
            pair_rows.append({"cond": cond["name"], "pair": "BV-wz~G-wz(clean logRV)", "W": W0, "jaccard": float(np.mean(jx))})
            bf = _baseline_frame(obs, ends, sorted({W for _, W in armWs}))
            bf.insert(0, "cond", cond["name"])
            base_frames.append(bf)
    frames = []
    for cname, aname, W, jac, fut in jobs:
        feats = fut.result()
        df = pd.DataFrame(feats.astype(np.float32), columns=FEAT_COLS)
        df.insert(0, "jaccard", jac.astype(np.float32))
        df.insert(0, "end", ends)
        df.insert(0, "W", W)
        df.insert(0, "arm", aname)
        df.insert(0, "cond", cname)
        if regime is not None:
            df["regime"] = regime.astype(np.int8)
        frames.append(df)
    out = {"features": pd.concat(frames, ignore_index=True), "pairs": pd.DataFrame(pair_rows)}
    if base_frames:
        bdf = pd.concat(base_frames, ignore_index=True)
        bdf["regime"] = np.tile(regime, len(bdf) // len(regime)).astype(np.int8)
        out["baselines"] = bdf
    print(f"E3a {system} seed {seed}: {len(out['features'])} windows in {time.time() - t0:.1f}s", flush=True)
    return out


def _e3a_key(spec: dict, system: str, seed: int) -> dict:
    keys = ["simulator", "synthetic", "W_primary", "W_secondary", "m", "tau", "rr", "theiler", "step",
            "train_days", "lrqa_H", "backend", "torch_dtype", "rp_fallback_arms"]
    return {"artifact": "e3a_seed", "system": system, "seed": int(seed), "conds": spec["conditions"][system],
            "arms": spec["arms"][system], **{k: spec.get(k) for k in keys}}


def e3a_seed(spec: dict, system: str, seed: int, cache_exp: str, pool=None) -> dict:
    def _fn():
        if pool is None:
            raise RuntimeError("E3a seed not cached and no worker pool given")
        return _compute_e3a_seed(spec, system, seed, cache_exp, pool)

    return cached(cache_exp, f"e3a_{system}_s{int(seed)}", _e3a_key(spec, system, seed), _fn, modules=E3_MODULES)


def _cond_meta(spec: dict, system: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"cond": c["name"], "factor": c["factor"], "level": c["level"]} for c in spec["conditions"][system]]
    )


def _drift_seed(feat: pd.DataFrame, sd_clean: pd.DataFrame) -> pd.DataFrame:
    """Per (cond, arm, W, feature): mean over windows of |f_cont - f_clean| / sd_clean."""
    clean = feat[feat["cond"] == "clean"].set_index(["arm", "W", "end"])[FEAT_COLS].sort_index()
    rows = []
    for (cname, aname, W), g in feat.groupby(["cond", "arm", "W"], sort=False, observed=True):
        c = clean.loc[(aname, W)].reindex(g["end"].to_numpy()).to_numpy(float)
        f = g[FEAT_COLS].to_numpy(float)
        sd = sd_clean.loc[(aname, W)].to_numpy(float)
        valid_sd = np.isfinite(sd) & (sd > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.abs(f - c) / np.where(valid_sd, sd, np.nan)
        means = np.nanmean(np.where(np.isfinite(d), d, np.nan), axis=0) if len(d) else np.full(len(FEAT_COLS), np.nan)
        for j, col in enumerate(FEAT_COLS):
            if valid_sd[j]:
                rows.append((cname, aname, W, col, float(means[j])))
        wr = [means[j] for j, col in enumerate(FEAT_COLS) if col in WR_COLS and valid_sd[j] and np.isfinite(means[j])]
        lr = [means[j] for j, col in enumerate(FEAT_COLS) if col in LR_COLS and valid_sd[j] and np.isfinite(means[j])]
        rows.append((cname, aname, W, "WRQA_mean", float(np.mean(wr)) if wr else np.nan))
        rows.append((cname, aname, W, "LRQA_mean", float(np.mean(lr)) if lr else np.nan))
    return pd.DataFrame(rows, columns=["cond", "arm", "W", "feature", "drift"])


def _ignore_empty_slice(fn, *a, **k):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*a, **k)


def run_e3a(spec: dict, out_dir: Path, cache_exp: str, pool) -> dict:
    """Features for every (system, seed, condition, arm, W), fidelity and drift tables."""
    from condpred import report

    fid_rows, pair_frames, clean_frames = [], [], []
    for system in SYSTEMS:
        for seed in spec["seeds"]:
            res = e3a_seed(spec, system, seed, cache_exp, pool)
            f = res["features"]
            jm = f.groupby(["cond", "arm", "W"], sort=False, observed=True)["jaccard"].mean().reset_index()
            jm.insert(0, "seed", seed)
            jm.insert(0, "system", system)
            fid_rows.append(jm)
            p = res["pairs"].copy()
            if len(p):
                p.insert(0, "seed", seed)
                p.insert(0, "system", system)
                pair_frames.append(p)
            c = f[f["cond"] == "clean"][["arm", "W"] + FEAT_COLS].copy()
            c.insert(0, "system", system)
            clean_frames.append(c)
    fid_seed = pd.concat(fid_rows, ignore_index=True)
    pair_seed = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    clean_all = pd.concat(clean_frames, ignore_index=True)
    sd_clean = clean_all.groupby(["system", "arm", "W"])[FEAT_COLS].std(ddof=1)
    del clean_all, clean_frames

    drift_rows = []
    for system in SYSTEMS:
        sdc = sd_clean.loc[system]
        for seed in spec["seeds"]:
            res = e3a_seed(spec, system, seed, cache_exp, pool)
            d = _ignore_empty_slice(_drift_seed, res["features"], sdc)
            d.insert(0, "seed", seed)
            d.insert(0, "system", system)
            drift_rows.append(d)
    drift_seed = pd.concat(drift_rows, ignore_index=True)

    meta = pd.concat([_cond_meta(spec, s).assign(system=s) for s in SYSTEMS], ignore_index=True)
    arm_input = pd.DataFrame(
        [{"system": s, "arm": a["name"], "input": a["input"]} for s in SYSTEMS for a in spec["arms"][s]]
    )
    fid = (
        fid_seed.groupby(["system", "cond", "arm", "W"], observed=True)["jaccard"]
        .agg(jaccard_mean="mean", jaccard_sd="std", n_seeds="count")
        .reset_index()
        .assign(kind="arm_clean_vs_contaminated")
    )
    fid = fid.merge(arm_input, on=["system", "arm"], how="left")
    if len(pair_seed):
        pr = (
            pair_seed.groupby(["system", "cond", "pair", "W"])["jaccard"]
            .agg(jaccard_mean="mean", jaccard_sd="std", n_seeds="count")
            .reset_index()
            .rename(columns={"pair": "arm"})
            .assign(kind="pair_same_condition", input="rv / logbv")
        )
        fid = pd.concat([fid, pr], ignore_index=True)
    fid = fid.merge(meta, on=["system", "cond"], how="left")
    fid = fid[["system", "cond", "factor", "level", "kind", "arm", "input", "W", "jaccard_mean", "jaccard_sd", "n_seeds"]]
    report.save_table(fid.sort_values(["system", "factor", "level", "cond", "arm", "W"]).reset_index(drop=True), out_dir, "fidelity")

    drift = (
        drift_seed.groupby(["system", "cond", "arm", "W", "feature"], observed=True)["drift"]
        .agg(drift_mean="mean", drift_sd="std", n_seeds="count")
        .reset_index()
        .merge(meta, on=["system", "cond"], how="left")
    )
    drift = drift[["system", "cond", "factor", "level", "arm", "W", "feature", "drift_mean", "drift_sd", "n_seeds"]]
    report.save_table(drift.sort_values(["system", "factor", "level", "cond", "arm", "W", "feature"]).reset_index(drop=True), out_dir, "feature_drift")
    return {"fid_seed": fid_seed, "pair_seed": pair_seed, "fidelity": fid, "drift": drift}


# ----------------------------------------------------------------------------- bootstrap helpers


def seed_boot(values: np.ndarray, B: int, seed: int) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    boots = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _per_seed(fid_seed: pd.DataFrame, system: str, cond: str, arm: str, W: int) -> pd.Series:
    s = fid_seed[(fid_seed["system"] == system) & (fid_seed["cond"] == cond) & (fid_seed["arm"] == arm) & (fid_seed["W"] == W)]
    return s.set_index("seed")["jaccard"]


def _diff(a: pd.Series, b: pd.Series) -> np.ndarray:
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    return (j["a"] - j["b"]).to_numpy(float)


def _row(gate, statistic, value, lo, hi, rule, verdict, note="", smoke_mode=False):
    return {"gate": gate, "statistic": statistic, "value": value, "ci_low": lo, "ci_high": hi, "rule": rule,
            "verdict": verdict, "note": note, "smoke": smoke_mode}


def _not_evaluable(gate, rule, missing, smoke_mode):
    return _row(gate, "", np.nan, np.nan, np.nan, rule, "not_evaluable", f"cell absent from this run: {missing}", smoke_mode)


def evaluate_e3a(spec: dict, e3a: dict) -> list[dict]:
    fs, ps = e3a["fid_seed"], e3a["pair_seed"]
    W0, B, sm = int(spec["W_primary"]), int(spec["bootstrap"]["B"]), bool(spec["smoke"])
    hyp = spec["hypotheses"]
    conds = {c["name"] for c in spec["conditions"]["ms_sv"]}
    rows = []

    cj = "J_l0.2_k5"
    if cj in conds:
        wz = _per_seed(fs, "ms_sv", cj, "G-wz", W0)
        stats_b = []
        for b in BETAS:
            d = _diff(_per_seed(fs, "ms_sv", cj, f"G-beta({_fmt(b)})", W0), wz)
            stats_b.append((b, *seed_boot(d, B, _crc(f"H1a{b}"))))
        worst = max(stats_b, key=lambda r: r[1])
        ok = all(r[1] <= 0.02 for r in stats_b)
        note = "; ".join(f"b={_fmt(r[0])}: {r[1]:+.4f} [{r[2]:+.4f}, {r[3]:+.4f}]" for r in stats_b)
        rows.append(_row("H1a", f"max_b mean(J_beta(b) - J_wz) at b={_fmt(worst[0])}", worst[1], worst[2], worst[3],
                         hyp["H1a"]["rule"], "supported" if ok else "not_supported", note, sm))

        arms_other = [a["name"] for a in spec["arms"]["ms_sv"] if a["name"] != "BV-wz"]
        means = {a: float(_per_seed(fs, "ms_sv", cj, a, W0).mean()) for a in arms_other}
        best = max(means, key=means.get)
        d = _diff(_per_seed(fs, "ms_sv", cj, "BV-wz", W0), _per_seed(fs, "ms_sv", cj, best, W0))
        m, lo, hi = seed_boot(d, B, _crc("H1b"))
        rows.append(_row("H1b", f"J_BV-wz - J_{best}", m, lo, hi, hyp["H1b"]["rule"],
                         "supported" if lo > 0 else "not_supported",
                         f"J_BV-wz={float(_per_seed(fs, 'ms_sv', cj, 'BV-wz', W0).mean()):.4f}; best other arm {best} J={means[best]:.4f}; J_G-wz={means['G-wz']:.4f}", sm))
    else:
        rows.append(_not_evaluable("H1a", hyp["H1a"]["rule"], cj, sm))
        rows.append(_not_evaluable("H1b", hyp["H1b"]["rule"], cj, sm))

    cn = "N_w0.05"
    if cn in conds:
        d = _diff(_per_seed(fs, "ms_sv", cn, "G-beta(5)", W0), _per_seed(fs, "ms_sv", cn, "G-beta(0)", W0))
        m, lo, hi = seed_boot(d, B, _crc("H2"))
        rows.append(_row("H2", "J_beta(5) - J_beta(0)", m, lo, hi, hyp["H2"]["rule"],
                         "supported" if (m >= 0.05 and lo > 0) else "not_supported", "", sm))
    else:
        rows.append(_not_evaluable("H2", hyp["H2"]["rule"], cn, sm))

    c3 = "J_l0.05_k5"
    if c3 in conds:
        d = _diff(_per_seed(fs, "ms_sv", c3, "G-eps", W0), _per_seed(fs, "ms_sv", c3, "G-wz", W0))
        m, lo, hi = seed_boot(d, B, _crc("H3"))
        rows.append(_row("H3", "J_eps - J_wz", m, lo, hi, hyp["H3"]["rule"], "supported" if lo > 0 else "not_supported", "", sm))
    else:
        rows.append(_not_evaluable("H3", hyp["H3"]["rule"], c3, sm))

    pc = ps[(ps["system"] == "ms_sv") & (ps["cond"] == "clean")]
    vals = {}
    for b in BETAS:
        v = pc[pc["pair"] == f"beta~pow({_fmt(b)})"]["jaccard"].to_numpy(float)
        vals[b] = seed_boot(v, B, _crc(f"H4{b}"))
    low_bs = [b for b in BETAS if b <= 3]
    ok = all(vals[b][0] >= 0.9 for b in low_bs) and vals[5.0][0] < 0.9
    bmin = min(low_bs, key=lambda b: vals[b][0])
    note = "; ".join(f"b={_fmt(b)}: {vals[b][0]:.4f} [{vals[b][1]:.4f}, {vals[b][2]:.4f}]" for b in BETAS)
    rows.append(_row("H4", f"min_(b<=3) mean J(beta(b), pow(b)) at b={_fmt(bmin)}", *vals[bmin], hyp["H4"]["rule"],
                     "supported" if ok else "not_supported", note, sm))
    return rows


# ----------------------------------------------------------------------------- E3b


def _fit_logistic(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), np.nan)
    keep = np.isfinite(Xtr).any(axis=0)
    if not keep.any():
        return np.full(len(Xte), np.nan)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(Xtr[:, keep], ytr)
    return pipe.predict_proba(Xte[:, keep])[:, 1]


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def _e3b_blocks(spec: dict, cache_exp: str) -> dict:
    """(cond, model, W) -> (seed, X, y) arrays across all seeds."""
    blocks: dict[tuple[str, str, int], list] = {}
    for seed in spec["seeds"]:
        res = e3a_seed(spec, "ms_sv", seed, cache_exp)
        f = res["features"]
        for (c, a, W), g in f.groupby(["cond", "arm", "W"], sort=False, observed=True):
            blocks.setdefault((c, a, int(W)), []).append((seed, g[FEAT_COLS].to_numpy(np.float64), g["regime"].to_numpy(int)))
        bl = res["baselines"]
        for (c, W), g in bl.groupby(["cond", "W"], sort=False):
            for name in ["B-rollRV", "B-rollBV", "B-PE"]:
                blocks.setdefault((c, name, int(W)), []).append((seed, g[[name]].to_numpy(np.float64), g["regime"].to_numpy(int)))
    out = {}
    for k, parts in blocks.items():
        out[k] = (
            np.concatenate([np.full(len(p[2]), p[0]) for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]),
        )
    return out


def run_e3b(spec: dict, out_dir: Path, cache_exp: str) -> dict:
    from joblib import Parallel, delayed

    from condpred import report

    blocks = _e3b_blocks(spec, cache_exp)
    tr_seeds = np.asarray(spec["e3b"]["train_seeds"])
    te_seeds = np.asarray(spec["e3b"]["test_seeds"])
    keys = []
    for c, model, W in blocks:
        for train_on in spec["e3b"]["train_on"]:
            if train_on == "clean" and c == "clean":
                continue
            keys.append((c, model, W, train_on))

    def _task(key):
        c, model, W, train_on = key
        sd, X, _ = blocks[(c, model, W)]
        src = blocks[("clean" if train_on == "clean" else c, model, W)]
        tr = np.isin(src[0], tr_seeds)
        return delayed(_fit_logistic)(src[1][tr], src[2][tr], X[np.isin(sd, te_seeds)])

    print(f"E3b: {len(keys)} logistic fits", flush=True)
    probs = Parallel(n_jobs=max(2, min(16, (os.cpu_count() or 4) - 4)), backend="loky")(_task(k) for k in keys)
    preds = {}
    rows = []
    inputs = {a["name"]: a["input"] for a in spec["arms"]["ms_sv"]}
    for key, p in zip(keys, probs):
        c, model, W, train_on = key
        sd, _, y = blocks[(c, model, W)]
        te = np.isin(sd, te_seeds)
        sd_te, y_te = sd[te], y[te]
        preds[key] = (sd_te, y_te, p)
        per = [_auc(y_te[sd_te == s], p[sd_te == s]) for s in te_seeds]
        rows.append({
            "cond": c, "model": model, "kind": "baseline" if model.startswith("B-") else "arm",
            "input": inputs.get(model, {"B-rollRV": "logrv", "B-rollBV": "logbv", "B-PE": "logrv"}.get(model)),
            "W": W, "train_on": train_on, "auc_pooled": _auc(y_te, p),
            "auc_seed_mean": float(np.nanmean(per)) if np.isfinite(per).any() else np.nan,
            "auc_seed_sd": float(np.nanstd(per, ddof=1)) if np.isfinite(per).sum() > 1 else np.nan,
            "n_test": int(len(y_te)),
        })
    auc = pd.DataFrame(rows).merge(_cond_meta(spec, "ms_sv"), on="cond", how="left")
    auc = auc[["cond", "factor", "level", "model", "kind", "input", "W", "train_on", "auc_pooled", "auc_seed_mean", "auc_seed_sd", "n_test"]]
    report.save_table(auc.sort_values(["factor", "level", "cond", "train_on", "W", "model"]).reset_index(drop=True), out_dir, "regime_auc")
    return {"auc": auc, "preds": preds}


def _boot_auc_draws(preds: dict, keys: list, te_seeds: np.ndarray, B: int, seed: int) -> np.ndarray:
    """[B, len(keys)] pooled AUCs over resampled test seeds (same draw for every key)."""
    rng = np.random.default_rng(seed)
    out = np.full((B, len(keys)), np.nan)
    idx_by_seed = [{s: np.flatnonzero(preds[k][0] == s) for s in te_seeds} for k in keys]
    for b in range(B):
        draw = rng.choice(te_seeds, size=len(te_seeds), replace=True)
        for j, k in enumerate(keys):
            sd, y, p = preds[k]
            ii = np.concatenate([idx_by_seed[j][s] for s in draw])
            out[b, j] = _auc(y[ii], p[ii])
    return out


def evaluate_e3b(spec: dict, e3b: dict) -> list[dict]:
    auc, preds = e3b["auc"], e3b["preds"]
    W0, B, sm = int(spec["W_primary"]), int(spec["bootstrap"]["B"]), bool(spec["smoke"])
    te = np.asarray(spec["e3b"]["test_seeds"])
    hyp = spec["hypotheses"]
    conds = {c["name"] for c in spec["conditions"]["ms_sv"]}
    rows = []
    arms = [a["name"] for a in spec["arms"]["ms_sv"]]

    cp = "C_JN_normal"
    if cp in conds:
        clean = auc[(auc["cond"] == "clean") & (auc["W"] == W0) & (auc["train_on"] == "same") & (auc["kind"] == "arm")]
        best = clean.sort_values("auc_pooled", ascending=False)["model"].iloc[0]
        keys = [("clean", "B-rollRV", W0, "same"), (cp, "B-rollRV", W0, "same"), ("clean", best, W0, "same"), (cp, best, W0, "same")]
        draws = _boot_auc_draws(preds, keys, te, B, _crc("H5"))
        point = np.array([_auc(preds[k][1], preds[k][2]) for k in keys])

        def stat(a):
            return (a[..., 0] - a[..., 1]) - (a[..., 2] - a[..., 3])

        d = stat(draws)
        rows.append(_row("H5", f"loss_rollRV - loss_{best}", float(stat(point)), float(np.nanpercentile(d, 2.5)),
                         float(np.nanpercentile(d, 97.5)), hyp["H5"]["rule"],
                         "supported" if np.nanpercentile(d, 2.5) > 0 else "not_supported",
                         f"AUC rollRV clean {point[0]:.4f} -> {point[1]:.4f}; {best} clean {point[2]:.4f} -> {point[3]:.4f}", sm))
    else:
        rows.append(_not_evaluable("H5", hyp["H5"]["rule"], cp, sm))

    cj = "J_l0.2_k5"
    if cj in conds:
        rv_arms = [a["name"] for a in spec["arms"]["ms_sv"] if a["input"] in ("logrv", "rv")]
        keys = [(cj, "B-rollBV", W0, "same")] + [(cj, a, W0, "same") for a in rv_arms]
        draws = _boot_auc_draws(preds, keys, te, B, _crc("E3b_BV"))
        point = np.array([_auc(preds[k][1], preds[k][2]) for k in keys])
        diffs = point[1:] - point[0]
        dd = draws[:, 1:] - draws[:, [0]]
        pvals = [(1 + np.sum(dd[:, j] <= 0)) / (1 + np.sum(np.isfinite(dd[:, j]))) for j in range(len(rv_arms))]
        adj = holm(pvals)
        j = int(np.nanargmax(diffs))
        yes = any((diffs[i] > 0) and (adj[i] < 0.05) for i in range(len(rv_arms)))
        note = f"AUC rollBV {point[0]:.4f}; " + "; ".join(f"{a}: {diffs[i]:+.4f} (p_holm {adj[i]:.3g})" for i, a in enumerate(rv_arms))
        rows.append(_row("E3b_jump_vs_BV", f"max AUC_arm - AUC_rollBV ({rv_arms[j]})", float(diffs[j]),
                         float(np.nanpercentile(dd[:, j], 2.5)), float(np.nanpercentile(dd[:, j], 97.5)),
                         hyp["E3b_jump_vs_BV"]["rule"], "yes" if yes else "no", note, sm))
    else:
        rows.append(_not_evaluable("E3b_jump_vs_BV", hyp["E3b_jump_vs_BV"]["rule"], cj, sm))
    return rows


# ----------------------------------------------------------------------------- E3d


def _hankel_rank(w_lin: np.ndarray, m: int, cap: int) -> int:
    H = completion._hankel(w_lin, m, 1)
    s = np.linalg.svd(H, compute_uv=False)
    cum = np.cumsum(s**2) / max((s**2).sum(), 1e-300)
    return int(min(cap, int(np.searchsorted(cum, 0.99) + 1)))


def fill_window(w: np.ndarray, method: str, m: int, cap: int = 10) -> np.ndarray:
    """Fill one window using only its own samples (all ≤ t)."""
    w = np.asarray(w, float)
    ok = np.isfinite(w)
    if ok.all():
        return w.copy()
    if not ok.any():
        return np.full_like(w, np.nan)
    if method == "ffill":
        return completion.ffill(w)
    if method == "linear":
        return completion.linear(w)
    if method == "hankel":
        rank = _hankel_rank(completion.linear(w), m, cap)
        return completion.hankel_complete(w, m, tau=1, rank=rank, n_iter=200, tol=1e-6)
    raise ValueError(method)


def _fill_windows(wins: np.ndarray, method: str, m: int) -> np.ndarray:
    return np.stack([fill_window(w, method, m) for w in wins])


def _window_stack(x: np.ndarray, ends: np.ndarray, W: int) -> np.ndarray:
    return np.stack([x[e - W + 1 : e + 1] for e in ends])


def _D_R(wins: np.ndarray, g: Geometry):
    torch = rp_batch._torch()
    dev = rp_batch.device()
    D, _ = rp_batch._direct_distances(wins, g, None, torch.float64, dev=dev)
    R = rp_batch.threshold_rr(D, g.rr, g.theiler_band())
    return D, R


def _pattern_def(name: str) -> dict:
    if name.startswith("M_"):
        _, mode, frac = name.split("_")
        return {"name": name, "kind": "missing", "mode": mode, "frac": float(frac)}
    return {"name": name, "kind": "stale", "mode": "random", "frac": float(name.split("_")[1])}


def _e3d_clean_series(spec: dict, series: str, seed: int, cache_exp: str) -> np.ndarray:
    if series == "logRV":
        return np.asarray(observed_all(spec, "ms_sv", seed, cache_exp)["clean"]["logrv"], float)
    return np.asarray(simulate_base(spec, "lorenz", seed, cache_exp)["x"], float)


def _compute_e3d_seed(spec: dict, series: str, seed: int, cache_exp: str, pool) -> pd.DataFrame:
    t0 = time.time()
    torch = rp_batch._torch()
    x = _e3d_clean_series(spec, series, seed, cache_exp)
    W, m, H = int(spec["W_primary"]), int(spec["m"]), int(spec["lrqa_H"])
    ends = feature_ends(spec, len(x))
    g = Geometry(scaling="wz", distance="euclid", threshold="rr", rr=float(spec["rr"]), theiler=int(spec["theiler"]), m=m, tau=int(spec["tau"]))
    clean_w = _window_stack(x, ends, W)
    D_clean, R_clean = _D_R(clean_w, g)
    nD = torch.linalg.vector_norm(D_clean, dim=(1, 2))
    R_clean_np = R_clean.cpu().numpy()
    records = [("clean", "clean", np.zeros(len(ends)), np.zeros(len(ends)), np.ones(len(ends)), submit_features(pool, R_clean_np, g, H))]

    pats = [_pattern_def(p) for p in spec["e3d"]["patterns"]]
    pending = []
    for p in pats:
        rng = _rng(10 + E3D_SERIES.index(series), seed, _crc(p["name"]))
        if p["kind"] == "missing":
            xo, _ = contaminate.mask_missing(x, p["frac"], p["mode"], 10, rng=rng)
            wo = _window_stack(xo, ends, W)
            miss = (~np.isfinite(wo)).mean(axis=1)
            fills = {}
            for meth in FILL_METHODS:
                if meth == "hankel":
                    chunks = np.array_split(np.arange(len(wo)), max(1, len(wo) // 20))
                    fills[meth] = [pool.submit(_fill_windows, wo[c], meth, m) for c in chunks]
                else:
                    fills[meth] = _fill_windows(wo, meth, m)
            pending.append((p, miss, fills))
        else:
            xs = contaminate.make_stale(x, p["frac"], "random", rng)
            ws = _window_stack(xs, ends, W)
            stale_frac = np.mean(ws != clean_w, axis=1)
            pending.append((p, stale_frac, {"none": ws}))

    for p, miss, fills in pending:
        for meth, val in fills.items():
            wf = np.concatenate([f.result() for f in val]) if isinstance(val, list) else val
            valid = np.isfinite(wf).all(axis=1)
            wf = np.where(valid[:, None], wf, clean_w)
            D_f, R_f = _D_R(wf, g)
            rel = (torch.linalg.vector_norm(D_f - D_clean, dim=(1, 2)) / nD.clamp(min=1e-300)).cpu().numpy()
            jac = rp_batch.jaccard(R_clean, R_f, g.theiler_band())
            rel = np.where(valid, rel, np.nan)
            jac = np.where(valid, jac, np.nan)
            records.append((p["name"], meth, miss, rel, jac, submit_features(pool, R_f.cpu().numpy(), g, H), valid))
    frames = []
    for rec in records:
        pname, meth, miss, rel, jac, fut = rec[:6]
        valid = rec[6] if len(rec) > 6 else np.ones(len(ends), bool)
        feats = fut.result()
        feats[~valid] = np.nan
        df = pd.DataFrame(feats.astype(np.float32), columns=FEAT_COLS)
        df.insert(0, "jaccard", np.asarray(jac, np.float32))
        df.insert(0, "rel_error", np.asarray(rel, np.float32))
        df.insert(0, "frac_in_window", np.asarray(miss, np.float32))
        df.insert(0, "end", ends)
        df.insert(0, "method", meth)
        df.insert(0, "pattern", pname)
        frames.append(df)
    print(f"E3d {series} seed {seed}: {sum(len(f) for f in frames)} windows in {time.time() - t0:.1f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def e3d_seed(spec: dict, series: str, seed: int, cache_exp: str, pool=None) -> pd.DataFrame:
    key = {"artifact": "e3d_seed", "series": series, "seed": int(seed), "e3d": spec["e3d"],
           **{k: spec.get(k) for k in ["simulator", "synthetic", "W_primary", "m", "tau", "rr", "theiler", "step", "train_days", "lrqa_H", "backend", "torch_dtype"]}}

    def _fn():
        if pool is None:
            raise RuntimeError("E3d seed not cached and no worker pool given")
        return _compute_e3d_seed(spec, series, seed, cache_exp, pool)

    return cached(cache_exp, f"e3d_{series}_s{int(seed)}", key, _fn, modules=E3_MODULES)


def run_e3d(spec: dict, out_dir: Path, cache_exp: str, pool) -> dict:
    from condpred import report

    per_seed = []
    for series in E3D_SERIES:
        frames = {seed: e3d_seed(spec, series, seed, cache_exp, pool) for seed in spec["seeds"]}
        clean = pd.concat([f[f["method"] == "clean"] for f in frames.values()])
        sd = clean[FEAT_COLS].std(ddof=1)
        valid_sd = np.isfinite(sd.to_numpy()) & (sd.to_numpy() > 0)
        for seed, f in frames.items():
            c = f[f["method"] == "clean"].set_index("end")[FEAT_COLS]
            for (pname, meth), g in f[f["method"] != "clean"].groupby(["pattern", "method"], sort=False):
                cf = c.reindex(g["end"].to_numpy()).to_numpy(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    d = np.abs(g[FEAT_COLS].to_numpy(float) - cf) / np.where(valid_sd, sd.to_numpy(), np.nan)
                dm = _ignore_empty_slice(np.nanmean, d, axis=0)
                rec = {"series": series, "seed": seed, "pattern": pname, "method": meth,
                       "frac_in_window": float(g["frac_in_window"].mean()),
                       "rel_error": float(np.nanmean(g["rel_error"])), "jaccard": float(np.nanmean(g["jaccard"])),
                       "n_valid": int(np.isfinite(g["rel_error"]).sum())}
                for j, col in enumerate(FEAT_COLS):
                    if valid_sd[j]:
                        rec[f"drift_{col}"] = float(dm[j])
                wr = [dm[j] for j, col in enumerate(FEAT_COLS) if col in WR_COLS and valid_sd[j] and np.isfinite(dm[j])]
                lr = [dm[j] for j, col in enumerate(FEAT_COLS) if col in LR_COLS and valid_sd[j] and np.isfinite(dm[j])]
                rec["drift_WRQA_mean"] = float(np.mean(wr)) if wr else np.nan
                rec["drift_LRQA_mean"] = float(np.mean(lr)) if lr else np.nan
                per_seed.append(rec)
    ps = pd.DataFrame(per_seed)
    pdef = pd.DataFrame([_pattern_def(p) for p in spec["e3d"]["patterns"]]).rename(columns={"name": "pattern"})
    metric_cols = [c for c in ps.columns if c in ("frac_in_window", "rel_error", "jaccard") or c.startswith("drift_")]
    agg = ps.groupby(["series", "pattern", "method"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = ps.groupby(["series", "pattern", "method"]).size()
    agg = agg.reset_index().merge(pdef, on="pattern", how="left")
    lead = ["series", "pattern", "kind", "mode", "frac", "method", "n_seeds"]
    agg = agg[lead + [c for c in agg.columns if c not in lead]]
    report.save_table(agg.sort_values(["series", "kind", "mode", "frac", "method"]).reset_index(drop=True), out_dir, "missing_fill")
    return {"per_seed": ps, "table": agg}


def evaluate_e3d(spec: dict, e3d: dict) -> list[dict]:
    ps = e3d["per_seed"]
    B, sm = int(spec["bootstrap"]["B"]), bool(spec["smoke"])
    hyp = spec["hypotheses"]
    pats = [p for p in ["M_random_0.4", "M_random_0.6"] if p in spec["e3d"]["patterns"]]
    rows = []
    for series in E3D_SERIES:
        if not pats:
            rows.append(_not_evaluable(f"H6_{series}", hyp["H6"]["rule"], "M_random_0.4/0.6", sm))
            rows.append(_not_evaluable(f"H7_{series}", hyp["H7"]["rule"], "M_random_0.4/0.6", sm))
            continue
        s = ps[ps["series"] == series]

        def diff(pat, col):
            a = s[(s["pattern"] == pat) & (s["method"] == "linear")].set_index("seed")[col]
            b = s[(s["pattern"] == pat) & (s["method"] == "hankel")].set_index("seed")[col]
            return _diff(a, b)

        res6 = [(p, *seed_boot(diff(p, "rel_error"), B, _crc(f"H6{series}{p}"))) for p in pats]
        ok6 = all(r[2] > 0 for r in res6)
        worst = min(res6, key=lambda r: r[2])
        rows.append(_row(f"H6_{series}", f"err_linear - err_hankel ({worst[0]}, lowest CI)", worst[1], worst[2], worst[3],
                         hyp["H6"]["rule"], "supported" if ok6 else "not_supported",
                         "; ".join(f"{r[0]}: {r[1]:+.4f} [{r[2]:+.4f}, {r[3]:+.4f}]" for r in res6), sm))
        res7 = [(p, col, *seed_boot(diff(p, f"drift_{col}"), B, _crc(f"H7{series}{p}{col}"))) for p in pats for col in ["DET", "LAM"]]
        ok7 = all(r[3] > 0 for r in res7)
        worst = min(res7, key=lambda r: r[3])
        rows.append(_row(f"H7_{series}", f"drift_linear - drift_hankel ({worst[0]} {worst[1]}, lowest CI)", worst[2], worst[3], worst[4],
                         hyp["H7"]["rule"], "supported" if ok7 else "not_supported",
                         "; ".join(f"{r[0]} {r[1]}: {r[2]:+.4f} [{r[3]:+.4f}, {r[4]:+.4f}]" for r in res7), sm))
    return rows


def e3c_row(spec: dict) -> dict:
    return _row("E3c", "", np.nan, np.nan, np.nan, "runs only if G1 passed", "not_run", spec["E3c"], bool(spec["smoke"]))


# ----------------------------------------------------------------------------- leakage


def run_leakage_e3(spec: dict) -> list[tuple[str, bool, str]]:
    """Perturb x[t+1:]; every arm's WRQA/LRQA and every within-window fill must not change.

    Also checks that the GPU RP + worker feature path equals the numpy reference on the same windows.
    """
    sim = dict(spec["simulator"], n_days=300)
    base = simulate_ms_sv(12345, sim)
    obs = daily_series(intraday_returns(base, "normal", int(sim["bars_per_day"])))
    W, m, tau, H = 60, int(spec["m"]), int(spec["tau"]), int(spec["lrqa_H"])
    lspec = dict(spec, train_days=150)
    t_list = [200, 251, 298]
    rng = np.random.default_rng(0)
    out = []
    one = ProcessPoolExecutor(max_workers=1)
    for arm in spec["arms"]["ms_sv"]:
        x = np.asarray(obs[arm["input"]], float)

        def feat(arr, t, arm=arm):
            g = prepare_geom(arm, arr, W, lspec)
            return {**window_rqa(arr, t, W, m, tau, g), **local_rqa(arr, t, W, m, tau, g, H=H)}

        out.append((f"leak_{arm['name']}", leakage_test(feat, x, t_list, rng), "WRQA+LRQA numpy, geometry fit on x[:150]"))
        g = prepare_geom(arm, x, W, lspec)
        ends = np.asarray(t_list)
        R = window_rp_batch(x, ends, W, g)
        got = submit_features(one, R, g, H).result()
        ref = np.array([[feat(x, int(t)).get(c, np.nan) for c in FEAT_COLS] for t in ends])
        same = np.allclose(got, ref, rtol=1e-9, atol=1e-12, equal_nan=True)
        out.append((f"parity_{arm['name']}", bool(same), "GPU RP + worker features vs numpy window_rqa/local_rqa"))
    one.shutdown()
    xl = np.asarray(obs["logrv"], float).copy()
    xl[np.random.default_rng(1).choice(len(xl), 120, replace=False)] = np.nan
    gw = Geometry(scaling="wz", rr=float(spec["rr"]), theiler=int(spec["theiler"]), m=m, tau=tau)
    for meth in FILL_METHODS:
        def ffeat(arr, t, meth=meth):
            w = fill_window(arr[t - W + 1 : t + 1], meth, m)
            _, R = gw.recurrence(w)
            return {"w": w, "lr": local_rqa_matrix(R, theiler=gw.theiler_band(), H=H)}

        out.append((f"leak_fill_{meth}", leakage_test(ffeat, xl, t_list, rng), "fill inside window ending at t"))
    return out


# ----------------------------------------------------------------------------- figures


def make_figures(out_dir: Path, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from condpred import report

    fid = pd.read_csv(out_dir / "fidelity.csv")
    fid = fid[(fid["kind"] == "arm_clean_vs_contaminated")]
    W0 = int(fid["W"].mode().iloc[0]) if len(fid) else 120
    for system, fs in fid[fid["W"] == W0].groupby("system"):
        panels = []
        j = fs[fs["factor"] == "jumps"]
        if len(j):
            k5 = j[j["cond"].str.endswith("_k5")]
            panels.append(("jumps (k=5): rate per day / sample", k5))
        for fac, lab in [("noise", "noise omega"), ("missing_random", "missing random frac (ffill)"),
                         ("missing_block", "missing block frac (ffill)"), ("stale", "stale frac")]:
            if (fs["factor"] == fac).any():
                panels.append((lab, fs[fs["factor"] == fac]))
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), sharey=True, squeeze=False)
        for ax, (lab, d) in zip(axes[0], panels):
            for arm, da in d.groupby("arm"):
                da = da.sort_values("level")
                ax.errorbar(da["level"], da["jaccard_mean"], yerr=da["jaccard_sd"], marker="o", ms=3, lw=1, capsize=2, label=arm)
            ax.set_xlabel(lab)
            ax.set_title(system)
        axes[0][0].set_ylabel("Jaccard(clean RP, contaminated RP)")
        axes[0][-1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        report.save_fig(fig, fig_dir, f"fidelity_{system}")
        plt.close(fig)

    auc = pd.read_csv(out_dir / "regime_auc.csv")
    a = auc[(auc["train_on"] == "same") & (auc["W"] == W0)]
    if len(a):
        piv = a.pivot_table(index="model", columns="cond", values="auc_pooled")
        order = [c for c in ["clean"] + sorted(set(piv.columns) - {"clean"}) if c in piv.columns]
        piv = piv[order]
        fig, ax = plt.subplots(figsize=(0.45 * len(order) + 3, 0.3 * len(piv) + 2))
        im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(piv)), piv.index, fontsize=7)
        fig.colorbar(im, ax=ax, label="regime AUC (pooled test seeds)")
        ax.set_title(f"E3b regime AUC, W={W0}, trained on same condition")
        report.save_fig(fig, fig_dir, "regime_auc_heatmap")
        plt.close(fig)

    mf = pd.read_csv(out_dir / "missing_fill.csv")
    mm = mf[mf["kind"] == "missing"]
    if len(mm):
        combos = list(mm.groupby(["series", "mode"]).groups)
        fig, axes = plt.subplots(1, len(combos), figsize=(4.2 * len(combos), 4), squeeze=False)
        for ax, (series, mode) in zip(axes[0], combos):
            d = mm[(mm["series"] == series) & (mm["mode"] == mode)]
            for meth, dm in d.groupby("method"):
                dm = dm.sort_values("frac")
                ax.errorbar(dm["frac"], dm["rel_error_mean"], yerr=dm["rel_error_std"], marker="o", ms=3, capsize=2, label=meth)
            ax.set_title(f"{series}, {mode} missing")
            ax.set_xlabel("missing fraction")
        axes[0][0].set_ylabel("||D_fill - D_clean||_F / ||D_clean||_F")
        axes[0][-1].legend()
        report.save_fig(fig, fig_dir, "e3d_rel_error")
        plt.close(fig)
