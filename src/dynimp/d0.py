"""D0 gate of the dynamics-preserving imputation direction.

Question: do the imputers that win on pointwise error also keep the recurrence structure (DET, LAM) of the
complete series? Every imputer fills the same 96-step windows. RQA is computed on the stitched test series and
compared with the RQA of the complete series.

Run from the repo root with ``PYTHONPATH=src``:

    .venv-dynimp/Scripts/python.exe -m dynimp.d0 --design       truth-only design tables, no imputer is run
    .venv-dynimp/Scripts/python.exe -m dynimp.d0 [--workers N]  spec.json first, then all units, then decision.csv

``.venv-dynimp`` is a venv on the conda base interpreter with ``--system-site-packages`` (conda's CUDA torch) plus PyPOTS.

``CONDPRED_SMOKE=1`` runs a tiny version into separate ``smoke/`` folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy import stats

from condpred import completion, contaminate, systems
from condpred.config import code_fingerprint, smoke, spec_hash
from condpred.report import save_fig, save_table, write_decision, write_spec
from recplay.embedding import average_mutual_information, embed, false_nearest_fraction, first_local_minimum
from recplay.recurrence import recurrence_matrix
from recplay.rqa import rqa

REPO = Path(__file__).resolve().parents[2]
MODULES = ["dynimp.d0", "condpred.completion", "condpred.contaminate", "condpred.systems",
           "recplay.embedding", "recplay.recurrence", "recplay.rqa"]
SPEC_VERSION = 1

SYSTEMS = ["lorenz", "rossler", "mackey_glass", "etth1"]
SYNTHETIC = ["lorenz", "rossler", "mackey_glass"]
_SYS_ID = {s: i + 1 for i, s in enumerate(SYSTEMS)}
_PATTERN_ID = {"point": 1, "block": 2}
ETT_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
RQA_CHANNEL = {"lorenz": 0, "rossler": 0, "mackey_glass": 0, "etth1": 6}
GATE_ARMS = ["linear", "hankel", "saits", "csdi"]
ARMS = GATE_ARMS + ["csdi_sample"]
POINT_ESTIMATORS = ["saits", "csdi"]
RQA_COLS = ["DET", "LAM", "L", "ENTR"]
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

N_STEPS = 96
RQA_W = 480
RR = 0.10
SPLITS = {"synthetic": (5760, 1440, 3840), "etth1": (8640, 2880, 2880)}
NOISE_CANDIDATES = [0.0, 0.05, 0.1, 0.2]

# Set from the --design step of 12 Nov 2026 (complete series only, no gaps and no imputers; design.json and
# design_truth_rqa.csv in the output folder). Embedding: AMI first minimum and false nearest neighbours.
# Noise: smallest positive candidate at which the mean DET of the complete test series is at most 0.95 in all
# three synthetic systems, so that an increase in DET is possible (0.05 gives 0.950 for Rossler; 0.10 gives
# 0.886 Lorenz, 0.901 Rossler, 0.833 Mackey-Glass).
EMBEDDING: dict[str, tuple[int, int]] | None = {"lorenz": (3, 4), "rossler": (3, 7), "mackey_glass": (3, 4), "etth1": (4, 15)}
NOISE_PRIMARY: float | None = 0.10

log = logging.getLogger("dynimp.d0")


# ----------------------------------------------------------------------------- paths


def out_dir() -> Path:
    base = REPO / "notebooks" / "tables" / "dynimp" / "D0"
    return base / "smoke" if smoke(False, True) else base


def cache_dir() -> Path:
    return REPO / "data" / "processed" / "dynimp" / "D0" / smoke("full", "smoke")


# ----------------------------------------------------------------------------- data


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng([11, *[int(p) for p in parts]])


def _crc(s: str) -> int:
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def mackey_glass(n, rng, beta=0.2, gamma=0.1, power=10, delay=17.0, h=0.1, sample_every=30, burn=10000):
    """x' = beta x(t - delay) / (1 + x(t - delay)^power) - gamma x, RK4 with step h, one sample per 3 time units.

    The delayed value at a half step is the mean of its two grid neighbours. History starts constant.
    """
    d = int(round(delay / h))
    total = burn + n * sample_every
    x = np.empty(d + 1 + total)
    x[: d + 1] = rng.uniform(0.6, 1.4)
    f = lambda v, vd: beta * vd / (1.0 + vd**power) - gamma * v
    for t in range(d, d + total):
        v, vd0, vd1 = x[t], x[t - d], x[t - d + 1]
        vdh = 0.5 * (vd0 + vd1)
        k1 = f(v, vd0)
        k2 = f(v + 0.5 * h * k1, vdh)
        k3 = f(v + 0.5 * h * k2, vdh)
        k4 = f(v + h * k3, vd1)
        x[t + 1] = v + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x[d + 1 + burn :: sample_every][:n, None]


def etth1() -> np.ndarray:
    """ETTh1 (Zhou et al. 2021), first 14,400 hours (the usual 12/4/4-month split). Downloaded once; SHA-256 checked."""
    folder = REPO / "data" / "raw" / "ett"
    path, manifest = folder / "ETTh1.csv", folder / "manifest.json"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        raw = urlopen(ETT_URL, timeout=120).read()
        path.write_bytes(raw)
        manifest.write_text(json.dumps({"file": "ETTh1.csv", "url": ETT_URL, "sha256": hashlib.sha256(raw).hexdigest(),
                                        "downloaded_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if json.loads(manifest.read_text(encoding="utf-8"))["sha256"] != sha:
        raise RuntimeError("ETTh1.csv does not match the SHA-256 in data/raw/ett/manifest.json")
    return pd.read_csv(path)[ETT_COLS].to_numpy(float)[:14400]


def split_sizes(system: str) -> tuple[int, int, int]:
    return SPLITS["etth1" if system == "etth1" else "synthetic"]


def complete_series(system: str, noise, seed: int) -> np.ndarray:
    """The series without gaps, shape [n, F]. Synthetic: clean trajectory plus observation noise of ``noise`` sd."""
    if system == "etth1":
        return etth1()
    n = sum(split_sizes(system))
    rng = _rng(1, _SYS_ID[system], seed)
    if system == "lorenz":
        x = systems.lorenz63(n, rng)[0]
    elif system == "rossler":
        x = systems.rossler(n, rng)
    else:
        x = mackey_glass(n, rng)
    if noise:
        sd = x[: split_sizes(system)[0]].std(axis=0)
        x = x + float(noise) * sd * _rng(2, _SYS_ID[system], seed).standard_normal(x.shape)
    return x


def missing_mask(n: int, n_features: int, frac: float, pattern: str, block_len: int, *rng_parts) -> np.ndarray:
    """True where missing. Each channel is masked on its own."""
    miss = np.zeros((n, n_features), bool)
    mode = "random" if pattern == "point" else "block"
    for c in range(n_features):
        _, observed = contaminate.mask_missing(np.zeros(n), frac, mode, block_len, rng=_rng(*rng_parts, c))
        miss[:, c] = ~observed
    return miss


def windows(x: np.ndarray, stride: int) -> np.ndarray:
    starts = range(0, len(x) - N_STEPS + 1, stride)
    return np.stack([x[s : s + N_STEPS] for s in starts]).astype(np.float32)


def unit_data(spec: dict, cell: dict, seed: int) -> dict:
    system, pattern = cell["system"], cell["pattern"]
    sid, pid = _SYS_ID[system], _PATTERN_ID[pattern]
    x = complete_series(system, cell["noise"], seed)
    n_tr, n_va, n_te = split_sizes(system)
    mp = spec["missing"]
    miss = missing_mask(len(x), x.shape[1], mp["frac"], pattern, mp["block_len"], 3, sid, seed, pid)
    xo = np.where(miss, np.nan, x)
    mu, sd = np.nanmean(xo[:n_tr], axis=0), np.nanstd(xo[:n_tr], axis=0)
    z_true, z_obs = (x - mu) / sd, (xo - mu) / sd

    va = slice(n_tr, n_tr + n_va)
    held = missing_mask(n_va, x.shape[1], mp["val_holdout_frac"], pattern, mp["block_len"], 4, sid, seed, pid) & ~miss[va]
    te = slice(n_tr + n_va, n_tr + n_va + n_te)
    return {
        "train": windows(z_obs[:n_tr], spec["windows"]["train_stride"]),
        "val_in": windows(np.where(held, np.nan, z_obs[va]), spec["windows"]["val_stride"]),
        "val_ori": windows(z_obs[va], spec["windows"]["val_stride"]),
        "val_held": windows(held.astype(float), spec["windows"]["val_stride"]) > 0.5,
        "test_in": windows(z_obs[te], N_STEPS),
        "test_true": z_true[te],
        "test_missing": miss[te],
    }


# ----------------------------------------------------------------------------- imputers


def _fill_channels(X: np.ndarray, fn) -> np.ndarray:
    out = X.astype(float).copy()
    for i in range(out.shape[0]):
        for c in range(out.shape[2]):
            w = out[i, :, c]
            if np.isfinite(w).all():
                continue
            out[i, :, c] = np.nan_to_num(fn(w), nan=0.0)
    return out


def _hankel_fill(w: np.ndarray, rows: int, cap: int) -> np.ndarray:
    lin = completion.linear(w)
    if np.isfinite(w).sum() < 2:
        return lin
    s = np.linalg.svd(completion._hankel(lin, rows, 1), compute_uv=False)
    cum = np.cumsum(s**2) / max((s**2).sum(), 1e-300)
    rank = int(min(cap, int(np.searchsorted(cum, 0.99) + 1)))
    return completion.hankel_complete(w, rows, tau=1, rank=rank, n_iter=200, tol=1e-6)


def _fit_meta(model, seconds: float) -> dict:
    return {"best_epoch": getattr(model, "best_epoch", None), "best_loss": getattr(model, "best_loss", None),
            "seconds": round(seconds, 1)}


def _saits(spec: dict, data: dict, device: str):
    from pypots.imputation import SAITS

    h = spec["saits"]
    n, L, F = data["train"].shape
    model = SAITS(n_steps=L, n_features=F, n_layers=h["n_layers"], d_model=h["d_model"], n_heads=h["n_heads"],
                  d_k=h["d_k"], d_v=h["d_v"], d_ffn=h["d_ffn"], dropout=h["dropout"], batch_size=h["batch_size"],
                  epochs=h["epochs"], patience=h["patience"], device=device, saving_path=None, verbose=False)
    t0 = time.time()
    model.fit({"X": data["train"]}, {"X": data["val_in"], "X_ori": data["val_ori"]})
    imp = model.predict({"X": data["test_in"]})["imputation"]
    return np.asarray(imp, float), _fit_meta(model, time.time() - t0)


def _csdi(spec: dict, data: dict, device: str):
    from pypots.imputation import CSDI

    h = spec["csdi"]
    n, L, F = data["train"].shape
    model = CSDI(n_steps=L, n_features=F, n_layers=h["n_layers"], n_heads=h["n_heads"], n_channels=h["n_channels"],
                 d_time_embedding=h["d_time_embedding"], d_feature_embedding=h["d_feature_embedding"],
                 d_diffusion_embedding=h["d_diffusion_embedding"], n_diffusion_steps=h["n_diffusion_steps"],
                 target_strategy=h["target_strategy"], schedule=h["schedule"], beta_start=h["beta_start"],
                 beta_end=h["beta_end"], batch_size=h["batch_size"], epochs=h["epochs"], patience=h["patience"],
                 device=device, saving_path=None, verbose=False)
    t0 = time.time()
    model.fit({"X": data["train"]}, {"X": data["val_in"], "X_ori": data["val_ori"]})
    samples = model.predict({"X": data["test_in"]}, n_sampling_times=h["n_sampling_times"])["imputation"]
    samples = np.asarray(samples, float)  # [windows, samples, steps, features]
    return np.median(samples, axis=1), samples[:, 0], _fit_meta(model, time.time() - t0)


# ----------------------------------------------------------------------------- RQA


def window_rqa(x: np.ndarray, m: int, tau: int) -> pd.DataFrame:
    """RQA of consecutive windows of RQA_W samples: window z-score, Euclidean, RR fixed, Theiler (m - 1) tau."""
    rows = []
    for w in range(len(x) // RQA_W):
        seg = np.asarray(x[w * RQA_W : (w + 1) * RQA_W], float)
        z = (seg - seg.mean()) / seg.std()
        r = rqa(recurrence_matrix(embed(z, m, tau), RR, theiler=(m - 1) * tau))
        rows.append({"window": w, **{k: float(r[k]) for k in RQA_COLS}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- one unit


def run_unit(spec: dict, cell: dict, seed: int) -> dict:
    """Train and apply every imputer for one (cell, seed). Returns per-arm errors and per-window RQA."""
    import torch
    from pypots.utils.logging import logger as pypots_logger
    from pypots.utils.random import set_random_seed

    pypots_logger.setLevel(logging.ERROR)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    data = unit_data(spec, cell, seed)
    X, miss, truth = data["test_in"], data["test_missing"], data["test_true"]
    n_te, F = truth.shape

    checks = [
        ("test_inputs_hide_truth", bool(np.isnan(X.reshape(n_te, F)[miss]).all())),
        ("val_inputs_hide_held_out", bool(np.isnan(data["val_in"][data["val_held"]]).all())),
        ("val_targets_are_observed", bool(np.isfinite(data["val_ori"][data["val_held"]]).all())),
        ("train_has_no_complete_truth", bool(np.isnan(data["train"]).any())),
    ]

    hk = spec["hankel"]
    filled = {"linear": _fill_channels(X, completion.linear),
              "hankel": _fill_channels(X, lambda w: _hankel_fill(w, hk["rows"], hk["rank_cap"]))}
    fit = {}
    set_random_seed(_crc(f"saits{cell['name']}{seed}") % 2**31)
    filled["saits"], fit["saits"] = _saits(spec, data, device)
    set_random_seed(_crc(f"csdi{cell['name']}{seed}") % 2**31)
    filled["csdi"], filled["csdi_sample"], fit["csdi"] = _csdi(spec, data, device)

    m, tau = spec["embedding"][cell["system"]]
    ch = RQA_CHANNEL[cell["system"]]
    arm_rows = [{"arm": "zero", "mae": float(np.abs(truth[miss]).mean()), "rmse": float(np.sqrt((truth[miss] ** 2).mean()))}]
    win = [window_rqa(truth[:, ch], m, tau).assign(arm="complete")]
    for arm in ARMS:
        z = np.where(miss, filled[arm].reshape(n_te, F), truth)
        checks.append((f"output_finite_{arm}", bool(np.isfinite(z).all())))
        err = z[miss] - truth[miss]
        arm_rows.append({"arm": arm, "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err**2).mean()))})
        win.append(window_rqa(z[:, ch], m, tau).assign(arm=arm))
    return {"cell": cell["name"], "seed": int(seed), "arms": pd.DataFrame(arm_rows), "windows": pd.concat(win, ignore_index=True),
            "fit": fit, "checks": checks, "missing_frac_test": float(miss.mean()), "seconds": round(time.time() - t0, 1)}


def run_unit_cached(spec: dict, cell: dict, seed: int) -> dict:
    key = spec_hash(spec) + code_fingerprint(MODULES)
    h12 = hashlib.sha256(key.encode()).hexdigest()[:12]
    path = cache_dir() / f"{cell['name'].replace('|', '_')}_s{seed}_{h12}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    res = run_unit(spec, cell, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(res))
    tmp.replace(path)
    return res


# ----------------------------------------------------------------------------- spec


def _cell(system, noise, pattern, role) -> dict:
    tag = "data" if noise is None else f"n{noise:g}"
    return {"name": f"{system}|{tag}|{pattern}", "system": system, "noise": noise, "pattern": pattern, "role": role}


def build_spec() -> dict:
    if EMBEDDING is None or NOISE_PRIMARY is None:
        raise RuntimeError("EMBEDDING / NOISE_PRIMARY are not set: run --design first and copy its choices into d0.py")
    cells = []
    for s in SYNTHETIC:
        cells += [_cell(s, NOISE_PRIMARY, "point", "primary"), _cell(s, NOISE_PRIMARY, "block", "secondary"),
                  _cell(s, 0.0, "point", "secondary")]
    cells += [_cell("etth1", None, "point", "primary"), _cell("etth1", None, "block", "secondary")]
    epochs, patience = smoke(200, 2), smoke(20, 2)
    return {
        "version": SPEC_VERSION,
        "experiment": "dynimp D0 gate: pointwise error vs preservation of recurrence structure under imputation",
        "smoke": bool(smoke(False, True)),
        "seeds": {"primary": list(range(smoke(8, 2))), "secondary": list(range(smoke(4, 1)))},
        "seed_meaning": "synthetic: trajectory, observation noise, masks and model initialisation all change with the seed; "
        "etth1: one fixed series, only masks and model initialisation change, so its intervals are conditional on the data",
        "cells": cells,
        "systems": {
            "lorenz": "condpred.systems.lorenz63 defaults (dt 0.01, every 5th sample), channels x, y, z",
            "rossler": "condpred.systems.rossler defaults (dt 0.05, every 4th sample), channels x, y, z",
            "mackey_glass": "beta 0.2, gamma 0.1, power 10, delay 17, RK4 h 0.1, one sample per 3 time units, 1 channel",
            "etth1": f"{ETT_URL}, first 14,400 hours, 7 channels, SHA-256 in data/raw/ett/manifest.json",
        },
        "noise": {"primary": NOISE_PRIMARY, "unit": "fraction of each channel's clean train sd, Gaussian, independent",
                  "secondary": 0.0, "candidates": NOISE_CANDIDATES,
                  "choice_rule": "smallest positive candidate at which the mean DET of the complete test series is at most "
                  "0.95 in all three synthetic systems (design_truth_rqa.csv); set before any imputer was run"},
        "splits": SPLITS,
        "missing": {"frac": 0.4, "point": "MCAR per entry, each channel on its own", "block": "runs of block_len per channel",
                    "block_len": 10, "applied_to": "train, validation and test alike",
                    "val_holdout_frac": 0.2,
                    "val_holdout": "same pattern applied again to the validation series; held-out entries that were observed "
                    "are the validation targets, so no truly missing value is used for model selection"},
        "scaling": "z-score per channel with mean and sd of the observed train entries; errors and RQA are in these units",
        "windows": {"n_steps": N_STEPS, "train_stride": 4, "val_stride": 24, "test_stride": N_STEPS,
                    "train_stride_note": "4 rather than 8 to give CSDI about 8,900 gradient steps in 200 epochs; chosen from a "
                    "runtime benchmark, not from results",
                    "context": "every imputer fills one 96-step window at a time; test windows do not overlap and are stitched"},
        "arms": {
            "linear": "condpred.completion.linear per channel",
            "hankel": "condpred.completion.hankel_complete per channel (rows, tau 1, n_iter 200, tol 1e-6)",
            "saits": "PyPOTS SAITS; trained with MAE on ORT + MIT (PyPOTS default), not squared error",
            "csdi": "PyPOTS CSDI; median over n_sampling_times samples (the usual point estimate)",
            "csdi_sample": "first of the same CSDI samples; secondary arm, not part of the gate",
        },
        "hankel": {"rows": 32, "rank_cap": 10, "rank": "smallest rank with 99 % energy of the linear-fill Hankel matrix, capped"},
        "saits": {"n_layers": 2, "d_model": 128, "n_heads": 4, "d_k": 32, "d_v": 32, "d_ffn": 256, "dropout": 0.1,
                  "batch_size": 32, "epochs": epochs, "patience": patience, "optimizer": "PyPOTS default (Adam, lr 1e-3)"},
        "csdi": {"n_layers": 4, "n_heads": 4, "n_channels": 64, "d_time_embedding": 128, "d_feature_embedding": 16,
                 "d_diffusion_embedding": 128, "n_diffusion_steps": 50, "target_strategy": "random", "schedule": "quad",
                 "beta_start": 1e-4, "beta_end": 0.5, "batch_size": 32, "epochs": epochs, "patience": patience,
                 "n_sampling_times": smoke(20, 3), "optimizer": "PyPOTS default (Adam, lr 1e-3)"},
        "hyperparameters": "fixed in advance, the same for every system; no search",
        "rqa": {"channel": RQA_CHANNEL, "embedding": "see 'embedding' (m, tau); AMI first local minimum (lags <= 40) and first "
                "m with false-nearest fraction < 0.05 (max 8), on the first 5000 train samples of seed 0 without gaps",
                "W": RQA_W, "scaling": "window z-score", "distance": "euclid", "rr": RR, "theiler": "(m - 1) * tau", "lmin": 2,
                "reference": "RQA of the complete test series (with observation noise): what the analysis would have given without gaps"},
        "embedding": {k: list(v) for k, v in EMBEDDING.items()},
        "metrics": {
            "mae": "mean |imputed - true| over the missing test entries, all channels",
            "drift_f": "f(imputed series) - f(complete series) per RQA window, signed; mean over windows per seed",
            "J": "mean over windows of (|drift_DET| + |drift_LAM|) / 2 per seed",
        },
        "interval": "95 % t-interval of the across-seed mean (paired differences for comparisons); unit = seed",
        "gate": {
            "primary_cells": "the four cells with role 'primary': point pattern; synthetic at the primary noise level, etth1 as is",
            "part1": "ranking by MAE differs from ranking by J. In a system, a reversal is a pair (a, b) of gate arms with "
            "MAE_a < MAE_b and J_a > J_b, both paired differences with a 95 % interval excluding zero. "
            "Part 1 passes iff at least 2 of the 4 primary systems have a reversal.",
            "part2": "point estimators inflate DET. Point estimators are saits and csdi (median). In a system an arm "
            "inflates DET iff the 95 % interval of its signed DET drift lies above zero. Part 2 passes iff saits and csdi "
            "each inflate DET in at least 3 of the 4 primary systems. The request's wording 'squared-error imputers' is "
            "operationalised as this set.",
            "evaluable": "a primary cell is evaluable iff saits and csdi both have a lower mean MAE than filling with zero "
            "(the train mean); if any primary cell is not evaluable the gate verdict is not_evaluable",
            "D0": "pass iff part 1 and part 2 pass",
            "gate_arms": GATE_ARMS,
        },
        "secondary": "block pattern, noise-free synthetic, LAM / L / ENTR drift, linear and hankel signed drift, and "
        "csdi_sample against csdi: reported with intervals, exploratory, not part of the gate",
        "leakage": "imputers receive NaN at every missing entry; truth is used only for scoring; validation targets are observed entries",
    }


# ----------------------------------------------------------------------------- design (truth only)


def choose_embedding(x: np.ndarray) -> tuple[int, int, list[float]]:
    x = np.asarray(x, float)[:5000]
    tau = first_local_minimum(average_mutual_information(x, 40)) or 1
    fnn = false_nearest_fraction(x, tau, max_m=8)
    m = next((i + 1 for i, f in enumerate(fnn) if f < 0.05), 8)
    return int(m), int(tau), [float(f) for f in fnn]


def design() -> None:
    """Embedding per system and RQA level of the complete test series per noise candidate. No gaps, no imputers."""
    out = out_dir()
    emb, rows = {}, []
    for system in SYSTEMS:
        n_tr, n_va, n_te = split_sizes(system)
        ch = RQA_CHANNEL[system]
        m, tau, fnn = choose_embedding(complete_series(system, 0.0, 0)[:n_tr, ch])
        emb[system] = {"m": m, "tau": tau, "fnn": fnn}
        for noise in [None] if system == "etth1" else NOISE_CANDIDATES:
            for seed in [0] if system == "etth1" else [0, 1, 2]:
                x = complete_series(system, noise, seed)[n_tr + n_va :, ch]
                r = window_rqa(x, m, tau)[RQA_COLS].mean()
                rows.append({"system": system, "noise": noise, "seed": seed, "m": m, "tau": tau, **r.to_dict()})
        log.info("%s: m %d tau %d", system, m, tau)
    df = pd.DataFrame(rows)
    save_table(df, out, "design_truth_rqa")
    summary = df.groupby(["system", "noise"], dropna=False)[RQA_COLS].mean().round(4)
    log.info("complete-series RQA (mean over seeds and test windows):\n%s", summary.to_string())
    (out / "design.json").write_text(json.dumps({"embedding": emb, "rr": RR, "W": RQA_W,
                                                 "written_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------- evaluation


def t_interval(v) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), np.nan, np.nan
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def seed_table(results: list[dict]) -> pd.DataFrame:
    """One row per (cell, seed, arm): MAE, RMSE, signed and absolute drift per RQA measure, J."""
    rows = []
    for res in results:
        w = res["windows"].set_index(["arm", "window"])
        ref = w.loc["complete"]
        errs = res["arms"].set_index("arm")
        for arm in ARMS:
            d = w.loc[arm][RQA_COLS] - ref[RQA_COLS]
            rec = {"cell": res["cell"], "seed": res["seed"], "arm": arm, "mae": errs.loc[arm, "mae"], "rmse": errs.loc[arm, "rmse"],
                   "mae_zero": errs.loc["zero", "mae"], "J": float((d["DET"].abs() + d["LAM"].abs()).mean() / 2)}
            for f in RQA_COLS:
                rec[f"drift_{f}"] = float(d[f].mean())
                rec[f"absdrift_{f}"] = float(d[f].abs().mean())
                rec[f"complete_{f}"] = float(ref[f].mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def evaluate(spec: dict, seeds: pd.DataFrame) -> dict:
    cells = {c["name"]: c for c in spec["cells"]}
    sm = bool(spec["smoke"])
    summary_rows, pair_rows, decision = [], [], []
    for name, g in seeds.groupby("cell", sort=False):
        for arm, ga in g.groupby("arm", sort=False):
            rec = {"cell": name, "role": cells[name]["role"], "arm": arm, "n_seeds": len(ga)}
            for col in ["mae", "rmse", "J"] + [f"drift_{f}" for f in RQA_COLS] + [f"absdrift_{f}" for f in RQA_COLS]:
                rec[col], rec[f"{col}_lo"], rec[f"{col}_hi"] = t_interval(ga[col])
            for f in RQA_COLS:
                rec[f"complete_{f}"] = float(ga[f"complete_{f}"].mean())
            rec["mae_zero"] = float(ga["mae_zero"].mean())
            summary_rows.append(rec)
        piv = {col: g.pivot(index="seed", columns="arm", values=col) for col in ["mae", "J"]}
        for a, b in list(combinations(GATE_ARMS, 2)) + [("csdi_sample", "csdi")]:
            dm, dm_lo, dm_hi = t_interval(piv["mae"][a] - piv["mae"][b])
            dj, dj_lo, dj_hi = t_interval(piv["J"][a] - piv["J"][b])
            reversal = bool((dm_hi < 0 and dj_lo > 0) or (dm_lo > 0 and dj_hi < 0))
            pair_rows.append({"cell": name, "role": cells[name]["role"], "a": a, "b": b, "d_mae": dm, "d_mae_lo": dm_lo,
                              "d_mae_hi": dm_hi, "d_J": dj, "d_J_lo": dj_lo, "d_J_hi": dj_hi, "reversal": reversal,
                              "in_gate": a in GATE_ARMS and b in GATE_ARMS})
    summary, pairs = pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)

    def row(gate, system, statistic, value, lo, hi, rule, verdict, note=""):
        return {"gate": gate, "system": system, "statistic": statistic, "value": value, "ci_low": lo, "ci_high": hi,
                "rule": rule, "verdict": verdict, "note": note, "smoke": sm}

    primary = [c for c in spec["cells"] if c["role"] == "primary"]
    n_rev, n_infl, evaluable = 0, {a: 0 for a in POINT_ESTIMATORS}, True
    for c in primary:
        s = summary[summary["cell"] == c["name"]].set_index("arm")
        p = pairs[(pairs["cell"] == c["name"]) & pairs["in_gate"]]
        ok = all(s.loc[a, "mae"] < s.loc[a, "mae_zero"] for a in POINT_ESTIMATORS)
        evaluable &= ok
        decision.append(row("evaluable", c["system"], "max MAE of saits, csdi vs MAE of zero fill",
                            float(max(s.loc[a, "mae"] for a in POINT_ESTIMATORS)), np.nan, np.nan,
                            "both below zero-fill MAE", "yes" if ok else "no", f"zero fill {s.loc['saits', 'mae_zero']:.4f}"))
        rev = p[p["reversal"]]
        n_rev += int(len(rev) > 0)
        tau_k = stats.kendalltau(s.loc[GATE_ARMS, "mae"], s.loc[GATE_ARMS, "J"]).statistic
        note = "; ".join(f"{r.a} vs {r.b}: dMAE {r.d_mae:+.4f} [{r.d_mae_lo:+.4f}, {r.d_mae_hi:+.4f}], "
                         f"dJ {r.d_J:+.4f} [{r.d_J_lo:+.4f}, {r.d_J_hi:+.4f}]" for r in rev.itertuples())
        order = lambda col: " < ".join(s.loc[GATE_ARMS, col].sort_values().index)
        decision.append(row("part1_system", c["system"], "number of significant reversals among 6 pairs", int(len(rev)), np.nan,
                            np.nan, "reversal: MAE_a < MAE_b and J_a > J_b, both 95 % intervals exclude zero",
                            "reversal" if len(rev) else "no_reversal",
                            f"MAE order {order('mae')}; J order {order('J')}; Kendall tau {tau_k:+.2f}" + (f"; {note}" if note else "")))
        for a in POINT_ESTIMATORS:
            infl = bool(s.loc[a, "drift_DET_lo"] > 0)
            n_infl[a] += int(infl)
            decision.append(row("part2_system", c["system"], f"signed DET drift of {a}", s.loc[a, "drift_DET"], s.loc[a, "drift_DET_lo"],
                                s.loc[a, "drift_DET_hi"], "inflates iff interval above zero", "inflates" if infl else "does_not_inflate",
                                f"complete-series DET {s.loc[a, 'complete_DET']:.4f}"))
    p1, p2 = n_rev >= 2, all(v >= 3 for v in n_infl.values())
    decision.append(row("part1", "all", "primary systems with a reversal", n_rev, np.nan, np.nan, "at least 2 of 4", "pass" if p1 else "fail"))
    decision.append(row("part2", "all", "primary systems where the arm inflates DET", min(n_infl.values()), np.nan, np.nan,
                        "saits and csdi each in at least 3 of 4", "pass" if p2 else "fail",
                        "; ".join(f"{a}: {v} of 4" for a, v in n_infl.items())))
    verdict = "not_evaluable" if not evaluable else ("pass" if p1 and p2 else "fail")
    if sm:
        verdict = f"{verdict} (smoke: mechanical only)"
    decision.append(row("D0", "all", "part 1 and part 2", float(p1 and p2), np.nan, np.nan, "pass iff both parts pass", verdict))
    return {"summary": summary, "pairs": pairs, "decision": decision}


def figure(spec: dict, summary: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [c for c in spec["cells"] if c["role"] == "primary"]
    fig, axes = plt.subplots(1, len(primary), figsize=(4.2 * len(primary), 3.8), layout="constrained")
    for ax, c in zip(np.atleast_1d(axes), primary):
        s = summary[summary["cell"] == c["name"]].set_index("arm")
        for arm in ARMS:
            ax.errorbar(s.loc[arm, "mae"], s.loc[arm, "J"], xerr=[[s.loc[arm, "mae"] - s.loc[arm, "mae_lo"]], [s.loc[arm, "mae_hi"] - s.loc[arm, "mae"]]],
                        yerr=[[s.loc[arm, "J"] - s.loc[arm, "J_lo"]], [s.loc[arm, "J_hi"] - s.loc[arm, "J"]]], fmt="o", label=arm, capsize=2)
        ax.set_title(c["name"])
        ax.set_xlabel("MAE on missing entries (z units)")
        ax.set_ylabel("J = mean (|dDET| + |dLAM|) / 2")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    save_fig(fig, out / "figures", "mae_vs_drift")
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def _setup_logging(out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(out / name, mode="a", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    out = out_dir()
    if args.design:
        _setup_logging(out, "design.log")
        design()
        return
    _setup_logging(out, smoke("run_full.log", "run_smoke.log"))
    spec = build_spec()
    write_spec(spec, out, modules=MODULES)  # before any imputer is run
    log.info("spec v%s written, hash %s, code %s", spec["version"], spec_hash(spec)[:12], code_fingerprint(MODULES)[:12])

    units = [(c, s) for role in ["primary", "secondary"] for c in spec["cells"] if c["role"] == role for s in spec["seeds"][role]]
    t0, results = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_unit_cached, spec, c, s): (c["name"], s) for c, s in units}
        for k, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            mae = res["arms"].set_index("arm")["mae"]
            log.info("[%d/%d] %s seed %d: %.0f s; epochs saits %s csdi %s; MAE %s", k, len(units), res["cell"], res["seed"], res["seconds"],
                     res["fit"]["saits"]["best_epoch"], res["fit"]["csdi"]["best_epoch"], " ".join(f"{a} {mae[a]:.3f}" for a in mae.index))
    results.sort(key=lambda r: (r["cell"], r["seed"]))
    log.info("all units done in %.0f s", time.time() - t0)

    checks = pd.DataFrame([{"cell": r["cell"], "seed": r["seed"], "check": n, "passed": ok} for r in results for n, ok in r["checks"]])
    save_table(checks, out, "leakage")
    if not checks["passed"].all():
        raise RuntimeError(f"leakage / sanity checks failed:\n{checks[~checks['passed']]}")
    save_table(pd.concat([r["windows"].assign(cell=r["cell"], seed=r["seed"]) for r in results], ignore_index=True), out, "per_window")
    save_table(pd.DataFrame([{"cell": r["cell"], "seed": r["seed"], "missing_frac_test": r["missing_frac_test"], "seconds": r["seconds"],
                              **{f"{a}_{k}": v for a, d in r["fit"].items() for k, v in d.items()}} for r in results]), out, "fit")
    seeds = seed_table(results)
    save_table(seeds, out, "per_seed")
    ev = evaluate(spec, seeds)
    save_table(ev["summary"], out, "summary")
    save_table(ev["pairs"], out, "pairs")
    write_decision(ev["decision"], out)
    figure(spec, ev["summary"], out)
    for d in ev["decision"]:
        log.info("%s | %s | %s = %s [%s, %s] -> %s %s", d["gate"], d["system"], d["statistic"], d["value"], d["ci_low"], d["ci_high"],
                 d["verdict"], d["note"])


if __name__ == "__main__":
    main()
