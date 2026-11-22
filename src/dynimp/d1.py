"""D1 gate: recurrence fill against learned imputers and low-rank Hankel completion.

Question: on signals whose states recur (noisy chaotic systems, PPG and ECG), does a recurrence fill that needs
no training do as well as learned imputers, and how does it compare with low-rank completion of the embedding
(Bousse et al. 2024)?

Run from the repo root with ``PYTHONPATH=src`` and ``.venv-dynimp/Scripts/python.exe -m dynimp.d1 --stage S``:

    design   download BIDMC, check subjects, choose embeddings (complete data only)
    cpu      recurrence fill, low-rank completion, interpolation, boosting (process pool)
    gpu      SAITS and CSDI (one process)
    eval     tables and decision.csv

``CONDPRED_SMOKE=1`` runs a tiny version into separate ``smoke/`` folders.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from condpred import completion, contaminate
from condpred.config import code_fingerprint, smoke, spec_hash
from condpred.report import save_table, write_decision, write_spec
from dynimp import d0
from dynimp.fills import lowrank_fill_gaps, lowrank_fill_tiled, nan_runs, recurrence_fill, spline_fill
from recplay.embedding import embed
from recplay.recurrence import euclidean_distances

REPO = Path(__file__).resolve().parents[2]
MODULES = ["dynimp.d1", "dynimp.fills", "dynimp.d0", "condpred.completion", "condpred.contaminate", "condpred.systems",
           "recplay.embedding", "recplay.recurrence"]
SPEC_VERSION = 1
BIDMC_URL = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_{:02d}_Signals.csv"
TASK_ID = {"ppg": 1, "ecg": 2}
TASKS = {
    "ppg": {"what": "gaps in PPG (PLETH); ECG lead II complete; both decimated to 62.5 Hz", "columns": ["PLETH", "II"], "decimate": 2,
            "n": 30000, "gaps": [62, 187], "L": 375, "Cs": [15, 31, 62, 125], "ctx": [125, 250], "helper_half_window": 62},
    "ecg": {"what": "gaps in ECG lead II at 125 Hz; no other channel", "columns": ["II"], "decimate": 1,
            "n": 60000, "gaps": [25, 75], "L": 375, "Cs": [12, 25, 62, 125], "ctx": [187, 375]},
}
CHAOTIC = ["lorenz", "rossler", "mackey_glass"]
LEARNED = ["saits", "csdi"]
RP_W = 500

log = logging.getLogger("dynimp.d1")


def out_dir() -> Path:
    base = REPO / "notebooks" / "tables" / "dynimp" / "D1"
    return base / "smoke" if smoke(False, True) else base


def cache_dir() -> Path:
    return REPO / "data" / "processed" / "dynimp" / "D1" / smoke("full", "smoke")


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng([13, *[int(p) for p in parts]])


def cached(spec: dict, name: str, fn):
    h12 = hashlib.sha256((spec_hash(spec) + code_fingerprint(MODULES)).encode()).hexdigest()[:12]
    path = cache_dir() / f"{name}_{h12}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    res = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(res))
    tmp.replace(path)
    return res


def blocks(n: int, frac: float, g: int, rng) -> np.ndarray:
    """True where missing: runs of g samples, about frac of n in total."""
    _, observed = contaminate.mask_missing(np.zeros(n), frac, "block", g, rng=rng)
    return ~observed


# ----------------------------------------------------------------------------- BIDMC


def bidmc_path(sid: int) -> Path:
    return REPO / "data" / "raw" / "bidmc" / f"bidmc_{sid:02d}_Signals.csv"


def download_bidmc() -> None:
    folder = bidmc_path(1).parent
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for sid in range(1, 54):
        p = bidmc_path(sid)
        if not p.exists():
            raw = urlopen(BIDMC_URL.format(sid), timeout=180).read()
            p.write_bytes(raw)
            manifest[p.name] = {"url": BIDMC_URL.format(sid), "sha256": hashlib.sha256(raw).hexdigest(),
                                "downloaded_at": datetime.now(timezone.utc).isoformat()}
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for name, rec in manifest.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != rec["sha256"]:
            raise RuntimeError(f"{name} does not match data/raw/bidmc/manifest.json")


def load_subject(sid: int, task: str) -> np.ndarray | None:
    """Complete signals [n, F], target first. None if a column is missing, not finite or constant."""
    from scipy.signal import decimate

    T = TASKS[task]
    df = pd.read_csv(bidmc_path(sid))
    df.columns = [c.strip() for c in df.columns]
    if any(c not in df.columns for c in T["columns"]):
        return None
    x = df[T["columns"]].to_numpy(float)
    if not np.isfinite(x).all():
        return None
    if T["decimate"] > 1:
        x = decimate(x, T["decimate"], axis=0, zero_phase=True)
    x = x[: T["n"]]
    if len(x) < T["n"] or (x.std(axis=0) < 1e-9).any():
        return None
    return np.ascontiguousarray(x)


def subject_series(task: str, g: int, sid: int) -> dict:
    x = load_subject(sid, task)
    n = len(x)
    miss = blocks(n, 0.2, g, _rng(31, TASK_ID[task], g, sid))
    xo = x.copy()
    xo[miss, 0] = np.nan
    n_hist = int(0.6 * n)
    mu, sd = np.nanmean(xo[:n_hist], axis=0), np.nanstd(xo[:n_hist], axis=0)
    return {"zo": (xo - mu) / sd, "zt": (x - mu) / sd, "miss": miss, "n_hist": n_hist, "n_lib": int(0.5 * n)}


def rp_error(filled: np.ndarray, truth: np.ndarray, miss: np.ndarray, m: int, tau: int) -> float:
    """Mean relative Frobenius error of the unthresholded RP over the windows of RP_W samples that contain a gap."""
    errs = []
    for s in range(0, len(truth) - RP_W + 1, RP_W):
        sl = slice(s, s + RP_W)
        if not miss[sl].any():
            continue
        mu, sd = truth[sl].mean(), truth[sl].std()
        D = euclidean_distances(embed((truth[sl] - mu) / sd, m, tau))
        Dh = euclidean_distances(embed((filled[sl] - mu) / sd, m, tau))
        errs.append(np.linalg.norm(Dh - D) / np.linalg.norm(D))
    return float(np.mean(errs)) if errs else np.nan


def _score(pred, truth, miss, emb) -> dict:
    z = np.where(miss, np.nan_to_num(pred, nan=0.0), truth)
    return {"mae": float(np.abs(z[miss] - truth[miss]).mean()), "rp": rp_error(z, truth, miss, *emb)}


def physio_cpu_unit(spec: dict, task: str, g: int, sid: int) -> dict:
    """Every method that needs no GPU, for one test subject. Choices are made on the subject's own history."""
    T = spec["tasks"][task]
    d = subject_series(task, g, sid)
    zo, n_hist, n_lib = d["zo"], d["n_hist"], d["n_lib"]
    hist, test, truth, miss = zo[:n_hist], zo[n_hist:], d["zt"][n_hist:, 0], d["miss"][n_hist:]
    lib, val = zo[:n_lib], zo[n_lib:n_hist]
    held = blocks(len(val), 0.2, g, _rng(32, TASK_ID[task], g, sid)) & np.isfinite(val[:, 0])
    v = val.copy()
    v[held, 0] = np.nan
    runs = [r for r in nan_runs(v[:, 0]) if held[r[0] : r[1]].all()][: spec["val_runs"]]
    idx = np.concatenate([np.arange(a, b) for a, b in runs])
    val_err = lambda pred: float(np.nanmean(np.abs(np.nan_to_num(pred[idx], nan=0.0) - val[idx, 0])))
    emb = spec["embedding"][task]
    arms, choices = {}, {}

    variants = [("recurrence", list(range(zo.shape[1])), None)]
    if zo.shape[1] > 1:
        variants += [("recurrence_own", [0], None), ("recurrence_helper", [0, 1], [0.0, 1.0])]
    for name, cols, w in variants:
        C = min(T["Cs"], key=lambda C: val_err(recurrence_fill(v[:, cols], lib[:, cols], 0, C=C, weights=w, runs=runs)))
        arms[name] = _score(recurrence_fill(test[:, cols], hist[:, cols], 0, C=C, weights=w), truth, miss, emb)
        choices[name] = C

    grid = [(ctx, r) for ctx in T["ctx"] for r in spec["lowrank"]["ranks_physio"]]
    ctx, rank = min(grid, key=lambda p: val_err(lowrank_fill_gaps(v[:, 0], p[0], p[1], runs=runs)))
    arms["lowrank"] = _score(lowrank_fill_gaps(test[:, 0], ctx, rank), truth, miss, emb)
    choices["lowrank"] = [ctx, rank]
    arms["linear"] = _score(completion.linear(test[:, 0]), truth, miss, emb)
    arms["spline"] = _score(spline_fill(test[:, 0]), truth, miss, emb)
    arms["mean"] = _score(np.zeros(len(test)), truth, miss, emb)

    if zo.shape[1] > 1:
        from numpy.lib.stride_tricks import sliding_window_view
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LinearRegression

        W = T["helper_half_window"]
        feats = lambda e: sliding_window_view(np.pad(e, W, mode="edge"), 2 * W + 1)[:, ::2]
        ok = np.isfinite(hist[:, 0])
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(feats(hist[:, 1])[ok], hist[ok, 0])
        arms["boosting_helper"] = _score(gb.predict(feats(test[:, 1])), truth, miss, emb)
        lr = LinearRegression().fit(hist[ok, 1:2], hist[ok, 0])
        arms["same_time_helper"] = _score(lr.predict(test[:, 1:2]), truth, miss, emb)
    return {"cell": f"{task}|g{g}", "unit": int(sid), "arms": arms, "choices": choices}


# ----------------------------------------------------------------------------- learned imputers


def _saits_model(cfg: dict, L: int, F: int, epochs: int, patience: int):
    from pypots.imputation import SAITS

    return SAITS(n_steps=L, n_features=F, n_layers=cfg["n_layers"], d_model=cfg["d_model"], n_heads=4, d_k=cfg["d_model"] // 4,
                 d_v=cfg["d_model"] // 4, d_ffn=2 * cfg["d_model"], dropout=0.1, batch_size=32, epochs=epochs, patience=patience,
                 device="cuda", saving_path=None, verbose=False)


def physio_deep_unit(spec: dict, task: str, g: int) -> dict:
    """SAITS (small grid, chosen on validation subjects) and CSDI, trained on other subjects, applied to the test subjects."""
    import logging as _l

    from pypots.imputation import CSDI
    from pypots.utils.logging import logger as pl
    from pypots.utils.random import set_random_seed

    pl.setLevel(_l.ERROR)
    T, S, D = spec["tasks"][task], spec["subjects"], spec["deep_physio"]
    L, emb = T["L"], spec["embedding"][task]
    win = lambda z, stride: np.stack([z[s : s + L] for s in range(0, len(z) - L + 1, stride)]).astype(np.float32)
    rng = _rng(41, TASK_ID[task], g)
    tr = np.concatenate([win(subject_series(task, g, s)["zo"], L // 3) for s in S["train"]])
    tr = tr[rng.permutation(len(tr))[: D["max_train_windows"]]]
    va_in, va_ori = [], []
    for s in S["val"]:
        z = subject_series(task, g, s)["zo"]
        held = blocks(len(z), 0.2, g, _rng(42, TASK_ID[task], g, s)) & np.isfinite(z[:, 0])
        zi = z.copy()
        zi[held, 0] = np.nan
        va_in.append(win(zi, L))
        va_ori.append(win(z, L))
    va = {"X": np.concatenate(va_in)[: D["max_val_windows"]], "X_ori": np.concatenate(va_ori)[: D["max_val_windows"]]}
    tests = {s: subject_series(task, g, s) for s in S["test"]}
    te = np.concatenate([win(d["zo"][d["n_hist"] :], L) for d in tests.values()])
    F = tr.shape[2]

    def per_subject(imp: np.ndarray) -> dict:
        out, k = {}, 0
        for s, d in tests.items():
            n = len(d["zo"]) - d["n_hist"]
            pred = imp[k : k + n // L, :, 0].reshape(-1)
            k += n // L
            out[s] = _score(pred, d["zt"][d["n_hist"] :, 0][: len(pred)], d["miss"][d["n_hist"] :][: len(pred)], emb)
        return out

    res, meta = {}, {}
    t0, best = time.time(), None
    for i, cfg in enumerate(D["saits_grid"]):
        set_random_seed(1000 * TASK_ID[task] + g + i)
        model = _saits_model(cfg, L, F, D["epochs"], D["patience"])
        model.fit({"X": tr}, va)
        loss = float(model.best_loss)
        if best is None or loss < best[0]:
            best = (loss, cfg, np.asarray(model.predict({"X": te})["imputation"], float), int(model.best_epoch))
    res["saits"] = per_subject(best[2])
    meta["saits"] = {"config": best[1], "val_loss": best[0], "best_epoch": best[3], "seconds": round(time.time() - t0)}
    t0 = time.time()
    set_random_seed(2000 * TASK_ID[task] + g)
    c = D["csdi"]
    model = CSDI(n_steps=L, n_features=F, n_layers=c["n_layers"], n_heads=c["n_heads"], n_channels=c["n_channels"],
                 d_time_embedding=128, d_feature_embedding=16, d_diffusion_embedding=128, n_diffusion_steps=50,
                 target_strategy=c["target_strategy"], schedule="quad", beta_start=1e-4, beta_end=0.5, batch_size=32,
                 epochs=D["epochs"], patience=D["patience"], device="cuda", saving_path=None, verbose=False)
    model.fit({"X": tr}, va)
    samples = np.asarray(model.predict({"X": te}, n_sampling_times=c["n_sampling_times"])["imputation"], float)
    res["csdi"] = per_subject(np.median(samples, axis=1))
    meta["csdi"] = {"best_epoch": int(model.best_epoch), "val_loss": float(model.best_loss), "seconds": round(time.time() - t0)}
    return {"cell": f"{task}|g{g}", "arms": res, "meta": meta, "n_train_windows": int(len(tr))}


# ----------------------------------------------------------------------------- chaotic systems (D0 data)


def _d0_cell(d0spec: dict, name: str) -> dict:
    return next(c for c in d0spec["cells"] if c["name"] == name)


def chaotic_cpu_unit(spec: dict, name: str, seed: int) -> dict:
    """Recurrence fill and tuned low-rank completion on a D0 cell. Same data, gaps and scoring as D0."""
    cell = _d0_cell(d0.build_spec(), name)
    system, pattern = cell["system"], cell["pattern"]
    sid, pid = d0._SYS_ID[system], d0._PATTERN_ID[pattern]
    n_tr, n_va, _ = d0.split_sizes(system)
    x = d0.complete_series(system, cell["noise"], seed)
    F = x.shape[1]
    miss = d0.missing_mask(len(x), F, 0.4, pattern, 10, 3, sid, seed, pid)
    xo = np.where(miss, np.nan, x)
    mu, sd = np.nanmean(xo[:n_tr], 0), np.nanstd(xo[:n_tr], 0)
    zt, zo = (x - mu) / sd, (xo - mu) / sd
    tr, va, test = zo[:n_tr], zo[n_tr : n_tr + n_va], zo[n_tr + n_va :]
    truth, m = zt[n_tr + n_va :], miss[n_tr + n_va :]
    held = d0.missing_mask(n_va, F, 0.2, pattern, 10, 4, sid, seed, pid) & np.isfinite(va)
    v = np.where(held, np.nan, va)
    verr = lambda f: float(np.nanmean(np.abs(np.nan_to_num(f, nan=0.0)[held] - va[held])))
    mae = lambda f: float(np.abs(np.nan_to_num(f, nan=0.0)[m] - truth[m]).mean())
    cols = lambda fn, z: np.column_stack([fn(z, c) for c in range(F)])

    C = min(spec["recurrence"]["Cs_chaotic"], key=lambda C: verr(cols(lambda z, c: recurrence_fill(z, tr, c, C=C), v)))
    lib = np.vstack([tr, va])
    grid = [(w, r) for w in spec["lowrank"]["windows_chaotic"] for r in spec["lowrank"]["ranks_chaotic"]]
    w, r = min(grid, key=lambda p: verr(cols(lambda z, c: lowrank_fill_tiled(z[:, c], p[0], p[1]), v)))
    return {"cell": name, "unit": int(seed),
            "arms": {"recurrence": {"mae": mae(cols(lambda z, c: recurrence_fill(z, lib, c, C=C), test))},
                     "lowrank": {"mae": mae(cols(lambda z, c: lowrank_fill_tiled(z[:, c], w, r), test))}},
            "choices": {"recurrence": C, "lowrank": [w, r]}}


def _chaotic_deep(d0spec: dict, cell: dict, seed: int, arm: str, override: dict) -> dict:
    from pypots.utils.random import set_random_seed

    data = d0.unit_data(d0spec, cell, seed)
    sp = copy.deepcopy(d0spec)
    sp[arm].update(override)
    truth, miss = data["test_true"], data["test_missing"]
    set_random_seed(d0._crc(f"d1{arm}{cell['name']}{seed}{json.dumps(override, sort_keys=True)}") % 2**31)
    if arm == "saits":
        imp, meta = d0._saits(sp, data, "cuda")
    else:
        imp, _, meta = d0._csdi(sp, data, "cuda")
    z = np.where(miss, imp.reshape(truth.shape), truth)
    return {"mae": float(np.abs(z[miss] - truth[miss]).mean()), **meta}


def chaotic_gpu(spec: dict) -> list[dict]:
    """D0 units (cached where they exist), a SAITS grid chosen on seed 0, and CSDI with longer patience where D0's stalled."""
    import logging as _l

    from pypots.utils.logging import logger as pl

    pl.setLevel(_l.ERROR)
    d0spec = d0.build_spec()
    C = spec["chaotic"]
    out = []
    for name in C["cells"]:
        cell = _d0_cell(d0spec, name)
        base = {s: d0.run_unit_cached(d0spec, cell, s) for s in C["seeds"]}
        log.info("%s: D0 units ready", name)
        s0 = C["seeds"][0]
        d0cfg = {k: d0spec["saits"][k] for k in ["n_layers", "d_model", "d_k", "d_v", "d_ffn"]}
        cands = [(float(base[s0]["fit"]["saits"]["best_loss"]), d0cfg, None)]
        for cfg in C["saits_grid"]:
            ov = {"n_layers": cfg["n_layers"], "d_model": cfg["d_model"], "d_k": cfg["d_model"] // 4, "d_v": cfg["d_model"] // 4, "d_ffn": 2 * cfg["d_model"]}
            r = cached(spec, f"saits_{name.replace('|', '_')}_s{s0}_{cfg['n_layers']}x{cfg['d_model']}", lambda: _chaotic_deep(d0spec, cell, s0, "saits", ov))
            cands.append((float(r["best_loss"]), ov, r))
        loss, ov, _ = min(cands, key=lambda c: c[0])
        retrain_csdi = name in C["csdi_long_patience_cells"]
        for s in C["seeds"]:
            arms0 = base[s]["arms"].set_index("arm")["mae"]
            arms = {a: {"mae": float(arms0[a])} for a in ["linear", "csdi"]}
            arms["mean"] = {"mae": float(arms0["zero"])}
            arms["saits_d0"] = {"mae": float(arms0["saits"])}
            if ov is d0cfg:
                arms["saits"] = {"mae": float(arms0["saits"])}
            else:
                key = f"saits_{name.replace('|', '_')}_s{s}_{ov['n_layers']}x{ov['d_model']}"
                arms["saits"] = {"mae": cached(spec, key, lambda: _chaotic_deep(d0spec, cell, s, "saits", ov))["mae"]}
            if retrain_csdi:
                arms["csdi_d0"] = arms["csdi"]
                r = cached(spec, f"csdi_{name.replace('|', '_')}_s{s}", lambda: _chaotic_deep(d0spec, cell, s, "csdi", {"patience": C["csdi_long_patience"]}))
                arms["csdi"] = {"mae": r["mae"]}
            out.append({"cell": name, "unit": int(s), "arms": arms, "choices": {"saits": [ov["n_layers"], ov["d_model"]], "saits_val_loss": loss}})
        log.info("%s: SAITS config %sx%s (val loss %.4f)", name, ov["n_layers"], ov["d_model"], loss)
    return out


# ----------------------------------------------------------------------------- spec


def build_spec() -> dict:
    design = json.loads((REPO / "notebooks" / "tables" / "dynimp" / "D1" / "design.json").read_text(encoding="utf-8"))
    valid = design["valid_subjects"]
    order = [valid[i] for i in np.random.default_rng(2026).permutation(len(valid))]
    n_test, n_val, n_train = smoke(20, 2), smoke(8, 2), smoke(25, 3)
    subjects = {"test": sorted(order[:n_test]), "val": sorted(order[n_test : n_test + n_val]),
                "train": sorted(order[n_test + n_val : n_test + n_val + n_train])}
    chaotic_cells = [f"{s}|n0.1|{p}" for s in CHAOTIC for p in ("point", "block")]
    epochs, patience = smoke(60, 2), smoke(8, 2)
    return {
        "version": SPEC_VERSION,
        "experiment": "dynimp D1 gate: recurrence fill against learned imputers and low-rank Hankel completion",
        "smoke": bool(smoke(False, True)),
        "cells": chaotic_cells + [f"{t}|g{g}" for t in TASKS for g in TASKS[t]["gaps"]],
        "chaotic": {
            "cells": chaotic_cells if not smoke(False, True) else chaotic_cells[4:],
            "seeds": list(range(smoke(6, 1))),
            "data": "exactly the D0 cells (10 % observation noise, 40 % missing, point or runs of 10): same series, gaps, "
            "scaling and scoring (MAE over the missing test entries of all channels)",
            "d0_arms": "linear, csdi and saits_d0 come from D0 units (cached for the seeds D0 ran, computed with the same code otherwise)",
            "saits_grid": [{"n_layers": 2, "d_model": 64}, {"n_layers": 2, "d_model": 256}, {"n_layers": 4, "d_model": 128}],
            "saits_choice": "lowest validation loss on seed 0 among the grid and the D0 configuration; that configuration is used for every seed",
            "csdi_long_patience_cells": ["mackey_glass|n0.1|point"], "csdi_long_patience": smoke(50, 1),
            "csdi_note": "CSDI keeps the D0 configuration (no search). Where D0's early stopping fired before it trained "
            "(Mackey-Glass, point gaps), it is retrained with patience 50; the D0 result is kept as csdi_d0",
        },
        "bidmc": {"source": "PhysioNet BIDMC PPG and Respiration Dataset 1.0.0, bidmc_csv/bidmc_XX_Signals.csv, 125 Hz, 8 min per subject",
                  "manifest": "data/raw/bidmc/manifest.json (SHA-256)", "excluded": design["excluded"]},
        "subjects": subjects,
        "subject_split": "random permutation (seed 2026) of the valid subjects: test, then validation, then train",
        "tasks": TASKS,
        "physio_protocol": {
            "gaps": "20 % of the target channel in runs of g samples, over the whole record of every subject",
            "scaling": "z-score per subject and channel with the observed samples of the first 60 % of the record",
            "history": "first 60 % of a test subject's record: library and tuning data of the training-free methods",
            "test": "last 40 % of the record; errors on the missing target samples there",
            "tuning": "training-free methods choose their parameters per subject by filling extra held-out runs of length g in "
            "the 50-60 % part of the record from the first 50 %",
            "learned": "trained on the train subjects' whole records (with their gaps), selected on the validation subjects; "
            "they never see a test subject's history",
        },
        "val_runs": 12,
        "recurrence": {"k": 8, "min_overlap": 0.3, "Cs_chaotic": [5, 10, 20, 40],
                       "state": "all channels on [a - C, b + C); chaotic: every channel is a target in turn",
                       "variants_ppg": "recurrence (PPG context + ECG), recurrence_own (PPG only), recurrence_helper (ECG only)"},
        "lowrank": {"method": "condpred.completion.hankel_complete, tau 1: iterative rank truncation with known entries re-imposed, "
                    "started from linear interpolation; an approximation of Bousse et al. (2024), not Tensorlab SDF",
                    "rows": "one third of the window (at most 150)",
                    "windows_chaotic": [96, 192], "ranks_chaotic": [2, 3, 4, 6, 8, 12],
                    "ranks_physio": [4, 8, 16, 24], "physio_window": "the gap plus ctx samples on each side; ctx from the task's list"},
        "deep_physio": {"epochs": epochs, "patience": patience, "max_train_windows": smoke(3000, 96), "max_val_windows": smoke(600, 32),
                        "saits_grid": smoke([{"n_layers": 2, "d_model": 64}, {"n_layers": 2, "d_model": 128}, {"n_layers": 2, "d_model": 256}],
                                            [{"n_layers": 2, "d_model": 64}]),
                        "csdi": {"n_layers": 4, "n_heads": 4, "n_channels": 64, "target_strategy": "mix", "n_sampling_times": smoke(5, 2)},
                        "note": "SAITS: grid chosen on validation subjects. CSDI: one configuration; target_strategy 'mix' lets it train "
                        "on the block patterns present in the data. Point estimate of CSDI = median of its samples"},
        "embedding": design["embedding"],
        "metrics": {"mae": "mean |filled - true| over the missing test samples, z units (primary)",
                    "rp": f"physiological cells only: relative Frobenius error of the unthresholded RP (Bousse et al.), windows of {RP_W} "
                    "samples of the test target that contain a gap, window z-scored with the complete window's mean and sd"},
        "interval": "95 % t-interval of the mean over units; unit = seed (chaotic, 6) or test subject (physiological, 20); paired differences",
        "gate": {
            "cells": "all ten cells",
            "part1": "non-inferiority: in a cell the recurrence fill is not worse than a learned imputer X (saits, csdi) iff the upper "
            "bound of MAE_recurrence - MAE_X is below 0.10 * mean MAE_X. Part 1 passes iff this holds for both learned imputers in at least 8 of 10 cells",
            "part2": "the recurrence fill is better than both learned imputers (upper bound of the difference below zero for each) in at least 3 of 10 cells",
            "D1": "pass iff part 1 and part 2 pass",
        },
        "secondary": {
            "H_LR1": "low-rank completion has a lower RP error than spline interpolation in both ECG cells (interval of the difference below zero)",
            "H_LR2": "recurrence fill against low-rank completion on MAE in every cell: reported with intervals",
            "other": "PPG variants of the recurrence fill, boosting on the ECG window, same-time regression, saits_d0, csdi_d0",
        },
        "leakage": "no method receives a missing value; tuning uses held-out observed samples only; learned imputers are scored on subjects they were not trained on",
    }


def design() -> None:
    """Download BIDMC, list usable subjects, choose RP embeddings. Uses complete signals only; no method is run."""
    download_bidmc()
    out = REPO / "notebooks" / "tables" / "dynimp" / "D1"
    out.mkdir(parents=True, exist_ok=True)
    valid, excluded = [], {}
    for sid in range(1, 54):
        bad = [t for t in TASKS if load_subject(sid, t) is None]
        if bad:
            excluded[sid] = bad
        else:
            valid.append(sid)
    emb = {}
    for t in TASKS:
        ms = [d0.choose_embedding(load_subject(s, t)[:, 0])[:2] for s in valid[:5]]
        emb[t] = [int(np.median([m for m, _ in ms])), int(np.median([tau for _, tau in ms]))]
        log.info("%s: (m, tau) of the first five valid subjects %s -> %s", t, ms, emb[t])
    log.info("valid subjects %d, excluded %s", len(valid), excluded)
    (out / "design.json").write_text(json.dumps({"valid_subjects": valid, "excluded": excluded, "embedding": emb,
                                                 "embedding_rule": "median over the first five valid subjects of the AMI / false-nearest-neighbour choice",
                                                 "written_at": datetime.now(timezone.utc).isoformat()}, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- evaluation


def collect(spec: dict) -> pd.DataFrame:
    rows = []
    h12 = hashlib.sha256((spec_hash(spec) + code_fingerprint(MODULES)).encode()).hexdigest()[:12]
    for p in sorted(cache_dir().glob(f"unit_*_{h12}.pkl")) + sorted(cache_dir().glob(f"stage_*_{h12}.pkl")):
        res = pickle.loads(p.read_bytes())
        for r in res if isinstance(res, list) else [res]:
            if "unit" in r:
                rows += [{"cell": r["cell"], "unit": r["unit"], "arm": a, **v} for a, v in r["arms"].items()]
            else:
                rows += [{"cell": r["cell"], "unit": int(u), "arm": a, **v} for a, per in r["arms"].items() for u, v in per.items()]
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["cell", "unit", "arm"]).sort_values(["cell", "arm", "unit"]).reset_index(drop=True)


def evaluate(spec: dict, df: pd.DataFrame) -> dict:
    sm = bool(spec["smoke"])
    summary, pairs, decision = [], [], []
    row = lambda gate, cell, stat, value, lo, hi, rule, verdict, note="": decision.append(
        {"gate": gate, "cell": cell, "statistic": stat, "value": value, "ci_low": lo, "ci_high": hi, "rule": rule, "verdict": verdict, "note": note, "smoke": sm})
    n_ni = n_sup = 0
    for cell in [c for c in spec["cells"] if c in set(df["cell"])]:
        g = df[df["cell"] == cell]
        for arm, ga in g.groupby("arm"):
            rec = {"cell": cell, "arm": arm, "n": len(ga)}
            for col in [c for c in ["mae", "rp"] if c in ga and ga[c].notna().any()]:
                rec[col], rec[f"{col}_lo"], rec[f"{col}_hi"] = d0.t_interval(ga[col])
            summary.append(rec)
        piv = g.pivot(index="unit", columns="arm", values="mae")
        diffs = {}
        for other in [a for a in piv.columns if a != "recurrence"]:
            dm, lo, hi = d0.t_interval((piv["recurrence"] - piv[other]).dropna())
            diffs[other] = (dm, lo, hi, float(piv[other].mean()))
            pairs.append({"cell": cell, "a": "recurrence", "b": other, "metric": "mae", "diff": dm, "lo": lo, "hi": hi, "mean_b": float(piv[other].mean())})
        if "rp" in g and g["rp"].notna().any():
            pr = g.pivot(index="unit", columns="arm", values="rp")
            for a, b in [("lowrank", "spline"), ("recurrence", "lowrank"), ("recurrence", "saits"), ("recurrence", "csdi")]:
                if a in pr and b in pr:
                    dm, lo, hi = d0.t_interval((pr[a] - pr[b]).dropna())
                    pairs.append({"cell": cell, "a": a, "b": b, "metric": "rp", "diff": dm, "lo": lo, "hi": hi, "mean_b": float(pr[b].mean())})
        if all(a in diffs for a in LEARNED):
            ni = all(diffs[a][2] < 0.10 * diffs[a][3] for a in LEARNED)
            sup = all(diffs[a][2] < 0 for a in LEARNED)
            n_ni, n_sup = n_ni + int(ni), n_sup + int(sup)
            note = "; ".join(f"vs {a}: {diffs[a][0]:+.4f} [{diffs[a][1]:+.4f}, {diffs[a][2]:+.4f}], {a} MAE {diffs[a][3]:.4f}" for a in LEARNED)
            row("cell", cell, "recurrence MAE", float(piv["recurrence"].mean()), np.nan, np.nan, "non-inferior: upper bound < 10 % of the other's MAE, both; better: upper bound < 0, both",
                ("better" if sup else "non_inferior" if ni else "not_shown"), note)
    p1, p2 = n_ni >= 8, n_sup >= 3
    row("part1", "all", "cells where recurrence is non-inferior to both learned imputers", n_ni, np.nan, np.nan, "at least 8 of 10", "pass" if p1 else "fail")
    row("part2", "all", "cells where recurrence is better than both learned imputers", n_sup, np.nan, np.nan, "at least 3 of 10", "pass" if p2 else "fail")
    row("D1", "all", "part 1 and part 2", float(p1 and p2), np.nan, np.nan, "pass iff both", ("pass" if p1 and p2 else "fail") + (" (smoke: mechanical only)" if sm else ""))
    pairs = pd.DataFrame(pairs)
    if len(pairs):
        lr = pairs[(pairs["metric"] == "rp") & (pairs["a"] == "lowrank") & (pairs["b"] == "spline") & pairs["cell"].str.startswith("ecg")]
        ok = len(lr) == 2 and bool((lr["hi"] < 0).all())
        row("H_LR1", "ecg", "low-rank RP error below spline RP error in both ECG cells", float(ok), np.nan, np.nan, "interval of the difference below zero in both",
            "supported" if ok else "not_supported", "; ".join(f"{r.cell}: {r.diff:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]" for r in lr.itertuples()))
    return {"summary": pd.DataFrame(summary), "pairs": pairs, "decision": decision}


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["design", "cpu", "gpu", "eval"])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(out / f"run_{args.stage}.log", mode="a", encoding="utf-8")):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    if args.stage == "design":
        design()
        return
    spec = build_spec()
    write_spec(spec, out, modules=MODULES)  # before any method is run
    log.info("spec v%s, hash %s, code %s", spec["version"], spec_hash(spec)[:12], code_fingerprint(MODULES)[:12])
    t0 = time.time()
    if args.stage == "cpu":
        jobs = [("chaotic", name, s) for name in spec["chaotic"]["cells"] for s in spec["chaotic"]["seeds"]]
        jobs += [("physio", t, g, s) for t in TASKS for g in TASKS[t]["gaps"] for s in spec["subjects"]["test"]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_cpu_job, spec, j): j for j in jobs}
            for k, f in enumerate(as_completed(futs), 1):
                r = f.result()
                log.info("[%d/%d] %s unit %s: %s | %s", k, len(jobs), r["cell"], r["unit"], " ".join(f"{a} {v['mae']:.3f}" for a, v in r["arms"].items()), r["choices"])
    elif args.stage == "gpu":
        cached(spec, "stage_chaotic_gpu", lambda: chaotic_gpu(spec))
        log.info("chaotic GPU stage done after %.0f s", time.time() - t0)
        for t in TASKS:
            for g in TASKS[t]["gaps"]:
                r = cached(spec, f"stage_deep_{t}_g{g}", lambda: physio_deep_unit(spec, t, g))
                log.info("%s|g%d: %s; train windows %d; after %.0f s", t, g, r["meta"], r["n_train_windows"], time.time() - t0)
    else:
        df = collect(spec)
        save_table(df, out, "per_unit")
        ev = evaluate(spec, df)
        save_table(ev["summary"], out, "summary")
        save_table(ev["pairs"], out, "pairs")
        write_decision(ev["decision"], out)
        for d in ev["decision"]:
            log.info("%s | %s | %s = %s -> %s %s", d["gate"], d["cell"], d["statistic"], d["value"], d["verdict"], d["note"])
    log.info("stage %s done in %.0f s", args.stage, time.time() - t0)


def _cpu_job(spec: dict, job: tuple) -> dict:
    if job[0] == "chaotic":
        return cached(spec, f"unit_chaotic_{job[1].replace('|', '_')}_s{job[2]}", lambda: chaotic_cpu_unit(spec, job[1], job[2]))
    return cached(spec, f"unit_{job[1]}_g{job[2]}_s{job[3]}", lambda: physio_cpu_unit(spec, job[1], job[2], job[3]))


if __name__ == "__main__":
    main()
