"""D2: learned imputers given the patient's own history.

Question: does the recurrence fill still match learned imputers on PPG and ECG once they are given the same
history (the test patient's first 50 %), a block training mask for SAITS (matching the test gaps) and a
larger budget? See plans/dynimp/D2_per_patient_learned.md and Writeups/dynimp_D1_gate.md.

Run from the repo root with ``PYTHONPATH=src`` (one GPU process at a time):

    .venv-dynimp/Scripts/python.exe -m dynimp.d2 --stage selfcheck
    .venv-dynimp/Scripts/python.exe -m dynimp.d2 --stage bench      # timing only, before the spec is frozen
    .venv-dynimp/Scripts/python.exe -m dynimp.d2 --stage cross
    .venv-dynimp/Scripts/python.exe -m dynimp.d2 --stage patient
    .venv-dynimp/Scripts/python.exe -m dynimp.d2 --stage eval

``CONDPRED_SMOKE=1`` runs a tiny version into separate ``smoke/`` folders. D1 arms are read from
``notebooks/tables/dynimp/D1/per_unit.csv`` and are never rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from condpred.config import code_fingerprint, smoke, spec_hash
from condpred.report import save_table, write_decision, write_spec
from dynimp import d0, d1

REPO = Path(__file__).resolve().parents[2]
MODULES = ["dynimp.d2", "dynimp.d1", "dynimp.fills", "dynimp.d0", "condpred.completion", "condpred.contaminate"]
SPEC_VERSION = 1
CELLS = [("ppg", 62), ("ppg", 187), ("ecg", 25), ("ecg", 75)]
GATE_ARMS = ["saits_ft", "csdi_ft"]
NEW_ARMS = ["saits_cross", "saits_own", "saits_ft", "csdi_cross", "csdi_ft", "csdi_own"]
REUSED = {"recurrence": "recurrence", "lowrank": "lowrank", "boosting_helper": "boosting_helper",
          "mean": "mean", "saits_d1": "saits", "csdi_d1": "csdi"}
BUDGET_GPU_HOURS = 12.0

log = logging.getLogger("dynimp.d2")


def out_dir() -> Path:
    base = REPO / "notebooks" / "tables" / "dynimp" / "D2"
    return base / "smoke" if smoke(False, True) else base


def cache_dir() -> Path:
    return REPO / "data" / "processed" / "dynimp" / "D2" / smoke("full", "smoke")


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


def win(z: np.ndarray, L: int, stride: int) -> np.ndarray:
    if len(z) < L:
        return np.empty((0, L, z.shape[1]), np.float32)
    return np.stack([z[s:s + L] for s in range(0, len(z) - L + 1, stride)]).astype(np.float32)


# ----------------------------------------------------------------------------- SAITS block mask


def block_mask_factory(g: int, frac: float = 0.2, target: int = 0):
    import torch

    def block_mcar(X, p=None):
        X = X.clone()
        n = X.shape[0]
        for _ in range(max(1, round(frac * n / g))):
            a = int(torch.randint(0, n - g + 1, (1,)).item())
            X[a:a + g, target] = float("nan")
        if torch.isnan(X[:, target]).all():  # never mask a whole window: put one observed segment back
            X[: max(1, g // 2), target] = 0.0
        return X

    return block_mcar


@contextmanager
def patched_mcar(g: int | None):
    """Replace PyPOTS SAITS's point mask with a block mask on the target channel (g=None: leave unpatched)."""
    if g is None:
        yield
        return
    import pypots.imputation.saits.data as saits_data

    orig = saits_data.mcar
    saits_data.mcar = block_mask_factory(g)
    try:
        yield
    finally:
        saits_data.mcar = orig


# ----------------------------------------------------------------------------- data


def patient_data(spec: dict, task: str, g: int, sid: int) -> dict:
    """Train / validation / test windows of one test patient. No window reaches into the test region."""
    T, L = spec["tasks"][task], spec["tasks"][task]["L"]
    d = d1.subject_series(task, g, sid)
    zo, n_hist, n_lib = d["zo"], d["n_hist"], d["n_lib"]
    val = zo[n_lib:n_hist]
    held = d1.blocks(len(val), 0.2, g, d1._rng(32, d1.TASK_ID[task], g, sid)) & np.isfinite(val[:, 0])
    vi = val.copy()
    vi[held, 0] = np.nan
    return {
        "d": d,
        "train": win(zo[:n_lib], L, spec["patient"]["stride"]),
        "train_own": win(zo[:n_lib], L, spec["patient"]["saits_own_stride"]),
        "val": {"X": win(vi, L, 75), "X_ori": win(val, L, 75)},
        "test": win(zo[n_hist:], L, L),
        "held": held,
    }


def cross_data(spec: dict, task: str, g: int) -> dict:
    """Windows of the train patients (training) and validation patients (model choice), as in D1."""
    T, L, S = spec["tasks"][task], spec["tasks"][task]["L"], spec["subjects"]
    rng = d1._rng(41, d1.TASK_ID[task], g)
    tr = np.concatenate([win(d1.subject_series(task, g, s)["zo"], L, 125) for s in S["train"]])
    tr = tr[rng.permutation(len(tr))[: spec["cross"]["max_train_windows"]]]
    va_in, va_ori = [], []
    for s in S["val"]:
        z = d1.subject_series(task, g, s)["zo"]
        held = d1.blocks(len(z), 0.2, g, d1._rng(42, d1.TASK_ID[task], g, s)) & np.isfinite(z[:, 0])
        zi = z.copy()
        zi[held, 0] = np.nan
        va_in.append(win(zi, L, L))
        va_ori.append(win(z, L, L))
    return {"train": tr, "val": {"X": np.concatenate(va_in)[: spec["cross"]["max_val_windows"]],
                                 "X_ori": np.concatenate(va_ori)[: spec["cross"]["max_val_windows"]]}}




def _score_pred(spec: dict, task: str, g: int, sid: int, imp: np.ndarray) -> dict:
    """imp: [n_windows, L, F] imputations of the non-overlapping test windows of one patient."""
    d = d1.subject_series(task, g, sid)
    L = spec["tasks"][task]["L"]
    pred = imp[:, :, 0].reshape(-1)
    n = len(d["zo"]) - d["n_hist"]
    return d1._score(pred, d["zt"][d["n_hist"]:, 0][:n], d["miss"][d["n_hist"]:][:n], spec["embedding"][task])


# ----------------------------------------------------------------------------- model wrappers


def _saits(cfg: dict, L: int, F: int, epochs: int, patience: int, lr: float | None = None):
    from pypots.imputation import SAITS

    kw = {}
    if lr is not None:
        from pypots.optim import Adam

        kw["optimizer"] = Adam(lr=lr)
    return SAITS(n_steps=L, n_features=F, n_layers=cfg["n_layers"], d_model=cfg["d_model"], n_heads=4,
                 d_k=cfg["d_model"] // 4, d_v=cfg["d_model"] // 4, d_ffn=2 * cfg["d_model"], dropout=0.1,
                 batch_size=32, epochs=epochs, patience=patience, device="cuda", saving_path=None,
                 verbose=False, **kw)
def _csdi(L: int, F: int, epochs: int, patience: int, lr: float | None = None):
    from pypots.imputation import CSDI

    kw = {}
    if lr is not None:
        from pypots.optim import Adam

        kw["optimizer"] = Adam(lr=lr)
    return CSDI(n_steps=L, n_features=F, n_layers=4, n_heads=4, n_channels=64, d_time_embedding=128,
                d_feature_embedding=16, d_diffusion_embedding=128, n_diffusion_steps=50,
                target_strategy="mix", schedule="quad", beta_start=1e-4, beta_end=0.5, batch_size=32,
                epochs=epochs, patience=patience, device="cuda", saving_path=None, verbose=False, **kw)


def _fit_record(arm: str, cell: str, unit, cfg, model, seconds: float, cap: int) -> dict:
    best = int(getattr(model, "best_epoch", -1))
    return {"cell": cell, "unit": unit, "arm": arm, "config": json.dumps(cfg) if cfg else "",
            "best_epoch": best, "best_val_loss": float(getattr(model, "best_loss", np.nan)),
            "epochs_run": int(getattr(model, "epoch", best) or best), "seconds": round(seconds),
            "cap": cap, "hit_cap": bool(best >= 0.95 * cap)}


def fit_saits(spec, task, g, cfg, tr, va, te, epochs, patience, lr=None, init=None, seed=0):
    from pypots.utils.random import set_random_seed

    set_random_seed(seed)
    L, F = tr.shape[1], tr.shape[2]
    model = _saits(cfg, L, F, epochs, patience, lr)
    if init is not None:
        model.model.load_state_dict(init)
    t0 = time.time()
    with patched_mcar(g):
        model.fit({"X": tr}, va)
    fit_sec = time.time() - t0
    imp = np.asarray(model.predict({"X": te})["imputation"], float)
    return imp, model, fit_sec


def fit_csdi(spec, task, g, tr, va, te, epochs, patience, lr=None, init=None, seed=0):
    from pypots.utils.random import set_random_seed

    set_random_seed(seed)
    L, F = tr.shape[1], tr.shape[2]
    model = _csdi(L, F, epochs, patience, lr)
    if init is not None:
        model.model.load_state_dict(init)
    t0 = time.time()
    model.fit({"X": tr}, va)
    fit_sec = time.time() - t0
    samples = np.asarray(model.predict({"X": te}, n_sampling_times=spec["csdi_n_samples"])["imputation"], float)
    return np.median(samples, axis=1), model, fit_sec


def _quiet_pypots():
    import logging as _l

    from pypots.utils.logging import logger as pl

    pl.setLevel(_l.ERROR)


# ----------------------------------------------------------------------------- stages: cross / patient


def cross_cell(spec: dict, task: str, g: int) -> dict:
    _quiet_pypots()
    cell = f"{task}|g{g}"
    data = cross_data(spec, task, g)
    tr, va = data["train"], data["val"]
    tr_csdi = tr[: spec["cross"]["csdi_max_train_windows"]]
    tests = spec["subjects"]["test"]
    te = np.concatenate([patient_data(spec, task, g, s)["test"] for s in tests])
    emb_n = {s: len(patient_data(spec, task, g, s)["test"]) for s in tests}

    def split(imp):
        out, k = {}, 0
        for s in tests:
            out[s] = _score_pred(spec, task, g, s, imp[k:k + emb_n[s]])
            k += emb_n[s]
        return out

    arms, fits, states = {}, [], {}
    B = spec["cross"]
    best = None
    for i, cfg in enumerate(B["saits_grid"]):
        imp, model, sec = fit_saits(spec, task, g, cfg, tr, va, te, B["saits_epochs"], B["saits_patience"],
                                    seed=1000 * d1.TASK_ID[task] + g + i)
        fits.append(_fit_record("saits_cross", cell, "cross", cfg, model, sec, B["saits_epochs"]))
        loss = float(model.best_loss)
        if best is None or loss < best[0]:
            best = (loss, cfg, imp, {k: v.detach().cpu() for k, v in model.model.state_dict().items()})
    arms["saits_cross"] = split(best[2])
    states["saits"] = best[3]
    chosen = best[1]

    if task in spec["csdi_tasks"]:
        imp, model, sec = fit_csdi(spec, task, g, tr_csdi, va, te, B["csdi_epochs"], B["csdi_patience"],
                                   seed=2000 * d1.TASK_ID[task] + g)
        arms["csdi_cross"] = split(imp)
        fits.append(_fit_record("csdi_cross", cell, "cross", None, model, sec, B["csdi_epochs"]))
        states["csdi"] = {k: v.detach().cpu() for k, v in model.model.state_dict().items()}
    return {"cell": cell, "arms": arms, "fits": fits, "chosen_saits": chosen, "states": states,
            "n_train_windows": int(len(tr))}


def patient_unit(spec: dict, task: str, g: int, sid: int, cross: dict) -> dict:
    _quiet_pypots()
    cell = f"{task}|g{g}"
    pd_ = patient_data(spec, task, g, sid)
    tr, tr_own, va, te = pd_["train"], pd_["train_own"], pd_["val"], pd_["test"]
    cfg = cross["chosen_saits"]
    P = spec["patient"]
    arms, fits, preds = {}, [], {}
    seed = 3000 * d1.TASK_ID[task] + g + sid

    imp, model, sec = fit_saits(spec, task, g, cfg, tr_own, va, te, P["saits_own_epochs"], P["saits_own_patience"], seed=seed)
    arms["saits_own"] = _score_pred(spec, task, g, sid, imp)
    preds["saits_own"] = imp[:, :, 0].reshape(-1)
    fits.append(_fit_record("saits_own", cell, sid, cfg, model, sec, P["saits_own_epochs"]))

    imp, model, sec = fit_saits(spec, task, g, cfg, tr, va, te, P["saits_ft_epochs"], P["saits_ft_patience"],
                                lr=1e-4, init=cross["states"]["saits"], seed=seed + 1)
    arms["saits_ft"] = _score_pred(spec, task, g, sid, imp)
    preds["saits_ft"] = imp[:, :, 0].reshape(-1)
    fits.append(_fit_record("saits_ft", cell, sid, cfg, model, sec, P["saits_ft_epochs"]))

    if task in spec["csdi_tasks"]:
        imp, model, sec = fit_csdi(spec, task, g, tr, va, te, P["csdi_ft_epochs"], P["csdi_ft_patience"],
                                   lr=1e-4, init=cross["states"]["csdi"], seed=seed + 2)
        arms["csdi_ft"] = _score_pred(spec, task, g, sid, imp)
        preds["csdi_ft"] = imp[:, :, 0].reshape(-1)
        fits.append(_fit_record("csdi_ft", cell, sid, None, model, sec, P["csdi_ft_epochs"]))

    if sid in spec["csdi_own_subjects"] and task in spec["csdi_own_tasks"]:
        imp, model, sec = fit_csdi(spec, task, g, tr, va, te, P["csdi_own_epochs"], P["csdi_own_patience"], seed=seed + 3)
        arms["csdi_own"] = _score_pred(spec, task, g, sid, imp)
        preds["csdi_own"] = imp[:, :, 0].reshape(-1)
        fits.append(_fit_record("csdi_own", cell, sid, None, model, sec, P["csdi_own_epochs"]))
    return {"cell": cell, "unit": int(sid), "arms": arms, "fits": fits,
            "pred": preds if sid == spec["subjects"]["test"][0] else {}}


# ----------------------------------------------------------------------------- self-checks


def _runs(mask1d: np.ndarray) -> list[int]:
    d = np.diff(np.concatenate([[0], mask1d.view(np.int8), [0]]))
    return [int(b - a) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def selfcheck(spec: dict) -> pd.DataFrame:
    _quiet_pypots()
    rows = []

    def row(check, passed, note):
        rows.append({"check": check, "pass": bool(passed), "note": note, "smoke": bool(spec["smoke"])})
        log.info("selfcheck %s: %s (%s)", check, passed, note)

    # 1. Block mask: runs >= g in channel 0, none in channel 1, no window fully masked in channel 0.
    import torch
    from pypots.imputation.saits.data import DatasetForSAITS

    ok, notes = True, []
    for task, g in CELLS:
        F = len(spec["tasks"][task]["columns"])
        rng = np.random.default_rng(7)
        X = rng.normal(size=(6, 375, F)).astype(np.float32)
        with patched_mcar(g):
            ds = DatasetForSAITS({"X": X}, return_X_ori=False, return_y=False)
            for i in range(len(ds)):
                _, Xi, missing, _, indicating = ds[i]
                ind = indicating.numpy()
                if F > 1 and ind[:, 1].any():
                    ok, notes = False, notes + [f"{task}|g{g}: indicating mask in channel 1"]
                r = _runs(ind[:, 0] > 0)
                if r and min(r) < g:
                    ok, notes = False, notes + [f"{task}|g{g}: run {min(r)} < {g}"]
                if missing.numpy()[:, 0].all():
                    ok, notes = False, notes + [f"{task}|g{g}: window fully masked"]
    row("block_mask", ok, "; ".join(notes) or "runs >= g in channel 0 only; no fully masked window")

    # 2. Leakage: patient windows never reach the test region; test inputs are NaN at every gap.
    ok, notes = True, []
    for task, g in CELLS:
        sid = spec["subjects"]["test"][0]
        pd_ = patient_data(spec, task, g, sid)
        d = pd_["d"]
        n_hist, L = d["n_hist"], spec["tasks"][task]["L"]
        if len(pd_["train"]) and (len(d["zo"][:d["n_lib"]]) < L or d["n_lib"] > n_hist):
            ok, notes = False, notes + [f"{task}|g{g}: train window beyond history"]
        te = d["zo"][n_hist:]
        if not np.isnan(te[d["miss"][n_hist:], 0]).all():
            ok, notes = False, notes + [f"{task}|g{g}: test input not NaN at a gap"]
        if np.isfinite(te[d["miss"][n_hist:], 0]).any():
            ok = False
    row("leakage", ok, "; ".join(notes) or "train/val windows end at n_hist; test inputs NaN at all gaps")

    # 4 (run before 3: cheap). Learnability on a noise-free sine with block gaps.
    row("sine_learnability", *sine_check(spec))

    # 3. Reproduction of D1's SAITS arm (unpatched mask, D1 budget and data), cell ppg|g62.
    row("reproduce_d1_saits", *repro_check(spec))
    return pd.DataFrame(rows)


def sine_check(spec: dict) -> tuple[bool, str]:
    import torch

    t = np.arange(6000, dtype=float)
    x = np.sin(2 * np.pi * t / 50.0)[:, None].astype(np.float32)
    g = 20
    rng = np.random.default_rng(3)
    tr = win(x[:4000], 100, 25)
    va_x = x[4000:5000].copy()
    held = d1.blocks(len(va_x), 0.2, g, rng) & np.isfinite(va_x[:, 0])
    vi = va_x.copy()
    vi[held, 0] = np.nan
    va = {"X": win(vi, 100, 100), "X_ori": win(va_x, 100, 100)}
    te = win(vi, 100, 100)
    epochs = smoke(80, 40)
    imp, model, sec = fit_saits(spec, "ppg", g, {"n_layers": 2, "d_model": 64}, tr, va, te, epochs, 15, seed=5)
    truth = win(va_x, 100, 100)[:, :, 0]
    m = np.isnan(te[:, :, 0])
    mae = float(np.abs(imp[:, :, 0][m] - truth[m]).mean())
    return mae < 0.1, f"sine block-gap SAITS MAE {mae:.4f} sd (threshold 0.1), {sec:.0f} s"


def repro_check(spec: dict) -> tuple[bool, str]:
    task, g = "ppg", 62
    d1spec = json.loads((REPO / "notebooks" / "tables" / "dynimp" / "D1" / "spec.json").read_text(encoding="utf-8"))
    S, L = spec["subjects"], spec["tasks"][task]["L"]
    rng = d1._rng(41, d1.TASK_ID[task], g)
    tr = np.concatenate([win(d1.subject_series(task, g, s)["zo"], L, 125) for s in S["train"]])
    tr = tr[rng.permutation(len(tr))[: smoke(3000, 96)]]
    va_in, va_ori = [], []
    for s in S["val"]:
        z = d1.subject_series(task, g, s)["zo"]
        held = d1.blocks(len(z), 0.2, g, d1._rng(42, d1.TASK_ID[task], g, s)) & np.isfinite(z[:, 0])
        zi = z.copy()
        zi[held, 0] = np.nan
        va_in.append(win(zi, L, L))
        va_ori.append(win(z, L, L))
    va = {"X": np.concatenate(va_in)[: smoke(600, 32)], "X_ori": np.concatenate(va_ori)[: smoke(600, 32)]}
    grid = smoke(d1spec["deep_physio"]["saits_grid"], d1spec["deep_physio"]["saits_grid"][:1])
    epochs, patience = smoke((60, 8), (2, 2))
    best = None
    tests = S["test"]
    te = np.concatenate([patient_data(spec, task, g, s)["test"] for s in tests])
    best = None
    for i, cfg in enumerate(grid):
        # Unpatched fit (original point mask): this is the D1 pipeline, so _saits is used directly.
        from pypots.utils.random import set_random_seed

        set_random_seed(1000 * d1.TASK_ID[task] + g + i)
        model2 = _saits(cfg, L, tr.shape[2], epochs, patience)
        model2.fit({"X": tr}, va)
        imp = np.asarray(model2.predict({"X": te})["imputation"], float)
        loss = float(model2.best_loss)
        if best is None or loss < best[0]:
            best = (loss, imp)
    maes, k = [], 0
    for s in S["test"]:
        n = len(patient_data(spec, task, g, s)["test"])
        maes.append(_score_pred(spec, task, g, s, best[1][k:k + n])["mae"])
        k += n
    d1df = pd.read_csv(REPO / "notebooks" / "tables" / "dynimp" / "D1" / "per_unit.csv")
    ref = float(d1df[(d1df["cell"] == "ppg|g62") & (d1df["arm"] == "saits") &
                     (d1df["unit"].isin(S["test"]))]["mae"].mean())
    diff = abs(float(np.mean(maes)) - ref)
    if spec["smoke"]:
        return np.isfinite(diff), f"smoke plumbing only: MAE {np.mean(maes):.4f} vs D1 {ref:.4f} (threshold in full run)"
    return diff <= 0.03, f"mean MAE {np.mean(maes):.4f} vs D1 {ref:.4f}, |diff| {diff:.4f} (threshold 0.03)"


# ----------------------------------------------------------------------------- bench


def bench(spec: dict) -> dict:
    """Time one epoch of each kind of fit on cell ppg|g62 and extrapolate. No test scoring."""
    _quiet_pypots()
    task, g = "ppg", 62
    data = cross_data(spec, task, g)
    tr, va = data["train"][:512], data["val"]
    sid = spec["subjects"]["test"][0]
    pd_ = patient_data(spec, task, g, sid)
    timings = {}
    def epoch_model(name, fn, full):
        """Seconds per epoch as fixed + per-window * N, fitted at two training-set sizes (validation cost is fixed)."""
        fn(full[:512])  # warmup: CUDA kernels and data loaders pay one-off costs here
        t1 = float(fn(full[:512])[2])
        t2 = float(fn(full[:2048])[2])
        slope = max((t2 - t1) / 1536.0, 0.0)
        fixed = max(t1 - slope * 512.0, 0.0)
        timings[name] = {"s_per_epoch_512win": round(t1, 2), "s_per_epoch_2048win": round(t2, 2),
                         "fixed_s": round(fixed, 2), "per_window_s": round(slope, 5)}
        log.info("bench %s: %.1f s/epoch at 512 windows, %.1f at 2048 (fixed %.1f s + %.4f s/window)",
                 name, t1, t2, fixed, slope)
        return lambda N: fixed + slope * N

    cfg_big = {"n_layers": 2, "d_model": 256}
    saits_cross_ep = epoch_model("saits_cross", lambda X: fit_saits(spec, task, g, cfg_big, X, va, X[:32], 1, 1, seed=1), data["train"])
    csdi_cross_ep = epoch_model("csdi_cross", lambda X: fit_csdi(spec, task, g, X, va, X[:32], 1, 1, seed=1), data["train"])
    saits_pat_ep = epoch_model("saits_patient", lambda X: fit_saits(spec, task, g, cfg_big, X, pd_["val"], X[:32], 1, 1, seed=1), pd_["train_own"])
    csdi_pat_ep = epoch_model("csdi_patient", lambda X: fit_csdi(spec, task, g, X, pd_["val"], X[:32], 1, 1, seed=1), pd_["train_own"])

    B, P = spec["cross"], spec["patient"]
    n_cross_tr = min(len(data["train"]), B["max_train_windows"])
    # SAITS in D1 stopped at best epochs 3-9 (patience 8); the block task is harder, so assume best 30 + patience 30.
    # CSDI in D1 had best epochs 57-59 at the 60 cap; assume best 60 + patience 25.
    exp = {"saits_cross": min(B["saits_epochs"], 60), "csdi_cross": min(B["csdi_epochs"], 85),
           "saits_own": min(P["saits_own_epochs"], 60), "saits_ft": min(P["saits_ft_epochs"], 45),
           "csdi_ft": min(P["csdi_ft_epochs"], 25), "csdi_own": min(P["csdi_own_epochs"], 80)}
    n_pat = len(spec["subjects"]["test"])
    est = {}
    est["saits_cross"] = 4 * 3 * saits_cross_ep(n_cross_tr) * exp["saits_cross"]  # grid of 3, 4 cells
    est["csdi_cross"] = 4 * csdi_cross_ep(B["csdi_max_train_windows"]) * exp["csdi_cross"]
    n_tr_pat = len(pd_["train"])
    est["saits_own"] = saits_pat_ep(len(pd_["train_own"])) * exp["saits_own"] * n_pat * 4
    est["saits_ft"] = saits_pat_ep(n_tr_pat) * exp["saits_ft"] * n_pat * 4
    est["csdi_ft"] = csdi_pat_ep(n_tr_pat) * exp["csdi_ft"] * n_pat * 4
    est["csdi_own"] = csdi_pat_ep(n_tr_pat) * exp["csdi_own"] * len(spec["csdi_own_subjects"]) * 4
    # CSDI prediction (sampling) is not in the fit timing; add 20 % on CSDI arms.
    for k in ("csdi_cross", "csdi_ft", "csdi_own"):
        est[k] *= 1.2
    total_h = sum(est.values()) / 3600.0
    reductions = []
    if total_h > BUDGET_GPU_HOURS and not spec["smoke"]:
        # Reduction order fixed by the plan: (a) drop csdi_own, (b) csdi_ft epochs 40->20,
        # (c) patient stride 15->25, (d) CSDI arms only on PPG cells.
        reductions.append("csdi_own dropped")
        total_h -= est.pop("csdi_own") / 3600.0
        if total_h > BUDGET_GPU_HOURS:
            reductions.append("csdi_ft epochs 40 -> 20")
            total_h -= est["csdi_ft"] / 3600.0 * 0.5
            est["csdi_ft"] *= 0.5
        if total_h > BUDGET_GPU_HOURS:
            reductions.append("patient stride 15 -> 25")
            f = 15 / 25
            for k in ("saits_own", "saits_ft", "csdi_ft"):
                total_h -= est[k] / 3600.0 * (1 - f)
                est[k] *= f
        if total_h > BUDGET_GPU_HOURS:
            reductions.append("CSDI arms restricted to PPG cells")
            total_h -= (est["csdi_cross"] + est["csdi_ft"]) / 3600.0 * 0.5
    out = {"timings_s_per_epoch_512win": timings, "expected_epochs": exp,
           "estimate_hours_by_arm": {k: round(v / 3600.0, 2) for k, v in est.items()},
           "estimate_total_hours": round(total_h, 2), "budget_hours": BUDGET_GPU_HOURS, "reductions": reductions}
    log.info("bench estimate: %.2f GPU-hours; reductions %s", total_h, reductions)
    return out


# ----------------------------------------------------------------------------- spec


def build_spec() -> dict:
    d1spec = json.loads((REPO / "notebooks" / "tables" / "dynimp" / "D1" / "spec.json").read_text(encoding="utf-8"))
    subjects_full = d1spec["subjects"]
    sm = bool(smoke(False, True))
    subjects = {"test": subjects_full["test"][: smoke(20, 2)], "val": subjects_full["val"][: smoke(8, 2)],
                "train": subjects_full["train"][: smoke(25, 3)]}
    bench_adj = {}
    bp = out_dir() / "bench.json"
    if bp.exists():
        bench_adj = json.loads(bp.read_text(encoding="utf-8")).get("reductions", [])
    csdi_own_subjects = [] if "csdi_own dropped" in bench_adj else subjects["test"][: smoke(5, 1)]
    csdi_tasks = ["ppg"] if "CSDI arms restricted to PPG cells" in bench_adj else ["ppg", "ecg"]
    return {
        "version": SPEC_VERSION,
        "experiment": "dynimp D2: learned imputers given the patient's own history (block mask, larger budget)",
        "smoke": sm,
        "cells": [f"{t}|g{g}" for t, g in CELLS],
        "subjects": subjects,
        "subject_split": "exactly the D1 split (notebooks/tables/dynimp/D1/spec.json)",
        "tasks": d1.TASKS,
        "embedding": d1spec["embedding"],
        "data": "d1.subject_series unchanged: same series, gaps, scaling, history (first 60 %), library (first 50 %)",
        "arms": {
            "saits_cross": "25 train patients, block mask, SAITS grid chosen on the 8 validation patients per cell",
            "saits_own": "test patient's first 50 % only, from scratch, chosen configuration",
            "saits_ft": "saits_cross weights fine-tuned on the patient's first 50 % (gate arm)",
            "csdi_cross": "25 train patients, target_strategy mix",
            "csdi_ft": "csdi_cross weights fine-tuned on the patient's first 50 % (gate arm)",
            "csdi_own": "patient's first 50 % only, from scratch; first 5 test patients only (secondary)",
            "reused_from_D1": REUSED,
        },
        "patient": {
            "train_windows": "zo[:n_lib], length 375, stride below; validation zo[n_lib:n_hist] with the D1 held mask, stride 75; test zo[n_hist:] non-overlapping",
            "stride": 25 if "patient stride 15 -> 25" in bench_adj else smoke(15, 75),
            "saits_own_stride": 5 if (not sm and "patient stride 15 -> 25" not in bench_adj) else smoke(15, 75),
            "saits_own_epochs": smoke(300, 3), "saits_own_patience": smoke(30, 2),
            "saits_ft_epochs": smoke(100, 3), "saits_ft_patience": smoke(15, 2), "saits_ft_lr": 1e-4,
            "csdi_ft_epochs": 20 if "csdi_ft epochs 40 -> 20" in bench_adj else smoke(40, 3),
            "csdi_ft_patience": smoke(8, 2), "csdi_ft_lr": 1e-4,
            "csdi_own_epochs": smoke(150, 3), "csdi_own_patience": smoke(20, 2),
        },
        "cross": {
            "saits_grid": smoke([{"n_layers": 2, "d_model": 128}, {"n_layers": 2, "d_model": 256},
                                 {"n_layers": 4, "d_model": 128}], [{"n_layers": 2, "d_model": 128}]),
            "saits_epochs": smoke(300, 3), "saits_patience": smoke(30, 2),
            "csdi_epochs": smoke(200, 3), "csdi_patience": smoke(25, 2),
            "csdi_max_train_windows": smoke(6000, 48),
            "max_train_windows": smoke(12000, 96), "max_val_windows": smoke(600, 32),
            "train_windows": "stride 125 over the train patients, random subset with a fixed seed",
        },
        "csdi": "D1 configuration: 4 layers, 4 heads, 64 channels, 50 diffusion steps, target_strategy mix; point estimate = median of 10 samples",
        "csdi_n_samples": smoke(10, 2),
        "csdi_own_subjects": csdi_own_subjects,
        "csdi_own_tasks": csdi_tasks,
        "csdi_tasks": csdi_tasks,
        "block_mask": "PyPOTS SAITS mcar replaced during every SAITS fit by runs of g masked samples in the target "
                      "channel only, about 20 % of the window; the helper channel is never masked",
        "bench_reductions": bench_adj,
        "gate": {
            "primary": "MAE on the missing test samples, paired over the 20 patients, 95 % t-interval",
            "not_worse": "in a cell, the recurrence fill is not worse than a gate arm X (saits_ft, csdi_ft) iff the upper bound of "
                         "MAE_recurrence - MAE_X is below 0.10 * mean MAE_X; it is worse iff the lower bound is above 0.10 * mean MAE_X",
            "evaluable": "a gate arm counts in a cell only if its mean MAE is below 0.75 * the mean-fill MAE of that cell; "
                         "if either gate arm is not evaluable in a cell, that cell is not_evaluable",
            "D2_pass": "the recurrence fill is not worse than both gate arms in at least 3 of the 4 cells and no cell is "
                       "not_evaluable for both arms",
            "D2_fail": "the recurrence fill is worse than at least one gate arm in at least 2 cells",
            "D2_else": "otherwise the verdict is inconclusive",
        },
        "secondary": "attribution saits_d1 -> saits_cross (mask + budget), saits_cross -> saits_ft (own history), saits_own vs "
                     "saits_ft (pretraining); same for CSDI where arms exist; RP error for every arm; recurrence vs "
                     "boosting_helper (PPG cells, from D1); per-patient scatter data (in per_unit.csv)",
        "leakage": "no arm sees any sample of the test region and no arm sees a value at a gap; per-patient training uses "
                   "the first 50 % only, early stopping the 50-60 % slice with the D1 held mask",
        "budget": f"full run capped at {BUDGET_GPU_HOURS} GPU-hours (bench stage); reductions applied in the plan's order are in bench_reductions",
    }


# ----------------------------------------------------------------------------- evaluation


def collect(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, fits = [], []
    d1df = pd.read_csv(REPO / "notebooks" / "tables" / "dynimp" / "D1" / "per_unit.csv")
    for (task, g) in CELLS:
        cell = f"{task}|g{g}"
        sub = d1df[(d1df["cell"] == cell) & (d1df["unit"].isin(spec["subjects"]["test"]))]
        for new, old in REUSED.items():
            for r in sub[sub["arm"] == old].itertuples():
                rows.append({"cell": cell, "unit": int(r.unit), "arm": new, "mae": r.mae, "rp": r.rp})
    h12 = hashlib.sha256((spec_hash(spec) + code_fingerprint(MODULES)).encode()).hexdigest()[:12]
    for p in sorted(cache_dir().glob(f"*_{h12}.pkl")):
        res = pickle.loads(p.read_bytes())
        if not isinstance(res, dict) or "arms" not in res:
            continue
        fits += res.get("fits", [])
        if "unit" in res:
            rows += [{"cell": res["cell"], "unit": int(res["unit"]), "arm": a, **v} for a, v in res["arms"].items()]
        else:
            rows += [{"cell": res["cell"], "unit": int(u), "arm": a, **v}
                     for a, per in res["arms"].items() for u, v in per.items()]
    df = pd.DataFrame(rows).drop_duplicates(["cell", "unit", "arm"]).sort_values(["cell", "arm", "unit"]).reset_index(drop=True)
    return df, pd.DataFrame(fits)


def evaluate(spec: dict, df: pd.DataFrame) -> dict:
    sm = bool(spec["smoke"])
    summary, pairs, decision = [], [], []
    not_worse_cells, worse_cells, both_unevaluable = 0, 0, 0
    for (task, g) in CELLS:
        cell = f"{task}|g{g}"
        gg = df[df["cell"] == cell]
        if not len(gg):
            continue
        for arm, ga in gg.groupby("arm"):
            rec = {"cell": cell, "arm": arm, "n": len(ga)}
            for col in ["mae", "rp"]:
                if ga[col].notna().any():
                    rec[col], rec[f"{col}_lo"], rec[f"{col}_hi"] = d0.t_interval(ga[col])
            summary.append(rec)
        piv = gg.pivot(index="unit", columns="arm", values="mae")
        if "rp" in gg:
            pr = gg.pivot(index="unit", columns="arm", values="rp")
            for a in pr.columns:
                if a != "recurrence" and "recurrence" in pr:
                    dm, lo, hi = d0.t_interval((pr["recurrence"] - pr[a]).dropna())
                    pairs.append({"cell": cell, "a": "recurrence", "b": a, "metric": "rp", "diff": dm, "lo": lo, "hi": hi,
                                  "mean_b": float(pr[a].mean())})
        if "recurrence" not in piv or "mean" not in piv:
            continue
        mean_fill = float(piv["mean"].mean())
        cell_not_worse, cell_worse, evaluable = True, False, {}
        for arm in GATE_ARMS:
            if arm not in piv:
                evaluable[arm] = False
                cell_not_worse = False
                decision.append({"gate": "cell_arm", "cell": cell, "statistic": f"recurrence - {arm} MAE", "value": np.nan,
                                 "ci_low": np.nan, "ci_high": np.nan, "rule": "upper bound < 0.10 * mean MAE of the gate arm",
                                 "verdict": "missing", "note": f"{arm} not run", "evaluable": False, "smoke": sm})
                continue
            ev = float(piv[arm].mean()) < 0.75 * mean_fill
            evaluable[arm] = ev
            dm, lo, hi = d0.t_interval((piv["recurrence"] - piv[arm]).dropna())
            margin = 0.10 * float(piv[arm].mean())
            verdict = "not_evaluable" if not ev else ("not_worse" if hi < margin else "worse" if lo > margin else "neither")
            if not ev or verdict != "not_worse":
                cell_not_worse = False
            if ev and verdict == "worse":
                cell_worse = True
            pairs.append({"cell": cell, "a": "recurrence", "b": arm, "metric": "mae", "diff": dm, "lo": lo, "hi": hi,
                          "mean_b": float(piv[arm].mean())})
            decision.append({"gate": "cell_arm", "cell": cell, "statistic": f"recurrence - {arm} MAE", "value": dm,
                             "ci_low": lo, "ci_high": hi, "rule": "not worse iff upper bound < 0.10 * mean MAE of the gate arm; "
                             "worse iff lower bound above it", "verdict": verdict,
                             "note": f"margin {margin:.4f}; {arm} MAE {piv[arm].mean():.4f}; mean-fill MAE {mean_fill:.4f}",
                             "evaluable": ev, "smoke": sm})
        # attribution pairs (secondary, no verdict)
        for a, b in [("saits_ft", "saits_cross"), ("saits_cross", "saits_d1"), ("saits_ft", "saits_own"),
                     ("csdi_ft", "csdi_cross"), ("csdi_cross", "csdi_d1"), ("csdi_ft", "csdi_own"),
                     ("recurrence", "boosting_helper")]:
            if a in piv and b in piv:
                dm, lo, hi = d0.t_interval((piv[a] - piv[b]).dropna())
                pairs.append({"cell": cell, "a": a, "b": b, "metric": "mae", "diff": dm, "lo": lo, "hi": hi,
                              "mean_b": float(piv[b].mean())})
        if not any(evaluable.values()):
            both_unevaluable += 1
        if cell_not_worse and all(evaluable.values()):
            not_worse_cells += 1
        if cell_worse:
            worse_cells += 1
    if worse_cells >= 2:
        verdict = "fail"
    elif not_worse_cells >= 3 and both_unevaluable == 0:
        verdict = "pass"
    else:
        verdict = "inconclusive"
    decision.append({"gate": "D2", "cell": "all", "statistic": "cells not worse than both gate arms / cells worse than a gate arm",
                     "value": not_worse_cells, "ci_low": np.nan, "ci_high": np.nan,
                     "rule": "pass iff not worse than both gate arms in at least 3 of 4 cells and no cell not_evaluable for both "
                             "arms; fail iff worse than at least one gate arm in at least 2 cells; else inconclusive",
                     "verdict": verdict + (" (smoke: mechanical only)" if sm else ""),
                     "note": f"not_worse cells {not_worse_cells}, worse cells {worse_cells}, cells with both arms not evaluable {both_unevaluable}",
                     "evaluable": "", "smoke": sm})
    return {"summary": pd.DataFrame(summary), "pairs": pd.DataFrame(pairs), "decision": decision}


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["selfcheck", "bench", "cross", "patient", "eval"])
    ap.add_argument("--only", action="append", default=[], help="cell like ppg|g62; repeatable (cross/patient stages)")
    args = ap.parse_args()
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(out / f"run_{args.stage}.log", mode="a", encoding="utf-8")):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    spec = build_spec()
    t0 = time.time()
    if args.stage == "bench":
        # Timing only, before the spec is frozen; reductions land in bench.json and are merged into the spec.
        res = bench(spec)
        (out / "bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        log.info("bench written to %s", out / "bench.json")
        return
    write_spec(spec, out, modules=MODULES)  # before any model of the experiment is trained
    log.info("spec v%s, hash %s, code %s", spec["version"], spec_hash(spec)[:12], code_fingerprint(MODULES)[:12])
    if args.stage == "selfcheck":
        df = selfcheck(spec)
        save_table(df, out, "selfcheck")
        if not bool(df["pass"].all()):
            raise SystemExit("selfcheck failed; see selfcheck.csv")
    elif args.stage in ("cross", "patient"):
        cells = [(t, g) for t, g in CELLS if not args.only or f"{t}|g{g}" in args.only]
        for task, g in cells:
            cross = cached(spec, f"stage_cross_{task}_g{g}", lambda: cross_cell(spec, task, g))
            log.info("%s: chosen SAITS %s, %d train windows, fits %s", cross["cell"], cross["chosen_saits"],
                     cross["n_train_windows"], [(f["arm"], f["best_epoch"], f["seconds"]) for f in cross["fits"]])
            if args.stage == "patient":
                for sid in spec["subjects"]["test"]:
                    r = cached(spec, f"unit_{task}_g{g}_s{sid}", lambda: patient_unit(spec, task, g, sid, cross))
                    log.info("%s unit %s: %s", r["cell"], r["unit"],
                             " ".join(f"{a} {v['mae']:.3f}" for a, v in r["arms"].items()))
    else:
        df, fits = collect(spec)
        save_table(df, out, "per_unit")
        if len(fits):
            save_table(fits.drop_duplicates(["cell", "unit", "arm"]), out, "fit")
        ev = evaluate(spec, df)
        save_table(ev["summary"], out, "summary")
        save_table(ev["pairs"], out, "pairs")
        write_decision(ev["decision"], out)
        for d in ev["decision"]:
            log.info("%s | %s | %s = %s -> %s %s", d["gate"], d["cell"], d["statistic"], d["value"], d["verdict"], d["note"])
    log.info("stage %s done in %.0f s", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
