"""D1/D2 rerun with only the recurrence fill changed: context-only overlap rule and averaging over C.

Question (plans/dynimp/D1_D2_C_Rerun.md): does the recurrence fill improve on its D1/D2 version when
(a) the minimum-overlap rule counts only entries that can be observed (context-only rule, thresholds
0.1 and 0.2) and (b) the fills from all C values in the grid are averaged instead of picking one C on
validation? Only the recurrence fill is rerun; every other arm is read from the stored D1/D2
``per_unit.csv`` tables. D2 never ran the recurrence fill itself (its rows are copied from D1), so
rerunning the D1 recurrence arms covers D2. CPU only. ``dynimp.fills``/``d0``/``d1``/``d2`` are imported,
never edited.

Run from the repo root with ``PYTHONPATH=src`` and ``.venv-dynimp/Scripts/python.exe -m dynimp.d1_d2_c_rerun``:

    --stage design   validation data only; chooses the primary arm; writes spec.json and design.json
    --stage run      test data; all arms; repro check of rec_w03_sel against the stored D1 rows
    --stage eval     improvement test, attribution, D1/D2 gates re-scored (needs a passing repro check)
    --stage write    Writeups/dynimp_D1_D2_C_Rerun.md from the tables

``CONDPRED_SMOKE=1`` runs a tiny version into separate ``smoke/`` folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from condpred.config import code_fingerprint, smoke, spec_hash
from condpred.report import save_table, write_decision, write_spec
from dynimp import d0, d1, d2
from dynimp.fills import nan_runs
from dynimp.fills_ctx import average_over_C, recurrence_fill_multi, rule_name

REPO = Path(__file__).resolve().parents[2]
MODULES = ["dynimp.fills_ctx", "dynimp.d1_d2_c_rerun", "dynimp.d1", "dynimp.d0", "dynimp.fills"]
SPEC_VERSION = 1

RULES = [("window", 0.3), ("context", 0.1), ("context", 0.2)]
KS = (8, 32)
CANDIDATES = ["rec_w03_avg", "rec_c01_sel", "rec_c01_avg", "rec_c02_sel", "rec_c02_avg"]
ARMS = ["rec_w03_sel", "rec_w03_avg", "rec_c01_sel", "rec_c01_avg", "rec_c02_sel", "rec_c02_avg",
        "rec_c01_k32_sel", "rec_c02_k32_sel"]
# arm -> (rule, k, C handling)
ARM_DEF = {
    "rec_w03_sel": (("window", 0.3), 8, "sel"), "rec_w03_avg": (("window", 0.3), 8, "avg"),
    "rec_c01_sel": (("context", 0.1), 8, "sel"), "rec_c01_avg": (("context", 0.1), 8, "avg"),
    "rec_c02_sel": (("context", 0.2), 8, "sel"), "rec_c02_avg": (("context", 0.2), 8, "avg"),
    "rec_c01_k32_sel": (("context", 0.1), 32, "sel"), "rec_c02_k32_sel": (("context", 0.2), 32, "sel"),
}
VARIANTS_PHYSIO = {"ppg": ["recurrence", "recurrence_own", "recurrence_helper"], "ecg": ["recurrence"]}
STORED_ARM = {"recurrence": "recurrence", "recurrence_own": "recurrence_own", "recurrence_helper": "recurrence_helper"}

log = logging.getLogger("dynimp.d1_d2_c_rerun")


def out_dir() -> Path:
    base = REPO / "notebooks" / "tables" / "dynimp" / "D1_D2_C_Rerun"
    assert base.name == "D1_D2_C_Rerun"
    return base / "smoke" if smoke(False, True) else base


def cache_dir() -> Path:
    return REPO / "data" / "processed" / "dynimp" / "D1_D2_C_Rerun" / smoke("full", "smoke")


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


# ----------------------------------------------------------------------------- spec


def build_spec(primary_arm: str | None) -> dict:
    d1spec = d1.build_spec()
    test_subjects = d1spec["subjects"]["test"][: smoke(len(d1spec["subjects"]["test"]), 1)]
    return {
        "version": SPEC_VERSION,
        "experiment": "dynimp D1/D2 rerun: recurrence fill with the context-only overlap rule and averaging over C",
        "smoke": bool(smoke(False, True)),
        "cells": d1spec["cells"],
        "chaotic": {"cells": d1spec["chaotic"]["cells"], "seeds": d1spec["chaotic"]["seeds"],
                    "data": "exactly the D1 chaotic protocol (D0 cells, 10 % noise, 40 % missing); data preparation "
                            "copied from d1.chaotic_cpu_unit without the low-rank arm"},
        "subjects": {"test": [int(s) for s in test_subjects]},
        "subject_split": "exactly the D1 split (notebooks/tables/dynimp/D1/spec.json)",
        "tasks": d1.TASKS,
        "embedding": d1spec["embedding"],
        "val_runs": d1spec["val_runs"],
        "chaotic_Cs": [5, 10, 20, 40],
        "rules": [{"kind": k, "frac": f, "name": rule_name((k, f))} for k, f in RULES],
        "rule_formulas": {"window": "cnt >= frac * F * (g + 2C) (D1 rule; denominator includes the g unobservable "
                                    "target entries inside the gap)",
                          "context": "cnt >= frac * (F * (g + 2C) - g) (only entries that can be observed)"},
        "ks": list(KS),
        "arms": {a: {"rule": list(ARM_DEF[a][0]), "k": ARM_DEF[a][1], "C": ARM_DEF[a][2]} for a in ARMS},
        "candidates_for_primary": CANDIDATES,
        "primary_arm": primary_arm,
        "primary_rule": "the candidate arm with the lowest mean validation MAE over the cells present "
                        "(per-cell mean over units first, then the unweighted mean over cells; PPG: recurrence "
                        "variant). The k = 32 controls are not eligible. Chosen at the design stage, on "
                        "validation data only, before any test data are scored.",
        "comparison_rule": {
            "per_cell": "paired over patients or seeds with d0.t_interval: MAE(primary) - MAE(D1 recurrence); "
                        "improved iff the upper bound < 0; worse iff the lower bound > 0",
            "verdict": "adopt iff improved in at least 5 of the 10 D1 cells and worse in none; reject iff worse "
                       "in at least 2 cells; otherwise inconclusive",
            "other_arms": "all other arms, PPG variants and RP error: reported with intervals, no verdict",
        },
        "variants_ppg": "every arm is run for recurrence (PPG context + ECG; the gate arm), recurrence_own "
                        "(PPG only) and recurrence_helper (ECG only, channel weights [0, 1]); variant arms are "
                        "named <arm>_own / <arm>_helper in per_unit.csv. The primary analysis uses recurrence.",
        "protocol": "everything else exactly as in D1: same cells, C grids, library splits (chaotic: train for "
                    "validation, train + validation for test; BIDMC: first 50 % for validation, first 60 % for "
                    "test), gap masks, validation runs and scoring (d1._score; chaotic MAE over all channels)",
        "rerun_scope": "only the recurrence fill is rerun; SAITS, CSDI, low-rank, boosting and the other arms "
                       "are read from notebooks/tables/dynimp/D1/per_unit.csv and D2/per_unit.csv. D2's recurrence "
                       "rows are copied from D1 in d2.collect, so the D1 rerun covers D2.",
        "disclosure": "the context-only rule was chosen from a validation-only probe; the averaging-over-C idea "
                      "comes from a probe scored on the D1 test patients (not formed blind to test data) and "
                      "the writeup must say so",
        "original_verdicts": "the original D1 (PASS) and D2 (INCONCLUSIVE) verdicts stand; re-scored verdicts "
                             "are a post hoc reanalysis",
        "leakage": "no method receives a missing value; C selection uses held-out observed samples only",
    }


def _frozen_primary() -> str:
    p = out_dir() / "spec.json"
    if not p.exists():
        raise RuntimeError(f"no spec at {p}: run --stage design first")
    primary = json.loads(p.read_text(encoding="utf-8"))["primary_arm"]
    if primary not in CANDIDATES:
        raise RuntimeError(f"spec primary_arm {primary!r} is not a candidate arm")
    return primary


# ----------------------------------------------------------------------------- fill computation


def _variants(task: str, F: int):
    """(variant, column indices, channel weights), exactly as in d1.physio_cpu_unit."""
    out = [("recurrence", list(range(F)), None)]
    if task == "ppg":
        out += [("recurrence_own", [0], None), ("recurrence_helper", [0, 1], [0.0, 1.0])]
    return out


def _sel_fill(per_C: dict, arm: str, C):
    rule, k, _ = ARM_DEF[arm]
    return per_C[C][(rule_name(rule), k)]


def _multi_series(series, lib, cols, weights, Cs, runs):
    """One recurrence_fill_multi per C. Returns ({C: fills}, [stats records])."""
    per_C, stats = {}, []
    for C in Cs:
        r = recurrence_fill_multi(series[:, cols], lib[:, cols], 0, C=C, rules=RULES, ks=KS,
                                  weights=weights, runs=runs)
        per_C[C] = r["fills"]
        for rn, st in r["stats"].items():
            stats.append({"C": int(C), "rule": rn, **st})
    return per_C, stats


def _pick_C(per_C: dict, err_fn) -> dict:
    """arm -> (chosen C, {arm: fill on this series}) for the sel arms, minimising err_fn(fill)."""
    chosen, fills = {}, {}
    for arm in ARMS:
        rule, k, mode = ARM_DEF[arm]
        if mode != "sel":
            continue
        rn = rule_name(rule)
        C = min(per_C, key=lambda C: err_fn(per_C[C][(rn, k)]))
        chosen[arm] = int(C)
        fills[arm] = per_C[C][(rn, k)]
    return chosen, fills


def _combine(per_C: dict, chosen: dict) -> dict:
    arms = {}
    for arm in ARMS:
        rule, k, mode = ARM_DEF[arm]
        arms[arm] = _sel_fill(per_C, arm, chosen[arm]) if mode == "sel" else average_over_C(
            {C: per_C[C][(rule_name(rule), k)] for C in per_C})
    return arms


# ----------------------------------------------------------------------------- validation (design)


def physio_val(spec: dict, task: str, g: int, sid: int) -> dict:
    """Validation fills for one BIDMC test subject, exactly the D1 tuning setup. No test data touched."""
    T = spec["tasks"][task]
    d = d1.subject_series(task, g, sid)
    zo, n_hist, n_lib = d["zo"], d["n_hist"], d["n_lib"]
    lib, val = zo[:n_lib], zo[n_lib:n_hist]
    held = d1.blocks(len(val), 0.2, g, d1._rng(32, d1.TASK_ID[task], g, sid)) & np.isfinite(val[:, 0])
    v = val.copy()
    v[held, 0] = np.nan
    runs = [r for r in nan_runs(v[:, 0]) if held[r[0]:r[1]].all()][: spec["val_runs"]]
    idx = np.concatenate([np.arange(a, b) for a, b in runs])
    err = lambda pred: float(np.nanmean(np.abs(np.nan_to_num(pred[idx], nan=0.0) - val[idx, 0])))
    per_arm, per_C_err, choices, stats = {}, {}, {}, []
    for variant, cols, w in _variants(task, zo.shape[1]):
        per_C, st = _multi_series(v, lib, cols, w, T["Cs"], runs)
        stats += [{"variant": variant, **s} for s in st]
        chosen, _ = _pick_C(per_C, err)
        arms = _combine(per_C, chosen)
        for arm, fill in arms.items():
            per_arm[f"{variant}|{arm}"] = err(fill)
        for C, fills in per_C.items():
            for (rn, k), fill in fills.items():
                per_C_err[f"{variant}|C{C}|{rn}|k{k}"] = err(fill)
        choices[variant] = chosen
    return {"cell": f"{task}|g{g}", "unit": int(sid), "per_arm": per_arm, "per_C": per_C_err,
            "choices": choices, "stats": stats}


def chaotic_prep(spec: dict, name: str, seed: int) -> dict:
    """Copied from d1.chaotic_cpu_unit (about 15 lines) without the low-rank arm, which is not needed."""
    cell = d1._d0_cell(d0.build_spec(), name)
    system, pattern = cell["system"], cell["pattern"]
    sid, pid = d0._SYS_ID[system], d0._PATTERN_ID[pattern]
    n_tr, n_va, _ = d0.split_sizes(system)
    x = d0.complete_series(system, cell["noise"], seed)
    F = x.shape[1]
    miss = d0.missing_mask(len(x), F, 0.4, pattern, 10, 3, sid, seed, pid)
    xo = np.where(miss, np.nan, x)
    mu, sd = np.nanmean(xo[:n_tr], 0), np.nanstd(xo[:n_tr], 0)
    zt, zo = (x - mu) / sd, (xo - mu) / sd
    tr, va, test = zo[:n_tr], zo[n_tr:n_tr + n_va], zo[n_tr + n_va:]
    held = d0.missing_mask(n_va, F, 0.2, pattern, 10, 4, sid, seed, pid) & np.isfinite(va)
    v = np.where(held, np.nan, va)
    return {"cell": cell, "F": F, "tr": tr, "va": va, "v": v, "held": held, "test": test,
            "truth": zt[n_tr + n_va:], "miss": miss[n_tr + n_va:]}


def chaotic_val(spec: dict, name: str, seed: int) -> dict:
    p = chaotic_prep(spec, name, seed)
    Cs = spec["chaotic_Cs"]
    verr = lambda f: float(np.nanmean(np.abs(np.nan_to_num(f, nan=0.0)[p["held"]] - p["va"][p["held"]])))
    per_C_cols, stats = {}, []
    for C in Cs:
        per_ch = []
        for c in range(p["F"]):
            r = recurrence_fill_multi(p["v"], p["tr"], c, C=C, rules=RULES, ks=KS)
            per_ch.append(r["fills"])
            for rn, st in r["stats"].items():
                stats.append({"variant": "recurrence", "C": int(C), "rule": rn, **st})
        per_C_cols[C] = per_ch
    # per_C[C][(rn,k)] as a full [n_va, F] array
    per_C = {C: {(rn, k): np.column_stack([per_C_cols[C][c][(rn, k)] for c in range(p["F"])])
                 for rn in [rule_name(r) for r in RULES] for k in KS} for C in Cs}
    chosen, _ = _pick_C(per_C, verr)
    arms = _combine(per_C, chosen)
    return {"cell": name, "unit": int(seed),
            "per_arm": {f"recurrence|{a}": verr(f) for a, f in arms.items()},
            "per_C": {f"recurrence|C{C}|{rn}|k{k}": verr(per_C[C][(rn, k)])
                      for C in Cs for (rn, k) in per_C[C]},
            "choices": {"recurrence": chosen}, "stats": stats}


# ----------------------------------------------------------------------------- test units


def physio_unit(spec: dict, task: str, g: int, sid: int) -> dict:
    T = spec["tasks"][task]
    d = d1.subject_series(task, g, sid)
    zo, n_hist = d["zo"], d["n_hist"]
    hist, test, truth, miss = zo[:n_hist], zo[n_hist:], d["zt"][n_hist:, 0], d["miss"][n_hist:]
    emb = spec["embedding"][task]
    val = physio_val(spec, task, g, sid)  # C choices from validation only
    arms, choices, stats = {}, {}, []
    for variant, cols, w in _variants(task, zo.shape[1]):
        per_C, st = _multi_series(test, hist, cols, w, T["Cs"], None)
        stats += [{"variant": variant, **s} for s in st]
        fills = _combine(per_C, val["choices"][variant])
        suffix = "" if variant == "recurrence" else "_" + variant.split("_", 1)[1]
        for arm, fill in fills.items():
            arms[f"{arm}{suffix}"] = d1._score(fill, truth, miss, emb)
        choices[variant] = val["choices"][variant]
    return {"cell": f"{task}|g{g}", "unit": int(sid), "arms": arms, "choices": choices, "stats": stats}


def chaotic_unit(spec: dict, name: str, seed: int) -> dict:
    p = chaotic_prep(spec, name, seed)
    Cs = spec["chaotic_Cs"]
    lib = np.vstack([p["tr"], p["va"]])
    val = chaotic_val(spec, name, seed)
    chosen = val["choices"]["recurrence"]
    mae = lambda f: float(np.abs(np.nan_to_num(f, nan=0.0)[p["miss"]] - p["truth"][p["miss"]]).mean())
    per_C, stats = {}, []
    for C in Cs:
        per_ch = []
        for c in range(p["F"]):
            r = recurrence_fill_multi(p["test"], lib, c, C=C, rules=RULES, ks=KS)
            per_ch.append(r["fills"])
            for rn, st in r["stats"].items():
                stats.append({"variant": "recurrence", "C": int(C), "rule": rn, **st})
        per_C[C] = {(rn, k): np.column_stack([per_ch[c][(rn, k)] for c in range(p["F"])])
                    for rn in [rule_name(r) for r in RULES] for k in KS}
    arms = {a: {"mae": mae(f)} for a, f in _combine(per_C, chosen).items()}
    return {"cell": name, "unit": int(seed), "arms": arms, "choices": {"recurrence": chosen}, "stats": stats}


# ----------------------------------------------------------------------------- jobs / collect


def _val_job(spec, job):
    return physio_val(spec, job[1], job[2], job[3]) if job[0] == "physio" else chaotic_val(spec, job[1], job[2])


def _run_job(spec, job):
    if job[0] == "physio":
        _, task, g, sid = job
        return cached(spec, f"unit_{task}_g{g}_s{sid}", lambda: physio_unit(spec, task, g, sid))
    return cached(spec, f"unit_chaotic_{job[1].replace('|', '_')}_s{job[2]}",
                  lambda: chaotic_unit(spec, job[1], job[2]))


def _jobs(spec):
    jobs = [("chaotic", name, s) for name in spec["chaotic"]["cells"] for s in spec["chaotic"]["seeds"]]
    jobs += [("physio", t, g, s) for t in d1.TASKS for g in d1.TASKS[t]["gaps"] for s in spec["subjects"]["test"]]
    return jobs


def _run_pool(spec, jobs, fn, workers):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, spec, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            log.info("[%d/%d] %s unit %s done", k, len(jobs), r["cell"], r["unit"])
    return out


def collect(spec) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, stats = [], []
    h12 = hashlib.sha256((spec_hash(spec) + code_fingerprint(MODULES)).encode()).hexdigest()[:12]
    for p in sorted(cache_dir().glob(f"unit_*_{h12}.pkl")):
        r = pickle.loads(p.read_bytes())
        rows += [{"cell": r["cell"], "unit": r["unit"], "arm": a, **v} for a, v in r["arms"].items()]
        stats += [{"cell": r["cell"], "unit": r["unit"], **s} for s in r.get("stats", [])]
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates(["cell", "unit", "arm"]).sort_values(["cell", "arm", "unit"]).reset_index(drop=True)
    return df, pd.DataFrame(stats)


def d1_stored() -> pd.DataFrame:
    return pd.read_csv(REPO / "notebooks" / "tables" / "dynimp" / "D1" / "per_unit.csv")


def repro_check(df: pd.DataFrame) -> pd.DataFrame:
    """rec_w03_sel (and its PPG variants) against the stored D1 recurrence rows, per unit."""
    stored = d1_stored()
    rows = []
    for (cell, unit), g in df.groupby(["cell", "unit"]):
        for variant, stored_arm in STORED_ARM.items():
            if variant != "recurrence" and not cell.startswith("ppg"):
                continue
            suffix = "" if variant == "recurrence" else "_" + variant.split("_", 1)[1]
            new = g[g["arm"] == f"rec_w03_sel{suffix}"]
            old = stored[(stored["cell"] == cell) & (stored["unit"] == unit) & (stored["arm"] == stored_arm)]
            if not len(new) or not len(old):
                continue
            d = abs(float(new["mae"].iloc[0]) - float(old["mae"].iloc[0]))
            rows.append({"cell": cell, "unit": int(unit), "variant": variant, "arm": f"rec_w03_sel{suffix}",
                         "stored_arm": stored_arm, "new_mae": float(new["mae"].iloc[0]),
                         "d1_mae": float(old["mae"].iloc[0]), "abs_diff": d, "pass": bool(d <= 1e-9)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- stages


def stage_design(workers: int) -> None:
    spec = build_spec(None)
    spec["chaotic_Cs"] = [5, 10, 20, 40]
    results = _run_pool(spec, _jobs(spec), _val_job, workers)
    # per-cell mean over units, then unweighted mean over cells (recurrence variant only)
    cells = sorted({r["cell"] for r in results})
    design = {"cells": cells, "per_arm": {}, "per_C": {}, "choices": []}
    for r in results:
        design["choices"].append({"cell": r["cell"], "unit": r["unit"],
                                  "choices": {v: {a: (int(c) if c is not None else None) for a, c in ch.items()}
                                              for v, ch in r["choices"].items()}})
    mean_val = {}
    for arm in CANDIDATES:
        key = f"recurrence|{arm}"
        per_cell = [np.mean([r["per_arm"][key] for r in results if r["cell"] == cell]) for cell in cells]
        mean_val[arm] = float(np.mean(per_cell))
        design["per_arm"][arm] = {"mean_validation_mae_over_cells": mean_val[arm],
                                  "per_cell": {c: float(v) for c, v in zip(cells, per_cell)}}
    for arm in ARMS:  # all arms reported in design.json, candidates or not
        if arm in design["per_arm"]:
            continue
        key = f"recurrence|{arm}"
        per_cell = [np.mean([r["per_arm"][key] for r in results if r["cell"] == cell]) for cell in cells]
        design["per_arm"][arm] = {"mean_validation_mae_over_cells": float(np.mean(per_cell)),
                                  "per_cell": {c: float(v) for c, v in zip(cells, per_cell)}}
    keys = sorted({k for r in results for k in r["per_C"] if k.startswith("recurrence|")})
    design["per_C"] = {k: {c: float(np.mean([r["per_C"][k] for r in results if r["cell"] == c]))
                           for c in cells if all(k in r["per_C"] for r in results if r["cell"] == c)}
                       for k in keys}
    primary = min(CANDIDATES, key=lambda a: mean_val[a])
    design["primary_arm"] = primary
    design["note"] = ("validation data only (D1 validation parts of the test subjects and the chaotic "
                      "validation segments); no test data scored")
    spec = build_spec(primary)
    spec["chaotic_Cs"] = [5, 10, 20, 40]
    out = out_dir()
    write_spec(spec, out, modules=MODULES)  # frozen before any test data are scored
    (out / "design.json").write_text(json.dumps(design, indent=1), encoding="utf-8")
    log.info("primary arm %s (validation MAE %s)", primary, {a: round(v, 4) for a, v in mean_val.items()})


def stage_run(workers: int) -> None:
    spec = build_spec(_frozen_primary())
    spec["chaotic_Cs"] = [5, 10, 20, 40]
    out = out_dir()
    write_spec(spec, out, modules=MODULES)
    _run_pool(spec, _jobs(spec), _run_job, workers)
    df, stats = collect(spec)
    save_table(df, out, "per_unit")
    if len(stats):
        save_table(_agg_stats(stats), out, "overlap_stats")
    rc = repro_check(df)
    save_table(rc, out, "repro_check")
    log.info("repro check: %d/%d pass", int(rc["pass"].sum()) if len(rc) else 0, len(rc))
    if len(rc) and not bool(rc["pass"].all()):
        log.error("repro check FAILED; eval will stop. Worst rows:\n%s",
                  rc.sort_values("abs_diff", ascending=False).head().to_string())


def _agg_stats(stats: pd.DataFrame) -> pd.DataFrame:
    g = stats.groupby(["cell", "variant", "C", "rule"], as_index=False)[
        ["n_gaps", "unfilled", "cand_total", "cand_rejected"]].sum()
    g["rejected_share"] = g["cand_rejected"] / g["cand_total"].replace(0, np.nan)
    return g


def _replace_recurrence(stored: pd.DataFrame, df: pd.DataFrame, arm: str, cells=None) -> pd.DataFrame:
    new = df[df["arm"] == arm].copy()
    if cells is not None:
        new = new[new["cell"].isin(cells)]
    new["arm"] = "recurrence"
    keep = [c for c in stored.columns if c in new.columns]
    covered = set(new["cell"])
    kept = stored[~((stored["arm"] == "recurrence") & (stored["cell"].isin(covered)))]
    return pd.concat([kept, new[keep]], ignore_index=True)


def stage_eval() -> None:
    out = out_dir()
    rc = pd.read_csv(out / "repro_check.csv")
    if not len(rc) or not bool(rc["pass"].all()):
        raise SystemExit("repro check failed or missing; eval stops (see repro_check.csv)")
    spec = build_spec(_frozen_primary())
    spec["chaotic_Cs"] = [5, 10, 20, 40]
    primary = spec["primary_arm"]
    df, stats = collect(spec)
    save_table(df, out, "per_unit")
    if len(stats):
        save_table(_agg_stats(stats), out, "overlap_stats")
    stored = d1_stored()

    summary, pairs = [], []
    for (cell, arm), ga in df.groupby(["cell", "arm"]):
        rec = {"cell": cell, "arm": arm, "n": len(ga)}
        for col in ["mae", "rp"]:
            if col in ga and ga[col].notna().any():
                rec[col], rec[f"{col}_lo"], rec[f"{col}_hi"] = d0.t_interval(ga[col])
        summary.append(rec)
    pivots = {}
    for cell, g in df.groupby("cell"):
        pivots[cell] = g.pivot(index="unit", columns="arm", values="mae")
    # paired differences of every new arm against its stored D1 counterpart (MAE; RP for physio)
    for cell, g in df.groupby("cell"):
        base_mae = stored[stored["cell"] == cell].pivot(index="unit", columns="arm", values="mae")
        for arm in sorted(g["arm"].unique()):
            variant = ("recurrence_helper" if arm.endswith("_helper") else
                       "recurrence_own" if arm.endswith("_own") else "recurrence")
            b = STORED_ARM[variant]
            if b not in base_mae:
                continue
            ga = g[g["arm"] == arm].set_index("unit")["mae"]
            common = ga.index.intersection(base_mae.index)
            dm, lo, hi = d0.t_interval((ga[common] - base_mae[b][common]).dropna())
            pairs.append({"cell": cell, "a": arm, "b": f"D1:{b}", "metric": "mae", "diff": dm, "lo": lo,
                          "hi": hi, "mean_b": float(base_mae[b][common].mean()), "kind": "vs_D1"})
            if "rp" in g and cell.startswith(("ppg", "ecg")):
                base_rp = stored[stored["cell"] == cell].pivot(index="unit", columns="arm", values="rp")
                gr = g[g["arm"] == arm].set_index("unit")["rp"]
                common = gr.index.intersection(base_rp.index)
                dm, lo, hi = d0.t_interval((gr[common] - base_rp[b][common]).dropna())
                pairs.append({"cell": cell, "a": arm, "b": f"D1:{b}", "metric": "rp", "diff": dm, "lo": lo,
                              "hi": hi, "mean_b": float(base_rp[b][common].mean()), "kind": "vs_D1"})
    # attribution: rule effect, averaging effect, donor-count control (primary variant arms)
    attr = [("rec_c01_sel", "rec_w03_sel", "rule"), ("rec_c02_sel", "rec_w03_sel", "rule"),
            ("rec_w03_avg", "rec_w03_sel", "averaging"), ("rec_c01_avg", "rec_c01_sel", "averaging"),
            ("rec_c02_avg", "rec_c02_sel", "averaging"),
            ("rec_c01_avg", "rec_c01_k32_sel", "donor_count"), ("rec_c02_avg", "rec_c02_k32_sel", "donor_count")]
    for cell, piv in pivots.items():
        for a, b, kind in attr:
            if a in piv and b in piv:
                dm, lo, hi = d0.t_interval((piv[a] - piv[b]).dropna())
                pairs.append({"cell": cell, "a": a, "b": b, "metric": "mae", "diff": dm, "lo": lo, "hi": hi,
                              "mean_b": float(piv[b].mean()), "kind": kind})
    pairs_df = pd.DataFrame(pairs)
    save_table(pd.DataFrame(summary), out, "summary")
    save_table(pairs_df, out, "pairs")

    # pre-registered improvement test on the primary arm
    decision, n_imp, n_worse = [], 0, 0
    for cell in spec["cells"]:
        if cell not in pivots or primary not in pivots[cell]:
            continue
        base = stored[(stored["cell"] == cell) & (stored["arm"] == "recurrence")].set_index("unit")["mae"]
        ga = pivots[cell][primary]
        common = ga.index.intersection(base.index)
        dm, lo, hi = d0.t_interval((ga[common] - base[common]).dropna())
        verdict = "improved" if hi < 0 else "worse" if lo > 0 else "neither"
        n_imp, n_worse = n_imp + int(verdict == "improved"), n_worse + int(verdict == "worse")
        decision.append({"gate": "primary", "cell": cell, "statistic": f"MAE {primary} - D1 recurrence",
                         "value": dm, "ci_low": lo, "ci_high": hi,
                         "rule": "improved iff upper bound < 0; worse iff lower bound > 0", "verdict": verdict,
                         "note": f"n={len(common)}; D1 recurrence MAE {float(base[common].mean()):.4f}", "smoke": spec["smoke"]})
    verdict = "adopt" if (n_imp >= 5 and n_worse == 0) else "reject" if n_worse >= 2 else "inconclusive"
    decision.append({"gate": "verdict", "cell": "all", "statistic": "improved / worse cells of 10",
                     "value": n_imp, "ci_low": np.nan, "ci_high": np.nan,
                     "rule": "adopt iff improved in at least 5 of 10 cells and worse in none; reject iff worse "
                             "in at least 2 cells; otherwise inconclusive",
                     "verdict": verdict + (" (smoke: mechanical only)" if spec["smoke"] else ""),
                     "note": f"improved {n_imp}, worse {n_worse}", "smoke": spec["smoke"]})
    write_decision(decision, out)

    # re-scored gates (post hoc reanalysis; the original verdicts stand)
    d1spec = d1.build_spec()
    rows1 = []
    for arm in ARMS:
        ev = d1.evaluate(d1spec, _replace_recurrence(stored, df, arm))
        rows1 += [{"arm": arm, **r} for r in ev["decision"]]
    save_table(pd.DataFrame(rows1), out, "d1_rescored")
    d2spec, stored2 = d2.build_spec(), pd.read_csv(REPO / "notebooks" / "tables" / "dynimp" / "D2" / "per_unit.csv")
    bidmc_cells = [f"{t}|g{g}" for t, g in d2.CELLS]
    rows2 = []
    for arm in ARMS:
        ev = d2.evaluate(d2spec, _replace_recurrence(stored2, df, arm, cells=bidmc_cells))
        rows2 += [{"arm": arm, **r} for r in ev["decision"]]
    save_table(pd.DataFrame(rows2), out, "d2_rescored")
    for r in decision:
        log.info("%s | %s | %s = %s -> %s %s", r["gate"], r["cell"], r["statistic"], r["value"], r["verdict"], r["note"])


# ----------------------------------------------------------------------------- writeup


def _fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def stage_write() -> None:
    out = out_dir()
    spec = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    design = json.loads((out / "design.json").read_text(encoding="utf-8"))
    primary = spec["primary_arm"]
    summary = pd.read_csv(out / "summary.csv")
    pairs = pd.read_csv(out / "pairs.csv")
    decision = pd.read_csv(out / "decision.csv")
    stored = d1_stored()
    lines = []
    lines.append("# dynimp D1/D2 rerun: context-only overlap rule and averaging over C\n")
    lines.append("Post hoc reanalysis of the D1/D2 recurrence fill. Only the recurrence fill was rerun; "
                 "every other arm is read from the stored D1/D2 tables. The original D1 verdict (PASS) and "
                 "D2 verdict (INCONCLUSIVE) stand.\n")

    lines.append("## 1. What changed in the method\n")
    lines.append("D1 rule: a library window counts only if the entries observed in both it and the query are "
                 "at least $0.3 \\cdot F (g + 2C)$. The denominator includes the $g$ target-channel entries "
                 "inside the gap, which are never observed in the query, so with one channel the rule requires "
                 "$C \\ge 0.21 g$; otherwise the gap is left unfilled and scored as zero fill. Worked example: "
                 "one channel, $g = 187$, $C = 15$: the window has $217$ entries, $187$ of which can never be "
                 "observed in the query, so at most $30$ entries can be counted and the threshold "
                 "$0.3 \\times 217 = 65.1$ can never be reached — every candidate is rejected. The context-only "
                 "rule counts only entries that can be observed: $\\mathrm{cnt} \\ge f \\,(F (g + 2C) - g)$, "
                 "tested at $f = 0.1$ and $0.2$. The second change averages the fills from all $C$ values in "
                 "the grid instead of picking one $C$ on validation. Donor count is $k = 8$ throughout, with "
                 "$k = 32$ selection arms as a control (as many donors as averaging 8 donors over the 4 $C$ "
                 "values).\n")

    lines.append("## 2. Where the two changes came from\n")
    lines.append("The context-only rule comes from a validation-only probe (D1 validation patients and chaotic "
                 "validation segments): the D1 rule rejected 46–58 % of library candidates in the chaotic "
                 "cells and 62 % in PPG-only g187, and the context-only rule at 0.1–0.2 lowered validation MAE "
                 "by 0.06–0.13 in all six chaotic cells and by 0.04–0.05 in PPG-only, with no difference for "
                 "PPG+ECG and a tie on ECG. The averaging idea comes from a probe that was scored on the D1 "
                 "test patients, so it was not formed blind to test data; averaging lowered MAE by 0.010–0.019 "
                 "in 3 of 4 physiological cells there.\n")

    lines.append("## 3. Design-stage choice of the primary arm\n")
    rows = [[a, _fmt(design["per_arm"][a]["mean_validation_mae_over_cells"])] for a in ARMS]
    lines.append(_md_table(["arm", "mean validation MAE over cells"], rows))
    lines.append(f"\nPrimary arm (lowest candidate, k = 32 controls not eligible): **{primary}**. "
                 "This was frozen in `spec.json` before any test data were scored.\n")

    lines.append("## 4. Results per cell\n")
    vd = decision[decision["gate"] == "verdict"].iloc[0]
    rows = []
    for cell in spec["cells"]:
        base = stored[(stored["cell"] == cell) & (stored["arm"] == "recurrence")]["mae"]
        s = summary[(summary["cell"] == cell) & (summary["arm"] == primary)]
        p = pairs[(pairs["cell"] == cell) & (pairs["a"] == primary) & (pairs["kind"] == "vs_D1") &
                  (pairs["metric"] == "mae")]
        if not len(s) or not len(p):
            continue
        s, p = s.iloc[0], p.iloc[0]
        rows.append([cell, _fmt(base.mean()), _fmt(s["mae"]),
                     f"{p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"])
    lines.append(_md_table(["cell", "D1 recurrence MAE", f"{primary} MAE", "paired difference [95 % interval]"], rows))
    lines.append(f"\nVerdict: **{vd['verdict']}** ({vd['note']}; rule: {vd['rule']}).\n")
    ties = decision[(decision["gate"] == "primary") & (decision["value"].abs() < 1e-9)]
    for r in ties.itertuples():
        lines.append(f"Note on {r.cell}: the paired difference is {r.value:.2e} (interval "
                     f"[{r.ci_low:.2e}, {r.ci_high:.2e}]), i.e. zero at the precision of the stored D1 table; "
                     "the arm's fills are identical to the D1 fills there (the paired difference against the "
                     "recomputed D1-rule arm is exactly 0). It is counted by the frozen rule above as written.\n")

    lines.append("All arms, paired MAE difference against D1 recurrence (negative is better):\n")
    rows = []
    for cell in spec["cells"]:
        r = {"cell": cell}
        for arm in ARMS:
            p = pairs[(pairs["cell"] == cell) & (pairs["a"] == arm) & (pairs["kind"] == "vs_D1") &
                      (pairs["metric"] == "mae")]
            r[arm] = f"{p.iloc[0]['diff']:+.4f}" if len(p) else "—"
        rows.append([r["cell"]] + [r[a] for a in ARMS])
    lines.append(_md_table(["cell"] + ARMS, rows) + "\n")

    lines.append("## 5. Attribution: rule vs averaging vs donor count\n")
    rows = []
    for kind, label in [("rule", "rule effect (context-only sel − window sel)"),
                        ("averaging", "averaging effect (avg − sel, same rule)"),
                        ("donor_count", "donor-count control (avg − k32 sel, same rule)")]:
        sub = pairs[pairs["kind"] == kind]
        for (a, b), g in sub.groupby(["a", "b"]):
            rows.append([label, f"{a} − {b}", _fmt(g["diff"].mean()),
                         f"{int((g['hi'] < 0).sum())}/{len(g)} cells with interval below 0"])
    lines.append(_md_table(["effect", "comparison", "mean paired difference over cells", "cells improved"], rows))
    lines.append("\nIf the k = 32 controls match the `_avg` arms, the averaging gain comes from more donors, "
                 "not from mixing context lengths.\n")

    lines.append("## 6. Overlap statistics\n")
    ov = pd.read_csv(out / "overlap_stats.csv")
    rows = []
    for (cell, rule), g in ov[ov["variant"] == "recurrence"].groupby(["cell", "rule"]):
        tot, rej, unf, gaps = g["cand_total"].sum(), g["cand_rejected"].sum(), g["unfilled"].sum(), g["n_gaps"].sum()
        rows.append([cell, rule, _fmt(rej / tot if tot else float("nan"), 3), f"{int(unf)}/{int(gaps)}"])
    lines.append(_md_table(["cell", "rule", "share of candidates rejected (pooled over C)", "unfilled / gaps"], rows) + "\n")

    lines.append("## 7. D1 and D2 gate tables re-scored\n")
    for name, orig in [("d1_rescored", "original D1 verdict: PASS"), ("d2_rescored", "original D2 verdict: INCONCLUSIVE")]:
        rs = pd.read_csv(out / f"{name}.csv")
        gate = "D1" if name.startswith("d1") else "D2"
        sub = rs[rs["gate"] == gate]
        rows = [[r.arm, r.verdict] for r in sub.itertuples()]
        lines.append(f"{orig}; re-scored with each arm replacing `recurrence` (post hoc reanalysis):\n")
        lines.append(_md_table(["arm used as recurrence", f"{gate} verdict"], rows) + "\n")

    lines.append("## 8. What changes in the D1/D2 conclusions\n")
    lines.append(f"Under the pre-registered comparison rule the primary arm {primary} is judged **{vd['verdict']}** "
                 f"against D1 recurrence ({vd['note']}). The re-scored gate tables above show whether either "
                 "original gate outcome would have changed with the revised fill; the original verdicts stand "
                 "regardless, because the arms and the primary arm were chosen after D1/D2 were run.\n")

    lines.append("## 9. Limits\n")
    lines.append("- The averaging-over-C idea was formed on the D1 test patients (section 2); its test-set "
                 "gain here is not independent confirmation.\n"
                 "- The re-scored D1/D2 gates reuse the stored learned-imputer rows; nothing about SAITS/CSDI "
                 "training was rerun, so any D2 budget caveat (CSDI caps) carries over unchanged.\n"
                 "- Chaotic scoring is MAE over all channels, as in D1; RP error is reported for the "
                 "physiological cells only.\n"
                 "- One distance pass per (gap, C) is shared by all arms, but the fill is still a CPU "
                 "nearest-neighbour search over the patient's own history; no timing claim beyond the D1/D2 "
                 "compute comparison is made.\n")
    path = REPO / "Writeups" / "dynimp_D1_D2_C_Rerun.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("writeup written to %s", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["design", "run", "eval", "write"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(out / f"run_{args.stage}.log", mode="a", encoding="utf-8")):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    t0 = time.time()
    if args.stage == "design":
        stage_design(args.workers)
    elif args.stage == "run":
        stage_run(args.workers)
    elif args.stage == "eval":
        stage_eval()
    else:
        stage_write()
    log.info("stage %s done in %.0f s", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
