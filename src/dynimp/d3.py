"""D3: revised recurrence fill, developed on 33 BIDMC patients, confirmed once on the 20 test patients.

See plans/dynimp/D3_fill_revision.md. Development uses the D1 train+val patients (33) and chaotic seeds
100-111; the final run uses the D1 test patients (20) and chaotic seeds 200-211, plus seeds 0-5 for the
gate re-score only. Every structural and numeric choice is made on the development units and frozen in
spec.json at ``select``; the test patients are loaded before ``final`` only for the base reproduction
check. Only the recurrence fill is run (CPU); SAITS/CSDI and the other arms are read from stored tables.
New files only: this module and ``dynimp.fills_v2``.

Run from the repo root with ``PYTHONPATH=src`` and ``.venv-dynimp/Scripts/python.exe -m dynimp.d3 --stage S``:

    bench   time one development unit per cell, extrapolate dev/final, base reproduction check (subset)
    dev     all (C, rule, w) fills for the development units (Tasos)
    select  choose and freeze the final configuration; cross-fitted development estimate (agent)
    final   fills for the test patients and final/gate seeds (Tasos; needs the frozen spec)
    eval    improvement test per family, ladder, re-scored gates (agent)
    write   Writeups/dynimp_D3_fill_revision.md (agent)

``CONDPRED_SMOKE=1`` runs a tiny version (one unit per cell) into separate ``smoke/`` folders.
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
from dynimp.fills import nan_runs, recurrence_fill
from dynimp.fills_v2 import RULES, recurrence_fills_v2

REPO = Path(__file__).resolve().parents[2]
MODULES = ["dynimp.fills_v2", "dynimp.d3", "dynimp.d1", "dynimp.d0", "dynimp.fills"]
SPEC_VERSION = 1
CAND_RULES = ["old_fb", "ctx10", "ctx20", "ctx40", "ctx60", "pen05", "pen10"]
MODES = ["pick", "pool", "avg_all", "avg20", "pool_avg20"]
WS = [0.0, 0.25, 0.5, 1.0]
GRIDS = {
    "chaotic": {"d1": [5, 10, 20, 40], "ext": [1, 2, 3, 5, 10, 20, 40, 80]},
    "ppg": {"d1": [15, 31, 62, 125], "ext": [4, 8, 15, 31, 62, 125, 250]},
    "ecg": {"d1": [12, 25, 62, 125], "ext": [6, 12, 25, 62, 125, 250]},
}
TOUCHED_PROBE = {1, 2, 3, 5}
MAX_VLOO_GAPS = smoke(40, 4)

log = logging.getLogger("dynimp.d3")


def out_dir() -> Path:
    base = REPO / "notebooks" / "tables" / "dynimp" / "D3"
    assert base.name == "D3"
    return base / "smoke" if smoke(False, True) else base


def cache_dir() -> Path:
    return REPO / "data" / "processed" / "dynimp" / "D3" / smoke("full", "smoke")


def _d1spec_file() -> dict:
    return json.loads((REPO / "notebooks" / "tables" / "dynimp" / "D1" / "spec.json").read_text(encoding="utf-8"))


def family(cell: str) -> str:
    return "chaotic" if "n0.1" in cell else cell.split("|")[0]


def grids_for(spec: dict, cell: str) -> dict:
    red = spec.get("bench_reductions", [])
    out = {}
    for name, Cs in spec["grids"][family(cell)].items():
        Cs = [c for c in Cs if not (name == "ext" and "drop C=250 (ECG)" in red and family(cell) == "ecg" and c == 250)]
        Cs = [c for c in Cs if not (name == "ext" and "drop C=80 (chaotic)" in red and family(cell) == "chaotic" and c == 80)]
        out[name] = Cs
    return out


def build_spec() -> dict:
    d1s = _d1spec_file()
    dev_full = [int(s) for s in d1s["subjects"]["train"]] + [int(s) for s in d1s["subjects"]["val"]]
    test_full = [int(s) for s in d1s["subjects"]["test"]]
    bench_red = []
    bp = out_dir() / "bench.json"
    if bp.exists():
        bench_red = json.loads(bp.read_text(encoding="utf-8")).get("reductions", [])
    return {
        "version": SPEC_VERSION,
        "experiment": "dynimp D3: revised recurrence fill, developed on 33 BIDMC patients, confirmed once on 20 test patients",
        "smoke": bool(smoke(False, True)),
        "cells": d1s["cells"],
        "tasks": d1.TASKS,
        "embedding": d1s["embedding"],
        "subjects": {"dev": dev_full[: smoke(len(dev_full), 1)], "dev_full_count": len(dev_full),
                     "val_patients": [int(s) for s in d1s["subjects"]["val"]],
                     "test": test_full[: smoke(len(test_full), 1)]},
        "chaotic_seeds": {"dev": list(range(100, 112))[: smoke(12, 1)], "final": list(range(200, 212))[: smoke(12, 1)],
                          "gate": list(range(6))[: smoke(6, 1)]},
        "grids": GRIDS,
        "bench_reductions": bench_red,
        "rules": RULES,
        "candidate_rules": CAND_RULES,
        "modes": MODES,
        "ecg_weights": WS,
        "k": 8,
        "validation": {
            "V_d1": "D1 protocol (library first 50 %, held blocks of the 50-60 % part); used by the base arm only",
            "V_loo": "validation gaps inside the whole history, each BIDMC gap hidden alone, candidate windows "
                     "overlapping the query window excluded; chaotic: 5 repeats each hiding 2 % of the observed "
                     "history entries with the cell pattern; BIDMC: runs of g fully observed target samples, "
                     "evenly spread, at most 40 per patient and cell",
        },
        "fallback": "same C with rule pen10, then condpred.completion.linear; after the fallback no gap is unfilled",
        "candidates": "7 rules x {D1, extended grid} x {pick, pool, avg_all, avg20, pool_avg20} = 70 configurations, "
                      "all with V_loo; one configuration for all 10 cells; C, averaged C set and ECG weight w are per cell",
        "base": "old rule, D1 grid, pick, V_d1, w = 1 (the D1 fill)",
        "selection": "statistic = mean over the 10 cells of MAE(config)/MAE(base) on the scored part of the "
                     "development units; among configurations within 0.01 of the best, the one differing from "
                     "(old_fb, D1 grid, pick, w = 1) in the fewest components; remaining ties to the lower statistic",
        "comparison": {
            "primary": "per cell MAE(final) - MAE(base) on the missing samples of the scored part, paired over the "
                       "20 test patients or seeds 200-211, 95 % interval from d0.t_interval",
            "identical": "a cell is identical if every unit's absolute difference is below 1e-9",
            "verdicts": "per family (6 chaotic, 4 BIDMC), separately, no combined verdict: adopt iff improved in at "
                        "least half of the family's cells and worse in none; reject iff worse in more cells than "
                        "improved; otherwise inconclusive",
        },
        "data_roles": "development: D1 train (25) + val (8) patients, chaotic seeds 100-111; final improvement: D1 "
                      "test patients (20), chaotic seeds 200-211; gate re-score only: chaotic seeds 0-5. The split "
                      "is read from notebooks/tables/dynimp/D1/spec.json and not re-drawn.",
        "prior_use": "patients 1, 2, 3, 5 were in the 2026-11-12 throwaway probe; the 50-60 % part of the 8 val "
                     "patients was scored in the overlap-rule probe (their last 40 % was not); the other 22 train "
                     "patients have never been filled by the recurrence fill. All 33 trained or selected the D1/D2 "
                     "cross-patient learned models; no learned imputer is run on them here.",
        "leakage": "test patients are loaded before final only for the base reproduction check; pooled choices and "
                   "w are frozen at select from development units; per-unit choices (pick, avg20) use the unit's "
                   "own history only",
    }


def cached(spec: dict, name: str, fn):
    h12 = hashlib.sha256((spec_hash(spec) + code_fingerprint(MODULES)).encode()).hexdigest()[:12]
    path = cache_dir() / f"{name}_{h12}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    alts = sorted(cache_dir().glob(f"{name}_*.pkl"), key=lambda p: p.stat().st_mtime)
    if alts:  # a post-hoc code fix must not orphan units computed under an earlier fingerprint
        return pickle.loads(alts[-1].read_bytes())
    res = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(res))
    tmp.replace(path)
    return res


def _cache_path(spec: dict, name: str) -> Path:
    h12 = hashlib.sha256((spec_hash(spec) + code_fingerprint(MODULES)).encode()).hexdigest()[:12]
    return cache_dir() / f"{name}_{h12}.pkl"


# ----------------------------------------------------------------------------- data preparation


def physio_prep(task: str, g: int, sid: int) -> dict:
    d = d1.subject_series(task, g, sid)
    zo, n_hist, n_lib = d["zo"], d["n_hist"], d["n_lib"]
    return {"d": d, "zo": zo, "hist": zo[:n_hist], "lib50": zo[:n_lib], "val_seg": zo[n_lib:n_hist],
            "scored": zo[n_hist:], "truth": d["zt"][n_hist:, 0], "miss": d["miss"][n_hist:]}


def chaotic_prep(name: str, seed: int) -> dict:
    """Copied from d1.chaotic_cpu_unit (data part only), for any seed."""
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
    return {"cell": cell, "system": system, "pattern": pattern, "sys_id": sid, "pat_id": pid, "F": F,
            "tr": zo[:n_tr], "va": zo[n_tr:n_tr + n_va], "hist": zo[:n_tr + n_va],
            "scored": zo[n_tr + n_va:], "truth": zt[n_tr + n_va:], "miss": miss[n_tr + n_va:],
            "n_tr": n_tr, "n_va": n_va, "seed": seed}


def weight_vectors(cell: str, F: int) -> dict:
    if family(cell) == "ppg":
        return {w: np.array([1.0, w]) for w in WS}
    return {1.0: np.ones(F)}


# ----------------------------------------------------------------------------- validation


def _vd1_physio(spec, task, g, sid, p, val_loo=None) -> dict:
    T = spec["tasks"][task]
    val = p["val_seg"]
    held = d1.blocks(len(val), 0.2, g, d1._rng(32, d1.TASK_ID[task], g, sid)) & np.isfinite(val[:, 0])
    v = val.copy()
    v[held, 0] = np.nan
    runs = [r for r in nan_runs(v[:, 0]) if held[r[0]:r[1]].all()][:12]  # D1 val_runs
    if not runs:
        # No usable held run ( every held block merged with an existing gap; patient 36 at g=187).
        # Fall back to the V_loo errors of the old rule for picking C; this affects only such units.
        if val_loo is None:
            raise ValueError(f"no V_d1 validation runs for {task}|g{g} patient {sid} and no fallback given")
        return {C: val_loo[(C, "old", 1.0)] for C in grids_for(spec, f"{task}|g{g}")["d1"]}
    idx = np.concatenate([np.arange(a, b) for a, b in runs])
    out = {}
    for C in grids_for(spec, f"{task}|g{g}")["d1"]:
        f = recurrence_fills_v2(v, p["lib50"], 0, C=C, rules=["old"], weight_vectors={1.0: np.ones(v.shape[1])},
                                runs=runs)["fills"][("old", 1.0)]
        out[C] = float(np.nanmean(np.abs(np.nan_to_num(f[idx], nan=0.0) - val[idx, 0])))
    return out


def _vd1_chaotic(spec, name, p) -> dict:
    held = d0.missing_mask(p["n_va"], p["F"], 0.2, p["pattern"], 10, 4, p["sys_id"], p["seed"], p["pat_id"]) & np.isfinite(p["va"])
    v = np.where(held, np.nan, p["va"])
    out = {}
    for C in grids_for(spec, name)["d1"]:
        errs = []
        for c in range(p["F"]):
            f = recurrence_fills_v2(v, p["tr"], c, C=C, rules=["old"], weight_vectors={1.0: np.ones(p["F"])})["fills"][("old", 1.0)]
            errs.append(np.abs(np.nan_to_num(f, nan=0.0)[held[:, c]] - p["va"][held[:, c], c]))
        out[C] = float(np.concatenate(errs).mean())
    return out


def _vloo_gaps_physio(hist: np.ndarray, g: int) -> list[tuple[int, int]]:
    fin = np.isfinite(hist[:, 0]).astype(int)
    cs = np.concatenate([[0], np.cumsum(fin)])
    starts = np.flatnonzero(cs[g:] - cs[:-g] == g)
    starts = starts[starts + g <= len(hist)]
    if len(starts) > MAX_VLOO_GAPS:
        starts = starts[np.unique(np.linspace(0, len(starts) - 1, MAX_VLOO_GAPS).round().astype(int))]
    return [(int(a), int(a + g)) for a in starts]


def _vloo_physio(spec, task, g, p) -> dict:
    hist = p["hist"]
    wv = weight_vectors(f"{task}|g{g}", hist.shape[1])
    gaps = _vloo_gaps_physio(hist, g)
    acc: dict[tuple, float] = {}
    for a, b in gaps:
        t1 = hist.copy()
        t1[a:b, 0] = np.nan
        for C in grids_for(spec, f"{task}|g{g}")["ext"]:
            r = recurrence_fills_v2(t1, t1, 0, C=C, rules=RULES, weight_vectors=wv, runs=[(a, b)], self_lib=True)
            for key, f in r["fills"].items():
                acc[(C, *key)] = acc.get((C, *key), 0.0) + float(
                    np.abs(np.nan_to_num(f[a:b], nan=0.0) - hist[a:b, 0]).sum())
    n = max(len(gaps), 1) * g
    return {key: v / n for key, v in acc.items()}


def _vloo_chaotic(spec, name, p) -> dict:
    hist = p["hist"]
    wv = {1.0: np.ones(p["F"])}
    acc: dict[tuple, float] = {}
    cnt: dict[tuple, int] = {}
    for rep in range(5):
        held = d0.missing_mask(len(hist), p["F"], 0.02, p["pattern"], 10, p["sys_id"], p["seed"], p["pat_id"],
                               900 + rep) & np.isfinite(hist)
        v = np.where(held, np.nan, hist)
        for C in grids_for(spec, name)["ext"]:
            for c in range(p["F"]):
                r = recurrence_fills_v2(v, v, c, C=C, rules=RULES, weight_vectors=wv, self_lib=True)
                for key, f in r["fills"].items():
                    k = (C, *key)
                    acc[k] = acc.get(k, 0.0) + float(np.abs(np.nan_to_num(f, nan=0.0)[held[:, c]] - hist[held[:, c], c]).sum())
                    cnt[k] = cnt.get(k, 0) + int(held[:, c].sum())
    return {k: acc[k] / max(cnt[k], 1) for k in acc}


def _old_vloo_worker(job):
    spec, cell, unit = job
    task, g = cell.split("|")[0], int(cell.split("|g")[1])
    return (cell, int(unit)), _vloo_physio_old_d1(spec, task, g, physio_prep(task, g, int(unit)))


def _vloo_physio_old_d1(spec, task, g, p) -> dict:
    """Corrected old-rule V_loo errors on the D1 grid (unfilled validation gaps scored as zero fill).

    The cached physio val_loo accumulated NaN for the old rule where a validation gap was unfilled
    (164 of 1360 entries), which made the ladder_loo C-pick arbitrary for those units. Only the old
    rule is affected: every candidate rule has the fallback, so its validation fills are finite.
    """
    hist = p["hist"]
    gaps = _vloo_gaps_physio(hist, g)
    acc: dict[tuple, float] = {}
    for a, b in gaps:
        t1 = hist.copy()
        t1[a:b, 0] = np.nan
        for C in grids_for(spec, f"{task}|g{g}")["d1"]:
            r = recurrence_fills_v2(t1, t1, 0, C=C, rules=["old"],
                                    weight_vectors={1.0: np.ones(hist.shape[1])}, runs=[(a, b)], self_lib=True)
            f = r["fills"][("old", 1.0)]
            acc[(C, "old", 1.0)] = acc.get((C, "old", 1.0), 0.0) + float(
                np.abs(np.nan_to_num(f[a:b], nan=0.0) - hist[a:b, 0]).sum())
    n = max(len(gaps), 1) * g
    return {k: v / n for k, v in acc.items()}


# ----------------------------------------------------------------------------- units


def _unit(cell: str, unit: int, prep: dict, base_val: dict, val_loo: dict, scored_fn) -> dict:
    per_C, fills_miss, truth_miss, stats = scored_fn()
    return {"cell": cell, "unit": int(unit), "base_val": {int(k): v for k, v in base_val.items()},
            "val_loo": {tuple(k): v for k, v in val_loo.items()},
            "per_C": per_C, "fills_miss": fills_miss, "truth_miss": truth_miss, "stats": stats}


def physio_unit(spec, task, g, sid) -> dict:
    cell = f"{task}|g{g}"
    p = physio_prep(task, g, sid)
    wv = weight_vectors(cell, p["zo"].shape[1])
    val_loo = _vloo_physio(spec, task, g, p)
    base_val = _vd1_physio(spec, task, g, sid, p, val_loo=val_loo)

    def scored_fn():
        per_C, fills_miss, stats = [], {}, []
        for C in grids_for(spec, cell)["ext"]:
            r = recurrence_fills_v2(p["scored"], p["hist"], 0, C=C, rules=RULES, weight_vectors=wv)
            for (rule, wl), f in r["fills"].items():
                fm = np.nan_to_num(f[p["miss"]], nan=0.0)
                fills_miss[(C, rule, wl)] = f[p["miss"]]
                per_C.append({"C": C, "rule": rule, "w": wl,
                              "scored_mae": float(np.abs(fm - p["truth"][p["miss"]]).mean())})
            for (rule, wl), st in r["stats"].items():
                stats.append({"C": C, "rule": rule, "w": wl, **st})
        return per_C, fills_miss, p["truth"][p["miss"]], stats

    return _unit(cell, sid, p, base_val, val_loo, scored_fn)


def chaotic_unit(spec, name, seed) -> dict:
    p = chaotic_prep(name, seed)
    wv = {1.0: np.ones(p["F"])}
    base_val = _vd1_chaotic(spec, name, p)
    val_loo = _vloo_chaotic(spec, name, p)

    def scored_fn():
        per_C, fills_miss, stats = [], {}, []
        lib = p["hist"]
        for C in grids_for(spec, name)["ext"]:
            per_ch = []
            for c in range(p["F"]):
                r = recurrence_fills_v2(p["scored"], lib, c, C=C, rules=RULES, weight_vectors=wv)
                per_ch.append(r)
                for (rule, wl), st in r["stats"].items():
                    stats.append({"C": C, "rule": rule, "w": wl, "channel": c, **st})
            for rule in RULES:
                fm = np.concatenate([np.nan_to_num(per_ch[c]["fills"][(rule, 1.0)], nan=0.0)[p["miss"][:, c]]
                                     for c in range(p["F"])])
                fills_miss[(C, rule, 1.0)] = np.concatenate(
                    [per_ch[c]["fills"][(rule, 1.0)][p["miss"][:, c]] for c in range(p["F"])])
                per_C.append({"C": C, "rule": rule, "w": 1.0, "scored_mae": float(np.abs(
                    fm - np.concatenate([p["truth"][p["miss"][:, c], c] for c in range(p["F"])])).mean())})
        truth_miss = np.concatenate([p["truth"][p["miss"][:, c], c] for c in range(p["F"])])
        return per_C, fills_miss, truth_miss, stats

    return _unit(name, seed, p, base_val, val_loo, scored_fn)


def _job(spec, job):
    kind, role = job[0], job[1]
    if kind == "physio":
        _, _, task, g, sid = job
        allowed = spec["subjects"]["dev"] if role == "dev" else spec["subjects"]["test"]
        assert int(sid) in allowed, f"patient {sid} not in role {role}"
        return cached(spec, f"unit_{role}_{task}_g{g}_s{sid}", lambda: physio_unit(spec, task, g, sid))
    _, _, name, seed = job
    assert int(seed) in spec["chaotic_seeds"][role], f"seed {seed} not in role {role}"
    return cached(spec, f"unit_{role}_chaotic_{name.replace('|', '_')}_s{seed}",
                  lambda: chaotic_unit(spec, name, seed))


def _jobs(spec, role: str):
    jobs = []
    if role == "dev":
        jobs += [("physio", "dev", t, g, s) for t in d1.TASKS for g in d1.TASKS[t]["gaps"]
                 for s in spec["subjects"]["dev"]]
        jobs += [("chaotic", "dev", name, s) for name in spec["cells"] if "n0.1" in name
                 for s in spec["chaotic_seeds"]["dev"]]
    else:
        jobs += [("physio", "final", t, g, s) for t in d1.TASKS for g in d1.TASKS[t]["gaps"]
                 for s in spec["subjects"]["test"]]
        jobs += [("chaotic", "final", name, s) for name in spec["cells"] if "n0.1" in name
                 for s in spec["chaotic_seeds"]["final"]]
        jobs += [("chaotic", "gate", name, s) for name in spec["cells"] if "n0.1" in name
                 for s in spec["chaotic_seeds"]["gate"]]
    return jobs


def _run_pool(spec, jobs, workers):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_job, spec, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            log.info("[%d/%d] %s unit %s done", k, len(jobs), r["cell"], r["unit"])
    return out


def collect_units(spec, roles) -> list[dict]:
    """Load cached units. Dev units are keyed by the v1 (unfrozen) spec, final/gate units by the frozen
    spec, so the hash spec is chosen per role. Each returned unit gets a ``role`` field."""
    out = []
    named = []
    for role in roles:
        hspec = build_spec() if role == "dev" else spec
        if role in ("dev", "final"):
            named += [(f"unit_{role}_{t}_g{g}_s{s}", role, hspec) for t in d1.TASKS for g in d1.TASKS[t]["gaps"]
                      for s in (spec["subjects"]["dev"] if role == "dev" else spec["subjects"]["test"])]
            named += [(f"unit_{role}_chaotic_{n.replace('|', '_')}_s{s}", role, hspec)
                      for n in spec["cells"] if "n0.1" in n
                      for s in spec["chaotic_seeds"]["dev" if role == "dev" else "final"]]
        if role == "gate":
            named += [(f"unit_gate_chaotic_{n.replace('|', '_')}_s{s}", role, hspec)
                      for n in spec["cells"] if "n0.1" in n for s in spec["chaotic_seeds"]["gate"]]
    for name, role, hspec in named:
        p = _cache_path(hspec, name)
        if not p.exists():  # fall back to any fingerprint: a post-hoc code fix must not orphan computed units
            alts = sorted(cache_dir().glob(f"{name}_*.pkl"))
            if not alts:
                raise RuntimeError(f"missing cached unit {p.name}; run the corresponding stage first")
            p = alts[-1]
        u = pickle.loads(p.read_bytes())
        u["role"] = role
        out.append(u)
    return out


# ----------------------------------------------------------------------------- configuration evaluation


def _pc(unit: dict) -> dict:
    return {(r["C"], r["rule"], r["w"]): r["scored_mae"] for r in unit["per_C"]}


def _mae_of_fill(unit: dict, fill: np.ndarray) -> float:
    return float(np.abs(np.nan_to_num(fill, nan=0.0) - unit["truth_miss"]).mean())


def config_fill(unit: dict, spec, rule, grid, mode, w, pool_info=None) -> np.ndarray:
    """Fill at the missing points of one unit under one configuration (V_loo choices from the unit itself)."""
    Cs = grids_for(spec, unit["cell"])[grid]
    if mode in ("pick", "pool"):
        if mode == "pick":
            C = min(Cs, key=lambda C: unit["val_loo"][(C, rule, w)])
        else:
            C = pool_info["C"]
        return unit["fills_miss"][(C, rule, w)]
    if mode == "avg_all":
        sel = Cs
    elif mode == "avg20":
        vals = {C: unit["val_loo"][(C, rule, w)] for C in Cs}
        best = min(vals.values())
        sel = [C for C in Cs if vals[C] <= 1.2 * best]
    else:  # pool_avg20
        sel = pool_info["set"]
    return np.nanmean(np.stack([unit["fills_miss"][(C, rule, w)] for C in sel]), axis=0)


def pool_from(units, spec, cell, rule, w, grid) -> dict:
    Cs = grids_for(spec, cell)[grid]
    pooled = {C: float(np.mean([u["val_loo"][(C, rule, w)] for u in units])) for C in Cs}
    best = min(pooled.values())
    return {"C": int(min(Cs, key=lambda C: pooled[C])),
            "set": [int(C) for C in Cs if pooled[C] <= 1.2 * best], "pooled_val": pooled}


def base_mae(unit: dict, spec) -> float:
    C = min(unit["base_val"], key=lambda C: unit["base_val"][C])
    return _pc(unit)[(C, "old", 1.0)]


def choose_on(train_by_cell: dict, spec, rule, grid, mode) -> dict:
    """Per-cell frozen choices (w, pooled C / C set) from training units only."""
    choices = {}
    for cell, units in train_by_cell.items():
        ws = WS if family(cell) == "ppg" else [1.0]
        best = None
        for w in ws:
            pool = pool_from(units, spec, cell, rule, w, grid)
            mae = float(np.mean([_mae_of_fill(u, config_fill(u, spec, rule, grid, mode, w, pool)) for u in units]))
            if best is None or mae < best[0]:
                best = (mae, w, pool)
        choices[cell] = {"w": float(best[1]), "pool": best[2], "train_mae": float(best[0])}
    return choices


def score_with(units_by_cell: dict, spec, rule, grid, mode, choices) -> dict:
    return {cell: [_mae_of_fill(u, config_fill(u, spec, rule, grid, mode, choices[cell]["w"], choices[cell]["pool"]))
                   for u in units] for cell, units in units_by_cell.items()}


def eval_config(train_by_cell, spec, rule, grid, mode) -> dict:
    choices = choose_on(train_by_cell, spec, rule, grid, mode)
    scored = score_with(train_by_cell, spec, rule, grid, mode, choices)
    base = {c: float(np.mean([base_mae(u, spec) for u in us])) for c, us in train_by_cell.items()}
    ratio = {c: float(np.mean(scored[c])) / base[c] for c in scored}
    return {"rule": rule, "grid": grid, "mode": mode, "statistic": float(np.mean(list(ratio.values()))),
            "per_cell_ratio": ratio, "per_cell_mae": {c: float(np.mean(v)) for c, v in scored.items()},
            "choices": choices}


def _by_cell(units) -> dict:
    out: dict[str, list] = {}
    for u in units:
        out.setdefault(u["cell"], []).append(u)
    return out


def _n_components(cfg: dict) -> int:
    n = int(cfg["rule"] != "old_fb") + int(cfg["grid"] != "d1") + int(cfg["mode"] != "pick")
    n += int(any(ch["w"] != 1.0 for ch in cfg["choices"].values()))
    return n


def select_final(evals: list[dict]) -> dict:
    best = min(e["statistic"] for e in evals)
    cands = [e for e in evals if e["statistic"] <= best + 0.01]
    return min(cands, key=lambda e: (_n_components(e), e["statistic"]))


# ----------------------------------------------------------------------------- base-only (reproduction)


def base_only_mae_physio(spec, task, g, sid) -> float:
    p = physio_prep(task, g, sid)
    bv = _vd1_physio(spec, task, g, sid, p)
    C = min(bv, key=lambda C: bv[C])
    f = recurrence_fills_v2(p["scored"], p["hist"], 0, C=C, rules=["old"],
                            weight_vectors={1.0: np.ones(p["zo"].shape[1])})["fills"][("old", 1.0)]
    return float(np.abs(np.nan_to_num(f, nan=0.0)[p["miss"]] - p["truth"][p["miss"]]).mean())


def base_only_mae_chaotic(spec, name, seed) -> float:
    p = chaotic_prep(name, seed)
    bv = _vd1_chaotic(spec, name, p)
    C = min(bv, key=lambda C: bv[C])
    errs = []
    for c in range(p["F"]):
        f = recurrence_fills_v2(p["scored"], p["hist"], c, C=C, rules=["old"],
                                weight_vectors={1.0: np.ones(p["F"])})["fills"][("old", 1.0)]
        errs.append(np.abs(np.nan_to_num(f, nan=0.0)[p["miss"][:, c]] - p["truth"][p["miss"][:, c], c]))
    return float(np.concatenate(errs).mean())


def repro_rows(spec, physio_units, chaotic_units) -> pd.DataFrame:
    stored = pd.read_csv(REPO / "notebooks" / "tables" / "dynimp" / "D1" / "per_unit.csv")
    rows = []
    for task, g, sid in physio_units:
        cell = f"{task}|g{g}"
        new = base_only_mae_physio(spec, task, g, sid)
        old = stored[(stored["cell"] == cell) & (stored["unit"] == sid) & (stored["arm"] == "recurrence")]["mae"]
        if len(old):
            rows.append({"cell": cell, "unit": int(sid), "new_mae": new, "d1_mae": float(old.iloc[0]),
                         "abs_diff": abs(new - float(old.iloc[0]))})
    for name, seed in chaotic_units:
        new = base_only_mae_chaotic(spec, name, seed)
        old = stored[(stored["cell"] == name) & (stored["unit"] == seed) & (stored["arm"] == "recurrence")]["mae"]
        if len(old):
            rows.append({"cell": name, "unit": int(seed), "new_mae": new, "d1_mae": float(old.iloc[0]),
                         "abs_diff": abs(new - float(old.iloc[0]))})
    df = pd.DataFrame(rows)
    if len(df):
        df["pass"] = df["abs_diff"] <= 1e-9
    return df


def exact_base_fill(spec, u) -> tuple[float, np.ndarray]:
    """MAE and fill-at-missing-points of the base arm, computed with the original D1 fill function.

    The cached old-rule fills come from the BLAS distance pass in fills_v2; its summation order can
    break exact donor ties differently from D1's direct differencing (one ECG unit: distances agreed
    to 3e-15, the 8th/9th donors were exactly tied, MAE differed by 5.6e-05). The base arm is defined
    as the exact D1 fill, so it is recomputed with dynimp.fills.recurrence_fill at the C the cached
    V_d1 validation picked (that choice matched D1's stored choice wherever both exist).
    """
    C = min(u["base_val"], key=lambda C: u["base_val"][C])
    if family(u["cell"]) != "chaotic":
        task, g = u["cell"].split("|")[0], int(u["cell"].split("|g")[1])
        p = physio_prep(task, g, int(u["unit"]))
        f = recurrence_fill(p["scored"], p["hist"], 0, C=C, k=8, min_overlap=0.3)
        mae = float(np.abs(np.nan_to_num(f, nan=0.0)[p["miss"]] - p["truth"][p["miss"]]).mean())
        return mae, f[p["miss"]]
    p = chaotic_prep(u["cell"], int(u["unit"]))
    fills = [recurrence_fill(p["scored"], p["hist"], c, C=C, k=8, min_overlap=0.3) for c in range(p["F"])]
    raw = np.concatenate([fills[c][p["miss"][:, c]] for c in range(p["F"])])
    return _mae_of_fill(u, raw), raw


# ----------------------------------------------------------------------------- stages: bench / dev


def stage_bench(workers: int) -> None:
    spec = build_spec()
    out = out_dir()  # bench writes bench.json only; dev/select write the spec (bench reductions enter via build_spec)
    cells_t = {}
    t_all = time.time()
    for cell in spec["cells"]:
        t0 = time.time()
        if family(cell) in ("ppg", "ecg"):
            task, g = cell.split("|")[0], int(cell.split("|g")[1])
            physio_unit(spec, task, g, spec["subjects"]["dev"][0])
        else:
            chaotic_unit(spec, cell, spec["chaotic_seeds"]["dev"][0])
        cells_t[cell] = round(time.time() - t0, 1)
        log.info("bench %s: %.1f s per unit", cell, cells_t[cell])
    # full unit counts per cell: BIDMC 33 dev / 20 final patients; chaotic 12 dev / (12 final + 6 gate) seeds
    est = {"dev_hours_8workers": round(sum(t * (33 if family(c) != "chaotic" else 12)
                                           for c, t in cells_t.items()) / 8 / 3600, 2),
           "final_hours_8workers": round(sum(t * (20 if family(c) != "chaotic" else 18)
                                             for c, t in cells_t.items()) / 8 / 3600, 2)}
    reductions = []
    if est["dev_hours_8workers"] > 4:
        reductions = ["drop C=250 (ECG)", "drop C=80 (chaotic)"]
    bench = {"seconds_per_unit": cells_t, "estimate": est, "workers": 8, "reductions": reductions,
             "note": "estimates assume the timed unit is typical; dev = 204 units, final = 188 units",
             "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "bench.json").write_text(json.dumps(bench, indent=1), encoding="utf-8")
    if reductions:
        log.warning("bench estimate above 4 h; reductions recorded in bench.json and applied via build_spec: %s", reductions)
    t_phys = [(t, g, spec["subjects"]["test"][0]) for t in d1.TASKS for g in d1.TASKS[t]["gaps"]]
    t_cha = [(n, spec["chaotic_seeds"]["gate"][0]) for n in spec["cells"] if "n0.1" in n]
    rc = repro_rows(spec, t_phys, t_cha)
    save_table(rc, out, "repro_check")
    log.info("bench repro: %d/%d pass (max diff %.2e); total bench %.0f s",
             int(rc["pass"].sum()) if len(rc) else 0, len(rc), rc["abs_diff"].max() if len(rc) else np.nan,
             time.time() - t_all)


def stage_dev(workers: int) -> None:
    spec = build_spec()
    out = out_dir()
    if (out / "spec.json").exists() and json.loads((out / "spec.json").read_text())["version"] > SPEC_VERSION:
        raise RuntimeError("spec already frozen by select; dev must not rerun after freezing")
    write_spec(spec, out, modules=MODULES)
    _run_pool(spec, _jobs(spec, "dev"), workers)


# ----------------------------------------------------------------------------- stage: select


def stage_select() -> None:
    spec = build_spec()
    out = out_dir()
    write_spec(spec, out, modules=MODULES)
    units = collect_units(spec, ["dev"])
    by_cell = _by_cell(units)
    base_cell = {c: float(np.mean([base_mae(u, spec) for u in us])) for c, us in by_cell.items()}
    evals = []
    for rule in CAND_RULES:
        for grid in ("d1", "ext"):
            for mode in MODES:
                evals.append(eval_config(by_cell, spec, rule, grid, mode))
                log.info("config %s/%s/%s statistic %.4f", rule, grid, mode, evals[-1]["statistic"])
    final_cfg = select_final(evals)
    log.info("final configuration: %s/%s/%s statistic %.4f", final_cfg["rule"], final_cfg["grid"],
             final_cfg["mode"], final_cfg["statistic"])

    # cross-fitted development estimate: repeat the whole selection on 2 folds, score the held-out fold
    dev_subj = spec["subjects"]["dev"]
    perm = np.random.default_rng(2027).permutation(len(dev_subj))
    folds_s = [[dev_subj[i] for i in perm[k::3]] for k in range(3)]
    dev_seeds = spec["chaotic_seeds"]["dev"]
    perm_s = np.random.default_rng(2027).permutation(len(dev_seeds))
    folds_seed = [[dev_seeds[i] for i in perm_s[k::3]] for k in range(3)]
    cf_rows = []
    for k in range(3):
        held_s, held_seed = set(folds_s[k]), set(folds_seed[k])
        train = [u for u in units if (u["unit"] not in held_s if family(u["cell"]) != "chaotic"
                                      else u["unit"] not in held_seed)]
        held = [u for u in units if (u["unit"] in held_s if family(u["cell"]) != "chaotic"
                                     else u["unit"] in held_seed)]
        tr_bc, he_bc = _by_cell(train), _by_cell(held)
        fold_evals = [eval_config(tr_bc, spec, r, gr, m) for r in CAND_RULES for gr in ("d1", "ext") for m in MODES]
        fcfg = select_final(fold_evals)
        scored = score_with(he_bc, spec, fcfg["rule"], fcfg["grid"], fcfg["mode"], fcfg["choices"])
        for cell, maes in scored.items():
            for u, mae in zip(he_bc[cell], maes):
                cf_rows.append({"cell": cell, "unit": int(u["unit"]), "fold": k,
                                "config": f"{fcfg['rule']}/{fcfg['grid']}/{fcfg['mode']}",
                                "mae_final": mae, "mae_base": base_mae(u, spec), "diff": mae - base_mae(u, spec)})
    cf = pd.DataFrame(cf_rows)
    save_table(cf, out, "dev_crossfit")
    dev_units_df = pd.DataFrame([{"cell": u["cell"], "unit": u["unit"],
                                  **{f"mae_{r['rule']}": r["scored_mae"] for r in []}} for u in []])
    design = {
        "statistic_per_config": [{k: e[k] for k in ("rule", "grid", "mode", "statistic", "per_cell_ratio",
                                                    "per_cell_mae")} | {"w_per_cell": {c: ch["w"] for c, ch in e["choices"].items()}}
                                 for e in evals],
        "base_mae_per_cell_dev": base_cell,
        "final_configuration": {k: final_cfg[k] for k in ("rule", "grid", "mode", "statistic", "per_cell_ratio",
                                                          "per_cell_mae")},
        "per_cell_choices": {c: {"w": ch["w"], "pool_C": ch["pool"]["C"], "pool_set": ch["pool"]["set"]}
                             for c, ch in final_cfg["choices"].items()},
        "crossfit": {
            "all33_and_chaotic": cf.groupby("cell")["diff"].agg(["mean", "count"]).round(5).to_dict("index"),
            "note": "3 folds; patients permuted with default_rng(2027), 11 per fold; seeds 4 per fold; the whole "
                    "selection (configuration, pooled C, w) repeated on the other two folds",
        },
        "written_before_final": True,
    }
    never = [s for s in _d1spec_file()["subjects"]["train"] if s not in TOUCHED_PROBE]
    cf_never = cf[cf["cell"].str.startswith(("ppg", "ecg")) & cf["unit"].isin(never)]
    design["crossfit"]["never_used_22_bidmc"] = (cf_never.groupby("cell")["diff"].mean().round(5).to_dict()
                                                 if len(cf_never) else {})
    design["never_used_patients"] = never
    (out / "design.json").write_text(json.dumps(design, indent=1, default=str), encoding="utf-8")

    frozen = dict(spec)
    frozen["version"] = SPEC_VERSION + 1
    frozen["final_configuration"] = {"rule": final_cfg["rule"], "grid": final_cfg["grid"], "mode": final_cfg["mode"],
                                     "statistic_dev": final_cfg["statistic"]}
    frozen["per_cell_choices"] = design["per_cell_choices"]
    frozen["ladder"] = [
        {"name": "base", "rule": "old", "grid": "d1", "mode": "pick", "validation": "V_d1", "w": "one"},
        {"name": "ladder_loo", "rule": "old", "grid": "d1", "mode": "pick", "validation": "V_loo", "w": "one"},
        {"name": "ladder_fallback", "rule": "old_fb", "grid": "d1", "mode": "pick", "validation": "V_loo", "w": "one"},
        {"name": "ladder_rule", "rule": final_cfg["rule"], "grid": "d1", "mode": "pick", "validation": "V_loo", "w": "one"},
        {"name": "ladder_grid", "rule": final_cfg["rule"], "grid": final_cfg["grid"], "mode": "pick",
         "validation": "V_loo", "w": "one"},
        {"name": "ladder_cmode", "rule": final_cfg["rule"], "grid": final_cfg["grid"], "mode": final_cfg["mode"],
         "validation": "V_loo", "w": "one"},
        {"name": "final", "rule": final_cfg["rule"], "grid": final_cfg["grid"], "mode": final_cfg["mode"],
         "validation": "V_loo", "w": "final"},
    ]
    write_spec(frozen, out, modules=MODULES)
    log.info("frozen spec v%s written; per-cell choices %s", frozen["version"], frozen["per_cell_choices"])


def _frozen_spec() -> dict:
    p = out_dir() / "spec.json"
    raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if "final_configuration" not in raw:
        raise RuntimeError("no frozen final configuration in spec.json; run --stage select first")
    spec = build_spec()
    for k in ("final_configuration", "per_cell_choices", "ladder"):
        spec[k] = raw[k]
    spec["version"] = raw["version"]
    return spec


# ----------------------------------------------------------------------------- stage: final


def stage_final(workers: int) -> None:
    spec = _frozen_spec()
    out = out_dir()
    _run_pool(spec, _jobs(spec, "final"), workers)
    units = collect_units(spec, ["final", "gate"])
    stored = pd.read_csv(REPO / "notebooks" / "tables" / "dynimp" / "D1" / "per_unit.csv")
    rows = []
    for u in units:
        old = stored[(stored["cell"] == u["cell"]) & (stored["unit"] == u["unit"]) & (stored["arm"] == "recurrence")]
        if not len(old):
            continue
        b = base_mae(u, spec)
        rows.append({"cell": u["cell"], "unit": u["unit"], "new_mae": b, "d1_mae": float(old["mae"].iloc[0]),
                     "abs_diff": abs(b - float(old["mae"].iloc[0]))})
    rc = pd.DataFrame(rows)
    rc["pass"] = rc["abs_diff"] <= 1e-9
    save_table(rc, out, "repro_check")
    log.info("final repro: %d/%d pass (max diff %.2e)", int(rc["pass"].sum()), len(rc), rc["abs_diff"].max())
    if not bool(rc["pass"].all()):
        raise SystemExit("reproduction check failed; eval will stop")


# ----------------------------------------------------------------------------- stage: eval


def _replace_recurrence(stored: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    new = new_rows.copy()
    new["arm"] = "recurrence"
    keep = [c for c in stored.columns if c in new.columns]
    covered = set(new["cell"])
    kept = stored[~((stored["arm"] == "recurrence") & (stored["cell"].isin(covered)))]
    return pd.concat([kept, new[keep]], ignore_index=True)


def _ladder_mae(u, spec, lad, dev_by_cell, final_choices):
    if lad["name"] == "base":
        return base_mae(u, spec)
    w = 1.0 if lad["w"] == "one" else final_choices[u["cell"]]["w"]
    pool = None
    if lad["mode"] in ("pool", "pool_avg20"):
        pool = pool_from(dev_by_cell[u["cell"]], spec, u["cell"], lad["rule"], w, lad["grid"])
    return _mae_of_fill(u, config_fill(u, spec, lad["rule"], lad["grid"], lad["mode"], w, pool))


def stage_eval() -> None:
    out = out_dir()
    spec = _frozen_spec()
    # Reproduction check on the exact-D1 base arm (see exact_base_fill). The stage-final check used the
    # cached BLAS fills and failed on one exact donor tie; this recomputation is the check as specified.
    units_all = collect_units(spec, ["final", "gate"])
    stored_d1 = pd.read_csv(REPO / "notebooks" / "tables" / "dynimp" / "D1" / "per_unit.csv")
    rc_rows, exact = [], {}
    for u in units_all:
        mae, fill = exact_base_fill(spec, u)
        exact[(u["cell"], int(u["unit"]), u["role"])] = (mae, fill)
        old = stored_d1[(stored_d1["cell"] == u["cell"]) & (stored_d1["unit"] == u["unit"])
                        & (stored_d1["arm"] == "recurrence")]
        if len(old):
            rc_rows.append({"cell": u["cell"], "unit": int(u["unit"]), "new_mae": mae,
                            "d1_mae": float(old["mae"].iloc[0]),
                            "abs_diff": abs(mae - float(old["mae"].iloc[0]))})
    rc = pd.DataFrame(rc_rows)
    rc["pass"] = rc["abs_diff"] <= 1e-9
    save_table(rc, out, "repro_check")
    log.info("eval repro (exact D1 base): %d/%d pass (max diff %.2e)",
             int(rc["pass"].sum()), len(rc), rc["abs_diff"].max())
    if not len(rc) or not bool(rc["pass"].all()):
        raise SystemExit("reproduction check failed; eval stops")
    fc = spec["final_configuration"]
    final_choices = {c: {"w": ch["w"], "pool": {"C": ch["pool_C"], "set": ch["pool_set"]}}
                     for c, ch in spec["per_cell_choices"].items()}
    units = collect_units(spec, ["final", "gate"])
    dev_by_cell = _by_cell(collect_units(spec, ["dev"]))
    corr_jobs = [(spec, u["cell"], int(u["unit"])) for u in units if family(u["cell"]) != "chaotic"]
    if corr_jobs:  # corrected old-rule V_loo for the ladder (see _vloo_physio_old_d1), in parallel
        with ProcessPoolExecutor(max_workers=8) as pool:
            corr_by_unit = dict(pool.map(_old_vloo_worker, corr_jobs))
        for u in units:
            if family(u["cell"]) != "chaotic":
                u["val_loo"] = {**u["val_loo"], **corr_by_unit[(u["cell"], int(u["unit"]))]}
    rows, per_C_rows, stats_rows = [], [], []
    for u in units:
        role = u["role"]
        arms = {}
        for lad in spec["ladder"]:
            arms[lad["name"]] = (exact[(u["cell"], int(u["unit"]), role)][0] if lad["name"] == "base"
                                 else _ladder_mae(u, spec, lad, dev_by_cell, final_choices))
        for name, mae in arms.items():
            rows.append({"cell": u["cell"], "unit": u["unit"], "role": role, "arm": name, "mae": mae})
        for r in u["per_C"]:
            per_C_rows.append({"cell": u["cell"], "unit": u["unit"], "role": role, **r,
                               "val_mae_loo": u["val_loo"].get((r["C"], r["rule"], r["w"]), np.nan)})
        for s in u["stats"]:
            stats_rows.append({"cell": u["cell"], "unit": u["unit"], "role": role, **s})
    per_unit = pd.DataFrame(rows)
    # RP error for base and final on BIDMC (reconstruct the scored series; test data after final)
    rp_rows = []
    for u in units:
        if family(u["cell"]) == "chaotic":
            continue
        task, g = u["cell"].split("|")[0], int(u["cell"].split("|g")[1])
        p = physio_prep(task, g, int(u["unit"]))
        for name, lad in [("base", spec["ladder"][0]), ("final", spec["ladder"][-1])]:
            if name == "base":
                fill = exact[(u["cell"], int(u["unit"]), u["role"])][1]
            else:
                fill = config_fill(u, spec, fc["rule"], fc["grid"], fc["mode"], final_choices[u["cell"]]["w"],
                                   final_choices[u["cell"]]["pool"])
            pred = p["scored"][:, 0].copy()
            pred[p["miss"]] = np.nan_to_num(fill, nan=0.0)
            sc = d1._score(pred, p["truth"], p["miss"], spec["embedding"][task])
            rp_rows.append({"cell": u["cell"], "unit": u["unit"], "role": u["role"], "arm": name, "rp": sc["rp"]})
    if rp_rows:
        per_unit = per_unit.merge(pd.DataFrame(rp_rows), on=["cell", "unit", "role", "arm"], how="left")
    save_table(per_unit, out, "per_unit")
    save_table(pd.DataFrame(per_C_rows), out, "per_C")
    st = pd.DataFrame(stats_rows)
    if len(st):
        agg = st.groupby(["cell", "role", "C", "rule", "w"], as_index=False)[
            ["n_gaps", "cand_total", "cand_rejected", "unfilled_before_fallback", "fb_pen", "fb_linear"]].sum()
        agg["rejected_share"] = agg["cand_rejected"] / agg["cand_total"].replace(0, np.nan)
        save_table(agg, out, "overlap_stats")

    final_df = per_unit[(per_unit["role"] == "final")]
    summary, pairs, decision = [], [], []
    for (cell, arm), ga in final_df.groupby(["cell", "arm"]):
        rec = {"cell": cell, "arm": arm, "n": len(ga), "mae": float(ga["mae"].mean())}
        m, lo, hi = d0.t_interval(ga["mae"])
        rec.update({"mae_lo": lo, "mae_hi": hi})
        if "rp" in ga and ga["rp"].notna().any():
            rec["rp"] = float(ga["rp"].mean())
        summary.append(rec)
    # fixed-C and best-C under the final rule (descriptive)
    pc = pd.DataFrame(per_C_rows)
    pc_fin = pc[(pc["role"] == "final") & (pc["rule"] == fc["rule"])]
    for cell, gpc in pc_fin.groupby("cell"):
        w = final_choices[cell]["w"]
        gw = gpc[np.isclose(gpc["w"], w)]
        for C, gc in gw.groupby("C"):
            summary.append({"cell": cell, "arm": f"fixed_C{int(C)}", "n": int(gc["unit"].nunique()),
                            "mae": float(gc["scored_mae"].mean())})
        best = gw.loc[gw.groupby("unit")["scored_mae"].idxmin()]
        summary.append({"cell": cell, "arm": "best_C_oracle", "n": int(best["unit"].nunique()),
                        "mae": float(best["scored_mae"].mean())})
    save_table(pd.DataFrame(summary), out, "summary")

    piv = {c: g.pivot(index="unit", columns="arm", values="mae") for c, g in final_df.groupby("cell")}
    lad_names = [l["name"] for l in spec["ladder"]]
    for cell, pv in piv.items():
        for a, b in zip(lad_names[1:], lad_names[:-1]):
            if a in pv and b in pv:
                dm, lo, hi = d0.t_interval((pv[a] - pv[b]).dropna())
                pairs.append({"cell": cell, "a": a, "b": b, "kind": "ladder", "diff": dm, "lo": lo, "hi": hi})
        if "final" in pv and "base" in pv:
            dm, lo, hi = d0.t_interval((pv["final"] - pv["base"]).dropna())
            pairs.append({"cell": cell, "a": "final", "b": "base", "kind": "primary", "diff": dm, "lo": lo, "hi": hi})
    save_table(pd.DataFrame(pairs), out, "pairs")

    fam_verdict = {}
    for fam, cells in [("chaotic", [c for c in spec["cells"] if family(c) == "chaotic"]),
                       ("bidmc", [c for c in spec["cells"] if family(c) != "chaotic"])]:
        n_imp = n_worse = 0
        for cell in cells:
            pv = piv.get(cell)
            if pv is None or "final" not in pv:
                continue
            diffs = (pv["final"] - pv["base"]).dropna()
            dm, lo, hi = d0.t_interval(diffs)
            verdict = ("identical" if bool((diffs.abs() < 1e-9).all()) else
                       "improved" if hi < 0 else "worse" if lo > 0 else "neither")
            n_imp, n_worse = n_imp + int(verdict == "improved"), n_worse + int(verdict == "worse")
            decision.append({"gate": "primary", "family": fam, "cell": cell, "statistic": "MAE final - base",
                             "value": dm, "ci_low": lo, "ci_high": hi, "verdict": verdict,
                             "note": f"n={len(diffs)}", "smoke": spec["smoke"]})
        verdict = ("adopt" if (n_imp >= len(cells) / 2 and n_worse == 0) else
                   "reject" if n_worse > n_imp else "inconclusive")
        fam_verdict[fam] = verdict
        decision.append({"gate": "verdict", "family": fam, "cell": "all", "statistic": "improved / worse cells",
                         "value": n_imp, "ci_low": np.nan, "ci_high": np.nan, "verdict": verdict,
                         "note": f"improved {n_imp}, worse {n_worse}, of {len(cells)} cells", "smoke": spec["smoke"]})
    write_decision(decision, out)

    # re-scored gates with final replacing recurrence (descriptive; original verdicts stand)
    stored = pd.read_csv(REPO / "notebooks" / "tables" / "dynimp" / "D1" / "per_unit.csv")
    stored_keys = set(zip(stored.loc[stored["arm"] == "recurrence", "cell"],
                          stored.loc[stored["arm"] == "recurrence", "unit"]))
    cols = ["cell", "unit", "mae"] + (["rp"] if "rp" in per_unit else [])
    new_rows = per_unit[per_unit["arm"] == "final"][cols].copy()
    new_rows = new_rows[[k in stored_keys for k in zip(new_rows["cell"], new_rows["unit"])]]
    d1ev = d1.evaluate(d1.build_spec(), _replace_recurrence(stored, new_rows))
    save_table(pd.DataFrame(d1ev["decision"]), out, "d1_rescored")
    stored2 = pd.read_csv(REPO / "notebooks" / "tables" / "dynimp" / "D2" / "per_unit.csv")
    bidmc_new = new_rows[new_rows["cell"].isin([f"{t}|g{g}" for t, g in d2.CELLS])]
    d2ev = d2.evaluate(d2.build_spec(), _replace_recurrence(stored2, bidmc_new))
    save_table(pd.DataFrame(d2ev["decision"]), out, "d2_rescored")
    for r in decision:
        log.info("%s | %s | %s -> %s %s", r["gate"], r.get("cell"), r["statistic"], r["verdict"], r["note"])


# ----------------------------------------------------------------------------- stage: write


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def stage_write() -> None:
    out = out_dir()
    spec = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    design = json.loads((out / "design.json").read_text(encoding="utf-8"))
    fc = spec["final_configuration"]
    decision = pd.read_csv(out / "decision.csv")
    summary = pd.read_csv(out / "summary.csv")
    pairs = pd.read_csv(out / "pairs.csv")
    cf = pd.read_csv(out / "dev_crossfit.csv")
    L = []
    L.append("# dynimp D3: revised recurrence fill\n")
    L.append("## 1. What changed and why\n")
    L.append("The D1 overlap rule counted entries that can never be observed (the target channel inside the "
             "gap) in its denominator, rejected many usable donors and left gaps unfilled to be scored as zero. "
             "D3 separates the corrected denominator from the admission threshold (ctx10-ctx60), adds penalised "
             "distances that charge for entries a candidate does not share with the query (pen05, pen10), a "
             "fallback (pen10 at the same C, then linear interpolation) so no gap is left unfilled, a wider C "
             "grid, restricted averaging over C (avg20, pool_avg20), per-cell ECG weight for PPG, and a "
             "leave-one-out-style validation (V_loo) whose missing rate equals the test rate. The D1_D2_C_Rerun "
             "verdict was decided by a 3.5e-17 rounding difference; D3 therefore defines a cell whose every "
             "unit differs by less than 1e-9 as identical.\n")
    L.append("## 2. Data roles and prior use\n")
    L.append("Development: the 33 D1 train+val BIDMC patients and chaotic seeds 100-111. Final: the 20 D1 test "
             "patients and chaotic seeds 200-211; seeds 0-5 only for the gate re-score. Prior use of the "
             "development patients: patients 1, 2, 3, 5 were in the first throwaway probe (2026-11-12); the "
             "50-60 % part of the 8 val patients was scored in the overlap-rule probe (their last 40 % was "
             "not); the other 22 train patients had never been filled by the recurrence fill. All 33 trained "
             "or selected the D1/D2 cross-patient learned models; no learned imputer is run on them here. "
             "Development results are reported for all 33 and for the 22 never-used patients in the "
             "cross-fitted estimate (section 4).\n")
    L.append("## 3. Development results and the chosen configuration\n")
    stats = sorted(design["statistic_per_config"], key=lambda e: e["statistic"])[:10]
    L.append(_md_table(["rule", "grid", "mode", "statistic (mean MAE ratio to base)"],
                       [[e["rule"], e["grid"], e["mode"], f"{e['statistic']:.4f}"] for e in stats]))
    L.append(f"\nChosen for all 10 cells: **{fc['rule']} / {fc['grid']} / {fc['mode']}** "
             f"(development statistic {fc['statistic_dev']:.4f}). Per-cell choices (pooled C, averaged C set, "
             "ECG weight w) are frozen in `spec.json`.\n")
    L.append("## 4. Cross-fitted development estimate\n")
    rows = [[c, f"{g['diff'].mean():+.4f}", int(g["diff"].count())] for c, g in cf.groupby("cell")]
    L.append(_md_table(["cell", "mean paired MAE difference (selected - base), held-out folds", "units"], rows))
    L.append("\nEach fold's configuration, pooled C and w were selected on the other two folds only, so these "
             "differences are free of selection on the scored units. The writeup tables in the repository "
             "(`dev_crossfit.csv`) also give the 22 never-used patients separately.\n")
    L.append("## 5. Final result and verdicts\n")
    for fam in ("chaotic", "bidmc"):
        vd = decision[(decision["gate"] == "verdict") & (decision["family"] == fam)].iloc[0]
        rows = []
        for r in decision[(decision["gate"] == "primary") & (decision["family"] == fam)].itertuples():
            rows.append([r.cell, f"{r.value:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]", r.verdict])
        L.append(f"### {fam}: **{vd['verdict']}** ({vd['note']})\n")
        L.append(_md_table(["cell", "MAE final - base [95 % interval]", "verdict"], rows) + "\n")
    L.append("## 6. Attribution ladder\n")
    rows = []
    for (a, b), g in pairs[pairs["kind"] == "ladder"].groupby(["a", "b"]):
        rows.append([f"{a} - {b}", f"{g['diff'].mean():+.4f}", f"{int((g['hi'] < 0).sum())}/{len(g)} cells improved"])
    L.append(_md_table(["step (against previous)", "mean paired difference over cells", "cells improved"], rows) + "\n")
    L.append("## 7. How much C matters\n")
    fc_rows = summary[summary["arm"].str.startswith(("fixed_C", "best_C"))]
    rows = [[r.cell, r.arm, f"{r.mae:.4f}"] for r in fc_rows.itertuples()]
    L.append(_md_table(["cell", "arm (final rule)", "MAE"], rows))
    L.append("\nFixed-C rows show the error if every unit used that C; best_C_oracle picks each unit's best C "
             "on the test part itself and is descriptive only (an upper bound on what C choice could give).\n")
    L.append("## 8. Gates re-scored\n")
    for name, orig in [("d1_rescored", "D1 (original verdict PASS)"), ("d2_rescored", "D2 (original verdict INCONCLUSIVE)")]:
        rs = pd.read_csv(out / f"{name}.csv")
        gate = "D1" if name.startswith("d1") else "D2"
        sub = rs[rs["gate"] == gate]
        if len(sub):
            L.append(f"{orig}: re-scored with the final fill replacing `recurrence` -> **{sub.iloc[0]['verdict']}**. "
                     "Descriptive only; the original verdict stands. The re-scored D2 gate is not a fair gate: the "
                     "fill gets a second round of tuning here while SAITS and CSDI stay at their D2 budget.\n")
    L.append("## 9. Limits\n")
    L.append("- The 20 test patients are not unseen. D1, D2, the multi-C probe and D1_D2_C_Rerun were all scored "
             "on them, and the ideas tested here (restricted averaging, ECG weight) came partly from those scores. "
             "What this plan guarantees is that every numeric and structural choice is made on other patients. "
             "The cross-fitted development estimate is the evidence that does not depend on the test patients; "
             "if it and the final run disagree in sign in a cell, both are reported above.\n"
             "- 11 of the 33 development patients were touched by earlier probes (section 2).\n"
             "- Reproduction note: with the BLAS distance pass used for the candidate fills, one ECG unit "
             "(g25, patient 9) broke an exact donor tie differently from D1 (distances agreed to 3e-15; MAE "
             "differed by 5.6e-05). The base arm reported here is therefore computed with the original D1 "
             "fill function, against which the reproduction check passes for every unit; the candidate fills "
             "keep the BLAS pass, whose tie-breaking among exactly tied donors is arbitrary in the same way.\n"
             "- The re-scored D2 gate is not a fair gate (section 8).\n"
             "- After this run, the 20 test patients should not be used for further development of the fill. A "
             "later confirmation needs new patients (the VitalDB cases under data/vitaldb/snuadc/ may serve; not "
             "examined for this plan).\n"
             "- V_loo lets donors come from after the validation gap, while at test time all donors precede the "
             "scored part.\n")
    path = REPO / "Writeups" / "dynimp_D3_fill_revision.md"
    path.write_text("\n".join(L), encoding="utf-8")
    log.info("writeup written to %s", path)


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["bench", "dev", "select", "final", "eval", "write"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(out / f"run_{args.stage}.log", mode="a", encoding="utf-8")):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    t0 = time.time()
    {"bench": lambda: stage_bench(args.workers), "dev": lambda: stage_dev(args.workers),
     "select": stage_select, "final": lambda: stage_final(args.workers), "eval": stage_eval,
     "write": stage_write}[args.stage]()
    log.info("stage %s done in %.0f s", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
