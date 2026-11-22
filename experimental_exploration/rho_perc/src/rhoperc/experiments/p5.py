"""P5: incremental value (00_CONSTITUTION.md P5).

Spec is written before any outcome is computed. Frozen core verified.
z transform only (Decision T). Seeds: np.random.default_rng([5, cell, rep]).
One file per (cell, rep) under cache/P5/v<version>/.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SMOKE = os.environ.get("RHOPERC_SMOKE") == "1"
PHASE = 5
VERSION = 1
DELTA = 0.05
M = 3
TAU = {"F1": 7, "F2": 4}
PROCESSES = ["S1", "S2", "S3", "S4", "S5", "S6", "D1", "D2", "D3", "F1", "F2"]
GATE_RULE = "Median unexplained share \u2265 0.20 and at least one task where adding \u03c1_perc raises AUC with a 95 % interval above 0."
BASE_N = 100 if SMOKE else 400
REPS = 2 if SMOKE else 100

from rhoperc import prep  # noqa: E402
from rhoperc.competitors import COMPETITORS, compute_competitors  # noqa: E402
from rhoperc.core import rho_perc  # noqa: E402
from rhoperc.processes import generate  # noqa: E402

C = list(COMPETITORS)
FEATURES = ["rho_perc"] + C

# cells: 0-10 P5a processes; 11 TaskA; 12 TaskB control; 13 TaskB shift; 14 TaskC pairs
R_FULL = np.round(np.arange(3.5, 4.0 + 1e-9, 0.001), 3)
TASKA_REPS = (list(range(0, 501, 25)) if SMOKE else list(range(501)))
TASK_REPS = 2 if SMOKE else 200


def _frozen_hash():
    frozen = json.loads((ROOT / "plans/FROZEN.json").read_text(encoding="utf-8"))
    expected = frozen["core_py_sha256"]
    actual = hashlib.sha256((ROOT / "src/rhoperc/core.py").read_bytes()).hexdigest()
    return expected, actual


def _cache_dir():
    d = ROOT / "cache/P5" / (f"smoke_v{VERSION}" if SMOKE else f"v{VERSION}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tau(pid):
    return TAU.get(pid, 1)


def _lyapunov(r):
    x, total, count = 0.5, 0.0, 0
    n_iter = 10_000 if SMOKE else 100_000
    for i in range(1000 + n_iter):
        d = abs(r * (1.0 - 2.0 * x))
        if i >= 1000:
            total += np.log(d) if d > 0 else -np.inf
            count += 1
        x = r * x * (1.0 - x)
    return float(total / count)


def _row(dataset, process, cell, rep, x, m, tau, label=None, label_name="", w_fixed=None, extra=None):
    xt = prep.transform_series(x, "z")
    from recplay.embedding import embed
    from recplay.recurrence import euclidean_distances
    states = embed(xt, m, tau)
    D = euclidean_distances(states)
    rule = prep.theiler_rule(x, m=m, tau=tau, n_states=len(D))
    w = int(w_fixed) if w_fixed is not None else int(rule["w"])
    core = rho_perc(D, w, delta=DELTA, connectivity=8)
    if not (core["rho_iso"] - 1e-12 <= core["rho"] <= core["rho_lag"] + 1e-12):
        raise AssertionError(f"bound violated {core}")
    comp = compute_competitors(D, w, xt, core_result=core)
    row = {"dataset": dataset, "process": process, "cell": int(cell), "rep": int(rep),
           "label": label, "label_name": label_name, "n_states": int(len(D)),
           "m": int(m), "tau": int(tau), "theiler_w": w,
           "tau_e_rule": rule["tau_e"], "too_persistent": bool(rule["too_persistent"]),
           "rho_perc": float(core["rho"]), "eps_perc": float(core["eps"])}
    row.update(comp)
    if extra:
        row.update(extra)
    return row


def _run_cell(args):
    cell, rep = args
    cache = _cache_dir() / f"cell{cell}_rep{rep}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rng = np.random.default_rng([PHASE, int(cell), int(rep)])
    if cell < 11:  # P5a
        pid = PROCESSES[cell]
        tau = _tau(pid)
        x = generate(pid, BASE_N + (M - 1) * tau, rng)
        rows = [_row("P5a", pid, cell, rep, x, M, tau)]
    elif cell == 11:  # TaskA, m=1, N=1000 full / 100 smoke
        from condpred.systems import logistic
        r_val = float(R_FULL[rep])
        n = BASE_N if SMOKE else 1000
        x = logistic(r_val, n, rng)
        lam = _lyapunov(r_val)
        rows = [_row("TaskA", "logistic", cell, rep, x, 1, 1,
                     label=int(lam > 0), label_name="chaotic" if lam > 0 else "periodic",
                     extra={"r": r_val, "lambda_lyapunov": lam})]
    elif cell in (12, 13):  # TaskB
        tau = 1
        x = generate("S3", BASE_N + (M - 1) * tau, rng)
        shift = (cell == 13)
        t_shift = None
        if shift:
            t_shift = int(rng.integers(len(x) // 4, 3 * len(x) // 4))
            x = np.asarray(x, dtype=float).copy()
            x[t_shift:] += 1.5
        rows = [_row("TaskB", "S3", cell, rep, x, M, tau,
                     label=int(shift), label_name="shift" if shift else "no_shift",
                     extra={"shift_time": t_shift})]
    elif cell == 14:  # TaskC: noisy F1 original + its IAAFT surrogate
        from condpred.contaminate import add_obs_noise
        from recplay.surrogates import iaaft
        tau = _tau("F1")
        clean = generate("F1", BASE_N + (M - 1) * tau, rng)
        noisy = add_obs_noise(clean, 0.25 * float(np.std(clean)), rng)
        rule = prep.theiler_rule(noisy, m=M, tau=tau, n_states=BASE_N)
        w0 = int(rule["w"])
        r_orig = _row("TaskC", "F1", cell, rep, noisy, M, tau,
                      label=0, label_name="noisy_original", w_fixed=w0,
                      extra={"pair_id": int(rep)})
        surr = iaaft(noisy, rng)
        r_surr = _row("TaskC", "F1", cell, rep, surr, M, tau,
                      label=1, label_name="iaaft_surrogate", w_fixed=w0,
                      extra={"pair_id": int(rep)})
        rows = [r_orig, r_surr]
    else:
        raise ValueError(cell)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def _tasks_list():
    t = [(c, r) for c in range(11) for r in range(REPS)]
    t += [(11, r) for r in TASKA_REPS]
    t += [(12, r) for r in range(TASK_REPS)] + [(13, r) for r in range(TASK_REPS)]
    t += [(14, r) for r in range(TASK_REPS)]
    return t


# ---------------------------------------------------------------- modelling
def _reg_models():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    ridge = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()), ("m", Ridge())])
    gbr = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("m", GradientBoostingRegressor(random_state=5))])
    return {"ridge": ridge, "gbr": gbr}


def _unexplained(features):
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    kf = KFold(2 if SMOKE else 5, shuffle=True, random_state=5)
    rows = []
    def one(df, scope):
        X, y = df[C], df["rho_perc"]
        r2 = {}
        for name, mdl in _reg_models().items():
            pred = cross_val_predict(mdl, X, y, cv=kf)
            r2[name] = float(r2_score(y, pred))
        u = 1.0 - max(max(r2.values()), 0.0)
        return {"scope": scope, "n": int(len(df)), "R2_ridge": r2["ridge"],
                "R2_gbr": r2["gbr"], "U": float(u)}
    p5a = features[features.dataset == "P5a"]
    for pid in PROCESSES:
        rows.append(one(p5a[p5a.process == pid], pid))
    med_u = float(np.median([r["U"] for r in rows]))
    rows.append({"scope": "MEDIAN", "n": len(rows),
                 "R2_ridge": float(np.median([r["R2_ridge"] for r in rows])),
                 "R2_gbr": float(np.median([r["R2_gbr"] for r in rows])), "U": med_u})
    pooled = one(p5a, "POOLED")
    rows.append(pooled)
    return pd.DataFrame(rows), med_u, pooled["U"]


def _logreg(feats):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=1000))])


def _oof(mdl, X, y, seed):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    nsp = 2 if SMOKE else 5
    cv = StratifiedKFold(nsp, shuffle=True, random_state=seed)
    return cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1]


def _tasks_table(features):
    from sklearn.metrics import roc_auc_score
    rows = []
    task_delta = {}
    n_rep = 2 if SMOKE else 5
    n_boot = 20 if SMOKE else 1000
    for ti, dataset in enumerate(["TaskA", "TaskB", "TaskC"]):
        df = features[features.dataset == dataset]
        y = df["label"].to_numpy(dtype=int)
        model_sets = [(f, [f]) for f in FEATURES] + [("C", C), ("C+rho_perc", C + ["rho_perc"])]
        aucs = {}
        for name, cols in model_sets:
            reps_auc = []
            for rep_i in range(n_rep):
                pred = _oof(_logreg(cols), df[cols], y, seed=5 + rep_i)
                reps_auc.append(float(roc_auc_score(y, pred)))
            aucs[name] = float(np.mean(reps_auc))
            rows.append({"task": dataset, "model": name, "features": "+".join(cols),
                         "auc": aucs[name], "delta_auc": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan})
        # delta CI from canonical (seed=5) pooled OOF predictions, bootstrap rows
        p_c = _oof(_logreg(C), df[C], y, seed=5)
        p_cr = _oof(_logreg(C + ["rho_perc"]), df[C + ["rho_perc"]], y, seed=5)
        rng = np.random.default_rng([PHASE, 900 + ti, 0])
        deltas = []
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            deltas.append(roc_auc_score(y[idx], p_cr[idx]) - roc_auc_score(y[idx], p_c[idx]))
        lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        delta = aucs["C+rho_perc"] - aucs["C"]
        task_delta[dataset] = (delta, lo, hi)
        for r in rows:
            if r["task"] == dataset and r["model"] == "C+rho_perc":
                r["delta_auc"], r["ci_lo"], r["ci_hi"] = delta, lo, hi
            if r["task"] == dataset and r["model"] == "C":
                r["delta_auc"] = 0.0
    return pd.DataFrame(rows), task_delta


def main():
    from condpred.report import save_table, write_decision, write_spec
    expected, actual = _frozen_hash()
    if actual != expected:
        (ROOT / "plans/STOP_P5.md").write_text(
            f"# STOP P5\n\nFrozen core hash mismatch: FROZEN.json={expected}, current={actual}. P5 not run.\n",
            encoding="utf-8")
        print(f"STOP_P5 frozen mismatch {expected} {actual}", flush=True)
        return 2
    out_dir = ROOT / "tables/P5" / ("smoke" if SMOKE else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": VERSION, "phase": "P5", "smoke": SMOKE,
        "gate": "G5", "gate_rule": GATE_RULE,
        "gate_code": "PASS iff median per-process U >= 0.20 AND at least one task deltaAUC 95% CI lower bound > 0; else STOP_P5.md",
        "transform": "z only (Decision T, notebooks/tables/rho_perc/P3/decision.csv)",
        "baseline": {"N_states": BASE_N, "m": 3, "delta": DELTA, "connectivity": 8,
                     "tau_by_process": {p: _tau(p) for p in PROCESSES},
                     "theiler": "w = rhoperc.prep.theiler_rule(series, m, tau, n_states=N) per run, except TaskC surrogate uses w of its original"},
        "competitors_C": C,
        "competitor_definitions": {
            "DET_LAM_L_ENTR_K2_lines": "recplay.rqa.rqa(recplay.recurrence.threshold_by_rate(D, 0.10, theiler=w))",
            "lag1_autocorr": "lag-1 autocorrelation of the z-transformed series",
            "tau_e": "smallest L>=1 with autocorr < 1/e, searched L<=N/10 (N states); censored at N//10+1 if none",
            "rho_iso_rho_lag": "Article 3.4 bounds from rhoperc.core.rho_perc",
            "rho_conn": "e_conn = largest edge weight of scipy.sparse.csgraph.minimum_spanning_tree over the complete graph on the N states with weights D; rho_conn = share of Omega points (upper triangle, lag>=w) with D_ij <= e_conn",
            "largest_share": "scipy.ndimage.label (8-neighbour 3x3) on the RR=10% plot; largest component points / total recurrence points (0 if none)"},
        "P5a": {"processes": PROCESSES, "runs_per_process": REPS, "N_states": BASE_N,
                "models": "per process: 5-fold CV (KFold shuffle, random_state=5; 2-fold in smoke), ridge = Pipeline(median imputer, StandardScaler, Ridge() defaults), gradient boosting = Pipeline(median imputer, GradientBoostingRegressor(random_state=5, otherwise sklearn defaults)); R2 from pooled out-of-fold predictions (cross_val_predict); U = 1 - max(max(R2_ridge, R2_gbr), 0)",
                "median_and_pooled": "MEDIAN row = median of the 11 per-process values; POOLED = same U computation on all P5a rows pooled, C only, no process indicator"},
        "P5b": {"model": "Pipeline(median imputer, StandardScaler, LogisticRegression(max_iter=1000, sklearn defaults otherwise))",
                "point_auc": "5-fold stratified CV repeated 5 times (repeat i uses StratifiedKFold shuffle random_state=5+i; 2 folds x 2 repeats in smoke); each repeat AUC from pooled out-of-fold predict_proba; reported AUC = mean over repeats; computed for each single feature (incl. rho_perc), C, and C+rho_perc",
                "delta_CI": "canonical pooled OOF predictions from one 5-fold stratified CV (random_state=5) for C and C+rho_perc; 1000 bootstrap resamples of rows (y, pred_C, pred_C+rho) with rng default_rng([5, 900+task_index, 0]), task_index TaskA=0,TaskB=1,TaskC=2 (20 resamples in smoke); CI = 2.5/97.5 percentiles of bootstrap deltaAUC; delta point = repeated-CV mean AUC(C+rho)-mean AUC(C)",
                "TaskA": "P3a logistic rows regenerated: r=3.5..4.0 step 0.001 (501), m=1, tau=1, N=1000 states (100 smoke, r every 25th), z; series logistic(r, n, rng=default_rng([5,11,r_idx])) (condpred logistic burn=1000, uniform x0 draw from rng); lambda = mean log|r(1-2x)| over 100000 iterates after 1000 transient from x0=0.5 as in P3; y=1 chaotic (lambda>0), 0 periodic",
                "TaskB": "S3 (ar1 0.9) N=400 states m=3 tau=1 z; control cell12 vs shift cell13, 200 runs each (2 smoke); shift = +1.5 sd (sd=1) from t_shift onward, t_shift drawn from run rng after generation, uniform in middle half [n_series//4, 3*n_series//4); y=1 shift",
                "TaskC": "F1 N=400 states m=3 tau=7 z; per pair (cell14, 200 pairs): clean rossler, noisy = add_obs_noise(clean, 0.25*std(clean), rng); row y=0 noisy original; row y=1 iaaft(noisy, same rng) surrogate of same run, using w of the noisy original; 400 rows total"},
        "features_csv": "one row per run; dataset column in {P5a, TaskA, TaskB, TaskC}; columns include process, cell, rep, label, rho_perc and all C",
        "seeds": "np.random.default_rng([5, cell, rep]); cells 0-10 = P5a processes in listed order, 11=TaskA (rep=r_index), 12/13=TaskB control/shift, 14=TaskC pairs (rep=pair). Workers receive only (cell, rep) ints.",
        "caches": "data/processed/rho_perc/P5/v<version>/ (smoke_v<version> in smoke), one csv per (cell, rep), rerun skips existing",
        "outputs": ["features.csv", "unexplained.csv", "tasks.csv", "decision.csv"],
        "core_py_sha256_frozen": expected, "core_py_sha256_at_spec": actual,
    }
    write_spec(spec, out_dir, modules=["rhoperc.core", "rhoperc.prep", "rhoperc.processes",
                                       "rhoperc.competitors", "rhoperc.experiments.p5"])
    from joblib import Parallel, delayed
    tasks = _tasks_list()
    t0 = time.time()
    frames = Parallel(n_jobs=8)(delayed(_run_cell)(a) for a in tasks)
    elapsed = time.time() - t0
    features = pd.concat(frames, ignore_index=True)
    print(f"features rows={len(features)} tasks={len(tasks)} elapsed={elapsed:.1f}s smoke={SMOKE}", flush=True)
    if not SMOKE:
        assert len(features[features.dataset == "P5a"]) == 1100
        assert len(features) == 2401, len(features)
    save_table(features, out_dir, "features")
    unexplained, med_u, pooled_u = _unexplained(features)
    save_table(unexplained, out_dir, "unexplained")
    tasks_df, task_delta = _tasks_table(features)
    save_table(tasks_df, out_dir, "tasks")
    print(f"unexplained medianU={med_u:.4f} pooledU={pooled_u:.4f} tasks={task_delta}", flush=True)
    if not SMOKE:
        any_task = any(lo > 0 for (_, lo, _) in task_delta.values())
        passed = bool(med_u >= 0.20 and any_task)
        write_decision([{"item": "G5_gate", "rule": GATE_RULE,
                         "result": "PASS" if passed else "FAIL",
                         "detail": f"median U={med_u:.4f} (>=0.20: {med_u>=0.20}); pooled U={pooled_u:.4f}; " +
                                   "; ".join(f"{k}: deltaAUC={d:.4f} CI=[{lo:.4f},{hi:.4f}]" for k, (d, lo, hi) in task_delta.items())}],
                       out_dir)
        if not passed:
            (ROOT / "plans/STOP_P5.md").write_text(
                f"# STOP P5 — measure redundant (Gate G5 failed)\n\nGate rule: {GATE_RULE}\n\n"
                f"Median unexplained share U = {med_u:.4f} (need >= 0.20). Pooled U = {pooled_u:.4f}.\n\n" +
                "\n".join(f"- {k}: deltaAUC = {d:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]" for k, (d, lo, hi) in task_delta.items()) +
                "\n\nEvidence: tables/P5/unexplained.csv and tasks.csv. "
                "See plans/reports/REPORT_P5.md.\n", encoding="utf-8")
        print(f"G5 {'PASS' if passed else 'FAIL'}", flush=True)
        r = subprocess.run([sys.executable, "-m", "rhoperc.selfcheck"], cwd=ROOT,
                           env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT.parents[1] / "src")))}, capture_output=True, text=True)
        print(r.stdout, flush=True)
        if r.stderr:
            print(r.stderr, flush=True)
        print(f"selfcheck exit={r.returncode}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
