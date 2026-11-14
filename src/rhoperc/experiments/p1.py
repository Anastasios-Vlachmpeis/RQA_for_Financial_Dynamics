"""P1: triviality audit — is rho_perc pinned to rho_iso or rho_lag?

Run from RecPlay/ with PYTHONPATH=src (Python312).
Smoke: RHOPERC_SMOKE=1 -> 2 reps, N=100, writes to smoke/ folders.
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
PHASE = 1
VERSION = 1
IN_SCOPE = ["S1", "S2", "S3", "S4", "S5", "S6", "D1", "D2", "D3", "F1", "F2"]
TRANSFORMS = ["z", "rank"]
SETTINGS = ["rule", "minimum"]
GATE_RULE = 'At least 6 of the 11 in-scope processes are "free" (not pinned to a bound).'

from rhoperc.processes import PIDS, generate  # noqa: E402
from rhoperc import prep  # noqa: E402
from rhoperc.core import rho_perc  # noqa: E402


def _frozen_hash():
    frozen = json.loads((ROOT / "plans/rho_perc/FROZEN.json").read_text(encoding="utf-8"))
    expected = frozen["core_py_sha256"]
    actual = hashlib.sha256((ROOT / "src/rhoperc/core.py").read_bytes()).hexdigest()
    return expected, actual


def _taus():
    spec = json.loads((ROOT / "notebooks/tables/rho_perc/P0/spec.json").read_text(encoding="utf-8"))
    tau = spec["tau"]
    return {"F1": int(tau["F1"]), "F2": int(tau["F2"])}


def _run_cell(args):
    pid, rep, tau, n_states = args
    cell = PIDS.index(pid)
    cache_dir = ROOT / "data/processed/rho_perc/P1" / ("smoke" if SMOKE else f"v{VERSION}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{pid}_rep{rep}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rng = np.random.default_rng([PHASE, cell, rep])
    m = 3
    n_series = n_states + (m - 1) * tau
    x = generate(pid, n_series, rng)
    rule = prep.theiler_rule(x, m=m, tau=tau, n_states=n_states)
    w_rule = int(rule["w"])
    w_min = int((m - 1) * tau + 1)
    ws = {"rule": w_rule, "minimum": w_min}
    rows = []
    for transform in TRANSFORMS:
        D = prep.distance(x, transform=transform, m=m, tau=tau)
        for setting in SETTINGS:
            w = ws[setting]
            r = rho_perc(D, w, delta=0.05, connectivity=8)
            eps, eps_iso, eps_lag = float(r["eps"]), float(r["eps_iso"]), float(r["eps_lag"])
            rho, rho_iso, rho_lag = float(r["rho"]), float(r["rho_iso"]), float(r["rho_lag"])
            pinned_high = bool(eps >= 0.95 * eps_lag)
            pinned_low = bool((eps <= 1.05 * eps_iso) or (eps == 0.0 and eps_iso == 0.0))
            denom = rho_lag - rho_iso
            pi = float((rho - rho_iso) / denom) if denom != 0.0 else np.nan
            rows.append({
                "process": pid, "transform": transform, "theiler_setting": setting,
                "theiler_w": int(w), "rep": int(rep),
                "seed_phase": PHASE, "seed_cell": int(cell), "seed_rep": int(rep),
                "seed": f"[{PHASE},{cell},{rep}]",
                "N": int(len(D)), "m": m, "tau": int(tau),
                "tau_e": (None if rule["tau_e"] is None else int(rule["tau_e"])),
                "too_persistent": bool(rule["too_persistent"]),
                "eps_perc": eps, "rho_perc": rho,
                "eps_iso": eps_iso, "rho_iso": rho_iso,
                "eps_lag": eps_lag, "rho_lag": rho_lag,
                "pi": pi, "pinned_high": pinned_high, "pinned_low": pinned_low,
                "pinned": bool(pinned_high or pinned_low),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def _by_process(df):
    from scipy.stats import spearmanr
    rows = []
    for (pid, transform, setting), g in df.groupby(["process", "transform", "theiler_setting"], sort=False):
        def _sp(a, b):
            if g[a].nunique() < 2 or g[b].nunique() < 2:
                return float("nan")
            v = spearmanr(g[a], g[b]).statistic
            return float(v) if np.isfinite(v) else float("nan")
        rows.append({
            "process": pid, "transform": transform, "theiler_setting": setting,
            "n_runs": int(len(g)),
            "share_pinned_high": float(g["pinned_high"].mean()),
            "share_pinned_low": float(g["pinned_low"].mean()),
            "share_pinned": float(g["pinned"].mean()),
            "median_pi": float(g["pi"].median()) if g["pi"].notna().any() else float("nan"),
            "spearman_rho_perc_vs_rho_iso": _sp("rho_perc", "rho_iso"),
            "spearman_rho_perc_vs_rho_lag": _sp("rho_perc", "rho_lag"),
            "median_rho_iso": float(g["rho_iso"].median()),
            "median_rho_perc": float(g["rho_perc"].median()),
            "median_rho_lag": float(g["rho_lag"].median()),
            "share_too_persistent": float(g["too_persistent"].mean()),
        })
    out = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(PIDS)}
    out["_o"] = out["process"].map(order)
    out["_t"] = out["transform"].map({t: i for i, t in enumerate(TRANSFORMS)})
    out["_s"] = out["theiler_setting"].map({s: i for i, s in enumerate(SETTINGS)})
    return out.sort_values(["_o", "_t", "_s"]).drop(columns=["_o", "_t", "_s"]).reset_index(drop=True)


def _classification(by):
    """Per in/out-scope process under rule+z: free / pinned high / pinned low."""
    primary = by[(by["transform"] == "z") & (by["theiler_setting"] == "rule")]
    cls = {}
    for _, r in primary.iterrows():
        if r["share_pinned"] < 0.5:
            cls[r["process"]] = "free"
        elif r["share_pinned_high"] >= r["share_pinned_low"]:
            cls[r["process"]] = "pinned high"
        else:
            cls[r["process"]] = "pinned low"
    return cls


def _figure(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from condpred.report import save_fig
    g = df[(df["transform"] == "z") & (df["theiler_setting"] == "rule")]
    med = g.groupby("process")[["rho_iso", "rho_perc", "rho_lag"]].median().reindex(PIDS)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    xpos = np.arange(len(PIDS))
    width = 0.25
    for i, (col, label) in enumerate([("rho_iso", "rho_iso"), ("rho_perc", "rho_perc"), ("rho_lag", "rho_lag")]):
        ax.bar(xpos + (i - 1) * width, med[col].to_numpy(), width=width, label=label)
    ax.set_xticks(xpos)
    ax.set_xticklabels(PIDS)
    ax.set_ylabel("recurrence rate (median over runs)")
    ax.set_xlabel("process")
    ax.set_title("P1: rho_iso, rho_perc, rho_lag per process (z transform, Theiler rule, N=400)")
    ax.legend()
    fig.tight_layout()
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "figures", "rho_bounds_by_process")
    plt.close(fig)


def main():
    from condpred.report import write_spec, write_decision, save_table

    expected, actual = _frozen_hash()
    if actual != expected:
        stop = ROOT / "plans/rho_perc/STOP_P1.md"
        stop.write_text(
            f"# STOP P1\n\nFrozen core hash mismatch: FROZEN.json={expected}, "
            f"current src/rhoperc/core.py={actual}. P1 not run.\n", encoding="utf-8")
        print(f"STOP_P1 frozen hash mismatch expected={expected} actual={actual}", flush=True)
        return 2

    taus = _taus()
    n_states = 100 if SMOKE else 400
    reps = 2 if SMOKE else 50
    out_dir = ROOT / "notebooks/tables/rho_perc/P1" / ("smoke" if SMOKE else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "version": VERSION,
        "phase": "P1",
        "smoke": SMOKE,
        "processes": PIDS,
        "in_scope_processes": IN_SCOPE,
        "transforms": TRANSFORMS,
        "theiler_settings": {
            "rule": "w = max((m-1)*tau+1, tau_e); tau_e = smallest lag L>=1 with autocorrelation < 1/e, searched L<=N/10; if none, too_persistent=True and w=(m-1)*tau+1 (rhoperc.prep.theiler_rule)",
            "minimum": "w = (m-1)*tau + 1",
        },
        "reps": reps,
        "N_states": n_states,
        "m": 3,
        "tau": {"stochastic_and_maps": 1, "F1": taus["F1"], "F2": taus["F2"],
                "source": "frozen in notebooks/tables/rho_perc/P0/spec.json"},
        "delta": 0.05,
        "connectivity": 8,
        "seeds": "np.random.default_rng([phase, cell, rep]) with phase=1, cell=index in rhoperc.processes.PIDS; same seed for all transform/Theiler cells of a (process, rep); seed recorded per row in runs.csv",
        "definitions": {
            "pinned_high": "eps_perc >= 0.95 * eps_lag",
            "pinned_low": "eps_perc <= 1.05 * eps_iso (or both eps_perc and eps_iso are 0)",
            "pinned": "pinned_high or pinned_low",
            "free": "process is free if fewer than half of its runs are pinned, under the Theiler rule and the z transform",
            "pi": "(rho_perc - rho_iso) / (rho_lag - rho_iso); if rho_lag == rho_iso the denominator is 0 and pi is recorded as NaN (undefined: all three values coincide, so no position between the bounds exists)",
        },
        "gate": "G1",
        "gate_rule": GATE_RULE,
        "gate_outcomes": ">=6 free: PASS; 3-5 free: continue, paper limited to those classes; 0-2 free: STOP",
        "core_py_sha256_frozen": expected,
        "core_py_sha256_at_spec": actual,
    }
    write_spec(spec, out_dir, modules=["rhoperc.core", "rhoperc.prep", "rhoperc.processes", "rhoperc.experiments.p1"])

    tasks = [(pid, rep, (taus[pid] if pid in taus else 1), n_states) for pid in PIDS for rep in range(reps)]
    t0 = time.time()
    from joblib import Parallel, delayed
    frames = Parallel(n_jobs=8)(delayed(_run_cell)(a) for a in tasks)
    elapsed = time.time() - t0
    df = pd.concat(frames, ignore_index=True)
    order = {p: i for i, p in enumerate(PIDS)}
    df["_o"] = df["process"].map(order)
    df["_t"] = df["transform"].map({t: i for i, t in enumerate(TRANSFORMS)})
    df["_s"] = df["theiler_setting"].map({s: i for i, s in enumerate(SETTINGS)})
    df = df.sort_values(["_o", "rep", "_t", "_s"]).drop(columns=["_o", "_t", "_s"]).reset_index(drop=True)
    save_table(df, out_dir, "runs")
    by = _by_process(df)
    save_table(by, out_dir, "by_process")
    print(f"runs rows={len(df)} by_process rows={len(by)} elapsed={elapsed:.1f}s smoke={SMOKE}", flush=True)

    if not SMOKE:
        _figure(df, out_dir)
        cls = _classification(by)
        free_in = [p for p in IN_SCOPE if cls.get(p) == "free"]
        n_free = len(free_in)
        if n_free >= 6:
            result, passed = "PASS", True
        elif n_free >= 3:
            result, passed = "CONTINUE_LIMITED", False
        else:
            result, passed = "STOP", False
        # Rerun selfcheck once to confirm frozen state.
        r = subprocess.run([sys.executable, "-m", "rhoperc.selfcheck"], cwd=ROOT,
                           env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
        print(r.stdout, flush=True)
        if r.stderr:
            print(r.stderr, flush=True)
        print(f"selfcheck exit={r.returncode}", flush=True)
        write_decision([{
            "gate": "G1", "result": result, "pass": bool(passed),
            "rule": GATE_RULE,
            "n_free_in_scope": int(n_free), "n_in_scope": len(IN_SCOPE),
            "free_processes": ",".join(free_in),
            "classification_rule_z": "; ".join(f"{p}={cls[p]}" for p in PIDS),
            "runs_rows": int(len(df)), "by_process_rows": int(len(by)),
            "core_py_sha256": actual,
            "selfcheck_exit": int(r.returncode),
            "selfcheck_all_pass": bool(r.returncode == 0 and "ALL PASS" in r.stdout),
        }], out_dir)
        if result == "STOP":
            stop = ROOT / "plans/rho_perc/STOP_P1.md"
            stop.write_text(
                f"# STOP P1\n\nGate G1: only {n_free} of 11 in-scope processes are free "
                f"(rule+z; see notebooks/tables/rho_perc/P1/decision.csv and by_process.csv). "
                f"Free: {', '.join(free_in) or 'none'}. Constitution Article 6: 0-2 free => STOP.\n",
                encoding="utf-8")
            print("STOP_P1 written", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
