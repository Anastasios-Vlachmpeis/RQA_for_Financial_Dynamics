"""P0: legacy vs span vs cover rule on the 13-process panel.

Run from RecPlay/ with PYTHONPATH=src.
Smoke: RHOPERC_SMOKE=1 -> 2 reps, N=100, writes to smoke/ folders.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SMOKE = os.environ.get("RHOPERC_SMOKE") == "1"
PHASE = 0
VERSION = 1

from rhoperc.processes import PIDS, generate  # noqa: E402
from rhoperc import prep  # noqa: E402
from rhoperc.core import rho_perc  # noqa: E402


def frozen_tau(pid):
    """First local minimum of AMI on one reference run (length 5000, seed 0)."""
    from recplay.embedding import average_mutual_information, first_local_minimum
    rng = np.random.default_rng(0)
    x = generate(pid, 5000, rng)
    tau = first_local_minimum(average_mutual_information(x, 100))
    return int(tau) if tau is not None else 1


def tau_for(pid, taus):
    return taus[pid] if pid in taus else 1


def _run_cell(args):
    pid, rep, tau, n_states = args
    from recplay.connectivity import rr_critical
    from condpred.features.percolation import rho_perc_span
    cell = PIDS.index(pid)
    cache_dir = ROOT / "data/processed/rho_perc/P0" / ("smoke" if SMOKE else f"v{VERSION}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{pid}_rep{rep}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rng = np.random.default_rng([PHASE, cell, rep])
    m = 3
    w = (m - 1) * tau + 1
    n_series = n_states + (m - 1) * tau
    x = generate(pid, n_series, rng)
    D = prep.distance(x, transform="z", m=m, tau=tau)
    cover = rho_perc(D, w, delta=0.05, connectivity=8)
    row = {"pid": pid, "rep": int(rep), "N": int(len(D)), "m": m, "tau": int(tau), "w": int(w),
           "legacy": float(rr_critical(D, theiler=w, delta=0.05)),
           "span": float(rho_perc_span(D, theiler=w, delta=0.05)),
           "cover": float(cover["rho"]), "cover_eps": float(cover["eps"]),
           "rho_iso": float(cover["rho_iso"]), "rho_lag": float(cover["rho_lag"])}
    df = pd.DataFrame([row])
    df.to_csv(cache, index=False)
    return df


def main():
    from condpred.report import write_spec, write_decision, save_table

    taus = {"F1": frozen_tau("F1"), "F2": frozen_tau("F2")}
    n_states = 100 if SMOKE else 400
    reps = 2 if SMOKE else 20
    out_dir = ROOT / "notebooks/tables/rho_perc/P0" / ("smoke" if SMOKE else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "version": VERSION,
        "phase": "P0",
        "smoke": SMOKE,
        "processes": PIDS,
        "reps": reps,
        "N_states": n_states,
        "m": 3,
        "tau": {"stochastic_and_maps": 1, "F1": taus["F1"], "F2": taus["F2"],
                "tau_rule": "first local minimum of AMI on reference run length 5000 seed 0, max_lag 100"},
        "delta": 0.05,
        "connectivity": 8,
        "transform": "z",
        "theiler": "w = (m-1)*tau + 1",
        "seeds": "np.random.default_rng([phase, cell, rep]) with phase=0",
        "rules_compared": ["legacy: recplay.connectivity.rr_critical",
                           "span: condpred.features.percolation.rho_perc_span",
                           "cover: rhoperc.core.rho_perc"],
        "gate": "G0",
        "gate_rule": "Every known-answer test passes.",
    }
    # Spec before any outcome computation in this run.
    write_spec(spec, out_dir, modules=["rhoperc.core", "rhoperc.prep", "rhoperc.processes", "rhoperc.experiments.p0"])

    tasks = [(pid, rep, tau_for(pid, taus), n_states) for pid in PIDS for rep in range(reps)]
    t0 = time.time()
    from joblib import Parallel, delayed
    frames = Parallel(n_jobs=8)(delayed(_run_cell)(a) for a in tasks)
    elapsed = time.time() - t0
    df = pd.concat(frames, ignore_index=True).sort_values(["pid", "rep"], key=lambda s: s.map(PIDS.index) if s.name == "pid" else s)
    save_table(df, out_dir, "rule_comparison")
    print(f"comparison rows={len(df)} elapsed={elapsed:.1f}s smoke={SMOKE}", flush=True)

    if not SMOKE:
        # Gate G0 by code from selfcheck output.
        r = subprocess.run([sys.executable, "-m", "rhoperc.selfcheck"], cwd=ROOT,
                           env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
        print(r.stdout, flush=True)
        ks = {}
        for line in r.stdout.splitlines():
            if ":" in line and line.split(":")[0].strip().startswith("K"):
                ks[line.split(":")[0].strip()] = "PASS" in line
        all_pass = len(ks) == 9 and all(ks.values()) and r.returncode == 0
        # Fast timing at N=2000 (saved in decision.csv).
        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000 + 2)
        D = prep.distance(x, transform="z", m=3, tau=1)
        t = time.time()
        rho_perc(D, 3)
        timing = time.time() - t
        print(f"N2000 timing={timing:.2f}s", flush=True)
        write_decision([{"gate": "G0", "pass": bool(all_pass),
                         "rule": "Every known-answer test passes.",
                         "k_tests": ",".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in sorted(ks.items())),
                         "selfcheck_exit": int(r.returncode),
                         "n2000_seconds": round(float(timing), 3),
                         "n2000_under_5s": bool(timing < 5.0)}], out_dir)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
