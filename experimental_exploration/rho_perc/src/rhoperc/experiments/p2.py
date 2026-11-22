"""P2: null behaviour (iid marginals) and size of the IAAFT surrogate test.

Spec (written before any outcome) is in tables/P2/spec.json.
See plans/00_CONSTITUTION.md, phase P2 and Articles 4/6 (gate G2).
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
PHASE = 2
VERSION = 1
DELTA = 0.05
TAU = 1
M_SIZE = 3  # Article 4 baseline embedding for P2b

GATE_RULE = "The IAAFT test rejects in at most 9 % of runs in every linear-null cell."
# Spec-before-results counting rule (assignment, stated here before any run):
# gate cells = ALL P2b cells in size_summary.csv (as-generated and exp(.)
# variants, both transforms, including the deduplicated exp(.)+rank cells).

MARGINALS = ["normal", "uniform", "student_t3", "lognormal", "two_peaked"]
NS_IID = [100, 200, 400, 800, 1600]
MS_IID = [1, 3]
TRANSFORMS = ["z", "rank"]
PROCESSES = ["S2", "S3", "S4"]
VARIANTS = ["as_generated", "exp"]
NS_SIZE = [200, 400]
N_SURR_FULL = 39


def _reps(n):
    return 200 if n <= 400 else (100 if n == 800 else 50)


def _frozen_hash():
    frozen = json.loads((ROOT / "plans/FROZEN.json").read_text(encoding="utf-8"))
    expected = frozen["core_py_sha256"]
    actual = hashlib.sha256((ROOT / "src/rhoperc/core.py").read_bytes()).hexdigest()
    return expected, actual


def _gen_marginal(name, n, rng):
    """Generate a marginal sample directly (two_peaked is a direct mixture,
    not a panel process)."""
    if name == "normal":
        return rng.standard_normal(n)
    if name == "uniform":
        return rng.random(n)
    if name == "student_t3":
        return rng.standard_t(3, n)
    if name == "lognormal":
        return rng.lognormal(0.0, 1.0, n)
    if name == "two_peaked":
        comp = rng.random(n) < 0.5
        return rng.standard_normal(n) + np.where(comp, -2.0, 2.0)
    raise ValueError(name)


def _cell_iid(marginal, N, m):
    return MARGINALS.index(marginal) * (len(NS_IID) * len(MS_IID)) + NS_IID.index(N) * len(MS_IID) + MS_IID.index(m)


def _cell_size(process, N):
    # offset 50 = number of P2a (marginal, N, m) cells; seed is for the base
    # series, both variants (as_generated / exp(.)) derive from it.
    return 50 + PROCESSES.index(process) * len(NS_SIZE) + (NS_SIZE.index(N) if N in NS_SIZE else 0)


def _iid_task(args):
    from rhoperc import prep
    from rhoperc.core import rho_perc

    marginal, N, m, rep = args
    cache_dir = ROOT / "cache/P2" / ("smoke" if SMOKE else f"v{VERSION}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"iid_{marginal}_N{N}_m{m}_rep{rep}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    cell = _cell_iid(marginal, N, m)
    rng = np.random.default_rng([PHASE, cell, rep])
    n_series = N + (m - 1) * TAU  # n giving N states after embedding (prep/P1 convention)
    x = _gen_marginal(marginal, n_series, rng)
    rule = prep.theiler_rule(x, m=m, tau=TAU, n_states=N)
    w = int(rule["w"])
    rows = []
    for transform in TRANSFORMS:
        D = prep.distance(x, transform=transform, m=m, tau=TAU)
        r = rho_perc(D, w, delta=DELTA, connectivity=8)
        rows.append({
            "marginal": marginal, "N": int(N), "m": int(m), "transform": transform,
            "rep": int(rep), "seed": f"[{PHASE},{cell},{rep}]", "seed_cell": int(cell),
            "n_series": int(n_series), "theiler_w": w,
            "tau_e": (None if rule["tau_e"] is None else int(rule["tau_e"])),
            "too_persistent": bool(rule["too_persistent"]),
            "eps_perc": float(r["eps"]), "rho_perc": float(r["rho"]),
            "eps_iso": float(r["eps_iso"]), "rho_iso": float(r["rho_iso"]),
            "eps_lag": float(r["eps_lag"]), "rho_lag": float(r["rho_lag"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def _size_task(args):
    from recplay.surrogates import iaaft
    from rhoperc import prep
    from rhoperc.core import rho_perc
    from rhoperc.processes import generate

    process, N, rep, n_surr = args
    cache_dir = ROOT / "cache/P2" / ("smoke" if SMOKE else f"v{VERSION}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"size_{process}_N{N}_rep{rep}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    cell = _cell_size(process, N)
    rng = np.random.default_rng([PHASE, cell, rep])
    n_series = N + (M_SIZE - 1) * TAU
    x_base = generate(process, n_series, rng)
    rows = []
    for variant in VARIANTS:
        x = x_base if variant == "as_generated" else np.exp(x_base)
        rule = prep.theiler_rule(x, m=M_SIZE, tau=TAU, n_states=N)
        w = int(rule["w"])  # w of the original (variant) series; surrogates inherit it
        end_gap = float(abs(x[-1] - x[0]))
        # observed
        obs = {}
        for transform in TRANSFORMS:
            if variant == "exp" and transform == "rank":
                continue  # deduplicated: copied from as_generated+rank at assembly
            D = prep.distance(x, transform=transform, m=M_SIZE, tau=TAU)
            r = rho_perc(D, w, delta=DELTA, connectivity=8)
            obs[transform] = r
        # surrogates: same transform/m/tau/delta and w of the original (Art. 4)
        surr = {t: [] for t in obs}
        for _ in range(n_surr):
            xs = iaaft(x, rng)
            for transform in obs:
                D = prep.distance(xs, transform=transform, m=M_SIZE, tau=TAU)
                surr[transform].append(float(rho_perc(D, w, delta=DELTA, connectivity=8)["rho"]))
        for transform, r in obs.items():
            stats = prep.surrogate_stats(float(r["rho"]), surr[transform])
            pct = stats["pct"]
            # Art. 4 test, exactly via pct: observed is smallest of the
            # n_surr+1 values iff no surrogate <= it (pct==0); largest iff
            # every surrogate <= it (pct==1).
            reject = bool(pct == 0.0 or pct == 1.0)
            rows.append({
                "process": process, "variant": variant, "N": int(N), "m": M_SIZE,
                "transform": transform, "rep": int(rep),
                "seed": f"[{PHASE},{cell},{rep}]", "seed_cell": int(cell),
                "n_series": int(n_series), "theiler_w": w,
                "tau_e": (None if rule["tau_e"] is None else int(rule["tau_e"])),
                "too_persistent": bool(rule["too_persistent"]),
                "eps_perc": float(r["eps"]), "rho_perc": float(r["rho"]),
                "rho_iso": float(r["rho_iso"]), "rho_lag": float(r["rho_lag"]),
                "mean_surrogate": stats["mean_surrogate"], "d_lin": stats["d"],
                "pct_lin": pct, "n_surrogates": int(n_surr),
                "reject": reject, "end_gap": end_gap, "deduped": False,
            })
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def _add_deduped(df):
    """exp(.)+rank rows reuse as_generated+rank (rank is unchanged by exp)."""
    src = df[(df["variant"] == "as_generated") & (df["transform"] == "rank")].copy()
    src["variant"] = "exp"
    src["deduped"] = True
    return pd.concat([df, src], ignore_index=True)


def _iid_summary(df):
    rows = []
    for (marginal, N, m, transform), g in df.groupby(["marginal", "N", "m", "transform"], sort=False):
        q1, med, q3 = g["rho_perc"].quantile([0.25, 0.5, 0.75])
        rows.append({
            "marginal": marginal, "N": int(N), "m": int(m), "transform": transform,
            "n_runs": int(len(g)), "median": float(med), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "sd": float(g["rho_perc"].std(ddof=1)) if len(g) > 1 else 0.0,
            "spread_definition": "spread = IQR (q3-q1); sd also given",
        })
    out = pd.DataFrame(rows)
    # range of medians across marginals per (N, m, transform)
    rng_map = out.groupby(["N", "m", "transform"])["median"].agg(lambda s: float(s.max() - s.min()))
    out["range_of_medians_across_marginals"] = [float(rng_map.loc[(r.N, r.m, r.transform)]) for r in out.itertuples()]
    # power-law fit of spread (IQR) vs N for uniform; large-N uniform value
    out["uniform_powerlaw_slope"] = np.nan
    out["uniform_powerlaw_intercept"] = np.nan
    out["uniform_largeN_median_at_1600"] = np.nan
    uni = out[out["marginal"] == "uniform"]
    for (m, transform), g in uni.groupby(["m", "transform"]):
        if g["N"].nunique() >= 2 and (g["iqr"] > 0).all():
            slope, intercept = np.polyfit(np.log(g["N"].astype(float)), np.log(g["iqr"]), 1)
        else:
            slope, intercept = np.nan, np.nan
        large = g.loc[g["N"] == g["N"].max(), "median"]
        large_val = float(large.iloc[0]) if len(large) else np.nan
        idx = g.index
        out.loc[idx, "uniform_powerlaw_slope"] = slope
        out.loc[idx, "uniform_powerlaw_intercept"] = intercept
        out.loc[idx, "uniform_largeN_median_at_1600"] = large_val
    order_m = {v: i for i, v in enumerate(MARGINALS)}
    out["_o"] = out["marginal"].map(order_m)
    out["_t"] = out["transform"].map({t: i for i, t in enumerate(TRANSFORMS)})
    return out.sort_values(["_o", "N", "m", "_t"]).drop(columns=["_o", "_t"]).reset_index(drop=True)


def _size_summary(df):
    rows = []
    for (process, variant, N, transform), g in df.groupby(["process", "variant", "N", "transform"], sort=False):
        rows.append({
            "process": process, "variant": variant, "N": int(N), "transform": transform,
            "n_runs": int(len(g)), "n_reject": int(g["reject"].sum()),
            "rejection_rate": float(g["reject"].mean()),
            "deduped": bool(g["deduped"].iloc[0]),
        })
    out = pd.DataFrame(rows)
    out["_p"] = out["process"].map({p: i for i, p in enumerate(PROCESSES)})
    out["_v"] = out["variant"].map({v: i for i, v in enumerate(VARIANTS)})
    out["_t"] = out["transform"].map({t: i for i, t in enumerate(TRANSFORMS)})
    return out.sort_values(["_p", "_v", "N", "_t"]).drop(columns=["_p", "_v", "_t"]).reset_index(drop=True)


def _endgap_summary(size_df, failing):
    rows = []
    for _, f in failing.iterrows():
        g = size_df[(size_df["process"] == f["process"]) & (size_df["variant"] == f["variant"])
                    & (size_df["N"] == f["N"]) & (size_df["transform"] == f["transform"])]
        med = g["end_gap"].median()
        sub = g[g["end_gap"] < med]  # strictly below the cell median
        rows.append({
            "process": f["process"], "variant": f["variant"], "N": int(f["N"]),
            "transform": f["transform"], "end_gap_definition": "|x_last - x_first| of the variant series",
            "end_gap_median": float(med), "n_runs_full": int(len(g)),
            "rejection_rate_full": float(g["reject"].mean()),
            "n_runs_below_median_end_gap": int(len(sub)),
            "n_reject_below_median_end_gap": int(sub["reject"].sum()) if len(sub) else 0,
            "rejection_rate_below_median_end_gap": float(sub["reject"].mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def write_decision(size_summary, out_dir):
    """Gate G2, computed by code. Counting rule (pre-registered in spec):
    ALL P2b cells count (as-generated and exp, both transforms, deduped incl.)."""
    from condpred.report import write_decision as _wd

    failing = size_summary[size_summary["rejection_rate"] > 0.09]
    n_cells = int(len(size_summary))
    n_fail = int(len(failing))
    fail_frac = n_fail / n_cells if n_cells else 0.0
    if n_fail == 0:
        result, passed = "PASS", True
    elif fail_frac <= 1.0 / 3.0:
        result, passed = "CONTINUE_LIMITED", False
    else:
        result, passed = "STOP", False
    fail_list = ";".join(
        f"{r.process}/{r.variant}/N{int(r.N)}/{r.transform}={r.rejection_rate:.3f}"
        for r in failing.itertuples()) if n_fail else ""
    _wd([{
        "gate": "G2", "result": result, "pass": bool(passed), "rule": GATE_RULE,
        "counting_rule": "all P2b cells in size_summary.csv counted (as_generated and exp variants, both transforms, including deduplicated exp+rank cells)",
        "n_cells": n_cells, "n_failing_cells": n_fail,
        "failing_fraction": float(fail_frac),
        "max_rejection_rate": float(size_summary["rejection_rate"].max()) if n_cells else np.nan,
        "failing_cells": fail_list,
    }], out_dir)
    return result, failing


def _fig_iid(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from condpred.report import save_fig

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ti, transform in enumerate(TRANSFORMS):
        for mi, m in enumerate(MS_IID):
            ax = axes[ti][mi]
            sub = summary[summary["transform"] == transform]
            sub = sub[sub["m"] == m] if not SMOKE or True else sub
            for marginal in MARGINALS:
                g = sub[sub["marginal"] == marginal].sort_values("N")
                if g.empty:
                    continue
                ax.plot(g["N"], g["median"], marker="o", label=marginal)
                ax.fill_between(g["N"], g["q1"], g["q3"], alpha=0.15)
            ax.set_xscale("log")
            ax.set_xticks(NS_IID if not SMOKE else sorted(sub["N"].unique()))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_title(f"transform={transform}, m={m}")
            ax.set_xlabel("N (states)")
            ax.set_ylabel("rho_perc median (band = IQR)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("P2a: rho_perc on iid data by marginal")
    fig.tight_layout()
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "figures", "iid_rho_by_marginal")
    plt.close(fig)


def _fig_size(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from condpred.report import save_fig

    s = summary.copy()
    s["cell"] = [f"{r.process}|{r.variant}|N{int(r.N)}|{r.transform}" for r in s.itertuples()]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(s)), s["rejection_rate"])
    ax.axhline(0.05, color="k", ls="--", lw=1, label="nominal 0.05")
    ax.axhline(0.09, color="r", ls="--", lw=1, label="gate limit 0.09")
    ax.set_xticks(np.arange(len(s)))
    ax.set_xticklabels(s["cell"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("rejection rate")
    ax.set_title("P2b: IAAFT test rejection rate per cell (39 surrogates)")
    ax.legend()
    fig.tight_layout()
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "figures", "size_rejection_by_cell")
    plt.close(fig)


def main():
    from condpred.report import save_table, write_spec

    expected, actual = _frozen_hash()
    if actual != expected:
        stop = ROOT / "plans/STOP_P2.md"
        stop.write_text(f"# STOP P2\n\nFrozen core hash mismatch: FROZEN.json={expected}, "
                        f"current src/rhoperc/core.py={actual}. P2 not run.\n", encoding="utf-8")
        print(f"STOP_P2 frozen hash mismatch expected={expected} actual={actual}", flush=True)
        return 2

    if SMOKE:
        ns_iid, ns_size, reps_iid = [100], [100], lambda n: 2  # noqa: E731
        n_surr = 3
    else:
        ns_iid, ns_size, reps_iid = NS_IID, NS_SIZE, _reps
        n_surr = N_SURR_FULL

    out_dir = ROOT / "tables/P2" / ("smoke" if SMOKE else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "version": VERSION, "phase": "P2", "smoke": SMOKE,
        "P2a_iid": {
            "marginals": MARGINALS,
            "marginal_generation": "generated directly per rep: normal=rng.standard_normal; uniform=rng.random; student_t3=rng.standard_t(3); lognormal=rng.lognormal(0,1); two_peaked=direct equal mixture: rng.standard_normal + (-2 if rng.random<0.5 else +2) (not a panel process)",
            "N_states": ns_iid, "m": MS_IID, "transforms": TRANSFORMS, "tau": TAU,
            "runs": "200 for N<=400, 100 at N=800, 50 at N=1600 (smoke: 2 reps, N=100 only)",
            "n_series_convention": "n = series length giving N states after embedding (Article 4 process-panel wording, following rhoperc.prep.measure / P1): n_series = N + (m-1)*tau; embedding then yields exactly N states",
            "theiler": "w from rhoperc.prep.theiler_rule on the generated series (n_states=N), computed once per (marginal,N,m,rep) and used for both transforms, as in P1/prep.measure",
            "summary_definitions": "iid_summary: per (marginal,N,m,transform) median, q1, q3, spread=IQR=q3-q1, sd (ddof=1); range_of_medians_across_marginals per (N,m,transform); for uniform, power-law fit log(IQR)=intercept+slope*log(N) per (m,transform) (columns uniform_powerlaw_slope/intercept, fit over all N in grid); uniform_largeN_median_at_1600 = uniform median at the largest N in grid (1600 full run)",
        },
        "P2b_size": {
            "processes": PROCESSES, "variants": VARIANTS,
            "variant_definition": "as_generated: series from rhoperc.processes.generate; exp: np.exp of the same generated series (same base seed per (process,N,rep))",
            "N_states": ns_size, "m": M_SIZE, "tau": TAU, "transforms": TRANSFORMS,
            "runs": 200 if not SMOKE else 2, "n_surrogates": n_surr,
            "surrogates": "recplay.surrogates.iaaft(x, rng) with rng continued from the run seed; each surrogate gets the same transform, m, tau, delta and the w of its original variant series (Article 4); d_lin/pct_lin via rhoperc.prep.surrogate_stats (d_lin on rho_perc)",
            "test": "reject at 5% when observed rho_perc is the smallest or largest of the n_surrogates+1 values (Article 4, 39 surrogates full run); implemented exactly via pct_lin = share of surrogates <= observed: reject iff pct_lin==0.0 (smallest) or pct_lin==1.0 (largest)",
            "dedupe": "under rank, exp(.) changes nothing: exp+rank cells are NOT computed separately; their size.csv/size_summary.csv rows are copies of the as_generated+rank rows for the same (process,N,rep) with deduped=True (including that cell's w and end_gap). Computed cells: as_generated z+rank and exp z only.",
            "end_gap_definition": "end_gap = |x_last - x_first| of the variant series x (for exp, of exp(series)). End-gap check (only if a cell fails): per failing cell, runs with end_gap strictly below that cell's median end_gap; rejection rate recomputed on that subset; test unchanged; saved as size_endgap.csv",
        },
        "delta": DELTA, "connectivity": 8,
        "seeds": "np.random.default_rng([phase, cell, rep]), phase=2. P2a cell = index of (marginal,N,m) over marginals x N x m in listed order (0..49), same seed for both transforms of a rep. P2b cell = 50 + index of (process,N) over processes x N (50..55), seed generates the base series; IAAFT surrogates draw from the same rng sequentially (as_generated variant first, then exp). Workers receive only seeds/parameters (joblib args), never arrays.",
        "gate": "G2", "gate_rule": GATE_RULE,
        "gate_counting_rule": "ALL P2b cells in size_summary.csv are counted for G2: as_generated and exp(.) variants, both transforms, including the deduplicated exp+rank cells (24 cells full run). This rule is fixed before any run; the report additionally breaks failing cells down by as-generated vs exp.",
        "gate_outcomes": "rejection <=0.09 in every counted cell: PASS; up to 1/3 of counted cells failing: continue, list them as limits (CONTINUE_LIMITED); more than 1/3 failing: STOP (STOP_P2.md)",
        "core_py_sha256_frozen": expected, "core_py_sha256_at_spec": actual,
    }
    write_spec(spec, out_dir, modules=["rhoperc.core", "rhoperc.prep", "rhoperc.processes", "rhoperc.experiments.p2"])

    from joblib import Parallel, delayed

    t0 = time.time()
    iid_tasks = [(mg, N, m, rep) for mg in MARGINALS for N in ns_iid for m in MS_IID
                 for rep in range(reps_iid(N))]
    frames = Parallel(n_jobs=8)(delayed(_iid_task)(a) for a in iid_tasks)
    iid = pd.concat(frames, ignore_index=True)
    om = {v: i for i, v in enumerate(MARGINALS)}
    iid["_o"] = iid["marginal"].map(om)
    iid["_t"] = iid["transform"].map({t: i for i, t in enumerate(TRANSFORMS)})
    iid = iid.sort_values(["_o", "N", "m", "_t", "rep"]).drop(columns=["_o", "_t"]).reset_index(drop=True)
    save_table(iid, out_dir, "iid")
    iid_sum = _iid_summary(iid)
    save_table(iid_sum, out_dir, "iid_summary")
    print(f"P2a rows={len(iid)} summary={len(iid_sum)} elapsed={time.time()-t0:.1f}s smoke={SMOKE}", flush=True)

    t1 = time.time()
    size_tasks = [(p, N, rep, n_surr) for p in PROCESSES for N in ns_size for rep in range(200 if not SMOKE else 2)]
    frames = Parallel(n_jobs=8)(delayed(_size_task)(a) for a in size_tasks)
    size = _add_deduped(pd.concat(frames, ignore_index=True))
    op = {p: i for i, p in enumerate(PROCESSES)}
    size["_p"] = size["process"].map(op)
    size["_v"] = size["variant"].map({v: i for i, v in enumerate(VARIANTS)})
    size["_t"] = size["transform"].map({t: i for i, t in enumerate(TRANSFORMS)})
    size = size.sort_values(["_p", "_v", "N", "_t", "rep"]).drop(columns=["_p", "_v", "_t"]).reset_index(drop=True)
    save_table(size, out_dir, "size")
    size_sum = _size_summary(size)
    save_table(size_sum, out_dir, "size_summary")
    print(f"P2b rows={len(size)} summary={len(size_sum)} elapsed={time.time()-t1:.1f}s smoke={SMOKE}", flush=True)

    if not SMOKE:
        _fig_iid(iid_sum, out_dir)
        _fig_size(size_sum, out_dir)
        result, failing = write_decision(size_sum, out_dir)
        print(f"G2 result={result} failing_cells={len(failing)}", flush=True)
        if len(failing):
            eg = _endgap_summary(size, failing)
            save_table(eg, out_dir, "size_endgap")
            print(f"size_endgap rows={len(eg)}", flush=True)
        if result == "STOP":
            stop = ROOT / "plans/STOP_P2.md"
            fl = "; ".join(f"{r.process}/{r.variant}/N{int(r.N)}/{r.transform}={r.rejection_rate:.3f}"
                           for r in failing.itertuples())
            stop.write_text(
                f"# STOP P2\n\nGate G2: {len(failing)} of {len(size_sum)} counted cells reject in "
                f"more than 9% of runs (more than one third of cells fail; see "
                f"tables/P2/decision.csv and size_summary.csv). Failing: {fl}.\n",
                encoding="utf-8")
            print("STOP_P2 written", flush=True)
        # selfcheck once (frozen state confirmation, as P1)
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
