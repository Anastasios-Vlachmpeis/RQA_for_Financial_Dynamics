"""P4: scaling and sensitivity (00_CONSTITUTION.md P4).

Run from RecPlay/ with PYTHONPATH=src (Python312).
Smoke: RHOPERC_SMOKE=1 -> 2 reps, 3 surrogates, N capped at 100, smoke/ folders.
Kept transform only (Decision T after P3: z).
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
PHASE = 4
VERSION = 3
PROCESSES = ["S1", "S3", "S4", "S5", "D1", "D3", "F1", "F2"]
FLOWS = ["F1", "F2"]
TAU = {"F1": 7, "F2": 4}  # frozen in P0 spec; all other processes tau=1
BASE_STEP = {"F1": 0.2, "F2": 0.05}  # condpred defaults: rossler dt .05 x4, lorenz dt .01 x5
N_GRID = [100, 200, 400, 800, 1600, 3200]
DELTA_GRID = [0.02, 0.05, 0.10, 0.20]
THEILER_LEVELS = ["1", "rule", "2x_rule"]
M_GRID = [1, 2, 3, 5]
CONN_GRID = [4, 8]
SAMP_FACTORS = [0.5, 1.0, 2.0, 4.0]
DLIN_N = [100, 200, 400, 800]
SETTING_ORDER = {"N": 0, "delta": 1, "theiler": 2, "m": 3, "neighbourhood": 4, "sampling": 5}

from rhoperc import prep  # noqa: E402
from rhoperc.core import rho_perc  # noqa: E402
from rhoperc.processes import generate  # noqa: E402


def _frozen_hash():
    frozen = json.loads((ROOT / "plans/rho_perc/FROZEN.json").read_text(encoding="utf-8"))
    expected = frozen["core_py_sha256"]
    actual = hashlib.sha256((ROOT / "src/rhoperc/core.py").read_bytes()).hexdigest()
    return expected, actual


def _cache_dir():
    d = ROOT / "data/processed/rho_perc/P4" / (f"smoke_v{VERSION}" if SMOKE else f"v{VERSION}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reps_N(n):
    if SMOKE:
        return 2
    return 50 if n <= 800 else (20 if n == 1600 else 10)


def _build_configs():
    """Unique compute configs; cell = cfg_id. Baseline cfg is shared (deduped)
    across every setting dimension it is the level of."""
    cfgs = []

    def add(**kw):
        kw["cfg_id"] = len(cfgs)
        cfgs.append(kw)
        return kw["cfg_id"]

    n_grid = [100] if SMOKE else N_GRID
    for pid in PROCESSES:
        tau = TAU.get(pid, 1)
        base = dict(process=pid, tau=tau, m=3, delta=0.05, connectivity=8,
                    theiler_mode="rule", sampling_factor=1.0, sampling_mode="none")
        baseline_id = None
        for n in n_grid:
            reps = _reps_N(n)
            cid = add(**base, N=n, n_series=n + 2 * tau, reps=reps,
                      do_dlin=(n in ([100] if SMOKE else DLIN_N)))
            if n == (100 if SMOKE else 400):
                baseline_id = cid
        if SMOKE:  # baseline for the other dimensions is the N=100 cfg above
            pass
        else:
            assert baseline_id is not None
        reps50 = 2 if SMOKE else 50
        n_base = 100 if SMOKE else 400
        for d in DELTA_GRID:
            if d != 0.05:
                add(**{**base, "delta": d}, N=n_base, n_series=n_base + 2 * tau,
                    reps=reps50, do_dlin=False)
        for mode in ["one", "twice"]:
            add(**{**base, "theiler_mode": mode}, N=n_base, n_series=n_base + 2 * tau,
                reps=reps50, do_dlin=False)
        for m in M_GRID:
            if m != 3:
                add(**{**base, "m": m}, N=n_base, n_series=n_base + (m - 1) * tau,
                    reps=reps50, do_dlin=False)
        add(**{**base, "connectivity": 4}, N=n_base, n_series=n_base + 2 * tau,
            reps=reps50, do_dlin=False)
        if pid in FLOWS:
            reps_s = 2 if SMOKE else 30
            base_n_series = n_base + 2 * tau
            for smode in ["fixed_N", "fixed_span"]:
                for f in ([0.5, 2.0] if SMOKE else SAMP_FACTORS):
                    if f == 1.0:
                        continue  # identical to baseline cfg (deduped)
                    if smode == "fixed_N":
                        ns, nst = base_n_series, n_base
                    else:
                        ns = int(round((base_n_series - 1) / f)) + 1
                        nst = ns - 2 * tau
                    add(**{**base, "sampling_factor": f, "sampling_mode": smode},
                        N=nst, n_series=ns, reps=reps_s, do_dlin=False)
    return cfgs


CONFIGS = _build_configs()
BASELINE_ID = {}
for _c in CONFIGS:
    if (_c["N"] == (100 if SMOKE else 400) and _c["m"] == 3 and _c["delta"] == 0.05
            and _c["connectivity"] == 8 and _c["theiler_mode"] == "rule"
            and _c["sampling_mode"] == "none"):
        BASELINE_ID[_c["process"]] = _c["cfg_id"]


def _dims(cfg):
    """(setting, level_label, level_sort) pairs this config provides rows for."""
    pid = cfg["process"]
    if cfg["cfg_id"] == BASELINE_ID[pid]:
        dims = [("N", str(cfg["N"]), float(cfg["N"])), ("delta", "0.05", 0.05),
                ("theiler", "rule", 1.0), ("m", "3", 3.0), ("neighbourhood", "8", 8.0)]
        if pid in FLOWS:
            dims += [("sampling", "fixed_N:x1", 1.0), ("sampling", "fixed_span:x1", 11.0)]
        return dims
    if cfg["sampling_mode"] != "none":
        lab = f"{cfg['sampling_mode']}:x{cfg['sampling_factor']:g}"
        srt = (0.0 if cfg["sampling_mode"] == "fixed_N" else 10.0) + cfg["sampling_factor"]
        return [("sampling", lab, srt)]
    if cfg["delta"] != 0.05:
        return [("delta", f"{cfg['delta']:g}", float(cfg["delta"]))]
    if cfg["theiler_mode"] != "rule":
        lab = "1" if cfg["theiler_mode"] == "one" else "2x_rule"
        return [("theiler", lab, 0.0 if lab == "1" else 2.0)]
    if cfg["m"] != 3:
        return [("m", str(cfg["m"]), float(cfg["m"]))]
    if cfg["connectivity"] != 8:
        return [("neighbourhood", "4", 4.0)]
    return [("N", str(cfg["N"]), float(cfg["N"]))]


def _gen_series(cfg, rng):
    pid, ns, f = cfg["process"], int(cfg["n_series"]), float(cfg["sampling_factor"])
    if pid in FLOWS and cfg["sampling_mode"] != "none":
        from condpred import systems
        # Step varied via sample_every at fixed integrator dt (dt*f destabilises
        # rossler at f=4). F2 x0.5 uses dt=0.005 so sample_every stays integer.
        if pid == "F1":
            return np.asarray(systems.rossler(ns, rng, dt=0.05, sample_every=int(4 * f)), dtype=float)[:, 0]
        if f == 0.5:
            return np.asarray(systems.lorenz63(ns, rng, dt=0.005, sample_every=5)[1], dtype=float)
        return np.asarray(systems.lorenz63(ns, rng, dt=0.01, sample_every=int(5 * f))[1], dtype=float)
    return generate(pid, ns, rng)


def _run_cell(args):
    cfg_id, rep = args
    cfg = CONFIGS[cfg_id]
    cache = _cache_dir() / f"cfg{cfg_id:03d}_rep{rep}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    t0 = time.time()
    rng = np.random.default_rng([PHASE, int(cfg_id), int(rep)])
    x = _gen_series(cfg, rng)
    m, tau, n = int(cfg["m"]), int(cfg["tau"]), int(cfg["N"])
    rule = prep.theiler_rule(x, m=m, tau=tau, n_states=n)
    w_rule = int(rule["w"])
    w = {"one": 1, "rule": w_rule, "twice": 2 * w_rule}[cfg["theiler_mode"]]
    D = prep.distance(x, transform="z", m=m, tau=tau)
    r = rho_perc(D, w, delta=float(cfg["delta"]), connectivity=int(cfg["connectivity"]))
    row = {
        "cfg_id": int(cfg_id), "process": cfg["process"], "rep": int(rep),
        "seed": f"[{PHASE},{int(cfg_id)},{int(rep)}]", "seed_cell": int(cfg_id),
        "N": int(len(D)), "n_series": int(cfg["n_series"]), "m": m, "tau": tau,
        "delta": float(cfg["delta"]), "connectivity": int(cfg["connectivity"]),
        "theiler_mode": cfg["theiler_mode"], "theiler_w": int(w), "theiler_w_rule": w_rule,
        "sampling_factor": float(cfg["sampling_factor"]), "sampling_mode": cfg["sampling_mode"],
        "sampling_step": (BASE_STEP[cfg["process"]] * float(cfg["sampling_factor"])
                          if cfg["process"] in FLOWS else np.nan),
        "transform": "z",
        "tau_e": (None if rule["tau_e"] is None else int(rule["tau_e"])),
        "too_persistent": bool(rule["too_persistent"]),
        "eps_perc": float(r["eps"]), "rho_perc": float(r["rho"]),
        "eps_iso": float(r["eps_iso"]), "rho_iso": float(r["rho_iso"]),
        "eps_lag": float(r["eps_lag"]), "rho_lag": float(r["rho_lag"]),
        "d_lin": np.nan, "pct_lin": np.nan, "mean_surrogate_rho_perc": np.nan,
        "n_surrogates": 0,
    }
    n_surr = 3 if SMOKE else 19
    dlin_reps = 2 if SMOKE else 30
    if cfg["do_dlin"] and rep < dlin_reps:
        from recplay.surrogates import iaaft
        vals = []
        for _ in range(n_surr):
            xs = iaaft(x, rng)
            Ds = prep.distance(xs, transform="z", m=m, tau=tau)
            vals.append(float(rho_perc(Ds, w, delta=float(cfg["delta"]),
                                       connectivity=int(cfg["connectivity"]))["rho"]))
        st = prep.surrogate_stats(float(r["rho"]), vals)
        row.update({"d_lin": st["d"], "pct_lin": st["pct"],
                    "mean_surrogate_rho_perc": st["mean_surrogate"],
                    "n_surrogates": int(n_surr)})
    row["elapsed_s"] = float(time.time() - t0)
    df = pd.DataFrame([row])
    df.to_csv(cache, index=False)
    return df


def _expand(config_rows):
    rows = []
    by_cfg = {cid: g for cid, g in config_rows.groupby("cfg_id")}
    for cfg in CONFIGS:
        g = by_cfg.get(cfg["cfg_id"])
        if g is None:
            continue
        for setting, level, srt in _dims(cfg):
            sub = g.copy()
            if setting == "sampling":
                sub = sub[sub["rep"] < (2 if SMOKE else 30)]
                sub["sampling_mode"] = level.split(":")[0]
            sub["setting"] = setting
            sub["level"] = level
            sub["_lsort"] = srt
            rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    df["_p"] = df["process"].map({p: i for i, p in enumerate(PROCESSES)})
    df["_s"] = df["setting"].map(SETTING_ORDER)
    df = df.sort_values(["_p", "_s", "_lsort", "rep"]).drop(columns=["_p", "_s", "_lsort"])
    return df.reset_index(drop=True)


def _summary(df):
    keys = ["process", "setting", "level", "N", "m", "delta", "connectivity",
            "theiler_mode", "sampling_factor", "sampling_mode"]
    rows = []
    for k, g in df.groupby(keys, sort=False):
        dl = g["d_lin"].dropna()
        rows.append({**dict(zip(keys, k)), "n_runs": int(len(g)),
                     "rho_perc_median": float(g["rho_perc"].median()),
                     "rho_perc_q1": float(g["rho_perc"].quantile(0.25)),
                     "rho_perc_q3": float(g["rho_perc"].quantile(0.75)),
                     "n_dlin": int(len(dl)),
                     "d_lin_median": float(dl.median()) if len(dl) else np.nan,
                     "d_lin_q1": float(dl.quantile(0.25)) if len(dl) else np.nan,
                     "d_lin_q3": float(dl.quantile(0.75)) if len(dl) else np.nan})
    out = pd.DataFrame(rows)
    out["_p"] = out["process"].map({p: i for i, p in enumerate(PROCESSES)})
    out["_s"] = out["setting"].map(SETTING_ORDER)
    return out.sort_values(["_p", "_s", "level"]).drop(columns=["_p", "_s"]).reset_index(drop=True)


def _stability(summary):
    rows = []
    for stat, col in [("rho_perc", "rho_perc_median"), ("d_lin", "d_lin_median")]:
        grid = N_GRID if stat == "rho_perc" else DLIN_N
        for pid in PROCESSES:
            g = summary[(summary["process"] == pid) & (summary["setting"] == "N")]
            med = {int(r.level): float(getattr(r, col)) for r in g.itertuples() if np.isfinite(getattr(r, col))}
            changes, stable = [], None
            for n in grid:
                if n in med and 2 * n in med:
                    ch = abs(med[2 * n] - med[n])
                    changes.append(f"{n}->{2*n}: {ch:.4f}")
                    if stable is None and ch < 0.02:
                        stable = (n, ch)
            rows.append({
                "process": pid, "statistic": stat,
                "smallest_N_from_which_doubling_changes_median_lt_0.02": (
                    int(stable[0]) if stable else "none"),
                "median_change_at_stable_step": (float(stable[1]) if stable else np.nan),
                "all_doubling_changes": "; ".join(changes),
                "note": ("" if stable else
                         f"no doubling within the tested grid ({grid[0]}..{grid[-1]}) "
                         f"changed the median by <0.02; stability N is >{grid[-1]}/none within grid"),
            })
    return pd.DataFrame(rows)


def _fig_setting(summary, setting, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from condpred.report import save_fig
    g = summary[summary["setting"] == setting]
    if setting == "sampling":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, mode in zip(axes, ["fixed_N", "fixed_span"]):
            for pid in FLOWS:
                s = g[(g["process"] == pid) & (g["sampling_mode"] == mode)].copy()
                s["f"] = s["level"].str.split(":x").str[1].astype(float)
                s = s.sort_values("f")
                ax.errorbar(s["f"], s["rho_perc_median"],
                            yerr=[s["rho_perc_median"] - s["rho_perc_q1"],
                                  s["rho_perc_q3"] - s["rho_perc_median"]],
                            marker="o", capsize=3, label=pid)
            ax.set_xscale("log", base=2); ax.set_xlabel("sampling step factor"); ax.set_title(mode)
        axes[0].set_ylabel("rho_perc median (Q1-Q3)")
    else:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for pid in PROCESSES:
            s = g[g["process"] == pid].copy()
            if setting == "N":
                s["xv"] = s["level"].astype(float); s = s.sort_values("xv"); ax.set_xscale("log", base=2)
            elif setting == "delta":
                s["xv"] = s["level"].astype(float); s = s.sort_values("xv")
            elif setting == "m":
                s["xv"] = s["level"].astype(float); s = s.sort_values("xv")
            elif setting == "neighbourhood":
                s["xv"] = s["level"].astype(float); s = s.sort_values("xv")
            else:  # theiler categorical
                order = {"1": 0, "rule": 1, "2x_rule": 2}
                s["xv"] = s["level"].map(order); s = s.sort_values("xv")
                ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["1", "rule", "2x_rule"])
            ax.errorbar(s["xv"], s["rho_perc_median"],
                        yerr=[s["rho_perc_median"] - s["rho_perc_q1"],
                              s["rho_perc_q3"] - s["rho_perc_median"]],
                        marker="o", capsize=3, label=pid)
        ax.set_xlabel(setting); ax.set_ylabel("rho_perc median (Q1-Q3)")
    fig.suptitle(f"P4 sensitivity: {setting} (z, other settings at Article 4 baseline)")
    axes = fig.axes
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / "figures", f"sensitivity_{setting}")
    plt.close(fig)


def main():
    from condpred.report import save_table, write_spec

    expected, actual = _frozen_hash()
    if actual != expected:
        stop = ROOT / "plans/rho_perc/STOP_P4.md"
        stop.write_text(f"# STOP P4\n\nFrozen core hash mismatch: FROZEN.json={expected}, "
                        f"current src/rhoperc/core.py={actual}. P4 not run.\n", encoding="utf-8")
        print(f"STOP_P4 frozen hash mismatch expected={expected} actual={actual}", flush=True)
        return 2

    out_dir = ROOT / "notebooks/tables/rho_perc/P4" / ("smoke" if SMOKE else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": VERSION, "phase": "P4", "smoke": SMOKE,
        "gate": "none", "gate_rule": "P4 has no gate (Article 6).",
        "transform": "z only (Decision T after P3: keep z, notebooks/tables/rho_perc/P3/decision.csv)",
        "processes": PROCESSES,
        "baseline_Article4": {"N_states": 400, "m": 3, "delta": 0.05, "connectivity": 8,
                              "tau_by_process": {p: TAU.get(p, 1) for p in PROCESSES},
                              "theiler": "w = rhoperc.prep.theiler_rule(series, m, tau, n_states=N) per run"},
        "settings_varied_one_at_a_time": {
            "N": {"levels": N_GRID, "runs": "50 up to 800; 20 at 1600; 10 at 3200"},
            "delta": {"levels": DELTA_GRID, "runs": 50},
            "theiler": {"levels": ["1", "rule", "2x_rule"], "runs": 50,
                        "preregistered": "2x_rule means w = 2 * w_rule where w_rule is the per-run theiler_rule value"},
            "m": {"levels": M_GRID, "runs": 50,
                  "preregistered": "tau frozen per process at the Article 4 baseline value (F1=7, F2=4, else 1); w = theiler_rule recomputed with that m (and that tau) per run; n_series = 400 + (m-1)*tau so N=400 states for every m"},
            "neighbourhood": {"levels": CONN_GRID, "runs": 50},
            "sampling": {"processes": FLOWS, "factors": SAMP_FACTORS, "runs": 30,
                         "preregistered": "base steps from condpred.systems defaults: F1 step 0.2 (rossler dt=0.05, sample_every=4), F2 step 0.05 (lorenz63 dt=0.01, sample_every=5); factor f multiplies the step via sample_every at fixed dt (F1 every=int(4f); F2 every=int(5f), except F2 x0.5 which uses dt=0.005, every=5, since 5*0.5 is not an integer). fixed_N: N=400 states at every f. fixed_span: total time span fixed at the baseline (x1, N=400) span, n_series = round((n_series_base-1)/f)+1, so N scales inversely with f. tau and theiler_rule as baseline. Smoke uses factors [0.5, 2.0] only (fixed_span x4 at the smoke base of N=100 would leave <20 states)."},
        },
        "d_lin": {"preregistered": "same 8 processes, baseline other settings, z; 19 IAAFT surrogates (3 in smoke), 30 runs, at N in DLIN_N; surrogate gets same transform, m, tau, delta and w of its original (Article 4); d_lin = rho_perc - mean(surrogate rho_perc). Computed inside the N-grid cells (rep<30) so the observed value is not recomputed.",
                  "N_levels": DLIN_N, "n_surrogates_full": 19, "runs": 30},
        "dedupe": "Caches are keyed by the full settings (one file per unique cfg_id x rep). The Article 4 baseline config is a single cfg per process and its rows are repeated in sensitivity.csv under each setting dimension it is the baseline level of (N=400, delta=0.05, theiler=rule, m=3, neighbourhood=8, and for F1/F2 sampling x1 in both modes, the latter truncated to 30 reps). No other recompute is shared across levels: each non-baseline level is its own cfg with its own seeds.",
        "configs": [{k: c[k] for k in ["cfg_id", "process", "N", "n_series", "m", "tau", "delta",
                                        "connectivity", "theiler_mode", "sampling_factor",
                                        "sampling_mode", "reps", "do_dlin"]} for c in CONFIGS],
        "seeds": "np.random.default_rng([phase, cell, rep]), phase=4, cell=cfg_id (index of the unique full-settings config in spec.configs), rep as listed. Surrogate draws continue from the run rng. Workers receive only (cfg_id, rep) ints (joblib), never arrays.",
        "caches": "data/processed/rho_perc/P4/v<version>/ (smoke/ under smoke), one file per (cfg, rep), rerun skips existing",
        "outputs": ["sensitivity.csv", "sensitivity_summary.csv (median/Q1/Q3 of rho_perc and d_lin per process x setting x level)",
                    "stability.csv (acceptance (1): per process smallest N from which doubling N changes the median by <0.02, for rho_perc and for d_lin, from sequential doublings of the N-grid medians)",
                    "figures/sensitivity_{N,delta,theiler,m,neighbourhood,sampling}.png (one figure per setting)"],
        "acceptance": "(1) stability.csv as above; (2) same for d_lin (in stability.csv); (3) report contains one sentence per setting on whether values at different levels are comparable. No gate, no decision.csv.",
        "smoke": "RHOPERC_SMOKE=1: 2 reps, 3 surrogates, N capped at 100, writes to smoke/ folders; smoke numbers never reported",
        "core_py_sha256_frozen": expected, "core_py_sha256_at_spec": actual,
    }
    write_spec(spec, out_dir, modules=["rhoperc.core", "rhoperc.prep", "rhoperc.processes",
                                       "rhoperc.experiments.p4"])

    from joblib import Parallel, delayed
    tasks = [(c["cfg_id"], rep) for c in CONFIGS for rep in range(
        (2 if SMOKE else c["reps"]))]
    t0 = time.time()
    frames = Parallel(n_jobs=8)(delayed(_run_cell)(a) for a in tasks)
    elapsed = time.time() - t0
    config_rows = pd.concat(frames, ignore_index=True)
    print(f"config rows={len(config_rows)} tasks={len(tasks)} elapsed={elapsed:.1f}s smoke={SMOKE}", flush=True)

    df = _expand(config_rows)
    save_table(df, out_dir, "sensitivity")
    summ = _summary(df)
    save_table(summ, out_dir, "sensitivity_summary")
    stab = _stability(summ)
    save_table(stab, out_dir, "stability")
    print(f"sensitivity rows={len(df)} summary={len(summ)} stability={len(stab)}", flush=True)

    if not SMOKE:
        for setting in ["N", "delta", "theiler", "m", "neighbourhood", "sampling"]:
            _fig_setting(summ, setting, out_dir)
        r = subprocess.run([sys.executable, "-m", "rhoperc.selfcheck"], cwd=ROOT,
                           env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
        print(r.stdout, flush=True)
        if r.stderr:
            print(r.stderr, flush=True)
        print(f"selfcheck exit={r.returncode}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
