"""P3: known transitions (logistic), breaks/regimes, and IAAFT power.

Spec (incl. pre-registered gate/Decision-T evaluation choices) is written
before any outcome is computed. See tables/P3/spec.json.
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
PHASE = 3
VERSION = 1
DELTA = 0.05
TRANSFORMS = ["z", "rank"]
MEASURES_A = ["rho_perc", "rho_iso", "rho_lag", "DET", "LAM", "L"]

GATE_RULE = "At least one of: (a) logistic periodic-vs-chaotic AUC \u2265 0.90; (b) permanent-break Spearman \u2265 0.5; (c) power \u2265 0.5 in at least 3 nonlinear cells."
DECISION_T_RULE = "Keep `rank` if its P3a AUC and its mean P3d power are each no more than 0.03 below those of `z`; otherwise keep `z`."

# P3a
R_FULL = np.round(np.arange(3.5, 4.0 + 1e-9, 0.001), 3)
N_LOGISTIC = 1000
MS_A = [1, 3]
TRANSITION_CENTRES = [(3.6786, "band_merging_3.6786"), (3.8284, "period3_window_3.8284"),
                      (3.8568, "interior_crisis_3.8568")]
LYAP_ITERS = 100_000
LYAP_TRANSIENT = 1000
LYAP_X0 = 0.5

# P3b
N_BREAKS = 400
PERM_SIZES = [0.0, 0.5, 1.0, 2.0, 4.0]   # in units of the S3 process sd (=1)
RECUR_GAPS = [0.0, 1.0, 2.0, 4.0]        # in units of the AR(1)0.8 process sd (=1)
N_SURR_BREAKS = 19

# P3c
N_POWER = 400
POWER_COMBOS = ([(p, nz) for p in ["D1", "D3", "F1"] for nz in [0.0, 0.25, 0.5, 1.0]]
                + [("S5", 0.0), ("S6", 0.0)])
N_SURR_POWER = 39
POWER_STATS = ["rho_perc", "DET", "LAM"]
TAU_BY_PROCESS = {"D1": 1, "D3": 1, "F1": 7, "S5": 1, "S6": 1, "S3": 1}


def _frozen_hash():
    frozen = json.loads((ROOT / "plans/FROZEN.json").read_text(encoding="utf-8"))
    expected = frozen["core_py_sha256"]
    actual = hashlib.sha256((ROOT / "src/rhoperc/core.py").read_bytes()).hexdigest()
    return expected, actual


def _cache_dir():
    d = ROOT / "cache/P3" / ("smoke" if SMOKE else f"v{VERSION}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _measures(D, w):
    """rho_perc (+bounds) and DET/LAM/L at RR=10% from one distance matrix."""
    from recplay.recurrence import threshold_by_rate
    from recplay.rqa import rqa
    from rhoperc.core import rho_perc

    r = rho_perc(D, w, delta=DELTA, connectivity=8)
    q = rqa(threshold_by_rate(D, 0.10, theiler=w))
    return {"rho_perc": float(r["rho"]), "eps_perc": float(r["eps"]),
            "rho_iso": float(r["rho_iso"]), "rho_lag": float(r["rho_lag"]),
            "DET": float(q["DET"]), "LAM": float(q["LAM"]), "L": float(q["L"])}


def _lyapunov(r):
    x = LYAP_X0
    n_iter = 10_000 if SMOKE else LYAP_ITERS
    total, count = 0.0, 0
    for i in range(LYAP_TRANSIENT + n_iter):
        d = abs(r * (1.0 - 2.0 * x))
        if i >= LYAP_TRANSIENT:
            total += np.log(d) if d > 0 else -np.inf
            count += 1
        x = r * x * (1.0 - x)
    return float(total / count)


def _window_name(r):
    for centre, name in TRANSITION_CENTRES:
        if abs(r - centre) <= 0.01 + 1e-12:
            return name
    return ""


# ---------------------------------------------------------------- P3a tasks
def _logistic_task(args):
    from condpred.systems import logistic
    from rhoperc import prep

    r_val, r_idx = args
    cache = _cache_dir() / f"logistic_r{r_idx:03d}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rng = np.random.default_rng([PHASE, int(r_idx), 0])
    lam = _lyapunov(float(r_val))
    rows = []
    for m in MS_A:
        n_series = N_LOGISTIC + (m - 1) * 1
        # same seed-derived series per m: regenerate with a fresh rng per m
        # so the underlying trajectory is identical for m=1 and m=3.
        rng_m = np.random.default_rng([PHASE, int(r_idx), 0])
        x = logistic(float(r_val), n_series if not SMOKE else min(n_series, 100 + (m - 1)),
                     rng_m)
        n_states = len(x) - (m - 1)
        rule = prep.theiler_rule(x, m=m, tau=1, n_states=n_states)
        w = int(rule["w"])
        for transform in TRANSFORMS:
            D = prep.distance(x, transform=transform, m=m, tau=1)
            vals = _measures(D, w)
            rows.append({
                "r": float(r_val), "r_index": int(r_idx), "N": int(n_states), "m": int(m),
                "transform": transform, "rep": 0, "seed": f"[{PHASE},{int(r_idx)},0]",
                "n_series": int(len(x)), "theiler_w": w,
                "tau_e": (None if rule["tau_e"] is None else int(rule["tau_e"])),
                "too_persistent": bool(rule["too_persistent"]),
                "lambda_lyapunov": float(lam),
                "lambda_class": ("periodic" if lam < 0 else ("chaotic" if lam > 0 else "zero")),
                "transition_window": _window_name(float(r_val)),
                **vals,
            })
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def _logistic_scores(df):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    rows = []
    for (m, transform), g in df.groupby(["m", "transform"], sort=False):
        gz = g[g["lambda_class"] == "zero"]
        gp = g[g["lambda_class"] == "periodic"]
        gc = g[g["lambda_class"] == "chaotic"]
        y = (g["lambda_class"] == "chaotic").to_numpy()
        usable = g["lambda_class"] != "zero"
        for meas in MEASURES_A:
            auc_raw = float(roc_auc_score(y[usable.to_numpy()], g.loc[usable, meas])) \
                if gp.shape[0] and gc.shape[0] else np.nan
            auc = max(auc_raw, 1.0 - auc_raw) if np.isfinite(auc_raw) else np.nan
            if len(gc) >= 3 and np.nanstd(gc[meas]) > 0:
                sp = spearmanr(gc["lambda_lyapunov"], gc[meas])
                rho_s, p_s = float(sp.statistic), float(sp.pvalue)
            else:
                rho_s, p_s = np.nan, np.nan
            rows.append({
                "m": int(m), "transform": transform, "measure": meas,
                "n_r": int(len(g)), "n_periodic": int(len(gp)), "n_chaotic": int(len(gc)),
                "n_lambda_zero_excluded": int(len(gz)),
                "auc_periodic_vs_chaotic": auc,
                "auc_raw_chaotic_as_positive_score_up": auc_raw,
                "auc_direction": ("measure_higher_when_chaotic" if auc_raw >= 0.5
                                  else "measure_lower_when_chaotic"),
                "spearman_lambda_chaotic_only": rho_s, "spearman_p": p_s,
            })
    out = pd.DataFrame(rows)
    out["_m"] = out["m"]; out["_t"] = out["transform"].map({t: i for i, t in enumerate(TRANSFORMS)})
    out["_meas"] = out["measure"].map({v: i for i, v in enumerate(MEASURES_A)})
    return out.sort_values(["_m", "_t", "_meas"]).drop(columns=["_m", "_t", "_meas"]).reset_index(drop=True)


# ---------------------------------------------------------------- P3b tasks
def _permanent_series(n, size_sd, rng):
    from condpred.systems import ar1
    x = ar1(0.9, n, rng)  # stationary (process) sd = 1, as X2 in rhoperc.processes
    x = x.copy()
    x[n // 2:] += size_sd * 1.0
    return x


def _recurring_series(n, gap_sd, rng, burn=500):
    # AR(1) 0.8 fluctuations (stationary sd 1) around a switching mean.
    phi = 0.8
    sigma = np.sqrt(1.0 - phi ** 2)
    total = n + burn
    level = 0.0
    y, out = 0.0, np.empty(total)
    for t in range(total):
        if t and rng.random() >= 0.98:
            level = gap_sd - level if gap_sd else 0.0  # switch 0 <-> gap
        y = phi * y + sigma * rng.standard_normal()
        out[t] = level + y
    return out[burn:]


def _breaks_task(args):
    from recplay.surrogates import iaaft
    from rhoperc import prep

    part, level, rep, n_surr = args
    part_idx = 0 if part == "permanent" else 1
    levels = PERM_SIZES if part == "permanent" else RECUR_GAPS
    cell = 1000 + part_idx * 100 + levels.index(level)
    cache = _cache_dir() / f"breaks_{part}_{level}_rep{rep}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rng = np.random.default_rng([PHASE, int(cell), int(rep)])
    n = 100 if SMOKE else N_BREAKS
    n_series = n + 2  # m=3, tau=1 -> N=n states
    if part == "permanent":
        x = _permanent_series(n_series, level, rng)
    else:
        x = _recurring_series(n_series, level, rng)
    rule = prep.theiler_rule(x, m=3, tau=1, n_states=n)
    w = int(rule["w"])
    obs, surr = {}, {t: [] for t in TRANSFORMS}
    for transform in TRANSFORMS:
        obs[transform] = _measures(prep.distance(x, transform=transform, m=3, tau=1), w)
    for _ in range(n_surr):
        xs = iaaft(x, rng)
        for transform in TRANSFORMS:
            D = prep.distance(xs, transform=transform, m=3, tau=1)
            from rhoperc.core import rho_perc
            surr[transform].append(float(rho_perc(D, w, delta=DELTA, connectivity=8)["rho"]))
    rows = []
    for transform in TRANSFORMS:
        st = prep.surrogate_stats(obs[transform]["rho_perc"], surr[transform])
        rows.append({
            "part": part, "size_sd": float(level), "N": int(n), "m": 3, "transform": transform,
            "rep": int(rep), "seed": f"[{PHASE},{int(cell)},{int(rep)}]", "seed_cell": int(cell),
            "n_series": int(n_series), "theiler_w": w,
            "tau_e": (None if rule["tau_e"] is None else int(rule["tau_e"])),
            "too_persistent": bool(rule["too_persistent"]),
            **obs[transform], "d_lin": st["d"], "pct_lin": st["pct"],
            "mean_surrogate_rho_perc": st["mean_surrogate"], "n_surrogates": int(n_surr),
        })
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def _breaks_summaries(df):
    from scipy.stats import spearmanr
    trend_rows, cell_rows = [], []
    for (part, transform), g in df.groupby(["part", "transform"], sort=False):
        for meas in ["rho_perc", "d_lin", "DET", "LAM"]:
            sp = spearmanr(g["size_sd"], g[meas])
            trend_rows.append({"part": part, "transform": transform, "measure": meas,
                               "spearman_size_vs_value": float(sp.statistic),
                               "spearman_p": float(sp.pvalue), "n_runs_pooled": int(len(g))})
        for level, gl in g.groupby("size_sd"):
            cell_rows.append({"part": part, "transform": transform, "size_sd": float(level),
                              "n_runs": int(len(gl)),
                              **{f"median_{c}": float(gl[c].median())
                                 for c in ["rho_perc", "d_lin", "DET", "LAM"]}})
    return pd.DataFrame(trend_rows), pd.DataFrame(cell_rows)


# ---------------------------------------------------------------- P3c tasks
def _power_task(args):
    from condpred.contaminate import add_obs_noise
    from recplay.surrogates import iaaft
    from rhoperc import prep
    from rhoperc.processes import generate

    combo_idx, rep, n_surr = args
    process, noise = POWER_COMBOS[combo_idx]
    cell = 2000 + int(combo_idx)
    cache = _cache_dir() / f"power_{process}_n{noise}_rep{rep}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rng = np.random.default_rng([PHASE, int(cell), int(rep)])
    tau = TAU_BY_PROCESS[process]
    n = 100 if SMOKE else N_POWER
    n_series = n + 2 * tau  # m=3 -> N=n states
    x_clean = generate(process, n_series, rng)
    omega = float(noise) * float(np.std(x_clean))
    x = add_obs_noise(x_clean, omega, rng) if True else x_clean
    rule = prep.theiler_rule(x, m=3, tau=tau, n_states=n)
    w = int(rule["w"])
    obs, surr = {}, {t: {s: [] for s in POWER_STATS} for t in TRANSFORMS}
    for transform in TRANSFORMS:
        obs[transform] = _measures(prep.distance(x, transform=transform, m=3, tau=tau), w)
    for _ in range(n_surr):
        xs = iaaft(x, rng)
        for transform in TRANSFORMS:
            vals = _measures(prep.distance(xs, transform=transform, m=3, tau=tau), w)
            for s in POWER_STATS:
                surr[transform][s].append(vals[s])
    rows = []
    for transform in TRANSFORMS:
        row = {"process": process, "noise_sd": float(noise), "noise_omega": omega,
               "N": int(n), "m": 3, "tau": int(tau), "transform": transform,
               "rep": int(rep), "seed": f"[{PHASE},{int(cell)},{int(rep)}]",
               "seed_cell": int(cell), "n_series": int(n_series), "theiler_w": w,
               "tau_e": (None if rule["tau_e"] is None else int(rule["tau_e"])),
               "too_persistent": bool(rule["too_persistent"]),
               "n_surrogates": int(n_surr), **obs[transform]}
        for s in POWER_STATS:
            st = prep.surrogate_stats(obs[transform][s], surr[transform][s])
            row[f"d_{s}"] = st["d"]; row[f"pct_{s}"] = st["pct"]
            row[f"reject_{s}"] = bool(st["pct"] == 0.0 or st["pct"] == 1.0)
        # d_lin/pct_lin names (Article 4) refer to rho_perc
        row["d_lin"] = row["d_rho_perc"]; row["pct_lin"] = row["pct_rho_perc"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def _power_summary(df):
    rows = []
    for (process, noise, transform), g in df.groupby(["process", "noise_sd", "transform"], sort=False):
        for s in POWER_STATS:
            rows.append({"process": process, "noise_sd": float(noise), "transform": transform,
                         "statistic": s, "n_runs": int(len(g)),
                         "n_reject": int(g[f"reject_{s}"].sum()),
                         "rejection_rate": float(g[f"reject_{s}"].mean())})
    out = pd.DataFrame(rows)
    out["_p"] = out["process"].map({p: i for i, p in enumerate(dict.fromkeys(c[0] for c in POWER_COMBOS))})
    out["_t"] = out["transform"].map({t: i for i, t in enumerate(TRANSFORMS)})
    out["_s"] = out["statistic"].map({v: i for i, v in enumerate(POWER_STATS)})
    return out.sort_values(["_p", "noise_sd", "_t", "_s"]).drop(columns=["_p", "_t", "_s"]).reset_index(drop=True)


# ---------------------------------------------------------------- gate / T
def write_decision(logistic_scores, breaks_summary, power_summary, out_dir):
    from condpred.report import write_decision as _wd

    sc = logistic_scores[(logistic_scores["m"] == 1) & (logistic_scores["measure"] == "rho_perc")]
    auc = {r.transform: float(r.auc_periodic_vs_chaotic) for r in sc.itertuples()}
    a_pass = any(v >= 0.90 for v in auc.values())
    perm = breaks_summary[(breaks_summary["part"] == "permanent")
                          & (breaks_summary["measure"] == "rho_perc")]
    spear = {r.transform: float(r.spearman_size_vs_value) for r in perm.itertuples()}
    b_pass = any(v >= 0.5 for v in spear.values())
    rho_cells = power_summary[power_summary["statistic"] == "rho_perc"]
    n_c_pass = int((rho_cells["rejection_rate"] >= 0.5).sum())
    c_pass = n_c_pass >= 3
    gate = "PASS" if (a_pass or b_pass or c_pass) else "STOP"
    # Decision T (constitution wording says P3d; P3 has parts a-c only, so
    # mean P3c power is used -- recorded in spec.json and DEVIATIONS.md).
    mean_pow = {t: float(rho_cells[rho_cells["transform"] == t]["rejection_rate"].mean())
                for t in TRANSFORMS}
    keep_rank = (auc["rank"] >= auc["z"] - 0.03) and (mean_pow["rank"] >= mean_pow["z"] - 0.03)
    kept = "rank" if keep_rank else "z"
    rows = [
        {"item": "G3_gate", "rule": GATE_RULE, "result": gate,
         "detail": (f"(a) m=1 rho_perc AUC z={auc['z']:.4f} rank={auc['rank']:.4f} pass={a_pass}; "
                    f"(b) permanent raw rho_perc Spearman z={spear['z']:.4f} rank={spear['rank']:.4f} pass={b_pass}; "
                    f"(c) rho_perc cells (process x noise x transform, all P3c cells) with rejection_rate>=0.5: "
                    f"{n_c_pass} of {len(rho_cells)} pass={c_pass}")},
        {"item": "Decision_T", "rule": DECISION_T_RULE, "result": f"keep_{kept}",
         "detail": (f"Uses P3c for 'P3d' (see spec/DEVIATIONS). m=1 rho_perc AUC rank={auc['rank']:.4f} vs "
                    f"z={auc['z']:.4f} (rank>=z-0.03: {auc['rank'] >= auc['z'] - 0.03}); mean P3c rho_perc "
                    f"rejection rate over all cells rank={mean_pow['rank']:.4f} vs z={mean_pow['z']:.4f} "
                    f"(rank>=z-0.03: {mean_pow['rank'] >= mean_pow['z'] - 0.03}); kept transform={kept}")},
    ]
    _wd(rows, out_dir)
    return gate, kept, {"auc": auc, "spearman": spear, "mean_power": mean_pow, "n_c_pass": n_c_pass}


# ---------------------------------------------------------------- figures
def _fig_logistic(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from condpred.report import save_fig

    g = df[(df["m"] == 1)].sort_values("r")
    panels = [("lambda_lyapunov", "Lyapunov lambda"), ("rho_perc", "rho_perc"),
              ("rho_iso", "rho_iso"), ("rho_lag", "rho_lag"), ("DET", "DET"),
              ("LAM", "LAM"), ("L", "L")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 12), sharex=True)
    for ax, (col, label) in zip(axes, panels):
        if col == "lambda_lyapunov":
            gz = g[g["transform"] == "z"].drop_duplicates("r")
            ax.plot(gz["r"], gz[col], lw=0.8); ax.axhline(0, color="k", lw=0.6)
        elif col in ("rho_perc", "rho_iso", "rho_lag"):
            for t, ls in [("z", "-"), ("rank", "--")]:
                gt = g[g["transform"] == t]
                ax.plot(gt["r"], gt[col], ls, lw=0.8, label=t)
            ax.legend(fontsize=7)
        else:
            gt = g[g["transform"] == "z"]
            ax.plot(gt["r"], gt[col], lw=0.8)
        ax.set_ylabel(label, fontsize=8)
        for centre, _ in TRANSITION_CENTRES:
            ax.axvline(centre, color="r", lw=0.5, alpha=0.6)
    axes[-1].set_xlabel("r")
    fig.suptitle("P3a logistic map, m=1, N=1000 (rho panels: z solid, rank dashed; RQA panels: z)")
    save_fig(fig, out_dir / "figures", "logistic")
    plt.close(fig)


def _fig_breaks(cell_summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from condpred.report import save_fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, part in zip(axes, ["permanent", "recurring"]):
        g = cell_summary[cell_summary["part"] == part]
        for t, ls in [("z", "-o"), ("rank", "--s")]:
            gt = g[g["transform"] == t].sort_values("size_sd")
            ax.plot(gt["size_sd"], gt["median_rho_perc"], ls, lw=1, label=t)
        ax.set_title(part); ax.set_xlabel("size / gap (sd)"); ax.legend(fontsize=8)
    axes[0].set_ylabel("median rho_perc")
    fig.suptitle("P3b breaks and regimes, N=400")
    save_fig(fig, out_dir / "figures", "breaks")
    plt.close(fig)


def _fig_power(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from condpred.report import save_fig

    g = summary[summary["statistic"] == "rho_perc"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, transform in zip(axes, TRANSFORMS):
        gt = g[g["transform"] == transform]
        for proc in ["D1", "D3", "F1"]:
            gp = gt[gt["process"] == proc].sort_values("noise_sd")
            ax.plot(gp["noise_sd"], gp["rejection_rate"], "-o", lw=1, label=proc)
        for proc in ["S5", "S6"]:
            gp = gt[gt["process"] == proc]
            ax.plot(gp["noise_sd"], gp["rejection_rate"], "s", ms=8, label=f"{proc} (noise 0)")
        ax.axhline(0.5, color="k", lw=0.6); ax.set_title(transform)
        ax.set_xlabel("observational noise (sd)"); ax.legend(fontsize=7)
    axes[0].set_ylabel("rejection rate, rho_perc")
    fig.suptitle("P3c IAAFT power, rho_perc, N=400")
    save_fig(fig, out_dir / "figures", "power")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    from condpred.report import save_table, write_spec

    expected, actual = _frozen_hash()
    if actual != expected:
        stop = ROOT / "plans/STOP_P3.md"
        stop.write_text(f"# STOP P3\n\nFrozen core hash mismatch: FROZEN.json={expected}, "
                        f"current src/rhoperc/core.py={actual}. P3 not run.\n", encoding="utf-8")
        print(f"STOP_P3 frozen hash mismatch expected={expected} actual={actual}", flush=True)
        return 2

    if SMOKE:
        r_grid = np.array([3.5, 3.7, 3.8284, 3.9, 4.0])
        breaks_reps, power_reps, n_surr_b, n_surr_p = 2, 2, 3, 3
        perm_levels, recur_levels = [0.0, 2.0], [0.0, 2.0]
        combos = [0, 12]  # D1 noise0, S5 noise0 indices into POWER_COMBOS
    else:
        r_grid = R_FULL
        breaks_reps, power_reps, n_surr_b, n_surr_p = 100, 60, N_SURR_BREAKS, N_SURR_POWER
        perm_levels, recur_levels = PERM_SIZES, RECUR_GAPS
        combos = list(range(len(POWER_COMBOS)))

    out_dir = ROOT / "tables/P3" / ("smoke" if SMOKE else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "version": VERSION, "phase": "P3", "smoke": SMOKE,
        "gate": "G3", "gate_rule": GATE_RULE,
        "gate_evaluation_preregistered": {
            "(a)": "rho_perc, m=1 (primary); AUC computed per transform (z and rank separately, see AUC definition); gate (a) passes if either transform AUC >= 0.90",
            "(b)": "permanent-break part only; Spearman of size_sd vs raw rho_perc pooled across all runs (all sizes, each run one point), per transform; gate (b) passes if either transform Spearman >= 0.5. d_lin Spearman is reported in breaks_summary.csv but does not count for the gate",
            "(c)": "cell = (process x noise_sd x transform) over ALL P3c cells listed in P3c.cells (28 cells full run); statistic = rho_perc only; gate (c) passes if rejection_rate >= 0.5 in at least 3 such cells",
        },
        "decision_T_rule": DECISION_T_RULE,
        "decision_T_evaluation_preregistered": {
            "note": "Constitution Decision T says 'mean P3d power' but P3 has parts a-c only; interpreted as mean P3c power (recorded in plans/rho_perc/DEVIATIONS.md, not silently renamed)",
            "AUC": "P3a rho_perc AUC (discrimination, below), m=1, per transform",
            "mean_power": "mean over all P3c cells (process x noise x transform as in gate (c)) of rho_perc rejection rate, per transform",
            "rule": "keep rank iff rank AUC >= z AUC - 0.03 AND rank mean power >= z mean power - 0.03, else keep z",
        },
        "P3a_logistic": {
            "r_grid": "3.5 to 4.0 step 0.001 (501 values)" if not SMOKE else "smoke subset [3.5, 3.7, 3.8284, 3.9, 4.0]",
            "N_states": N_LOGISTIC if not SMOKE else "<=100 (smoke)",
            "n_series_convention": "n = N + (m-1)*tau series length giving N states after embedding, tau=1; series regenerated per m with a fresh rng seeded [3, r_index, 0] so m=1 and m=3 use the same underlying trajectory",
            "generator": "condpred.systems.logistic(r, n, rng) (includes its own burn=1000); return checked: 1D array of length n",
            "m": MS_A, "transforms": TRANSFORMS, "tau": 1,
            "measures": "rho_perc, rho_iso, rho_lag via rhoperc.core.rho_perc on rhoperc.prep.distance; DET, LAM, L via recplay.rqa.rqa(recplay.recurrence.threshold_by_rate(D, 0.10, theiler=w))",
            "theiler": "w from rhoperc.prep.theiler_rule on the series per (r, m), used for both transforms, rho_perc and the RQA threshold",
            "lyapunov": "lambda = mean of ln|r(1-2x)| over 100000 iterates of a deterministic trajectory x0=0.5 after discarding 1000 transient iterates (smoke: 10000 iterates); computed once per r, independent of the measured series",
            "lambda_classes": "periodic if lambda<0, chaotic if lambda>0; lambda==0 excluded from AUC and Spearman and counted in logistic_scores.csv (n_lambda_zero_excluded)",
            "AUC_definition": "per (m, transform, measure): auc_raw = roc_auc_score(is_chaotic, measure) over lambda!=0 rows; auc_periodic_vs_chaotic = max(auc_raw, 1-auc_raw) (discrimination, >=0.5); direction recorded in auc_direction",
            "spearman": "Spearman of measure vs lambda over chaotic (lambda>0) r only, per (m, transform, measure)",
            "transition_windows": "logistic.csv column transition_window flags |r-centre|<=0.01 for centres 3.6786 (band_merging), 3.8284 (period3_window), 3.8568 (interior_crisis); transitions are confirmed in the report from the sign/size of lambda in logistic.csv, not assumed from these numbers",
        },
        "P3b_breaks": {
            "N_states": N_BREAKS if not SMOKE else 100, "runs_per_cell": breaks_reps,
            "n_surrogates": n_surr_b, "m": 3, "tau": 1, "transforms": TRANSFORMS,
            "permanent": "S3 = condpred.systems.ar1(0.9, n, rng) plus level shift size_sd * sd_process added from midpoint; sd_process = stationary sd of S3 = 1 (innovation sd sqrt(1-0.9^2) chosen by ar1 so stationary sd is 1; same convention as X2 in rhoperc.processes: +4.0 for 4 sd); sizes PERM_SIZES",
            "permanent_sizes_sd": PERM_SIZES if not SMOKE else perm_levels,
            "recurring": "AR(1) 0.8 fluctuations y (innovation sd sqrt(1-0.8^2), stationary sd 1) around a mean level switching between 0 and gap_sd*1 with stay probability 0.98 per step (gap in units of the same process sd=1); burn 500 discarded; gaps RECUR_GAPS",
            "recurring_gaps_sd": RECUR_GAPS if not SMOKE else recur_levels,
            "expectation": "permanent: rho_perc rises with size (stated); recurring: no stated expectation, trend reported either way",
            "statistics": "raw rho_perc/rho_iso/rho_lag/DET/LAM/L plus d_lin and pct_lin for rho_perc vs IAAFT surrogates (same transform, m, tau, delta and w of the original, Article 4)",
            "trend": "breaks_summary.csv: Spearman of size_sd vs value pooled across all runs per (part, transform) for rho_perc, d_lin, DET, LAM; breaks_cell_summary.csv: medians per cell",
        },
        "P3c_power": {
            "N_states": N_POWER if not SMOKE else 100, "runs_per_cell": power_reps,
            "n_surrogates": n_surr_p, "m": 3, "transforms": TRANSFORMS,
            "tau_by_process": TAU_BY_PROCESS,
            "cells": [{"process": p, "noise_sd": nz} for p, nz in POWER_COMBOS],
            "noise": "condpred.contaminate.add_obs_noise(x_clean, omega, rng) with omega = noise_sd * std(x_clean) (np.std, ddof=0), i.e. noise levels are in units of the clean series sd; S5 and S6 enter at noise 0 only",
            "generators": "rhoperc.processes.generate(pid, n, rng) (D1 logistic r=4, D3 Henon x, F1 Rossler x step 0.2, S5 GARCH, S6 MS-AR1)",
            "statistics": "rho_perc, DET, LAM each tested against the same IAAFT surrogates of the same run (same transform, m, tau, delta and w of the original); DET/LAM surrogate values computed with the same RR=10% threshold_by_rate + rqa method as the observed",
            "test": "reject at 5% when the observed statistic is the smallest or largest of the n_surrogates+1 values (pct==0 or pct==1)",
            "power_summary": "rejection rate per (process, noise_sd, transform, statistic)",
        },
        "delta": DELTA, "connectivity": 8,
        "seeds": "np.random.default_rng([phase, cell, rep]), phase=3. P3a cell=r_index (0-based over r grid), rep=0. P3b cell=1000+100*part_index+level_index (part_index 0=permanent,1=recurring). P3c cell=2000+combo_index over POWER_COMBOS. Surrogate/noise draws continue from the run rng. Workers receive only seeds/parameters (joblib args), never arrays.",
        "caches": "data/processed/rho_perc/P3/v<version>/ (smoke/ under smoke), one file per cell, rerun skips existing",
        "core_py_sha256_frozen": expected, "core_py_sha256_at_spec": actual,
    }
    write_spec(spec, out_dir, modules=["rhoperc.core", "rhoperc.prep", "rhoperc.processes",
                                       "rhoperc.experiments.p3"])

    from joblib import Parallel, delayed

    t0 = time.time()
    frames = Parallel(n_jobs=8)(delayed(_logistic_task)((float(r), i)) for i, r in enumerate(r_grid))
    logistic = pd.concat(frames, ignore_index=True)
    logistic = logistic.sort_values(["r_index", "m", "transform"]).reset_index(drop=True)
    save_table(logistic, out_dir, "logistic")
    scores = _logistic_scores(logistic)
    save_table(scores, out_dir, "logistic_scores")
    print(f"P3a rows={len(logistic)} scores={len(scores)} elapsed={time.time()-t0:.1f}s smoke={SMOKE}", flush=True)

    t1 = time.time()
    btasks = ([("permanent", lv, rep, n_surr_b) for lv in perm_levels for rep in range(breaks_reps)]
              + [("recurring", lv, rep, n_surr_b) for lv in recur_levels for rep in range(breaks_reps)])
    frames = Parallel(n_jobs=8)(delayed(_breaks_task)(a) for a in btasks)
    breaks = pd.concat(frames, ignore_index=True)
    breaks = breaks.sort_values(["part", "size_sd", "transform", "rep"]).reset_index(drop=True)
    save_table(breaks, out_dir, "breaks")
    b_sum, b_cell = _breaks_summaries(breaks)
    save_table(b_sum, out_dir, "breaks_summary")
    save_table(b_cell, out_dir, "breaks_cell_summary")
    print(f"P3b rows={len(breaks)} elapsed={time.time()-t1:.1f}s smoke={SMOKE}", flush=True)

    t2 = time.time()
    ptasks = [(ci, rep, n_surr_p) for ci in combos for rep in range(power_reps)]
    frames = Parallel(n_jobs=8)(delayed(_power_task)(a) for a in ptasks)
    power = pd.concat(frames, ignore_index=True)
    power = power.sort_values(["process", "noise_sd", "transform", "rep"]).reset_index(drop=True)
    save_table(power, out_dir, "power")
    p_sum = _power_summary(power)
    save_table(p_sum, out_dir, "power_summary")
    print(f"P3c rows={len(power)} summary={len(p_sum)} elapsed={time.time()-t2:.1f}s smoke={SMOKE}", flush=True)

    if not SMOKE:
        _fig_logistic(logistic, out_dir)
        _fig_breaks(b_cell, out_dir)
        _fig_power(p_sum, out_dir)
        gate, kept, detail = write_decision(scores, b_sum, p_sum, out_dir)
        print(f"G3={gate} DecisionT=keep_{kept} detail={detail}", flush=True)
        if gate == "STOP":
            stop = ROOT / "plans/STOP_P3.md"
            stop.write_text(
                "# STOP P3\n\nGate G3 failed: none of (a) logistic periodic-vs-chaotic AUC >= 0.90 "
                "(m=1 rho_perc, either transform), (b) permanent-break Spearman >= 0.5 (raw rho_perc, "
                "either transform), (c) rho_perc power >= 0.5 in at least 3 P3c cells held. See "
                "tables/P3/decision.csv, logistic_scores.csv, breaks_summary.csv "
                "and power_summary.csv.\n", encoding="utf-8")
            print("STOP_P3 written", flush=True)
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
