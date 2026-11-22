# REVIEW P1 round 1

Reviewer: independent, 2026-11-14. Scope: Constitution Part I, P1, Part III §4–5; `REPORT_P1.md`; `notebooks/tables/rho_perc/P1/{spec.json,decision.csv,runs.csv,by_process.csv}`; `FROZEN.json`. No fixes made, P2 not started.

## Checklist

1. **Spec before tables + G1 verbatim — PASS.** `spec.json` mtime 2026-11-14 00:52:28.989 (its `written_at` agrees), per-cell caches start 00:52:30.6, `runs.csv` 00:52:45, `by_process.csv` 00:52:45, `decision.csv` 00:52:51. `spec.json:61-62` `gate_rule` is exactly Article 6: "At least 6 of the 11 in-scope processes are \"free\" (not pinned to a bound)."
2. **FROZEN hash == core.py — PASS.** SHA-256 of `src/rhoperc/core.py` recomputed as `3ad7c180…c636`, identical to `plans/rho_perc/FROZEN.json` and to `spec.json` (`core_py_sha256_frozen` = `core_py_sha256_at_spec`).
3. **`python -m rhoperc.selfcheck` passes now — PASS.** Run fresh with Python312, `PYTHONPATH=src` (only the harness runner processes were present): exit 0, K1–K9 PASS, "ALL PASS".
4. **Three random rows reproduced — PASS.** Chosen by `default_rng(20261114)`: row indices 2468 (X2 rep 17, z/rule), 1948 (F1 rep 37, z/rule), 2436 (X2 rep 9, z/rule). Each regenerated directly (`default_rng([1, cell, rep])` → `processes.generate` → `prep.theiler_rule`/`prep.distance` → `core.rho_perc`); all six ε/ρ values, `theiler_w` and `too_persistent` match `runs.csv` to ≤ 4.4e-16.
5. **Every report number in the named CSV — PASS.** Independent recomputation from `runs.csv` reproduces `by_process.csv` exactly (pinned shares to 0.0, median π to 1e-16, Spearmans to 1e-16). Verified: pinned shares rule+z (S3 0.18 low; S6, F1 0.08; others 0.00–0.02; X1 0.38 = 0.22 high + 0.16 low; X2 0.46 high); ρ_iso ≤ ρ_perc ≤ ρ_lag in 0 of 2,600 violated; median π in-scope 0.296 (S3) to 0.753 (F1); Spearman vs ρ_iso −0.489 (D1) to 0.790 (S5), vs ρ_lag −0.146 (S6) to 0.929 (F1, itself 0.929); `too_persistent` 0.00 in-scope except S3 0.02, X1 0.66, X2 0.98; Theiler claims: w raised in 184/1300 = 14.2 % of (process, transform, rep) cells (S3 98 %, S6 40 %), max pinned-share change 0.14 (S3 z, pinned-low 0.04 → 0.18), no classification change. The 16.5 s compute claim matches `run.log`.
6. **decision.csv follows Article 6 G1 — PASS.** `src/rhoperc/experiments/p1.py`: free = share pinned < 0.5 under rule+z (`_classification`, lines 124–135), PASS iff ≥ 6 of the 11 `IN_SCOPE` free (lines 232–240); X1/X2 are computed and listed in `classification_rule_z` but excluded from `n_free_in_scope`. `decision.csv`: G1 PASS, 11/11 free. Pinned definitions exact vs P1 spec: pinned high ε_perc ≥ 0.95 ε_lag, pinned low ε_perc ≤ 1.05 ε_iso or both 0 (spec definitions; `p1.py` lines 70–71, recomputed pinned flags match `runs.csv` on all 2,600 rows). π guard present: π = NaN when ρ_lag = ρ_iso (`p1.py` line 73); occurred in 0 runs.
7. **git status: no forbidden changes — PASS.** No modified tracked files (`git diff` empty). Untracked entries are `src/rhoperc/` (created by this plan) and pre-existing `src/dynimp/`, `Writeups/dynimp_*` dated 2026-11-12, before the P1 run — not touched by P1. No existing notebook, `src/recplay/**`, `src/condpred/**`, `src/rpjepa/**`, README or write-up path changed.
8. **No smoke numbers — PASS.** Smoke run is separate (`P1/smoke/`, N=100, 2 reps, values differ, e.g. S3 pinned 0.5); every number in `REPORT_P1.md` matches the full 2,600-row tables instead.

## Extra checks

- `runs.csv`: 2,600 rows = 13 processes × 2 transforms × 2 Theiler settings × 50 reps, 24 columns including ε and ρ for perc/iso/lag, `theiler_w`, seed parts and `[1, cell, rep]` string, `N`=400, `m`=3, `tau`, `tau_e`, `too_persistent`, `pi`, pinned flags. `by_process.csv`: 52 rows.
- Figure `figures/rho_bounds_by_process.png` exists (1309×528) and its bars match the rule+z medians recomputed from `runs.csv` (e.g. F1 0.021/0.205/0.237; S5 0.371/0.650/0.965; X2 0.380/0.620/0.716).

## Verdict

**ACCEPT.** Gate G1 PASS (11/11 free) is supported end to end. No fixes required.
