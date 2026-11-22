# REVIEW P4 round 1 — FIX-REQUIRED

**Verdict: FIX-REQUIRED** (report text only; no rerun needed — tables, spec and code check out).

## Checks passed

1. **Spec versions (§4.1, adapted).** History verified by `written_at`, file times and caches: v1 smoke only (failed plumbing, archived as `P4/smoke/spec_v1.json`); v2 smoke + full run (caches `data/processed/rho_perc/P4/v2/`, 01:37–01:42); v3 spec written 2026-11-14T00:44:15Z, first v3 cache one second later, final tables 01:49. Final results were fully recomputed under the final v3 spec (5,800 fresh v3 caches), not reused from v2; v2→v3 values are identical (max |Δρ| = 0.0 over all 5,800 cache rows), confirming the "labelling fix only" claim. `DEVIATIONS.md` has exactly one P4 entry covering both bumps. Interpretations (2×rule, m with τ frozen, fixed-span, `d_lin`, dedupe) are pre-registered inside `spec.json`.
2. **Frozen hash (§4.2).** `FROZEN.json` = current `src/rhoperc/core.py` SHA-256 = `3ad7c180…c636`, also recorded in v3 `spec.json`.
3. **Selfcheck (§4.3).** Re-ran `python -m rhoperc.selfcheck` (Python312, `PYTHONPATH=src`): K1–K9 all PASS, exit 0.
4. **Direct reproduction (§4.4).** Three rows recomputed from seeds `[4, cfg, rep]`, bypassing caches, all matching `sensitivity.csv` exactly: S1 N=3200 cfg5/rep0 (ρ=0.546070884544); F1 sampling `fixed_N:x2` cfg106/rep0 (ρ=0.194186124756); S3 N=400 cfg17/rep0 with 19 IAAFT surrogates (ρ=0.660583045944, `d_lin`=0.073781831259).
5. **Report numbers (§4.5, recomputed).** `sensitivity_summary.csv` reproduces from `sensitivity.csv` (168 rows, max median diff 1e-16). Stability lists in the report match `stability.csv` for both ρ_perc and `d_lin`; the code (`p4.py:_stability`) takes the **first sequential doubling** 100→200→… with change < 0.02, not the minimum over arbitrary pairs. All one-sentence ranges verified: N up to 0.29 (S5), δ F1 0.168 / S5 0.105, F2 Theiler 0.147 vs 0.393, m F2 0.337, neighbourhood D3 0.534, sampling F1 0.198 / F2 0.225 at fixed N and 0.058–0.211 at fixed span.
6. **Grid accounting.** 132 unique configs, Σreps = 5,800 designed unique (cfg, rep) runs; `sensitivity.csv` has 7,520 rows = those 5,800 expanded, with only the baseline config repeated (×5 settings per process, ×2 extra sampling modes for F1/F2). Summaries group per setting, so the baseline is counted once per level — no double-counting.
7. **No gate (§4.6, adapted).** No `decision.csv` in P4 and none claimed; acceptance items (1)–(3) are all present. Six figures exist in `figures/`.
8. **Git (§4.7) / smoke (§4.8).** No tracked file modified; nothing under `src/recplay`, `src/condpred`, `src/rpjepa` changed (untracked `src/dynimp/`, `Writeups/dynimp_*` are outside P4's outputs). No smoke number appears in the report (smoke `stability.csv` differs from the full one cited).

## Fixes required (report wording, `REPORT_P4.md`)

1. **Finding 1 is misstated.** "In all 7520 saved runs with the 8-neighbourhood … 0 violations": 7,520 is the expanded row count, including 400 4-neighbourhood rows and baseline repeats. Correct figures: 5,400 unique 8-neighbourhood runs, 0 violations; the 13 bound violations in the table are all 4-neighbourhood rows (expected — ε_lag is not an upper bound there, P0 deviation). Restate with these counts.
2. **Finding 3 miscounts.** From the report's own table, `d_lin` stabilises no later than ρ_perc for **six** of eight processes (S1, S4, S5, D1, D3, F1), not five.
3. **"Stability N" overclaims.** The criterion is the *first* doubling below 0.02, not sustained stability: S1 qualifies at 200 yet 400→800 changes 0.024; S4 qualifies at 100 yet 200→400 changes 0.023 (`stability.csv` discloses all steps). The report (acceptance text and final paragraph) must call it the first-doubling N and drop the implication that values from that N onward are mutually stable.
