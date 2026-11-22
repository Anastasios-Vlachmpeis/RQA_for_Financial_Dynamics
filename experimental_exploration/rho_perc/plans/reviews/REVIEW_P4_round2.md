# REVIEW P4 round 2 — ACCEPT

**Verdict: ACCEPT** (all three round-1 fixes correctly applied; report text only).

## Fix-by-fix verification (recomputed, not taken from the report)

1. **Finding 1 counts — PASS.** From `notebooks/tables/rho_perc/P4/sensitivity.csv` (7,520 rows), grouping by unique `(cfg_id, rep)` per `connectivity`: 8-neighbourhood = 5,400 unique runs, 0 violations of ρ_iso ≤ ρ_perc ≤ ρ_lag; 4-neighbourhood = 400 unique runs, 13 violations, all ρ_perc > ρ_lag. Report Finding 1 now states exactly these counts, explains that the ρ_lag bound proof uses a straight (1,1) diagonal chain — 8-connected, not 4-connected — so ρ_lag is not an upper bound under the 4-neighbourhood (P0 caveat), and explicitly claims "no universal bound" for the 4-neighbourhood. No universal 4-neighbourhood bound claim remains anywhere in the report.
2. **Finding 3 count — PASS.** From `stability.csv`: d_lin first-doubling N ≤ ρ_perc's for S1 (100≤200), S4 (100≤100), S5 (200≤1600), D1 (100≤100), D3 (100≤400), F1 (400≤400) — six of eight (exceptions S3, F2). Report names all six.
3. **First-doubling wording — PASS.** Acceptance heading, the paragraph after the table, Acceptance (3) item 1, Finding 3 and the final paragraph all say "first-doubling N", "not sustained stability", "not a guarantee", "not guaranteed minimum lengths". Relapse numbers verified against `stability.csv`: S1 qualifies at 200 (0.008), later 0.024 (400→800) and 0.026 (800→1600); S4 qualifies at 100 (0.001), later 0.023 (200→400) and 0.028 (400→800). Remaining "minimum"/"stable" occurrences are all negated; no "stability N" phrasing survives.

## No collateral changes / no new numbers

- `git status`: no tracked file modified; P4 tables and `src/rhoperc/` are untracked/unchanged. File times: all P4 CSVs/figures 01:49, `src/rhoperc` code ≤01:43 — both predate round 1 (01:51); only `reports/REPORT_P4.md` changed after it (01:52).
- Every number newly introduced by the fixes (5,400 / 400 / 13 / 0, the six names, 0.024 / 0.026 / 0.023 / 0.028) recomputes from `sensitivity.csv`/`stability.csv` as above; no unsupported numbers added. (Table display 0.020 for S5 d_lin is the round-1-passed rounding of 0.0196, which meets <0.02.)

No rerun needed. P4 report accepted as written.
