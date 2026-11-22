# REPORT P0 — definition and implementation

**What was run.** Self-check K1–K9 (`python -m rhoperc.selfcheck`); then legacy (`rr_critical`), span (`rho_perc_span`) and cover (`rhoperc.core.rho_perc`) rules on the 13 Article-4 processes × 20 runs, N = 400 states, m = 3, δ = 0.05, 8-neighbourhood, z-transform, w = (m−1)τ+1. Frozen flow delays from AMI on a length-5000 seed-0 reference run: τ = 7 (F1 Rössler), τ = 4 (F2 Lorenz), recorded in `notebooks/tables/rho_perc/P0/spec.json` (written before the results). Full compute took 13.3 s for the comparison plus the self-check (run.log in the same folder).

**Gate G0: PASS.** Every known-answer test passes (`notebooks/tables/rho_perc/P0/decision.csv`: K1–K9 all PASS, selfcheck exit 0). Fast method at N = 2000: 1.302 s, under the 5 s limit (same file). `src/rhoperc/core.py` is frozen; its SHA-256 in `plans/rho_perc/FROZEN.json` matches the file.

**Findings** (all from `notebooks/tables/rho_perc/P0/rule_comparison.csv`, 260 rows, unless stated):

1. The legacy rule is far below the cover rule in most runs: cover − legacy median difference is 0.385 over all 260 runs, and legacy < cover in 94.2 % of runs. Per-process legacy medians are 0.003–0.007 for the 11 in-scope processes, against cover medians of 0.21–0.68.
2. The span rule is above the cover rule in 90.4 % of runs (median span − cover = 0.103). The gap is largest for the flows: median span − cover = 0.670 for F1 (span median 0.870 vs cover 0.212) and 0.312 for F2.
3. The cover value sat between its two bounds in every run: ρ_iso ≤ ρ_perc ≤ ρ_lag for all 260 rows (0 violations in `rule_comparison.csv`, columns `rho_iso`, `cover`, `rho_lag`).
4. On iid normal data (S1) the cover median is 0.499; on the out-of-scope panel, X2 (permanent level shift) has cover median 0.668 against 0.428 for S3 without the shift. These are single-panel medians from `rule_comparison.csv`; later phases test them properly.
5. K2 confirms the cycle failure of the span rule is fixed: for exact periods 7, 40 and 90 the cover rule gives ε_perc = 0 and ρ_perc equal to the closed form to 1e-12 (`decision.csv`, K2 PASS).

**Deviations.** Two entries in `plans/rho_perc/DEVIATIONS.md`: the fast search bracket above ε_lag for the 4-neighbourhood (a single diagonal is not 4-connected, so ε_lag only brackets the 8-neighbourhood), and exact-tie test cases built on an integer grid so tied distances are exact in floating point. Neither changes Article 3.

**What it means for the paper.** The corrected cover rule now has a tested, frozen implementation whose two predecessors are measurably wrong on the standard panel, so the methods paper has a sound object to study. Whether ρ_perc says anything beyond its bounds ρ_iso and ρ_lag is still open — that is exactly what P1 measures.
