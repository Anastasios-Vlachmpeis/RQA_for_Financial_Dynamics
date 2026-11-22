# REPORT P4 — scaling and sensitivity

**What was run.** Kept transform only (`z`, Decision T in `notebooks/tables/rho_perc/P3/decision.csv`). Processes S1, S3, S4, S5, D1, D3, F1, F2, each varied one setting at a time from the Article 4 baseline (N = 400 states, m = 3, τ per process — F1 = 7, F2 = 4, else 1 — δ = 0.05, Theiler rule, 8-neighbourhood): N ∈ {100, 200, 400, 800, 1600, 3200} (50 runs up to 800, 20 at 1600, 10 at 3200); δ ∈ {0.02, 0.05, 0.10, 0.20} (50); Theiler ∈ {1, rule, 2×rule} (50); m ∈ {1, 2, 3, 5} with τ frozen and `w` recomputed by `theiler_rule` at that m (50); neighbourhood ∈ {4, 8} (50); sampling step ×{0.5, 1, 2, 4} for F1/F2 only (30, at fixed N and at fixed time span). Also `d_lin` (19 IAAFT surrogates, 30 runs, same 8 processes, baseline settings) at N ∈ {100, 200, 400, 800}, computed inside the N-grid cells. The full grid is in `notebooks/tables/rho_perc/P4/spec.json` (written before the results), the per-run values in `sensitivity.csv` (7520 rows, baseline rows repeated across the dimensions they anchor), medians and quartiles in `sensitivity_summary.csv`, and the acceptance table in `stability.csv`; one figure per setting is in `figures/`. The full run took about 5 minutes (`run.log`); per-(config, rep) caches are under `data/processed/rho_perc/P4/v3/`.

**Gate.** P4 has no gate and no `decision.csv` (Article 6).

**Acceptance (1) and (2) — first doubling where |median change| < 0.02** (`stability.csv`, first doubling in the sequence 100→200→… that meets the criterion — not sustained stability from that N onward; the `d_lin` grid ends at 800):

| Process | ρ_perc: first-doubling N | change there | `d_lin`: first-doubling N | change there |
|---|---|---|---|---|
| S1 | 200 | 0.008 | 100 | 0.014 |
| S3 | 100 | 0.005 | 200 | 0.002 |
| S4 | 100 | 0.001 | 100 | 0.007 |
| S5 | 1600 | 0.006 | 200 | 0.020 |
| D1 | 100 | 0.004 | 100 | 0.002 |
| D3 | 400 | 0.004 | 100 | 0.016 |
| F1 | 400 | 0.017 | 400 | 0.012 |
| F2 | 200 | 0.013 | 400 | 0.009 |

Every process meets the criterion at some tested doubling for both statistics, so no entry is "none"; all individual doubling changes are listed in `stability.csv` (e.g. S5's ρ_perc median still moves by 0.104 between N = 400 and 800, and F1's by 0.217 between 100 and 200, in `sensitivity_summary.csv`). This first-doubling N is not sustained stability: S1 qualifies at 200 (0.008) yet later doublings change by 0.024 (400→800) and 0.026 (800→1600), and S4 qualifies at 100 (0.001) yet later doublings change by 0.023 (200→400) and 0.028 (400→800) (`stability.csv`).

**Acceptance (3) — one sentence per setting on whether values at different levels are comparable** (median ranges across levels, per process, from `sensitivity_summary.csv`):

1. **N:** Values at different N are not comparable in general — median ρ_perc ranges across the N grid reach 0.29 (S5: 0.50 at N = 100 to 0.79 at 1600) and 0.23 (F1), and stay below 0.02 only for D1 (0.016), so comparisons should keep N fixed; the first-doubling N in `stability.csv` is only the first point where one doubling happened to move the median by less than 0.02, not a length from which later doublings stay below 0.02.
2. **δ:** Values at different δ are broadly comparable for most processes (median range ≤ 0.043 for S1, S3, S4, D1, D3, F2) but not for F1 (0.168, falling to 0.057 at δ = 0.20) or S5 (0.105), so δ should still be reported and held fixed for flows and GARCH.
3. **Theiler window:** Values at w = 1, the rule, and 2×rule are comparable for the stochastic processes and maps (median range ≤ 0.037) but not for the flows (F2: 0.147 at w = 1 against 0.393 at the rule; F1 range 0.109), so flow values obtained with a small Theiler window must not be compared with rule values.
4. **m:** Values at different m are not comparable — median ranges reach 0.34 (F2: 0.150 at m = 1 against 0.487 at m = 2), 0.15 (S4), 0.14 (D3) and 0.09–0.10 (S1, S3, F1), with only D1 (0.031) and S5 (0.043) changing little.
5. **Neighbourhood:** Values from the 4- and 8-neighbourhoods are not comparable — the 4-neighbourhood median is higher by 0.53 for D3 (0.874 vs 0.340), 0.41 for D1, 0.30 for S1 and 0.22 for S5, even though S3, S4, F1 and F2 change by at most 0.05.
6. **Sampling step (F1, F2):** Values at different sampling steps are not comparable in either mode — across the ×0.5–×4 levels the median ranges are 0.198 for F1 and 0.225 for F2 at fixed N, and 0.058–0.211 at fixed time span (`sensitivity_summary.csv`, setting `sampling`).

**Further findings.**

1. Among the 5,400 unique 8-neighbourhood runs (unique = distinct (cfg_id, rep) in `sensitivity.csv`; the 7,520 saved rows include baseline repeats), ρ_iso ≤ ρ_perc ≤ ρ_lag held with 0 violations. The 13 bound violations in the table are all in the 400 unique 4-neighbourhood runs (all with ρ_perc > ρ_lag). That is expected, not a contradiction: the ρ_lag bound proof uses one straight diagonal — a chain of (1,1) steps — which is 8-connected but not 4-connected, so ε_lag (and hence ρ_lag) is not an upper bound under the 4-neighbourhood. This is the P0 caveat in `plans/rho_perc/DEVIATIONS.md`, also noted at the bisection bracket in `src/rhoperc/core.py`; no universal bound is claimed for the 4-neighbourhood.
2. ρ_perc itself is the most N-sensitive for S5 and F1: S5's median rises from 0.499 (N = 100) to 0.788 (N = 1600) and F1's median at N = 200 (0.051) sits far below its values at every other N (0.225–0.281), both in `sensitivity_summary.csv`.
3. `d_lin` reaches its first-doubling threshold no later than ρ_perc for six of the eight processes (S1, S4, S5, D1, D3, F1) and at N = 400 for both flows (`stability.csv`), so the surrogate correction does not remove the need to check series length — and, as for ρ_perc, that first-doubling N is not a guarantee of stability at larger N.

**Deviations.** One, recorded in `plans/rho_perc/DEVIATIONS.md`: the spec was bumped from version 1 to 2 before any full result after the v1 smoke failed on plumbing — the sampling step is varied via `sample_every` at fixed integrator `dt` (not `dt`×f, which made the Rössler integration overflow), smoke sampling factors are [0.5, 2.0] only, and smoke caches live in `smoke_v<N>/` — and then to version 3 for a labelling fix only (baseline ×1 sampling rows now carry their `fixed_N`/`fixed_span` mode in the tables; grid, seeds and values unchanged, the v3 medians reproduce the v2 ones). The pre-registered interpretations for 2×rule, m (τ frozen, `w` recomputed), fixed-span sampling, `d_lin` processes and baseline dedupe are written in the `spec.json` itself. No STOP file was written; the frozen core hash matched `FROZEN.json` at spec time and `rhoperc.selfcheck` passed after the run (`run.log`).

**What it means for the paper.** A ρ_perc value is only comparable with another when N, m, neighbourhood and (for flows) the Theiler window and sampling step match, with series length reported and held fixed unless length-dependence is checked; the per-process first-doubling thresholds in `stability.csv` are a starting point for that check, not guaranteed minimum lengths from which values are stable. δ is the one setting that can vary modestly for most processes without moving the value much. The Limits section should state these as conditions for comparison rather than as defaults a reader may change freely.
