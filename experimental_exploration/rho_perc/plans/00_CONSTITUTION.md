# ρ_perc — definition fix and validation plan

Governing document for every model that works on the Recurrence Percolation Threshold ρ_perc. Written 13 Nov 2026.
If any other file, prompt or habit disagrees with this document, this document wins. Only Tasos may change it.

**How to read it.** Part I (Articles 1–6) is binding for everyone: read all of it before touching anything.
Part II is the work, one phase per agent. Part III is how to run, report and stop. Appendices are reference.

---

# Part I — Foundations

## Article 1. Purpose, scope, and what "done" means

**Purpose.** Produce the evidence a short methods paper on ρ_perc needs, or show that the paper should not be written.
Both outcomes count as success. A negative result reported plainly is worth more than a positive one obtained by
adjusting things after the fact.

**The measure in one sentence.** Raise the recurrence threshold until one connected structure of recurrence points
involves every moment of the record; ρ_perc is the recurrence rate at which that first happens.

**In scope.** ρ_perc of one series: the corrected definition, a reference implementation, and seven checks
(triviality, null behaviour, known transitions, scaling, incremental value, robustness, one application), then the
write-up package.

**Out of scope. Do not work on these.**

| Topic | Why it is out |
|---|---|
| Cross recurrence plots | The CRP version is the discrete weak Fréchet distance, a known quantity. One paragraph in the paper, no experiments. |
| Third-order plots (TORP) | Tested on 13 Nov 2026: redundant with the pairwise values (`Write-up/torp_perc_gap/cases.csv`). Dropped. |
| Joint recurrence plots | Gets its own plan after gate G5. Only one unit test here (P0). |
| Forecasting, `condpred` experiments, CPI | Unrelated paused work. |
| New variants of the measure | See Article 5, rule 1. |

**Done** means: every phase P0–P8 has an accepted report, or a STOP file exists and Tasos has been told.

## Article 2. Facts already established (13 Nov 2026)

These were measured with the prototype (`plans/rho_perc/prototype_cover.py`) on 6–10 runs each. They explain the
design. **They are not results: the phases re-measure them, and none of these numbers may appear in the paper.**

| # | Fact | Evidence |
|---|---|---|
| F1 | The write-up's definition (`recplay.connectivity.rr_critical`) is wrong. A is the first k rows, B the last k columns; they overlap in the corner, so one early–late recurrence point ends the search. | For iid and AR(1) 0.9 series it equals "RR at the smallest early–late distance" in 8 of 8 runs. |
| F2 | The first repair attempt (`condpred.features.percolation.rho_perc_span`: a point with both indices early linked to a point with both late) is also wrong. It forces the path to start at a lag below k, so a slow cycle cannot link up along its own period. | Exact period-40 series, N = 300: span rule 0.763, correct value 0.022. |
| F3 | The **cover rule** (Article 3) passes every known-answer test, including F1 and F2. | `prototype_cover.py` output, tests K1–K7. |
| F4 | ρ_perc always lies between two simple statistics, ρ_iso and ρ_lag (Article 3.4). | 0 violations in several hundred prototype cases. |
| F5 | For persistent series with a small Theiler window the value sits close to the upper bound ρ_lag. | Rössler, w = 1: 0.035 against 0.040. AR(1) 0.97: 0.32 against 0.48. |
| F6 | On iid data the value depends on the marginal distribution. | Normal 0.52, uniform 0.38 (N = 300). |
| F7 | On iid uniform data the value is stable in N. | Medians 0.368, 0.374, 0.366, 0.378 at N = 100, 200, 400, 800; spread falls from 0.033 to 0.006. |
| F8 | A permanent break raises the value; recurring regimes do not. | AR(1) 0.9 with one level shift: 0.69 against 0.39. Markov-switching level, gap 0–4 sd: 0.47–0.52, no trend. |
| F9 | Speed of the fast method: about 0.2 s at N = 400, 0.45 s at N = 1000, 2.2 s at N = 2000. The machine has 20 logical CPUs. | Timed once. |

F5, F6 and F8 are the reasons phases P1, P2 and P3 exist. F8 already contradicts one sentence of the old write-up
("a series with distinct regimes should only link up late"); the paper must say "permanent break", if P3 confirms it.

## Article 3. The definition (frozen)

### 3.1 Inputs

- A symmetric N × N matrix D of distances between states (smaller = more similar).
- A Theiler window w, an integer with 1 ≤ w ≤ N/2.
- An edge fraction δ with 0 < δ < 1/2. Let k = max(1, ⌈δN⌉).
- A neighbourhood: 8 (default) or 4.

### 3.2 Definition

Indices are 1-based here; i is the row (earlier time), j the column (later time).

- **Domain.** Ω = {(i, j) : j − i ≥ w}.
- **Recurrence points.** P(ε) = {(i, j) ∈ Ω : D_ij ≤ ε}. Recurrence rate RR(ε) = |P(ε)| / |Ω|.
- **Structure.** A maximal set of recurrence points connected through neighbouring points
  (8-neighbourhood: max(|i − i′|, |j − j′|) = 1).
- **Extent of a structure C.** r_min, r_max = smallest and largest row in C; c_min, c_max = smallest and largest column.
- **C covers the record** when all three hold:
  1. r_min ≤ k (C contains a point in the first k rows: it involves an early moment);
  2. c_max > N − k (C contains a point in the last k columns: it involves a late moment);
  3. c_min ≤ r_max + 1 (no moment in between is left out).

Then

    ε_perc = the smallest ε at which some structure of P(ε) covers the record
    ρ_perc = RR(ε_perc)          (points with D_ij = ε_perc are counted)

**Why condition 3 is the repair.** Conditions 1 and 2 are the old write-up. Because neighbouring points differ by at
most one row and one column, the rows of a structure form an unbroken range r_min…r_max and its columns an unbroken
range c_min…c_max. The moments that take part in C are therefore the union of those two ranges. Condition 3 says the two
ranges meet, so every moment from k to N − k + 1 takes part. A single corner point has r_max ≤ k and c_min > N − k,
so it fails.

**Existence.** At the largest ε, P = Ω, which is one structure with rows 1…N − w and columns w + 1…N. It covers the
record exactly when w ≤ N/2. Outside that range the function must raise an error.

**Properties that must hold exactly** (they are unit tests in P0): unchanged when D is replaced by any strictly
increasing function of D; unchanged when the series is reversed in time; 8-neighbour value ≤ 4-neighbour value.

### 3.3 Algorithms

Two implementations must exist and agree to the last tied value.

- **Reference** (slow, plain): add the points of Ω in order of increasing D; merge each new point with its active
  neighbours by union–find; each root stores r_min, r_max, c_min, c_max; stop at the first point whose structure covers
  the record. ρ_perc is the share of Ω with D ≤ that point's distance.
- **Fast**: bisection over the sorted distinct values of D between ε_iso and ε_lag (3.4). At each step label the
  thresholded plot once (`scipy.ndimage.label`) and test the three conditions on the structures that touch both the
  first k rows and the last k columns. Valid because "covers the record" can only switch from false to true as ε grows.

Both are in `prototype_cover.py` and have been checked against each other, including with tied distances.

### 3.4 The two bounds (always computed, always reported)

    ε_iso = max over t in [k, N − k + 1] of  min over s with |s − t| ≥ w of D_ts
    ε_lag = min over lags ℓ in [w, ⌊N/2⌋] of  max over i of D_{i, i+ℓ}
    ε_iso ≤ ε_perc ≤ ε_lag,   ρ_iso = RR(ε_iso),   ρ_lag = RR(ε_lag)

- **ρ_iso, "most isolated moment".** Every moment between the edges must have at least one recurrence, so the moment
  that is hardest to match sets a floor. One outlier can raise it.
- **ρ_lag, "best single lag".** One straight diagonal at lag ℓ already covers the record, so the best such lag sets a
  ceiling. For smooth series and a small w this is the diagonal right next to the excluded band.

If ρ_perc equals one of these in some class of systems, it adds nothing there. P1 measures this.

### 3.5 Names

Keep "Recurrence Percolation Threshold", symbol ρ_perc, distance threshold ε_perc. The rule is called the
**cover rule**. `rr_critical` is the **legacy rule**; `rho_perc_span` is the **span rule**. Both appear only in the
P0 comparison table.

## Article 4. Standard settings

Every experiment uses these unless its phase says otherwise. A phase that varies one setting keeps the others here.

| Setting | Value |
|---|---|
| δ | 0.05 |
| Neighbourhood | 8 |
| Baseline length | N = 400 states |
| Embedding | m = 3. τ = 1 for maps and stochastic processes; for flows τ = first local minimum of the average mutual information on one reference run (length 5000, seed 0), frozen in the P0 spec. |
| Distance | Euclidean between embedded states (`recplay`). |
| Theiler window | w = max((m − 1)τ + 1, τ_e). τ_e = smallest lag L ≥ 1 with autocorrelation below 1/e, searched for L ≤ N/10. If none exists, set the flag `too_persistent` and use w = (m − 1)τ + 1. |
| Input transform | `z`: z-score. `rank`: ranks divided by N + 1. Both are computed in P1–P3; Decision T after P3 keeps one. |
| Surrogates | `shuffle` (random permutation) and `iaaft` (`recplay.surrogates.iaaft`). A surrogate gets the same transform, m, τ, δ and the **w of its original**. |
| Surrogate statistics | `d_lin` = ρ_perc − mean over IAAFT surrogates; `pct_lin` = share of surrogates with a value ≤ the observed one. `d_shuf`, `pct_shuf` likewise. |
| Surrogate test | 39 surrogates; reject at the 5 % level when the observed value is the smallest or the largest of the 40. |
| Seeds | `np.random.default_rng([phase, cell, rep])`. No global seed, no reuse across phases. |
| Parallelism | joblib, 8 workers. Pass seeds and parameters to workers, never arrays or data frames. |

**Process panel.** `n` below is the series length that gives N states after embedding.

| Id | Process | Generator | Class |
|---|---|---|---|
| S1 | iid normal | `rng.standard_normal` | stochastic |
| S2 | AR(1), φ = 0.5 | `condpred.systems.ar1` | stochastic |
| S3 | AR(1), φ = 0.9 | `condpred.systems.ar1` | stochastic |
| S4 | AR(2), φ = (1.5, −0.75), period about 12 | new, in `rhoperc.processes` | stochastic |
| S5 | GARCH(1,1), persistence 0.97 | `condpred.systems.garch11` | stochastic |
| S6 | Markov-switching AR(1), φ = (0.2, 0.95) | `condpred.systems.ms_ar1` | stochastic |
| D1 | Logistic map, r = 4.0 | `condpred.systems.logistic` | map |
| D2 | Logistic map, r = 3.7 | `condpred.systems.logistic` | map |
| D3 | Hénon map, x coordinate | `condpred.systems.henon` | map |
| F1 | Rössler, x coordinate, sampling step 0.2 | `condpred.systems.rossler` | flow |
| F2 | Lorenz 63, x coordinate, sampling step 0.05 | `condpred.systems.lorenz63` | flow |
| X1 | Random walk | cumulative sum of S1 | out of scope |
| X2 | S3 plus one permanent level shift of 4 sd at the midpoint | new | out of scope |

X1 and X2 are run and reported but never counted in a gate: the measure is defined for series that revisit their
states.

## Article 5. Hard rules

1. **The definition does not change.** No new variant, no extra condition, no different edge rule, no tuned δ. If a
   result suggests the definition is flawed, stop and report (Part III, §3). Changing it is Tasos's decision alone.
2. **`rhoperc/core.py` is frozen after G0.** Its SHA-256 goes into `plans/rho_perc/FROZEN.json`. Every later
   `spec.json` records it. If the file changes, every later phase is void and is rerun.
3. **Spec before results.** Each phase writes `spec.json` (grid, seeds, sizes, the gate rule copied from this
   document, code fingerprint) with `condpred.report.write_spec` before computing any outcome. A spec is never edited;
   a changed spec gets a new version number and an entry in `DEVIATIONS.md`.
4. **Gates are computed by code** from the saved tables and written with `condpred.report.write_decision`. Nobody
   rewords a gate after seeing numbers.
5. **Smoke first.** `RHOPERC_SMOKE=1` shrinks every grid (2 reps, 3 surrogates, N ≤ 100) and writes to `smoke/`
   folders. Smoke numbers are for plumbing only and are never reported.
6. **One heavy job at a time.** Never two compute jobs at once, across all agents. Procedure in Part III, §2.
7. **Do not touch:** `src/recplay/**`, `src/condpred/**`, `src/rpjepa/**`, `src/dynimp/**` (import only);
   every existing notebook; `README.md`; `Writeups/**`; `Write-up/RRc_percolation_measure.tex`;
   `Write-up/percolation_explainer.png`; existing files in `data/raw/**`; anything `git status` shows as
   modified or untracked that this plan did not create.
8. **Reuse before writing.** Embedding, distances, RQA measures, IAAFT, test systems, noise and missing-data
   helpers, spec and table writers already exist (Appendix A). New code is limited to what Appendix A lists as new.
9. **No `rm -rf`** (it is blocked). To redo something, write to a new folder name.
10. **Read narrowly.** Do not read whole modules over 300 lines or whole log files. Use Appendix A and targeted
    searches.
11. **Every number in a report comes from a saved CSV** and names the file. A statement not backed by a saved table
    is marked `[INFERENCE]`.
12. **No novelty claims.** The only allowed wording is the "safe claim" in
    `Relevant Research/RP Connectedness - Literature Review.md`. No citation enters the paper unverified (P8).
13. **Do not end a turn with "next I will…".** Finish the phase's acceptance list or write a STOP file.

## Article 6. Gates

| Gate | After | Pass rule | If it fails |
|---|---|---|---|
| G0 | P0 | Every known-answer test passes. | STOP. |
| G1 | P1 | At least 6 of the 11 in-scope processes are "free" (not pinned to a bound). | 3–5 free: continue; the paper is limited to those classes. 0–2 free: STOP. |
| G2 | P2 | The IAAFT test rejects in at most 9 % of runs in every linear-null cell. | Up to one third of cells fail: continue and list them as limits. More: STOP. |
| G3 | P3 | At least one of: (a) logistic periodic-vs-chaotic AUC ≥ 0.90; (b) permanent-break Spearman ≥ 0.5; (c) power ≥ 0.5 in at least 3 nonlinear cells. | STOP. |
| G5 | P5 | Median unexplained share ≥ 0.20 **and** at least one task where adding ρ_perc raises AUC with a 95 % interval above 0. | STOP: the measure is redundant with existing ones. |

P4, P6 and P7 have no gate. They produce limits, a comparability rule and an illustration.

**Decision T (after P3).** Keep `rank` if its P3a AUC and its mean P3d power are each no more than 0.03 below those of
`z`; otherwise keep `z`. Written to `decision.csv` by code. P4–P8 use only the kept transform.

---

# Part II — Phases

Order is by importance: each phase can make the later ones pointless. Run them in this order, one at a time.
Each phase section lists Goal, Build, Run, Outputs, Acceptance. Time estimates assume 8 workers and Article 2, F9;
measure the real cost in smoke and extrapolate before every full run.

## P0. Definition and implementation

**Goal.** A correct, tested `rho_perc`, and a record of how it differs from the two wrong rules.

**Build** (`src/rhoperc/`; start from `prototype_cover.py`):
- `core.py`: `rho_perc(D, theiler, delta=0.05, connectivity=8)` returning a dict with `eps, rho, eps_iso, rho_iso,
  eps_lag, rho_lag, n, k, theiler`; `rho_perc_reference(D, theiler, delta)`. Both read **only the upper triangle of
  D at lag ≥ w** (`omega_ranks` leaves the rest at 0), bounds included.
- `prep.py`: `theiler_rule`, `distance` (transform → `recplay.embedding.embed` → `recplay.recurrence.euclidean_distances`),
  `measure` (one call from series to result dict), `surrogate_stats`.
- `processes.py`: the panel of Article 4, `generate(pid, n, rng)`.
- `selfcheck.py`: the tests below; `python -m rhoperc.selfcheck` prints PASS or FAIL per test and exits non-zero on
  any FAIL.

**Known-answer tests.**

| Test | Input | Required result |
|---|---|---|
| K1 | N = 100, all distances about 1 (tiny random differences), except D[1, N] = 0.01 | legacy rule < 0.001; `rho_perc` > 0.2 |
| K2 | x_t = t mod p, m = 1, N = 300, p ∈ {7, 40, 90}, w ∈ {1, 5} | ε_perc = 0 and ρ_perc = Σ_{ℓ ≥ w, p divides ℓ} (N − ℓ) / |Ω| to 1e-12 |
| K3 | 200 random cases: N 30–200, w ∈ {1, 3, 10}, δ ∈ {0.02, 0.05, 0.2}, neighbourhood 4 and 8, half with distances rounded to create ties | fast = reference (reference is 8-neighbour: add a 4-neighbour option to it for this test) |
| K4 | same cases | ε_iso ≤ ε_perc ≤ ε_lag |
| K5 | same cases | unchanged under D², log(1 + D), and time reversal of the series |
| K6 | same cases | 8-neighbour value ≤ 4-neighbour value |
| K7 | noise-free step: x = 0 for the first half, 1 after; m = 1 | ρ_perc = 1.0 exactly |
| K8 | constant series; and w > N/2 | 1.0; `ValueError` |
| K9 | joint plot: `np.maximum(omega_ranks(D1, w), omega_ranks(D2, w))` of two copies of the same series | equals ρ_perc of the single series |

**Run.** Self-check; then the comparison table: legacy, span and cover rule on the 13 panel processes, 20 runs each,
N = 400, w = (m − 1)τ + 1.

**Outputs.** `notebooks/tables/rho_perc/P0/` with `spec.json` (including the frozen τ of F1 and F2),
`rule_comparison.csv`, `decision.csv`; `plans/rho_perc/FROZEN.json`; `reports/REPORT_P0.md`.

**Acceptance.** All K tests PASS (G0). Fast method at N = 2000 takes under 5 s. About 5 minutes of compute.

## P1. Triviality audit — is ρ_perc just one of its bounds?

**Goal.** Find the classes of systems where ρ_perc carries something that ρ_iso and ρ_lag do not.

**Run.** 13 processes × 2 transforms × 2 Theiler settings (the rule; the minimum (m − 1)τ + 1) × 50 runs, N = 400.
Save ε and ρ for the measure and both bounds, plus `too_persistent`.

**Definitions.** A run is *pinned high* if ε_perc ≥ 0.95 ε_lag, *pinned low* if ε_perc ≤ 1.05 ε_iso (or both are 0).
A process is **free** if fewer than half of its runs are pinned, under the Theiler rule and the `z` transform.
Position π = (ρ_perc − ρ_iso) / (ρ_lag − ρ_iso).

**Outputs.** `runs.csv`; `by_process.csv` (share pinned high, share pinned low, median π, Spearman of ρ_perc with each
bound across runs, for each transform and Theiler setting); one figure: ρ_iso, ρ_perc, ρ_lag per process.

**Acceptance.** Gate G1 computed. The report names each process as free, pinned high or pinned low, and states in one
sentence what the Theiler rule changed. About 10 minutes.

## P2. Null behaviour and the surrogate test

**Goal.** Know what ρ_perc is on structureless data, and whether the IAAFT test can be trusted.

**P2a, iid data.** Marginals: normal, uniform, Student t with 3 degrees of freedom, lognormal, two-peaked (equal
mix of N(−2, 1) and N(2, 1)). N ∈ {100, 200, 400, 800, 1600}; m ∈ {1, 3}; both transforms. Runs: 200 for N ≤ 400,
100 at 800, 50 at 1600. Report median and spread per cell; the range of medians across marginals per transform; a
fit of the spread to a power of N for the uniform case; the large-N value for the uniform case.

**P2b, size of the test.** Null processes: S2, S3, S4, each as generated and after exp(·). N ∈ {200, 400}.
200 runs × 39 IAAFT surrogates, both transforms. (Under `rank`, exp(·) changes nothing: run those cells once.)
Report the rejection rate per cell.

**Outputs.** `iid.csv`, `iid_summary.csv`, `size.csv`, `size_summary.csv`, two figures.

**Acceptance.** Gate G2 computed. If cells fail, check one known cause before reporting: IAAFT assumes the series
wraps around, so a large gap between first and last value distorts it. Report the failure rate again on the subset of
runs whose end gap is below the median; do not change the test. About 45 minutes.

## P3. Known transitions and power

**Goal.** Show what the measure responds to, next to DET and LAM.

**P3a, logistic map.** r from 3.5 to 4.0 in steps of 0.001; N = 1000; m = 1 (primary) and m = 3; both transforms.
Per r: ρ_perc, ρ_iso, ρ_lag; DET, LAM, L from `recplay.rqa.rqa` at RR = 10 %
(`recplay.recurrence.threshold_by_rate`); Lyapunov exponent λ = mean of ln|r(1 − 2x)| over 100 000 iterates.
Scores: AUC for periodic (λ < 0) against chaotic, for each measure; Spearman with λ over chaotic r; a table of all
measures in a window of ±0.01 around r = 3.6786 (band merging), 3.8284 (period-3 window opens), 3.8568 (interior
crisis). Confirm each transition from the sign and size of λ, not from these numbers.

**P3b, breaks and regimes.** N = 400, 100 runs per cell, `d_lin` with 19 surrogates as well as raw values.
- Permanent break: S3 plus one level shift at the midpoint, size ∈ {0, 0.5, 1, 2, 4} sd. Stated expectation:
  ρ_perc rises with the size.
- Recurring regimes: AR(1) 0.8 whose mean switches between two levels (stay probability 0.98), gap ∈ {0, 1, 2, 4} sd.
  No stated expectation (Article 2, F8). Report the trend either way.

**P3c, power.** Alternatives: D1, D3, F1 each with observational noise of 0, 0.25, 0.5, 1.0 sd
(`condpred.contaminate.add_obs_noise`); S5; S6. N = 400; 60 runs × 39 IAAFT surrogates; both transforms. Test
statistics: ρ_perc, DET, LAM, each against the same surrogates. Report the rejection rate per cell and statistic.

**Outputs.** `logistic.csv`, `logistic_scores.csv`, `breaks.csv`, `power.csv`, `power_summary.csv`, three figures
(the logistic figure in the layout of the standard RQA benchmark: λ on top, one panel per measure below).

**Acceptance.** Gate G3 and Decision T computed. The report states plainly which of DET, LAM, ρ_perc is best in each
part. About 60 minutes.

## P4. Scaling and sensitivity

**Goal.** A rule for when two values may be compared.

**Run.** Kept transform only. Processes S1, S3, S4, S5, D1, D3, F1, F2. Vary one setting at a time from Article 4:

| Setting | Values | Runs |
|---|---|---|
| N | 100, 200, 400, 800, 1600, 3200 | 50 up to 800; 20 at 1600; 10 at 3200 |
| δ | 0.02, 0.05, 0.10, 0.20 | 50 |
| Theiler window | 1; the rule; twice the rule | 50 |
| m | 1, 2, 3, 5 | 50 |
| Neighbourhood | 4, 8 | 50 |
| Sampling step (F1, F2 only) | ×0.5, ×1, ×2, ×4, once at fixed N and once at fixed time span | 30 |

Also `d_lin` (19 surrogates, 30 runs) at N ∈ {100, 200, 400, 800}.

**Outputs.** `sensitivity.csv`, `sensitivity_summary.csv` (median, quartiles), one figure per setting.

**Acceptance.** The report contains (1) for each process the smallest N from which doubling N changes the median by
less than 0.02; (2) the same for `d_lin`; (3) one sentence per setting saying whether values at different levels of
it are comparable. About 40 minutes.

## P5. Incremental value

**Goal.** Decide whether ρ_perc says anything that existing measures do not.

**Competitor set C** (`rhoperc/competitors.py`), all on the same embedded states:
DET, LAM, L, ENTR, K2_lines at RR = 10 % (`recplay.rqa.rqa`); lag-1 autocorrelation; τ_e; ρ_iso; ρ_lag;
ρ_conn = RR at the largest edge of the minimum spanning tree of D (the threshold at which the recurrence network
becomes connected; `scipy.sparse.csgraph.minimum_spanning_tree`); share of recurrence points in the largest structure
at RR = 10 %.

**P5a, unexplained share.** 11 in-scope processes × 100 runs, N = 400. Within each process, predict ρ_perc from C with
5-fold cross-validation, once with ridge regression and once with gradient boosting (scikit-learn defaults).
U = 1 − max(R², 0), taking the larger R² of the two. Report U per process, the median, and the pooled value.
A high U alone proves nothing (noise is also unexplained); it only counts together with P5b.

**P5b, tasks.** Logistic regression on standardised features, 5-fold stratified cross-validation repeated 5 times.
ΔAUC = AUC(C plus ρ_perc) − AUC(C). Interval: 1000 bootstrap resamples of the rows.
- Task A: periodic against chaotic, the P3a rows (m = 1).
- Task B: S3 with against without one level shift of 1.5 sd at a random time in the middle half; 200 runs each.
- Task C: F1 with 0.25 sd noise against an IAAFT surrogate of the same run; 200 runs each.

**Outputs.** `features.csv`, `unexplained.csv`, `tasks.csv` (AUC of each single feature, of C, of C plus ρ_perc, ΔAUC
and its interval).

**Acceptance.** Gate G5 computed. The report lists, for each task, the best single competitor and its AUC next to
ρ_perc. About 20 minutes.

## P6. Robustness

**Goal.** Numbers for the Limits section.

**Run.** Processes S3, D1, F1 and X2 (shift 1.5 sd); N = 400; 50 runs; raw value and `d_lin` (19 surrogates).
- Observational noise: 0, 0.1, 0.25, 0.5, 1.0 sd.
- One outlier of 5 or 10 sd at a random interior time. Stated expectation: ρ_perc moves up to about ρ_iso.
- Missing values: 5 % and 10 % at random (`condpred.contaminate.mask_missing`), filled by linear interpolation.

**Outputs.** `robustness.csv`, `robustness_summary.csv` (median change against the clean run), one figure.

**Acceptance.** Tables complete; the report states the noise level at which the median changes by more than 0.05 and
whether the outlier expectation held. About 20 minutes.

## P7. Illustration on real data

**Goal.** One honest worked example. It does not gate the paper, and with 12 recessions it cannot prove much.

**Data.** `data/raw/fred/UNRATE.csv` (945 months, 1948-01 to 2026-09). Recession months: download FRED series
`USREC` to the new file `data/raw/fred/USREC.csv`; if there is no network, enter the NBER peak and trough dates by
hand in the spec and cite the NBER page.

**Run.** Kept transform; m and τ chosen once by the Article 4 rules on the full series and frozen in the spec.
- Full sample: ρ_perc, ρ_iso, ρ_lag and `pct_lin` (99 surrogates), for the level and for the monthly change.
- Rolling: windows of 240 months, step 12; per window ρ_perc, `pct_lin` (99 surrogates), ρ_iso, largest absolute
  monthly change, standard deviation.
- Stated question: are values higher in windows whose last 12 months contain a recession month? One-sided
  Mann–Whitney statistic; p-value from 2000 circular shifts of the recession indicator (the windows overlap, so the
  textbook p-value is invalid). Same test for each competitor column.

**Outputs.** `full_sample.csv`, `rolling.csv`, `recession_test.csv`, one figure (series with recessions shaded,
rolling ρ_perc and ρ_iso below).

**Acceptance.** Tables complete; the report gives the result without softening it. About 10 minutes.

## P8. Write-up package

**Goal.** Everything Tasos needs to write the paper, with every statement traceable.

1. `Write-up/rho_perc_v2/rho_perc.tex`: a copy of the old `.tex`, changed only where this plan requires:
   - Definition: add condition 3 and the sentence explaining it (Article 3.2).
   - Algorithm: each structure also stores its largest row and smallest column; the stop test is the three conditions.
   - Figure 1: regenerate with `rhoperc/figures.py` under the cover rule; fix the caption's numbers.
   - "What to report next to it": ρ_iso, ρ_lag, `d_lin`, the Theiler rule.
   - "Where it could be useful" and "Limits": rewrite from P1, P3, P4, P6. Remove any claim a phase did not support.
   - "Related work": add the items of Appendix C that passed verification.
2. `plans/rho_perc/CLAIMS.md`: one row per factual sentence of the new text → file, column, row.
3. `plans/rho_perc/CITATIONS.md`: every new reference with a publisher or DOI link that was opened and matches
   authors, year, title and the claim it supports. Anything not verified is left out of the `.tex`.
4. `plans/rho_perc/README_rows_proposed.md` and `plans/rho_perc/recplay_promotion.md` (a proposed diff that moves
   `rho_perc` into `recplay.connectivity` and retires `rr_critical`; **not applied**).
5. If a LaTeX compiler is on the machine, build the PDF and copy it to `Writeups/rho_perc_v2.pdf` (new file).
   None was on PATH on 13 Nov 2026; if that is still so, deliver the `.tex` and say so.

**Acceptance.** Every row of `CLAIMS.md` resolves to a real table cell; no unverified citation in the `.tex`.

---

# Part III — Operating procedure

## §1 Roles

- **Orchestrator** (the main model): starts one phase agent at a time, in order, with this file and the previous
  phase reports as its only briefing; checks gates; talks to Tasos.
- **Phase agent**: a fresh agent per phase. Builds, runs smoke, runs the full job, writes the report.
- **Reviewer**: a different fresh agent after each phase. Works through §4 and writes
  `plans/rho_perc/reviews/REVIEW_P<k>_round<n>.md`. Fixes go to a new fix agent, then a new review round.

## §2 Running jobs

- Interpreter: `C:\Users\skous\AppData\Local\Programs\Python\Python312\python.exe` (numpy 2.2, scipy 1.16,
  pandas 2.3, scikit-learn 1.8, joblib, matplotlib, pytest; no numba, no statsmodels). Run from `RecPlay/` with
  `PYTHONPATH=src`. `.venv` lacks scipy: do not use it. Shell is bash: forward slashes, quote paths with spaces.
- Each experiment is a module with a `main()` under `if __name__ == "__main__":` (Windows starts workers by
  re-importing the file).
- Before any job: list running `python.exe` processes (PowerShell `Get-CimInstance Win32_Process`, called through a
  Python subprocess because the bash tool strips `$_`). If a compute job is running, wait for it.
- Start a long job synchronously with no timeout, output redirected to `run.log`. If the tool reports a timeout, the
  job is still running: do not start it again. Check the log's last lines only.
- Results are saved one file per cell under `data/processed/rho_perc/<phase>/v<spec version>/`; a rerun skips cells
  that exist. That makes every job resumable.
- **Run authority.** Agents run smoke always. Before a full run, tell Tasos the command, the log path and the time
  estimate from smoke. If the estimate is 45 minutes or less, proceed; if more, hand him the command and wait.

## §3 Stopping

Write `plans/rho_perc/STOP_P<k>.md` and stop work when: a gate says STOP; a known-answer test cannot be made to pass
without changing Article 3; a bound in 3.4 is violated in any run; two implementations disagree; or a phase needs
something this document forbids. The file states what was observed, the table that shows it, and nothing about how the
definition "could be fixed" beyond one short paragraph marked as a suggestion.

Smaller departures (a grid shortened for time, a failed download) go to `plans/rho_perc/DEVIATIONS.md` with date,
phase, what, why. Undeclared departures void the phase.

## §4 Review checklist

1. `spec.json` was written before the result tables (file times) and contains the gate rule verbatim.
2. `FROZEN.json` matches the current hash of `core.py`.
3. `python -m rhoperc.selfcheck` passes now.
4. Three rows picked at random from the main table are reproduced by a direct call with the recorded seed.
5. Every number in the report is found in the named CSV.
6. `decision.csv` follows from the tables by the rule in Article 6.
7. `git status` shows no change to the paths in Article 5, rule 7.
8. No smoke number appears in the report.

## §5 Report format

`plans/rho_perc/reports/REPORT_P<k>.md`, one page: what was run (grid and sizes), the gate result, three to six
findings each with its number and file, deviations, and what the result means for the paper in two sentences. Plain
words; no adjectives about the size of an effect without the number.

---

# Appendix A — API sheet

**Existing, import only.**

| Call | Returns |
|---|---|
| `recplay.embedding.embed(x, m, tau)` | states, shape (len(x) − (m − 1)τ, m) |
| `recplay.embedding.average_mutual_information(x, max_lag, bins=16)`, `first_local_minimum(values)` | AMI per lag; index of first minimum or None |
| `recplay.recurrence.euclidean_distances(states)` | N × N distance matrix |
| `recplay.recurrence.threshold_by_rate(d, rr, theiler=1)` | boolean RP at recurrence rate rr |
| `recplay.rqa.rqa(R)` | dict: RR, DET, L, ENTR, LAM, DET3_DET2, K2_lines |
| `recplay.surrogates.iaaft(x, rng, n_iter=200)` | one IAAFT surrogate |
| `recplay.connectivity.rr_critical(D, theiler, delta)` | legacy rule (comparison only) |
| `recplay.connectivity.omega_ranks(D, theiler)` | ranks within Ω, for joint plots |
| `condpred.features.percolation.rho_perc_span(D, theiler, delta)` | span rule (comparison only) |
| `condpred.systems`: `logistic(r, n, rng)`, `henon(n, rng)`, `lorenz63(n, rng, dt, sample_every)`, `rossler(n, rng, dt, sample_every)`, `ar1(phi, n, rng)`, `ms_ar1(phis, p_stay, n, rng)`, `garch11(persistence, n, rng)` | test systems; check each return shape with a three-line call, some return several columns |
| `condpred.contaminate`: `add_obs_noise(x, omega, rng)`, `mask_missing(x, frac, mode, block_len, rng)` | noisy copy; (x with NaN, observed mask) |
| `condpred.report`: `write_spec(spec, out_dir, modules=[...])`, `write_decision(rows, out_dir)`, `save_table(df, out_dir, name)`, `save_fig(fig, out_dir, name)` | read `report.py` lines 21–65 once before use |
| `condpred.stats.holm(pvalues)` | adjusted p-values |

**New, to be written** (`src/rhoperc/`): `core.py`, `prep.py`, `processes.py`, `competitors.py`, `selfcheck.py`,
`figures.py`, `experiments/p0.py … p7.py`. Nothing else.

# Appendix B — Layout

```
RecPlay/
├── src/rhoperc/                          new package
├── plans/rho_perc/                       this file, prototype_cover.py, FROZEN.json, DEVIATIONS.md,
│   ├── reports/  reviews/                CLAIMS.md, CITATIONS.md, STOP_*.md
├── notebooks/tables/rho_perc/<phase>/    spec.json, decision.csv, tables, figures/, smoke/
└── data/processed/rho_perc/<phase>/      per-cell caches
Write-up/rho_perc_v2/                     new .tex and figure (outside the repository)
```

# Appendix C — References to verify in P8

All of these were recalled from memory on 13 Nov 2026. None has been checked. Treat each as a lead, not a fact.

| Lead | Claim it would support |
|---|---|
| Site percolation with nearest and next-nearest neighbours on the square lattice has threshold ≈ 0.407 | Reference value for independent pixels (compare P2a) |
| Winkler (2000), "Dependent percolation and colliding random walks"; Balister, Bollobás, Stacey (2000), "Dependent percolation in two dimensions" | A recurrence plot of iid data is coordinate percolation, not independent site percolation |
| Alt and Godau (1995), Fréchet distance and the free-space diagram; Eiter and Mannila (1994), discrete Fréchet distance | The CRP version of ρ_perc is a known distance |
| Marwan, Thiel, Nowaczyk (2002), line of synchronisation in cross recurrence plots | Same |
| Eroglu, Marwan, Prasad, Kurths (2014), connectivity threshold of recurrence networks | Origin of the competitor ρ_conn (already cited in the old write-up) |
| Marwan, Wessel, Meyerfeldt, Schirdewan, Kurths (2002), logistic-map benchmark that introduced LAM | Layout and purpose of the P3a figure |
| Schreiber and Schmitz (1996), IAAFT; Theiler et al. (1992), surrogate data | The null hypothesis tested in P2b and P3c |

# Appendix D — Open questions for Tasos

Work does not wait for these; the default in brackets applies until he answers.

1. Article 3 replaces both the write-up's rule and the span rule recommended earlier. [Approved by adopting this plan.]
2. Run authority: agents run full jobs estimated at 45 minutes or less. [Yes.]
3. P7 uses US unemployment. A physics journal may prefer a physical or physiological record. [Unemployment only.]
4. Promotion of `rho_perc` into `recplay` and retiring `rr_critical`, which six `02_macro` notebooks and the
   `data/processed/rp_screening/*/rrc` caches use. [Not done; proposal file only.]
