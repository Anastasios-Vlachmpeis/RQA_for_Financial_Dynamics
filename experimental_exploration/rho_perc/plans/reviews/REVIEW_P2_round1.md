# REVIEW P2 round 1 — null behaviour and the surrogate test

**Verdict: ACCEPT.** Gate G2 result **CONTINUE_LIMITED (not PASS)** stands: 4 of 24 counted
cells reject above 9 %, i.e. 1/6 of cells, within the "up to one third" band of Article 6.
No STOP file is due. Checklist §4 items 1–8 all pass; two minor observations below do not
block acceptance and must not be "fixed" by rerunning under a changed spec.

## §4 checklist

1. **Spec before results — PASS.** `notebooks/tables/rho_perc/P2/spec.json` mtime
   2026-11-14 00:59:03 (its `written_at` = 2026-11-13T23:59:03Z agrees), before
   `iid.csv`/`iid_summary.csv` (01:02:22) and `size.csv` (01:09:59); the code
   `src/rhoperc/experiments/p2.py` (00:58:38) also predates the spec. The spec contains the
   G2 rule verbatim ("The IAAFT test rejects in at most 9 % of runs in every linear-null
   cell."), plus, stated before any run: the **counting rule over all 24 P2b cells**
   (as-generated and exp(·), both transforms, deduplicated exp+rank included),
   `gate_outcomes` (PASS / CONTINUE_LIMITED at ≤ 1/3 failing / STOP above), the rejection
   rule (`pct_lin == 0` or `== 1`), seeds, dedupe and end-gap definitions.
2. **FROZEN hash — PASS.** SHA-256 of `src/rhoperc/core.py` recomputed now =
   `3ad7c180…c636`, matching `plans/rho_perc/FROZEN.json` and both hash fields in spec.json.
3. **Selfcheck now — PASS.** `python -m rhoperc.selfcheck` (Python312, PYTHONPATH=src)
   rerun by this reviewer: K1–K9 PASS, exit 0.
4. **Three random rows reproduced — PASS (exact).** Picked with `random.seed(20261114)`:
   - `iid.csv` row 1049 (normal, N=200, m=1, rank, rep 49, seed [2,2,49]): regenerated the
     series and recomputed w=1, ρ_perc=0.384572864321608, ε, ρ_iso, ρ_lag — all match.
   - `size.csv` row 1973 (S3, as-generated, N=200, rank, rep 173): **full surrogate
     recompute** — replayed the run's rng through all 39 IAAFT surrogates; ρ_perc, ε,
     pct_lin=0.410256…, mean_surrogate, reject=False, end_gap all match to full precision.
   - `size.csv` row 4534 (S4, exp, N=400, z, rep 134): **full surrogate recompute including
     the preceding as-generated variant** (the exp surrogates draw from the rng after it);
     ρ_perc=0.990013…, pct_lin=1.0, reject=True, end_gap all match.
   Scope statement: surrogate recomputation was done for both size rows, not sampled or
   approximated; no other rows were rerun.
5. **Every report number in the CSVs — PASS.** Recomputed from `iid.csv`/`size.csv`:
   rejection rates 0.095 / 0.100 / 0.165 / 0.150 (19/200, 20/200, 33/200, 30/200) and the
   4/24 failing count; all other cells 0.035–0.085, all 12 rank cells in that range;
   `reject` equals `pct_lin ∈ {0, 1}` on all 4,800 rows, n_surrogates = 39 throughout.
   iid medians match `iid_summary.csv` to 1e-16; ranges of medians at N=1600: z 0.606 (m=1),
   0.538 (m=3), rank 0.0005 / 0.004; uniform medians (m=1: 0.368–0.376 across N and
   transforms; m=3 rising to 0.391 at N=1600) and power-law slopes −0.70 / −0.53 / −0.39 /
   −0.40 (m1-z, m1-rank, m3-z, m3-rank) all recompute as reported; end-gap subset rates
   0.080 / 0.090 / 0.180 / 0.160 recompute exactly (strictly below the cell median, 100 of
   200 runs each, test itself unchanged — the subset only re-averages the saved `reject`
   column); ρ_iso ≤ ρ_perc ≤ ρ_lag holds with 0 violations in all 19,800 rows.
   Grids complete: `iid.csv` 15,000 rows = 5 marginals × 5 N × 2 m × 2 transforms with
   200/200/200/100/50 reps at N=100/200/400/800/1600, no duplicate keys; `size.csv` 4,800
   rows = 24 cells × 200, no duplicate (process, variant, N, transform, rep) keys, of
   which 1,200 rows (6 exp+rank cells) are `deduped=True` copies, identical to their
   as-generated+rank sources on every shared field. Both figures exist and are non-empty
   (`figures/iid_rho_by_marginal.png`, `figures/size_rejection_by_cell.png`).
6. **decision.csv follows Article 6 — PASS.** Computed by code (`write_decision` in p2.py):
   PASS requires every cell ≤ 9 %; 4/24 = 0.1667 ≤ 1/3, so CONTINUE_LIMITED per "Up to one
   third of cells fail: continue and list them as limits. More: STOP." `decision.csv` has
   `pass=False`, result CONTINUE_LIMITED, counts all 24 cells, lists the four failing
   cells; the report's wording is "CONTINUE_LIMITED (not PASS)" and never claims PASS.
7. **Git forbidden paths — PASS.** `git status --short` shows no modified tracked files at
   all; only untracked entries (`src/rhoperc/**`, created by this plan, and pre-existing
   unrelated `src/dynimp/**`, `Writeups/dynimp_*`). Nothing under `src/recplay/**`,
   `src/condpred/**`, `src/rpjepa/**`, notebooks, `README.md` or `Write-up/**` is touched.
8. **No smoke numbers — PASS.** Smoke tables live only under `P2/smoke/` (N=100, 2 reps,
   rates 0.0/0.5); no such value appears in the report, and the report states this.

## Minor observations (not fixes; do not rerun)

- **Dedupe caveat, constitution-mandated.** The exp+rank copy is exactly what the
  constitution's P2b text orders ("Under rank, exp(·) changes nothing: run those cells
  once") and it was pre-registered. Note for the paper's honesty: the copied rows also
  carry the as-generated w and end_gap. The rank distance matrix given w is indeed
  unchanged by exp(·), but `theiler_rule` recomputed on exp(x) gives a different w in
  352 of 1,200 reps (checked on the computed z rows). The six deduped cells all reject at
  0.045–0.085, far from their as-generated z counterparts' failures, and any change here
  would need Tasos to amend the constitution — it is not a P2 defect.
- **Tie asymmetry in the rejection rule** is the pre-registered one: pct==1 counts an
  observed value tied for largest as a rejection, pct==0 requires it to be strictly
  smallest. Declared verbatim in the spec before results; leave as is.
- Finding 2's "0.369–0.376" for uniform m=1 is 0.368–0.376 in the CSV (0.3681 at N=100,
  rank). A 0.001 rounding slip in prose only; the CSV is authoritative.

## What was not done

Review only: no code or table was changed, no phase rerun, nothing in P3 touched.
