# REVIEW P5 round 1 — ACCEPT (STOP stands)

**Gate logic, exact.** G5 (Art. 6, copied verbatim into `spec.json` and
`decision.csv`) needs median U ≥ 0.20 **and** ≥1 task with ΔAUC 95% CI lower
bound > 0. From `tasks.csv`, unrounded lower bounds: TaskA
−1.1102230246251565e-16, TaskB −0.0017521620083074563, TaskC
−0.0014501891507143732. TaskA's printed `[-0.0000, …]` is a genuinely negative
bound (−1 ulp-scale value), not a rounded positive one; no task passes.
Median U condition holds (below), so G5 FAIL and STOP is correct.

**Spec before results / frozen.** `spec.json` (v1) precedes all P5 tables by
file time on the disclosed cache-rerun; `write_spec` raises if the same-version
spec hash changes, so the rerun's identical spec is enforced by code.
G5 rule verbatim; preregistrations present: ρ_conn (largest MST edge of D,
RR over Ω lag ≥ w — matches `competitors.py:99-103` and the constitution),
TaskA = P3a rows (labels and λ in `features.csv` match `P3/logistic.csv`
m=1 exactly, 501 rows), pooled U (all 1,100 P5a rows, C only, no process
indicator), bootstrap on pooled OOF rows. `FROZEN.json` hash
3ad7c180…c636 equals current `core.py` SHA-256.

**Recomputation (Python312, own script from `features.csv`).**
- `unexplained.csv` arithmetic: U = 1 − max(R²_ridge, R²_gbr, 0) holds to
  1.1e-16 on every row; median of 11 per-process U = 0.5615962076274326
  (reported 0.5616); pooled U = 0.1009.
- Independent logistic-regression recomputation (standardised, 5-fold
  stratified, repeated 5×; 1000-row bootstrap of canonical seed-5 OOF
  predictions, percentile CI — method matches spec/code exactly):
  TaskA AUC(C)=0.9991386735572781, C+ρ=0.9992474726868851, Δ=0.00010879912960692906,
  CI [−1.1102230246251565e-16, 0.0005872689980802315];
  TaskB 0.887170 / 0.892465, Δ=0.005295, CI [−0.0017521620083074563, 0.017567624621142783];
  TaskC 0.997535 / 0.997335, Δ=−0.000200, CI [−0.0014501891507143732, 2.50058764784844e-05].
  All equal `tasks.csv` to the last digit. Best single competitors confirmed:
  TaskA K2_lines 0.9986, TaskB τ_e 0.8757, TaskC K2_lines 0.9870 — each above
  ρ_perc alone (0.9604 / 0.6279 / 0.7351).

**Other §4 checks.** `python -m rhoperc.selfcheck` passes now (K1–K9).
Five feature rows reproduced by direct `rho_perc`+`compute_competitors`
calls with recorded seeds (P5a S1 rep0, TaskA rep0, TaskB control rep0,
TaskC pair0 both rows): max diff 4.4e-16. `features.csv` = 2,401 rows
(1,100+501+400+400). Every report number is in its named CSV; no smoke
numbers. `git status`: no tracked forbidden path modified (only
plan-created `src/rhoperc/` and unrelated pre-existing untracked files).
`STOP_P5.md` states the observed gate failure and its tables, with no fix
suggestion (§3-compliant).

**Verdict.** ACCEPT. G5 FAIL is arithmetically and procedurally sound;
STOP_P5 stands.
