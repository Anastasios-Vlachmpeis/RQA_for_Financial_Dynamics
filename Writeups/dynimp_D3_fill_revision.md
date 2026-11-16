# dynimp D3: revised recurrence fill

## 1. What changed and why

The D1 overlap rule counted entries that can never be observed (the target channel inside the gap) in its denominator, rejected many usable donors and left gaps unfilled to be scored as zero. D3 separates the corrected denominator from the admission threshold (ctx10-ctx60), adds penalised distances that charge for entries a candidate does not share with the query (pen05, pen10), a fallback (pen10 at the same C, then linear interpolation) so no gap is left unfilled, a wider C grid, restricted averaging over C (avg20, pool_avg20), per-cell ECG weight for PPG, and a leave-one-out-style validation (V_loo) whose missing rate equals the test rate. The D1_D2_C_Rerun verdict was decided by a 3.5e-17 rounding difference; D3 therefore defines a cell whose every unit differs by less than 1e-9 as identical.

## 2. Data roles and prior use

Development: the 33 D1 train+val BIDMC patients and chaotic seeds 100-111. Final: the 20 D1 test patients and chaotic seeds 200-211; seeds 0-5 only for the gate re-score. Prior use of the development patients: patients 1, 2, 3, 5 were in the first throwaway probe (2026-11-12); the 50-60 % part of the 8 val patients was scored in the overlap-rule probe (their last 40 % was not); the other 22 train patients had never been filled by the recurrence fill. All 33 trained or selected the D1/D2 cross-patient learned models; no learned imputer is run on them here. Development results are reported for all 33 and for the 22 never-used patients in the cross-fitted estimate (section 4).

## 3. Development results and the chosen configuration

| rule | grid | mode | statistic (mean MAE ratio to base) |
|---|---|---|---|
| ctx20 | ext | avg20 | 0.8998 |
| ctx20 | ext | pool_avg20 | 0.9012 |
| ctx20 | d1 | avg20 | 0.9023 |
| ctx20 | d1 | pool_avg20 | 0.9069 |
| old_fb | ext | pool_avg20 | 0.9092 |
| old_fb | ext | avg20 | 0.9103 |
| ctx10 | ext | avg20 | 0.9110 |
| ctx10 | d1 | avg20 | 0.9117 |
| ctx10 | ext | pool_avg20 | 0.9134 |
| old_fb | d1 | avg20 | 0.9144 |

Chosen for all 10 cells: **ctx20 / d1 / avg20** (development statistic 0.9023). Per-cell choices (pooled C, averaged C set, ECG weight w) are frozen in `spec.json`.

## 4. Cross-fitted development estimate

| cell | mean paired MAE difference (selected - base), held-out folds | units |
|---|---|---|
| ecg|g25 | -0.0107 | 33 |
| ecg|g75 | -0.0081 | 33 |
| lorenz|n0.1|block | -0.0194 | 12 |
| lorenz|n0.1|point | -0.0072 | 12 |
| mackey_glass|n0.1|block | -0.0252 | 12 |
| mackey_glass|n0.1|point | -0.0120 | 12 |
| ppg|g187 | -0.0192 | 33 |
| ppg|g62 | -0.0320 | 33 |
| rossler|n0.1|block | -0.0085 | 12 |
| rossler|n0.1|point | -0.0092 | 12 |

Each fold's configuration, pooled C and w were selected on the other two folds only, so these differences are free of selection on the scored units. The writeup tables in the repository (`dev_crossfit.csv`) also give the 22 never-used patients separately.

## 5. Final result and verdicts

### chaotic: **adopt** (improved 6, worse 0, of 6 cells)

| cell | MAE final - base [95 % interval] | verdict |
|---|---|---|
| lorenz|n0.1|point | -0.0069 [-0.0078, -0.0060] | improved |
| lorenz|n0.1|block | -0.0201 [-0.0262, -0.0140] | improved |
| rossler|n0.1|point | -0.0079 [-0.0099, -0.0060] | improved |
| rossler|n0.1|block | -0.0116 [-0.0148, -0.0083] | improved |
| mackey_glass|n0.1|point | -0.0156 [-0.0193, -0.0119] | improved |
| mackey_glass|n0.1|block | -0.0290 [-0.0480, -0.0100] | improved |

### bidmc: **adopt** (improved 2, worse 0, of 4 cells)

| cell | MAE final - base [95 % interval] | verdict |
|---|---|---|
| ppg|g62 | -0.0232 [-0.0311, -0.0153] | improved |
| ppg|g187 | -0.0246 [-0.0373, -0.0119] | improved |
| ecg|g25 | -0.0065 [-0.0144, +0.0015] | neither |
| ecg|g75 | -0.0074 [-0.0227, +0.0079] | neither |

## 6. Attribution ladder

| step (against previous) | mean paired difference over cells | cells improved |
|---|---|---|
| final - ladder_cmode | -0.0032 | 2/10 cells improved |
| ladder_cmode - ladder_grid | -0.0031 | 6/10 cells improved |
| ladder_fallback - ladder_loo | -0.0049 | 6/10 cells improved |
| ladder_grid - ladder_rule | +0.0000 | 0/10 cells improved |
| ladder_loo - base | -0.0040 | 1/10 cells improved |
| ladder_rule - ladder_fallback | -0.0001 | 6/10 cells improved |

## 7. How much C matters

| cell | arm (final rule) | MAE |
|---|---|---|
| ecg|g25 | fixed_C6 | 0.3074 |
| ecg|g25 | fixed_C12 | 0.2283 |
| ecg|g25 | fixed_C25 | 0.1750 |
| ecg|g25 | fixed_C62 | 0.1419 |
| ecg|g25 | fixed_C125 | 0.1712 |
| ecg|g25 | fixed_C250 | 0.2105 |
| ecg|g25 | best_C_oracle | 0.1413 |
| ecg|g75 | fixed_C6 | 0.3309 |
| ecg|g75 | fixed_C12 | 0.2611 |
| ecg|g75 | fixed_C25 | 0.2154 |
| ecg|g75 | fixed_C62 | 0.2118 |
| ecg|g75 | fixed_C125 | 0.2138 |
| ecg|g75 | fixed_C250 | 0.2282 |
| ecg|g75 | best_C_oracle | 0.1923 |
| lorenz|n0.1|block | fixed_C1 | 0.1260 |
| lorenz|n0.1|block | fixed_C2 | 0.1183 |
| lorenz|n0.1|block | fixed_C3 | 0.1164 |
| lorenz|n0.1|block | fixed_C5 | 0.1145 |
| lorenz|n0.1|block | fixed_C10 | 0.1175 |
| lorenz|n0.1|block | fixed_C20 | 0.1347 |
| lorenz|n0.1|block | fixed_C40 | 0.2101 |
| lorenz|n0.1|block | fixed_C80 | 0.4132 |
| lorenz|n0.1|block | best_C_oracle | 0.1140 |
| lorenz|n0.1|point | fixed_C1 | 0.2138 |
| lorenz|n0.1|point | fixed_C2 | 0.1279 |
| lorenz|n0.1|point | fixed_C3 | 0.1097 |
| lorenz|n0.1|point | fixed_C5 | 0.1010 |
| lorenz|n0.1|point | fixed_C10 | 0.1022 |
| lorenz|n0.1|point | fixed_C20 | 0.1223 |
| lorenz|n0.1|point | fixed_C40 | 0.1680 |
| lorenz|n0.1|point | fixed_C80 | 0.3106 |
| lorenz|n0.1|point | best_C_oracle | 0.1009 |
| mackey_glass|n0.1|block | fixed_C1 | 0.6917 |
| mackey_glass|n0.1|block | fixed_C2 | 0.4906 |
| mackey_glass|n0.1|block | fixed_C3 | 0.2352 |
| mackey_glass|n0.1|block | fixed_C5 | 0.1997 |
| mackey_glass|n0.1|block | fixed_C10 | 0.1172 |
| mackey_glass|n0.1|block | fixed_C20 | 0.0977 |
| mackey_glass|n0.1|block | fixed_C40 | 0.0966 |
| mackey_glass|n0.1|block | fixed_C80 | 0.1048 |
| mackey_glass|n0.1|block | best_C_oracle | 0.0964 |
| mackey_glass|n0.1|point | fixed_C1 | 0.6076 |
| mackey_glass|n0.1|point | fixed_C2 | 0.6728 |
| mackey_glass|n0.1|point | fixed_C3 | 0.2668 |
| mackey_glass|n0.1|point | fixed_C5 | 0.2500 |
| mackey_glass|n0.1|point | fixed_C10 | 0.1086 |
| mackey_glass|n0.1|point | fixed_C20 | 0.0959 |
| mackey_glass|n0.1|point | fixed_C40 | 0.0968 |
| mackey_glass|n0.1|point | fixed_C80 | 0.1070 |
| mackey_glass|n0.1|point | best_C_oracle | 0.0957 |
| ppg|g187 | fixed_C4 | 0.2931 |
| ppg|g187 | fixed_C8 | 0.2860 |
| ppg|g187 | fixed_C15 | 0.2796 |
| ppg|g187 | fixed_C31 | 0.2795 |
| ppg|g187 | fixed_C62 | 0.2840 |
| ppg|g187 | fixed_C125 | 0.3077 |
| ppg|g187 | fixed_C250 | 0.3372 |
| ppg|g187 | best_C_oracle | 0.2634 |
| ppg|g62 | fixed_C4 | 0.2474 |
| ppg|g62 | fixed_C8 | 0.2396 |
| ppg|g62 | fixed_C15 | 0.2348 |
| ppg|g62 | fixed_C31 | 0.2335 |
| ppg|g62 | fixed_C62 | 0.2448 |
| ppg|g62 | fixed_C125 | 0.2654 |
| ppg|g62 | fixed_C250 | 0.2986 |
| ppg|g62 | best_C_oracle | 0.2224 |
| rossler|n0.1|block | fixed_C1 | 0.1414 |
| rossler|n0.1|block | fixed_C2 | 0.1293 |
| rossler|n0.1|block | fixed_C3 | 0.1212 |
| rossler|n0.1|block | fixed_C5 | 0.1158 |
| rossler|n0.1|block | fixed_C10 | 0.1008 |
| rossler|n0.1|block | fixed_C20 | 0.0948 |
| rossler|n0.1|block | fixed_C40 | 0.0962 |
| rossler|n0.1|block | fixed_C80 | 0.1027 |
| rossler|n0.1|block | best_C_oracle | 0.0948 |
| rossler|n0.1|point | fixed_C1 | 0.3751 |
| rossler|n0.1|point | fixed_C2 | 0.1818 |
| rossler|n0.1|point | fixed_C3 | 0.1245 |
| rossler|n0.1|point | fixed_C5 | 0.1021 |
| rossler|n0.1|point | fixed_C10 | 0.0970 |
| rossler|n0.1|point | fixed_C20 | 0.0953 |
| rossler|n0.1|point | fixed_C40 | 0.0976 |
| rossler|n0.1|point | fixed_C80 | 0.1038 |
| rossler|n0.1|point | best_C_oracle | 0.0953 |

Fixed-C rows show the error if every unit used that C; best_C_oracle picks each unit's best C on the test part itself and is descriptive only (an upper bound on what C choice could give).

## 8. Gates re-scored

D1 (original verdict PASS): re-scored with the final fill replacing `recurrence` -> **pass**. Descriptive only; the original verdict stands. The re-scored D2 gate is not a fair gate: the fill gets a second round of tuning here while SAITS and CSDI stay at their D2 budget.

D2 (original verdict INCONCLUSIVE): re-scored with the final fill replacing `recurrence` -> **inconclusive**. Descriptive only; the original verdict stands. The re-scored D2 gate is not a fair gate: the fill gets a second round of tuning here while SAITS and CSDI stay at their D2 budget.

## 9. Limits

- The 20 test patients are not unseen. D1, D2, the multi-C probe and D1_D2_C_Rerun were all scored on them, and the ideas tested here (restricted averaging, ECG weight) came partly from those scores. What this plan guarantees is that every numeric and structural choice is made on other patients. The cross-fitted development estimate is the evidence that does not depend on the test patients; if it and the final run disagree in sign in a cell, both are reported above.
- 11 of the 33 development patients were touched by earlier probes (section 2).
- Reproduction note: with the BLAS distance pass used for the candidate fills, one ECG unit (g25, patient 9) broke an exact donor tie differently from D1 (distances agreed to 3e-15; MAE differed by 5.6e-05). The base arm reported here is therefore computed with the original D1 fill function, against which the reproduction check passes for every unit; the candidate fills keep the BLAS pass, whose tie-breaking among exactly tied donors is arbitrary in the same way.
- The re-scored D2 gate is not a fair gate (section 8).
- After this run, the 20 test patients should not be used for further development of the fill. A later confirmation needs new patients (the VitalDB cases under data/vitaldb/snuadc/ may serve; not examined for this plan).
- V_loo lets donors come from after the validation gap, while at test time all donors precede the scored part.
