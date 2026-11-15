# dynimp D1/D2 rerun: context-only overlap rule and averaging over C

Post hoc reanalysis of the D1/D2 recurrence fill. Only the recurrence fill was rerun; every other arm is read from the stored D1/D2 tables. The original D1 verdict (PASS) and D2 verdict (INCONCLUSIVE) stand.

## 1. What changed in the method

D1 rule: a library window counts only if the entries observed in both it and the query are at least $0.3 \cdot F (g + 2C)$. The denominator includes the $g$ target-channel entries inside the gap, which are never observed in the query, so with one channel the rule requires $C \ge 0.21 g$; otherwise the gap is left unfilled and scored as zero fill. Worked example: one channel, $g = 187$, $C = 15$: the window has $217$ entries, $187$ of which can never be observed in the query, so at most $30$ entries can be counted and the threshold $0.3 \times 217 = 65.1$ can never be reached — every candidate is rejected. The context-only rule counts only entries that can be observed: $\mathrm{cnt} \ge f \,(F (g + 2C) - g)$, tested at $f = 0.1$ and $0.2$. The second change averages the fills from all $C$ values in the grid instead of picking one $C$ on validation. Donor count is $k = 8$ throughout, with $k = 32$ selection arms as a control (as many donors as averaging 8 donors over the 4 $C$ values).

## 2. Where the two changes came from

The context-only rule comes from a validation-only probe (D1 validation patients and chaotic validation segments): the D1 rule rejected 46–58 % of library candidates in the chaotic cells and 62 % in PPG-only g187, and the context-only rule at 0.1–0.2 lowered validation MAE by 0.06–0.13 in all six chaotic cells and by 0.04–0.05 in PPG-only, with no difference for PPG+ECG and a tie on ECG. The averaging idea comes from a probe that was scored on the D1 test patients, so it was not formed blind to test data; averaging lowered MAE by 0.010–0.019 in 3 of 4 physiological cells there.

## 3. Design-stage choice of the primary arm

| arm | mean validation MAE over cells |
|---|---|
| rec_w03_sel | 0.2063 |
| rec_w03_avg | 0.2047 |
| rec_c01_sel | 0.1597 |
| rec_c01_avg | 0.1894 |
| rec_c02_sel | 0.1616 |
| rec_c02_avg | 0.1732 |
| rec_c01_k32_sel | 0.1608 |
| rec_c02_k32_sel | 0.1622 |

Primary arm (lowest candidate, k = 32 controls not eligible): **rec_c01_sel**. This was frozen in `spec.json` before any test data were scored.

## 4. Results per cell

| cell | D1 recurrence MAE | rec_c01_sel MAE | paired difference [95 % interval] |
|---|---|---|---|
| lorenz|n0.1|point | 0.1039 | 0.1012 | -0.0027 [-0.0052, -0.0002] |
| lorenz|n0.1|block | 0.1250 | 0.1162 | -0.0088 [-0.0182, +0.0007] |
| rossler|n0.1|point | 0.1017 | 0.0950 | -0.0067 [-0.0087, -0.0046] |
| rossler|n0.1|block | 0.1055 | 0.0952 | -0.0103 [-0.0194, -0.0013] |
| mackey_glass|n0.1|point | 0.1063 | 0.0966 | -0.0097 [-0.0154, -0.0040] |
| mackey_glass|n0.1|block | 0.1336 | 0.1005 | -0.0331 [-0.0648, -0.0013] |
| ppg|g62 | 0.2510 | 0.2510 | +0.0000 [+0.0000, +0.0000] |
| ppg|g187 | 0.2947 | 0.2947 | +0.0000 [-0.0000, +0.0000] |
| ecg|g25 | 0.1469 | 0.1445 | -0.0025 [-0.0111, +0.0062] |
| ecg|g75 | 0.2021 | 0.2153 | +0.0132 [-0.0037, +0.0301] |

Verdict: **inconclusive** (improved 5, worse 1; rule: adopt iff improved in at least 5 of 10 cells and worse in none; reject iff worse in at least 2 cells; otherwise inconclusive).

Note on ppg|g62: the paired difference is 3.54e-17 (interval [2.12e-17, 4.96e-17]), i.e. zero at the precision of the stored D1 table; the arm's fills are identical to the D1 fills there (the paired difference against the recomputed D1-rule arm is exactly 0). It is counted by the frozen rule above as written.

All arms, paired MAE difference against D1 recurrence (negative is better):

| cell | rec_w03_sel | rec_w03_avg | rec_c01_sel | rec_c01_avg | rec_c02_sel | rec_c02_avg | rec_c01_k32_sel | rec_c02_k32_sel |
|---|---|---|---|---|---|---|---|---|
| lorenz|n0.1|point | +0.0000 | +0.0065 | -0.0027 | +0.0038 | -0.0034 | +0.0040 | -0.0051 | -0.0059 |
| lorenz|n0.1|block | +0.0000 | +0.0110 | -0.0088 | +0.0002 | -0.0120 | -0.0004 | -0.0091 | -0.0118 |
| rossler|n0.1|point | +0.0000 | -0.0079 | -0.0067 | -0.0094 | -0.0055 | -0.0096 | -0.0088 | -0.0078 |
| rossler|n0.1|block | +0.0000 | -0.0092 | -0.0103 | -0.0046 | -0.0107 | -0.0098 | -0.0107 | -0.0109 |
| mackey_glass|n0.1|point | +0.0000 | -0.0064 | -0.0097 | +0.0776 | -0.0096 | +0.0075 | -0.0125 | -0.0129 |
| mackey_glass|n0.1|block | +0.0000 | -0.0371 | -0.0331 | +0.0548 | -0.0344 | -0.0206 | -0.0364 | -0.0381 |
| ppg|g62 | +0.0000 | -0.0037 | +0.0000 | -0.0037 | +0.0000 | -0.0037 | +0.0071 | +0.0071 |
| ppg|g187 | +0.0000 | -0.0101 | +0.0000 | -0.0101 | +0.0000 | -0.0101 | +0.0011 | +0.0011 |
| ecg|g25 | +0.0000 | -0.0157 | -0.0025 | +0.0253 | +0.0020 | +0.0085 | +0.0049 | +0.0043 |
| ecg|g75 | +0.0000 | -0.0191 | +0.0132 | +0.0109 | +0.0125 | -0.0013 | +0.0156 | +0.0071 |

## 5. Attribution: rule vs averaging vs donor count

| effect | comparison | mean paired difference over cells | cells improved |
|---|---|---|---|
| rule effect (context-only sel − window sel) | rec_c01_sel − rec_w03_sel | -0.0060 | 5/10 cells with interval below 0 |
| rule effect (context-only sel − window sel) | rec_c02_sel − rec_w03_sel | -0.0061 | 6/10 cells with interval below 0 |
| averaging effect (avg − sel, same rule) | rec_c01_avg − rec_c01_sel | 0.0205 | 2/10 cells with interval below 0 |
| averaging effect (avg − sel, same rule) | rec_c02_avg − rec_c02_sel | 0.0026 | 3/10 cells with interval below 0 |
| averaging effect (avg − sel, same rule) | rec_w03_avg − rec_w03_sel | -0.0092 | 6/10 cells with interval below 0 |
| donor-count control (avg − k32 sel, same rule) | rec_c01_avg − rec_c01_k32_sel | 0.0199 | 2/10 cells with interval below 0 |
| donor-count control (avg − k32 sel, same rule) | rec_c02_avg − rec_c02_k32_sel | 0.0032 | 2/10 cells with interval below 0 |

If the k = 32 controls match the `_avg` arms, the averaging gain comes from more donors, not from mixing context lengths.

## 6. Overlap statistics

| cell | rule | share of candidates rejected (pooled over C) | unfilled / gaps |
|---|---|---|---|
| ecg|g25 | context_0.1 | 0.001 | 1/15148 |
| ecg|g25 | context_0.2 | 0.006 | 18/15148 |
| ecg|g25 | window_0.3 | 0.057 | 273/15148 |
| ecg|g75 | context_0.1 | 0.001 | 2/5144 |
| ecg|g75 | context_0.2 | 0.004 | 3/5144 |
| ecg|g75 | window_0.3 | 0.318 | 1412/5144 |
| lorenz|n0.1|block | context_0.1 | 0.001 | 0/10268 |
| lorenz|n0.1|block | context_0.2 | 0.023 | 1/10268 |
| lorenz|n0.1|block | window_0.3 | 0.254 | 65/10268 |
| lorenz|n0.1|point | context_0.1 | 0.000 | 0/66656 |
| lorenz|n0.1|point | context_0.2 | 0.006 | 0/66656 |
| lorenz|n0.1|point | window_0.3 | 0.107 | 158/66656 |
| mackey_glass|n0.1|block | context_0.1 | 0.008 | 0/3496 |
| mackey_glass|n0.1|block | context_0.2 | 0.045 | 7/3496 |
| mackey_glass|n0.1|block | window_0.3 | 0.289 | 161/3496 |
| mackey_glass|n0.1|point | context_0.1 | 0.001 | 0/22048 |
| mackey_glass|n0.1|point | context_0.2 | 0.019 | 0/22048 |
| mackey_glass|n0.1|point | window_0.3 | 0.260 | 140/22048 |
| ppg|g187 | context_0.1 | 0.000 | 0/1000 |
| ppg|g187 | context_0.2 | 0.000 | 0/1000 |
| ppg|g187 | window_0.3 | 0.000 | 0/1000 |
| ppg|g62 | context_0.1 | 0.000 | 0/3144 |
| ppg|g62 | context_0.2 | 0.000 | 0/3144 |
| ppg|g62 | window_0.3 | 0.000 | 0/3144 |
| rossler|n0.1|block | context_0.1 | 0.002 | 2/10412 |
| rossler|n0.1|block | context_0.2 | 0.026 | 3/10412 |
| rossler|n0.1|block | window_0.3 | 0.264 | 88/10412 |
| rossler|n0.1|point | context_0.1 | 0.000 | 0/66424 |
| rossler|n0.1|point | context_0.2 | 0.007 | 0/66424 |
| rossler|n0.1|point | window_0.3 | 0.110 | 211/66424 |

## 7. D1 and D2 gate tables re-scored

original D1 verdict: PASS; re-scored with each arm replacing `recurrence` (post hoc reanalysis):

| arm used as recurrence | D1 verdict |
|---|---|
| rec_w03_sel | pass |
| rec_w03_avg | fail |
| rec_c01_sel | pass |
| rec_c01_avg | fail |
| rec_c02_sel | pass |
| rec_c02_avg | pass |
| rec_c01_k32_sel | pass |
| rec_c02_k32_sel | pass |

original D2 verdict: INCONCLUSIVE; re-scored with each arm replacing `recurrence` (post hoc reanalysis):

| arm used as recurrence | D2 verdict |
|---|---|
| rec_w03_sel | inconclusive |
| rec_w03_avg | inconclusive |
| rec_c01_sel | inconclusive |
| rec_c01_avg | inconclusive |
| rec_c02_sel | inconclusive |
| rec_c02_avg | inconclusive |
| rec_c01_k32_sel | inconclusive |
| rec_c02_k32_sel | inconclusive |

## 8. What changes in the D1/D2 conclusions

Under the pre-registered comparison rule the primary arm rec_c01_sel is judged **inconclusive** against D1 recurrence (improved 5, worse 1). The re-scored gate tables above show whether either original gate outcome would have changed with the revised fill; the original verdicts stand regardless, because the arms and the primary arm were chosen after D1/D2 were run.

## 9. Limits

- The averaging-over-C idea was formed on the D1 test patients (section 2); its test-set gain here is not independent confirmation.
- The re-scored D1/D2 gates reuse the stored learned-imputer rows; nothing about SAITS/CSDI training was rerun, so any D2 budget caveat (CSDI caps) carries over unchanged.
- Chaotic scoring is MAE over all channels, as in D1; RP error is reported for the physiological cells only.
- One distance pass per (gap, C) is shared by all arms, but the fill is still a CPU nearest-neighbour search over the patient's own history; no timing claim beyond the D1/D2 compute comparison is made.
