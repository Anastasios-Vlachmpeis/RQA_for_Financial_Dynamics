# STOP P5 — measure redundant (Gate G5 failed)

Gate rule: Median unexplained share ≥ 0.20 and at least one task where adding ρ_perc raises AUC with a 95 % interval above 0.

Median unexplained share U = 0.5616 (need >= 0.20). Pooled U = 0.1009.

- TaskA: deltaAUC = 0.0001, 95% CI [-0.0000, 0.0006]
- TaskB: deltaAUC = 0.0053, 95% CI [-0.0018, 0.0176]
- TaskC: deltaAUC = -0.0002, 95% CI [-0.0015, 0.0000]

Evidence: notebooks/tables/rho_perc/P5/unexplained.csv and tasks.csv. See plans/rho_perc/reports/REPORT_P5.md.
