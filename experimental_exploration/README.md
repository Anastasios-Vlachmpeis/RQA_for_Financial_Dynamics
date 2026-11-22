# Experimental exploration

Dedicated notebooks, code, and findings are grouped by research direction. Generated tables, caches, and virtual environments stay local; shared library code and raw inputs stay in the main project.

## JEPA pretraining

[Notebooks and experiment code](jepa_pretraining/) · local tables: `jepa_pretraining/tables/step1` through `step6` · cache: `jepa_pretraining/cache/`.

| Notebook | Finding |
|---|---|
| [01 Synthetic benchmark](jepa_pretraining/01_synthetic_benchmark.ipynb) | Six synthetic tasks with trivial, RQA, hand-written, random-encoder, and supervised baselines. |
| [02 RP vs attention](jepa_pretraining/02_rp_vs_attention.ipynb) | At width 32, recall R² was 0.998 for RP + CNN, 0.763 for attention, and 0.006 for convolution, at about 140× the attention FLOPs per window. |
| [03 JEPA probes](jepa_pretraining/03_jepa_probes.ipynb) | RP-JEPA beat TS-JEPA on mean probe score (0.524 vs 0.477), but improved only 0.011 over the same untrained encoder. |
| [04 TORP and multiscale](jepa_pretraining/04_torp_multiscale.ipynb) | Generalised-cosine TORPs recovered sign (0.88–0.89 on two of three seeds); perimeter TORPs and pairwise distances were at chance. Multiscale JEPA targets did not improve mean probes (0.504 vs 0.510). |
| [05 Nearest-pattern rule](jepa_pretraining/05_nearest_pattern_rule.ipynb) | Nearest-pattern retrieval solved noise-free recall (R² ≥ 0.999); RP + CNN did not beat the best hand-written rule. |
| [06 Series-only RP](jepa_pretraining/06_series_only_rp.ipynb) | Removing distances reduced recall R² from 0.998 to 0.933 (0.973 with a deeper CNN); with 5,000 training windows the comparison was 0.970 with distances vs 0.000 without. |

The dedicated environment is `jepa_pretraining/.venv-jepa/`. To recreate it, run from `jepa_pretraining/` with Python 3.12:

```sh
python -m venv .venv-jepa
.venv-jepa/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-jepa/Scripts/python.exe -m pip install numpy scipy pandas scikit-learn matplotlib ipykernel nbformat
.venv-jepa/Scripts/python.exe -m ipykernel install --user --name rqa-jepa --display-name 'RQA JEPA (.venv-jepa)'
JEPA_SMOKE=1 .venv-jepa/Scripts/python.exe -X utf8 run_notebook.py 01_synthetic_benchmark.ipynb tables/step1/smoke/figures
```

The runner does not write back to the notebook. Full runs take hours and resume from cached results. Shared encoders remain in [`src/rpjepa`](../src/rpjepa/).

## TORPs

[Notebooks and dashboard builder](torps/) · [static dashboard](torps/dashboard/torp_triplets.html) · local tables: `torps/tables/`.

- [Return triplets](torps/01_triplet_perimeter.ipynb): same-time perimeter dissimilarity followed the pairwise-correlation prediction (Spearman 0.991). Thresholded same-time recurrence separated high mixed-sign from uncorrelated triplets (2.4× vs 1.3× their base rates).
- [Individual RQA](torps/02_individual_rqa_torp.ipynb): RQA-vector proximity tracked mean absolute correlation, not its sign (Spearman −0.81 for Mahalanobis dissimilarity, eight triplets).
- [Joint RQA](torps/03_joint_rqa_torp.ipynb): excess joint RR tracked absolute return correlation (Spearman 0.94 over 23 pairs).

Use the shared Python setup. Run notebooks from `torps/`, then run `python triplet_dashboard.py` there to regenerate the dashboard. The shared [TORP implementation](../src/recplay/torp.py) and Yahoo price inputs remain in place.

## `ρ_perc`

[Source](rho_perc/src/rhoperc/) · [phase reports](rho_perc/plans/reports/) · [stop decision](rho_perc/plans/STOP_P5.md) · local tables and caches in `rho_perc/tables/` and `rho_perc/cache/`.

The cover rule fixes the single-corner failure of the retired RRc definition. The added-value gate failed: median unexplained share was 0.562, but no tested task had a positive AUC gain with a 95% interval above zero. P0 is historical evidence; its retired runner is not restored.

Use the shared Python setup. From the repository root in Git Bash on Windows:

```sh
PYTHONPATH='experimental_exploration/rho_perc/src;src' python -m rhoperc.selfcheck
PYTHONPATH='experimental_exploration/rho_perc/src;src' RHOPERC_SMOKE=1 python -m rhoperc.experiments.p1
```

Use `:` rather than `;` in `PYTHONPATH` on POSIX Python. Historical plans and reports retain original paths: `plans/rho_perc/`, `notebooks/tables/rho_perc/`, and `data/processed/rho_perc/` now correspond to `rho_perc/plans/`, `rho_perc/tables/`, and `rho_perc/cache/` here.

## CPI direction

[Six historical CPI direction notebooks](cpi_direction/notebooks/) · local archived caches: `cpi_direction/cache/`.

The [cross-country confirmation](cpi_direction/notebooks/CPI_direction_crosscountry.ipynb) did not support the combined RQA/RRc feature group (12-month pooled ΔAUC −0.033; 95% interval −0.059 to −0.008). This does not establish a result for RQA alone. These are snapshots, not runnable experiments: their corner-based RRc dependency was retired. Shared FRED/OECD inputs remain under `data/raw/`. The separate [E9 model-error test](../notebooks/07_conditional_predictability/E9_cpi_model_error.ipynb) stays with conditional predictability; it tested when a reference forecast is wrong, not whether the direction forecast improves.
