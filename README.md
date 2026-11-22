# RecPlay

Recurrence plots (RPs) and recurrence quantification analysis (RQA) on financial and macroeconomic time series.
Part of the MaRBLe RQA research project (Tracks 2, 4 and 5: new RQA measures, RPs built with different
distance/divergence measures, and RPs as inputs to deep-learning models). The central question is whether
turning a time series into an RP improves outcome modelling over using the raw series.

## Research questions

- There is enough data in a single trading day to build RPs. Which combinations of data (price, volume,
  rolling volatility, indicators, rolling averages) give recurrence structure that carries useful information?
- Can an RP capture enough information to be used as input to a machine-learning model?
- Can entropy measures be used, and if so, how?
- Can RPs model outcomes probabilistically, rather than classifying whether the next move is up or down?

## Setup

Python 3.12. Third-party packages used by the notebooks:

```
pip install numpy pandas scipy matplotlib scikit-learn torch yfinance openpyxl jupyter
```

`torch` is needed for the reproduction notebooks; `EucRPRepro` expects a CUDA device.
`Stock_trend_RQA_transposed_transformer` has its own pinned
`notebooks/01_reproductions/requirements-stock-trend.txt` and Python 3.12 environment
(`.venv-stock-repro312`, kernel **Stock trend reproduction**). Its setup cells document the
separate CUDA PyTorch installation. Historical execution requires `TUSHARE_TOKEN` and
Tushare historical-minute, calendar and daily-data permissions. `STOCK_REPRO_MODE=smoke`
explicitly exercises synthetic minute bars; those outputs are not a financial replication.
Its implementation is split into classes in the sibling
`notebooks/01_reproductions/Stock_trend_RQA_transposed_transformer/` package:
`ExperimentConfig`, `DataService`, `RQAExtractor`, PyTorch model classes, `Trainer`,
`Reporter` and `ReproductionChecks`. The notebook contains imports and orchestration;
restart its kernel after editing the modules. Cache identities include module sources.

See [Experimental exploration](experimental_exploration/) for JEPA setup and the archived experiment
findings, notebooks, and run commands.


Run notebooks from their own folders. Shared raw data lives under `data/raw/`.

## Layout

```
RecPlay/
├── src/recplay/          shared RP / RQA code
├── src/rpjepa/           shared neural encoders
├── experimental_exploration/  JEPA pretraining, TORPs, rho_perc and CPI direction
├── notebooks/
│   ├── 00_foundations/     exercises and first experiments
│   ├── 01_reproductions/   reproductions of published results
│   ├── 02_macro/           US macro series
│   ├── 03_equity_indices/  stock indices and RP computation benchmarks
│   └── 04_predictability/  RQA measures as predictability gauges
├── data/raw/             source data, not tracked in git
├── data/processed/       notebook caches, not tracked in git
└── results/tables/       tables written by notebooks
```

## Notebooks

| Folder | Notebook | What it does | Data |
|---|---|---|---|
| 00 | `ResearchBook` | Supervisor exercises: ε choice, noisy and random RPs, joint recurrence rates, the D₃ triple-dependence score | synthetic |
| 00 | `XOR_recurrence_demo` | Z = X ⊕ Y: dependence visible in the triple recurrence but not in any pair | synthetic |
| 00 | `recplay` | First distance matrix and thresholded RP on intraday prices; 3D recurrence onset surface over continuous ε (up to RR = 100 %) for the full series and a 20-point chunk, with timings | Yahoo, `CSG.AS` 2-min |
| 00 | `recplay_3d_viewer.py` (script) | Rotatable 3D onset surfaces for every sliding window of the day (default 20 samples, m = 2, τ = 1), slider / arrow keys to step through time, price panel shading the current window; `--help` for ticker, dates, window length, step, ε, fixed z scale | Yahoo, `CSG.AS` 2-min |
| 01 | `EucRPRepro` | Reproduction of Arratia & Sepúlveda: Euclidean RPs + CNN for the sign of the S&P 500 equity premium | Goyal–Welch 2025 |
| 01 | `Stock_trend_RQA_transposed_transformer` | Su et al. (2025): 13 RQA features + variable-token Transformer, all six named baselines, branch removals and six RQA-input arms; chronological selection/refit and paper comparison tables | Tushare Pro, seven A-share stocks, 1-minute OHLCV, 2018–2022; token/permissions required |
| 02 | `CPI_UNRATE_FEDFUNDS_RR` | Inflation, unemployment and fed funds rate, 1954–2026: individual, pairwise and triple joint RPs; independence ratios, Theiler window, joint covariance/correlation; β-divergence RPs | FRED |
| 03 | `RUT_SP500_NDX_RR` | Monthly log-return recurrences of the Russell 2000, S&P 500 and Nasdaq-100 | Yahoo, `^RUT` `^GSPC` `^NDX` |
| 03 | `NDX_2025_5min_RP_benchmark` | Sliding-window RP compute time, Euclidean vs β-divergence (β = 1, 3, 7), calendar year 2025 | 5-min `^NDX` from a local CSV (not supplied) |
| 03 | `NDX_60d_5min_RP_benchmark` | Same benchmark on the last 60 days available from Yahoo | Yahoo, 5-min `^NDX` |
| 04 | `DETPredictability` | Whether DET combined with a second RQA measure tracks the actual predictability of chaotic and stochastic systems | synthetic (logistic map, Lorenz-63, …) |

## Library (`src/recplay`)

| Module | Contents |
|---|---|
| `embedding` | `embed`: time-delay embedding; `average_mutual_information`, `first_local_minimum` and `false_nearest_fraction` to choose τ and m |
| `recurrence` | distance matrices, symmetric β-divergence (any real β), ε set by a target recurrence rate (`threshold_by_rate`), single and joint recurrence matrices |
| `rqa` | diagonal/vertical line lengths, `rqa` measures, `k2_lines` |
| `connectivity` | `omega_ranks` for joint RPs |
| `surrogates` | `iaaft` surrogates |
| `windowed` | sliding-window RQA and current-state recurrence rate (`RR_last`); Euclidean on z or exp(z), or β-divergence distance |
| `analogue` | `analogue_frame`: the current state's recurrences with the whole past and the share of them followed by a rise h months later |
| `dependence` | independence ratios and joint covariance/correlation for pairs and triples |
| `torp` | `cross_distances` between two state arrays; `perimeter_tensor` (sum of pairwise dissimilarities of three state arrays, any `distance`; a = b = c gives the single-series TORP); `threshold_tensor_by_rate`; `whitening_matrix` (Mahalanobis via Σ^(-1/2)); `same_time_stats` (same-time diagonal vs whole tensor) |

Conventions: Euclidean distance unless stated; ε chosen per series so the recurrence rate equals a target
(default 0.10); rates counted over upper-triangle pairs (i < j); a Theiler window is applied by only counting
pairs with j − i ≥ `min_sep`.

Dedicated JEPA tasks, training and probes live under [Experimental exploration](experimental_exploration/#jepa-pretraining).

## Data

`data/` is gitignored; recreate it as follows.

| Path | Source | Notes |
|---|---|---|
| `data/raw/fred/` | FRED CSV downloads: `CPIAUCSL`, `UNRATE`, `FEDFUNDS` (monthly) | Downloaded 18 Oct 2026. October 2025 CPI and UNRATE are blank: BLS did not collect them during the government shutdown. `CPI_UNRATE_FEDFUNDS_RR` interpolates and flags that month. |
| `data/raw/oecd_mei/` | FRED CSV downloads of OECD MEI CPI indices `<ISO>CPIALLMINMEI` (monthly, not seasonally adjusted) | Downloaded automatically by `CPI_direction_crosscountry` on first run (28 Oct 2026); most series end in early 2025, Japan in 2021 |
| `data/raw/goyal_welch/Goyal_Welch_2025.xlsx` | Goyal & Welch equity-premium predictor data, 2025 update, `Monthly` sheet | 1871–2025 |
| `data/raw/yahoo/` | yfinance timezone cache | Created automatically; prices are downloaded live and change between runs |
| `data/raw/yahoo/triplet_prices_2015_2024.csv` | yfinance daily adjusted closes of the [TORP triplet](experimental_exploration/torps/) tickers, US trading days | Written by `01_triplet_perimeter` on first run; delete to re-download |
| `data/raw/stock_trend_tushare/` | Tushare `stk_mins`, `trade_cal`, `daily` | Raw gzip CSV responses with metadata/SHA256 checks; no historical data supplied |
| `data/processed/stock_trend_repro/{full,smoke}/` | Stock-trend notebook | Code/input-fingerprinted RQA caches, epoch selections, neural checkpoints and predictions; synthetic smoke kept separate |

## Results

| File | Written by |
|---|---|
| `results/tables/det_predictability_results.csv` | `DETPredictability` |
| `results/tables/det_predictability_window_sweep.csv` | `DETPredictability` |
| `results/tables/stock_trend_repro/{full,smoke}/<run hash>/` | `Stock_trend_RQA_transposed_transformer`: predictions, seed/stock summaries, published-reference comparisons, ablations, coverage, manifest and diagnostic figures; full-mode `missing_data.csv` lists unavailable source ranges |

The original corner-based RRc could count a single corner pixel as a connection from start to finish. Its PDF and implementation were retired, along with the P0 runner that compared it against other definitions. The saved P0 results remain historical evidence; the separate cover-rule implementation and its negative added-value result are unaffected.

## References

- Marwan, N., Romano, M. C., Thiel, M., Kurths, J. (2007). Recurrence plots for the analysis of complex systems. *Physics Reports* 438, 237–329.
- Arratia, A., Sepúlveda, E. (2020). Convolutional neural networks, image recognition and financial time series forecasting. In *Mining Data for Financial Applications*, 60–69. DOI: 10.1007/978-3-030-37720-5_5.
- Theiler, J. (1986). Spurious dimension from correlation algorithms applied to limited time-series data. *Physical Review A* 34, 2427.
- Goyal, A., Welch, I. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies* 21, 1455–1508. Data updated annually.
- Schreiber, T., Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. *Physical Review Letters* 77, 635–638.
- Fraser, A. M., Swinney, H. L. (1986). Independent coordinates for strange attractors from mutual information. *Physical Review A* 33, 1134.
- Kennel, M. B., Brown, R., Abarbanel, H. D. I. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. *Physical Review A* 45, 3403.
- Federal Reserve Bank of St. Louis, FRED: https://fred.stlouisfed.org
