# Recurrence Geometry for Learning Financial Dynamics

This is a research project on volatility forecasting. The question is whether a model can learn more from the same financial history when similarity between past market states is measured by β-divergence rather than Euclidean distance.

A recurrence plot (RP) marks pairs of times at which the observed states are similar. The project compares three inputs: the raw time series, its Euclidean RP and its β-divergence RP. The comparison uses the same observations and lookback for all three, the same 2D CNN for both plots, and a 1D CNN with comparable tuning on the raw series. Evaluation is chronological, with overlapping windows kept apart across splits.

## Volatility and measurement noise

The simulation measures how much recurrence structure survives noise. The financial question is whether preserving that structure improves a forecast. The repository does not report a matched raw-series, Euclidean-RP and β-RP comparison on future realized variance.

- **Simulation.** [E3: contamination robustness](notebooks/07_conditional_predictability/E3_contamination_robustness.ipynb) generates five-minute returns with persistent calm and stressed periods. It adds price jumps and observation noise separately and together, so each observed plot can be compared with a clean plot from the same simulated market. At 0.05% price noise, β = 5 retained more of its clean recurrence pairs than β = 0 (a 0.169 increase in Jaccard overlap). Under jumps, using bipower variation instead of realized variance preserved more structure than changing β.
- **Financial forecasts.** The [Arratia–Sepúlveda reproduction](notebooks/01_reproductions/EucRPRepro.ipynb) compares raw monthly windows with Euclidean RPs for the sign of the S&P 500 equity premium. Walk-forward AUC was 0.546 for the raw CNN and 0.479 for the RP-CNN. This is a comparison on return direction, not volatility. [E4](notebooks/07_conditional_predictability/E4_har_rv_loss.ipynb) uses Oxford–Man daily realized variance and a HAR volatility forecast. Its target is the *error* of that forecast, not next-day variance itself; only the smoke run is available.
- **β-divergence on returns.** The [Nasdaq-100 benchmark](notebooks/03_equity_indices/NDX_60d_5min_RP_benchmark.ipynb) builds Euclidean and β-RPs from identical five-minute gross returns, `P[t] / P[t-1]`. These are positive, as required by the β implementation, while values below and above one distinguish losses from gains. The benchmark measures plot construction time, not predictive accuracy.

## Methods and data

[Shared RP and RQA code](src/recplay/) implements delay embedding, Euclidean distance, symmetric β-divergence, recurrence-rate thresholds and recurrence quantification analysis (RQA). The [foundations notebooks](notebooks/00_foundations/) illustrate how embedding and thresholds change a plot. The [macro notebook](notebooks/02_macro/CPI_UNRATE_FEDFUNDS_RR.ipynb) uses individual and joint RPs to describe recurrence across inflation, unemployment and the federal funds rate; it does not forecast them. The [predictability notebooks](notebooks/04_predictability/) test how well RQA measures track predictability in systems with known dynamics.

Source data and generated tables are not tracked. The equity-premium reproduction uses the monthly Goyal–Welch workbook; E4 uses the archived Oxford–Man Realized Library. The [simulation and evaluation code](src/condpred/) is shared by E3 and E4.

## References

**Recurrence geometry**

- Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). [Recurrence plots for the analysis of complex systems](https://doi.org/10.1016/j.physrep.2006.11.001). *Physics Reports*, 438, 237–329.
- Deckert, E., Dreesen, P., Marwan, N., & Boussé, M. (2025). [Beta-divergence-based recurrence plots for audio time-series analysis](https://doi.org/10.23919/EUSIPCO63237.2025.11226301). *EUSIPCO*, 291–295. The source for the β-RP method; the application in this paper is audio, not finance.
- Romano, M. C., Thiel, M., Kurths, J., & von Bloh, W. (2004). [Multivariate recurrence plots](https://doi.org/10.1016/j.physleta.2004.07.066). *Physics Letters A*, 330, 214–223.

**Financial applications of recurrence plots**

- Arratia, A., & Sepúlveda, E. (2020). [Convolutional neural networks, image recognition and financial time series forecasting](https://doi.org/10.1007/978-3-030-37720-5_5). In *Mining Data for Financial Applications*, 60–69. The equity-premium study reproduced in `EucRPRepro`.
- Shabani, M., Magris, M., Tzagkarakis, G., Kanniainen, J., & Iosifidis, A. (2023). [Predicting the state of synchronization of financial time series using cross recurrence plots](https://doi.org/10.1007/s00521-023-08674-y). *Neural Computing and Applications*, 35, 18519–18531.

**Financial data and forecast evaluation**

- Welch, I., & Goyal, A. (2008). [A comprehensive look at the empirical performance of equity premium prediction](https://doi.org/10.1093/rfs/hhm014). *Review of Financial Studies*, 21, 1455–1508. The monthly predictors in the reproduction come from the updated dataset.
- Patton, A. J. (2011). [Volatility forecast comparison using imperfect volatility proxies](https://doi.org/10.1016/j.jeconom.2010.03.034). *Journal of Econometrics*, 160, 246–256. Basis for using QLIKE to compare variance forecasts when realized variance is a noisy proxy.
