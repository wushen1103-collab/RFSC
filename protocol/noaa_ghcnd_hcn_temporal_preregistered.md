# NOAA GHCN-Daily HCN temporal graph-signal preregistration

This non-molecular experiment tests only the shared computational statement:
a graph signal is locally smooth but may change sharply across sparse spatial
boundaries. It does not assert that weather fronts and molecular activity
cliffs share a scientific cause.

The source is the official NOAA/NCEI GHCN-Daily v3.34 U.S. Historical
Climatology Network archive, pinned at update `3.34-upd-2026071718`. The archive
contains 1,219 station files, is 315,799,709 bytes, passed gzip validation, and
has SHA-256 `2d6742b0191bf15766b577963a47be1e7961cf14c4640a9d1d88e746b5f2b929`.
NOAA data are public-domain government information; NOAA and the GHCN-Daily DOI
must be credited. No third-party GraphDAU code is copied.

Freeze before station eligibility or performance inspection:

1. Retain contiguous-U.S. HCN stations with finite elevation and daily TMAX and
   TMIN whose quality flags are both blank. Temperature is their arithmetic
   mean in Celsius. Require at least 90% coverage independently in 2019--2022,
   2023, and 2024, and at least 300 eligible stations.
2. Use 2019--2022 for training, 2023 only for diagnostics, and 2024 once as
   test. The task is causal one-day-ahead forecasting. A prediction on date
   `d` may use observed temperature from `d-1` but never the response on `d` or
   any future date.
3. Build one response-free union 8-nearest-neighbor station graph using
   haversine distance, discard edges longer than 500 km, and weight remaining
   edges by `exp(-distance/median_distance)`.
4. Fit a four-harmonic per-station ridge climatology on training dates. Predict
   residuals with persistence, frozen Ridge, and frozen histogram gradient
   boosting; combine them with weights 0.50/0.25/0.25. Missing lags fall back
   to zero residual. The frozen Ridge and histogram boosting inputs are the
   previous-day station residual, weighted-neighbor mean residual,
   weighted-neighbor residual standard deviation, absolute station-neighbor
   residual gap (all divided by the global training residual scale), current
   day four-harmonic sine/cosine terms, and response-free station latitude,
   longitude, elevation, and graph degree. Ridge standardization is fit on
   training rows only. Hyperparameters in the YAML are fixed without a search.
5. Transfer the chemical operator without tuning: activate station edges whose
   unscaled predicted residual difference is at least 0.5 global training
   residual-scale units and expand active-edge differences by 1.5. Use the
   same base-disagreement precision center and no clipping.
6. Freeze validation and test predictions with hashes before an evaluator may
   read same-date validation/test responses. Define a boundary edge by the
   frozen 90th percentile of absolute training residual differences. Report
   all dates, plus boundary endpoints and boundary-edge pair deltas.
7. Primary uncertainty resamples contiguous seven-day test blocks. A cross-
   domain GO requires at least 3% pair-delta improvement, a negative bootstrap
   upper bound, at least 60% favorable test dates, and no more than 1% overall
   or boundary-endpoint point cost. A failure is retained and narrows the paper
   claim; it cannot be repaired by changing the graph, threshold, or scale on
   2024.

Run a 20-station parser/graph smoke first. Smoke output and metrics, if any, are
not paper evidence.
