# RF/GINE fixed-equal scale calibration preregistration

This is one bounded, arithmetic-only test of the under-dispersion diagnosis. It
does not retrain RF or GINE and does not reopen their ensemble weight.

- Base prediction: the already frozen 0.5 RF + 0.5 GINE prediction.
- Center: the median response among the outer-training molecules for that
  target/fold. Outer validation and test responses are prohibited from the
  center.
- Candidate scales: 1.00, 1.25, 1.50, and 2.00. Scale 1.00 is the identity
  safeguard.
- Transformation: `center + scale * (base_prediction - center)`.
- Selection partition: the next SCG component, identical to the existing RF
  and GINE validation partition.
- Selection objective: cliff-endpoint MAE + 0.25 overall MAE + 0.25 MMP-cliff
  pair-delta MAE. Ties prefer the scale closest to identity, then the lower
  scale.
- All candidate outer-test predictions and the selected prediction are frozen
  with hashes before any outer-test response is materialized.
- Aggregation: seed, then fold, then target macro over the frozen six-target
  panel.

The selected method is promoted only if all frozen criteria in the YAML config
pass: at least 3% cliff improvement versus GINE, no overall or pair-delta
regression versus GINE, cliff improvement on at least four targets, no more
than 1% cliff/overall regression versus the fixed-equal ensemble, no pair-delta
regression versus fixed equal, and pair-delta improvement versus fixed equal on
at least four targets. Candidate-scale results are sensitivity controls and
cannot replace the validation-selected primary result.

Smoke is restricted to one target/fold/seed. It checks partitions, component
hashes, scale selection, and prediction freezing but does not evaluate any
outer-test response. Only a successful smoke may unlock the 87 arithmetic
builds.

