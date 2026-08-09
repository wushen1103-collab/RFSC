# Confidence-weighted BPS v2 exploratory freeze

This direction was formulated after the ChEMBL 36 primary result and is
therefore exploratory on ChEMBL 36. It cannot replace the primary row. Its exact
formula is frozen before constructing a v2 prediction.

For each test node, let `p_svm`, `p_rf`, and `p_gine` be the already frozen base
predictions and let

`y = 0.5 p_svm + 0.25 p_rf + 0.25 p_gine`.

Define label-free weighted ensemble disagreement

`u_i = 0.5(p_svm_i-y_i)^2 + 0.25(p_rf_i-y_i)^2 + 0.25(p_gine_i-y_i)^2`,

`u_0 = median_i(u_i) + 1e-6`, and precision `w_i = 1/(u_i+u_0)`.

Use the unchanged main MMP active-edge rule `|y_i-y_j| >= 0.5` and scale 1.5.
For each non-singleton active component `K`, replace the ordinary center by

`mu_K = sum_(i in K) w_i y_i / sum_(i in K) w_i`,

`z_i = mu_K + 1.5(y_i-mu_K)`.

Inactive singleton nodes remain unchanged. There is no clipping, grid,
target-specific parameter, or label-derived uncertainty calibration.

## Exploratory decision

- Pair-delta improvement versus the unscaled reference must remain at least 3%.
- Overall and cliff-endpoint point costs must both be no larger than the frozen
  ordinary-center BPS costs.
- The active-edge predicted differences must match ordinary BPS to numerical
  precision.

Passing only authorizes a separately preregistered DockedAC confirmation. A
ChEMBL 36 v2 result is never described as confirmatory.

## Theory claim to verify

For positive diagonal `W=diag(w)`, the operator uniquely minimizes
`||z-y||_W` subject to `B_A z = 1.5 B_A y`. It preserves the W-weighted mean of
each active component and reduces to ordinary BPS when all weights are equal.
