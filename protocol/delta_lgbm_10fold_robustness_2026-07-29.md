# Delta-LightGBM ten-fold robustness protocol (frozen 2026-07-29)

Status: frozen before folds 2--10 are trained or evaluated. Fold 1 was exposed
under the earlier preregistration and will be rerun unchanged for a uniform
matrix.

## Reviewer gap

The second direct-pair architecture currently has one held-out fold per ten
DeepDelta datasets. This supports architecture breadth but not repeated-split
uncertainty. The expansion changes only the held-out fold index and runs all ten
folds already defined by `KFold(10, shuffle=True, random_state=29)`.

## Locked model and data contract

- Ten pinned DeepDelta ADMET datasets and their unchanged row order.
- Radius-2, 2048-bit Morgan fingerprints.
- At most 250,000 balanced ordered training pairs, both directions and all
  diagonals, sampled with random state 29017.
- Direct and absolute LightGBM controls: 100 trees, row subsampling 0.1,
  deterministic random state 29, force-column-wise training.
- Complete held-out flow; diagonal excluded from predictive metrics.
- Raw, antisymmetric, Hodge, and absolute-difference methods.
- High-delta threshold: within-fold upper quartile of absolute truth deltas,
  materialized only after every prediction artifact for that fold passes its
  hash and structure-partition preflight.
- No fold may tune features, pair budget, LightGBM parameters, projection, or
  decision thresholds.

The compute-only `n_jobs` value may be lowered to allow safe parallel folds; it
does not change the deterministic fitted model.

## Matrix and statistical units

- 10 datasets x 10 held-out folds = 100 dataset-fold units.
- Every unit must pass the exact Pythagorean identity audit.
- For each dataset, average its ten folds first. Bootstrap the ten datasets for
  the primary Hodge-vs-direct relative effect.
- Separately report the mean and standard deviation across the ten fold-level
  dataset-macro effects.
- Direction counts are reported both by dataset after fold averaging (x/10)
  and descriptively by dataset-fold (x/100). The x/100 count is not used as an
  independent-sample significance test.

## Claims and gates

The experiment strengthens repeated-split robustness only. It does not create
an independent data source and does not turn Delta-LightGBM into a published
external model result. A robustness GO requires:

1. all 100 prediction/hash/partition audits complete;
2. all 100 Pythagorean checks pass at `1e-10`;
3. fold-averaged Hodge RMSE and MAE improve on at least 8/10 datasets;
4. the dataset-bootstrap 95% upper bound is below zero for RMSE and MAE; and
5. the ten-fold mean high-delta MAE is not worse than direct prediction.

All failures remain visible and the original fold-1 decision is not rewritten.

