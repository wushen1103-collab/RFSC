# DeepDelta cycle-consistency external protocol

Status: frozen before loading any ADMET response or computing any new metric.

## Motivation and collision boundary

DeepDelta directly predicts ordered molecular property differences and reports
violations of identity, reversal antisymmetry, and three-molecule additivity as
diagnostics. It does not correct those violations. The proposed extension is a
response-free orthogonal projection of the published pair-flow onto the exact
gradient-flow subspace. This is distinct from training a pair model and from the
node-signal BPS boundary sharpening used in the activity experiments.

## Source and split

- Use the official DeepDelta repository at commit
  `cd9b131559d54be639fcb9455e4c057a3dfd2c6b`.
- Use all ten released ADMET benchmark datasets and the ten individual
  DeepDelta folds generated with shuffled `KFold(10, random_state=1)`.
- Candidate construction reads only the individual-fold prediction CSVs, which
  contain predictions but no response row.
- Reconstruct the official fold response differences only after all candidate
  node potentials and hashes are committed.

## Frozen operators

For an ordered complete pair-flow `F_ij` predicting `y_j-y_i`:

1. Antisymmetric ablation: `A=(F-F^T)/2`.
2. Hodge candidate: find centered node potentials `x` minimizing
   `sum_ij (x_j-x_i-F_ij)^2`, then return `G_ij=x_j-x_i`.
3. On a complete directed graph,
   `x_k=(sum_i F_ik-sum_j F_kj)/(2n)`.

No scale, threshold, label, validation response, or dataset-specific parameter
is used.

## Theorem-level check

Every true property-difference flow is a gradient flow. Orthogonality therefore
requires, fold by fold,

`||F-Y||_F^2 = ||G-Y||_F^2 + ||F-G||_F^2`.

The numerical relative discrepancy must be at most `1e-10`. Diagonal self-pairs
are excluded from performance metrics so that setting the diagonal to zero
cannot create a trivial performance claim.

## Metrics and decision

- Primary: off-diagonal pair RMSE, aggregated fold then dataset.
- Secondary: off-diagonal pair MAE, top-quartile absolute-delta MAE within each
  fold, and Pearson correlation.
- The Hodge projection must improve RMSE on all 10 datasets with a dataset
  bootstrap upper bound below zero.
- It must improve MAE on at least 8/10 datasets with bootstrap upper below zero,
  not worsen top-quartile MAE, and improve RMSE beyond antisymmetrization with a
  bootstrap upper bound below zero.
- A failed check is retained as a negative result. No setting exists to tune.

