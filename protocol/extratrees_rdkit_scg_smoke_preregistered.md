# RDKit-descriptor ExtraTrees strict-SCG smoke preregistration

Frozen before any descriptor-ExtraTrees validation metric is computed.

## Evidence status

The June 2026 v4 benchmark *Do Larger Models Really Win in Drug Discovery?*
reports a 300-tree ExtraTrees ensemble on RDKit descriptors as a strong compact
baseline across 26 endpoints and three split types. The primary paper and its only
supplementary PDF do not release the claimed reproducibility package and do not
specify descriptor failure handling or non-default ExtraTrees parameters. No
author-controlled public repository was found on 2026-07-18. This experiment is
therefore a **configuration-level adaptation**, not an official reproduction.

The benchmark's SVD/KMeans structure-separated split is also not equivalent to our
label-free Structure-Component Grouped split. Any result here is specific to the
strict SCG protocol.

## Authorized smoke

Only `CHEMBL287_Ki/fold_0/seed_0` is authorized. Fold 0 remains the inaccessible
outer test; fold 1 is validation and the remaining components are training. The run
may read training and validation responses, but it must neither read outer-test
responses nor generate outer-test predictions.

- Representation: every scalar descriptor in pinned RDKit's registered descriptor
  list, in registry order.
- Preprocessing: train-only median imputation; drop train-all-missing and
  train-constant columns; no scaling.
- Model: `ExtraTreesRegressor` with 300 trees, squared-error criterion,
  `max_features=1.0`, `min_samples_leaf=1`, no bootstrap, seed 0, and 16 CPU workers.
- No parameter selection, extra seed, alternate descriptor subset, or weight search
  is authorized.

The sole ensemble candidate is fixed before metrics: 0.50 frozen GINE + 0.25 frozen
ECFP-RF + 0.25 RDKit-ExtraTrees. Its comparator is the frozen 0.50 GINE + 0.50
ECFP-RF ensemble.

## Joint GO gate

All conditions must pass:

1. ExtraTrees validation RMSE is no more than 5% worse than ECFP-RF.
2. The diversified ensemble improves overall MAE by at least 0.5%.
3. It improves cliff-endpoint MAE by at least 0.5%.
4. Its cliff-pair delta MAE is no more than 0.5% worse.

Failure is a NO-GO: stop without outer-test evaluation, descriptor/parameter tuning,
or further targets. Passing permits only a separately preregistered two-point
validation replication.
