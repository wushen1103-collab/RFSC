# RDKit-descriptor ExtraTrees two-point replication preregistration

Frozen after the `CHEMBL287_Ki/fold_0/seed_0` smoke passed, and before either
replication task computes an ExtraTrees validation metric.

## Fixed tasks

1. `CHEMBL2971_Ki/fold_0/seed_0` tests cross-target transfer.
2. `CHEMBL4616_EC50/fold_0/seed_0` tests cross-target and cross-endpoint-type
   transfer.

For each task, outer fold 0 is inaccessible test, outer fold 1 is validation,
and all remaining SCG components are training. These tasks were selected to span
target and endpoint type, not from descriptor-ExtraTrees results. No outer-test
response may be read and no outer-test prediction may be generated.

## Frozen method and comparator

The representation, train-only imputation and constant filtering, 300-tree
ExtraTrees parameters, seed, and 16-worker limit are byte-equivalent to the smoke.
The only candidate remains 0.50 frozen GINE + 0.25 frozen ECFP-RF + 0.25
RDKit-ExtraTrees. The comparator remains frozen 0.50 GINE + 0.50 ECFP-RF. No
descriptor subset, model parameter, seed, task, or ensemble weight may be changed.

## Per-task gate and aggregate decision

Each task must independently satisfy all four smoke gates:

1. ExtraTrees RMSE no more than 5% worse than ECFP-RF.
2. Diversified ensemble overall MAE improves by at least 0.5%.
3. Diversified ensemble cliff-endpoint MAE improves by at least 0.5%.
4. Diversified ensemble cliff-pair delta MAE is no more than 0.5% worse.

Panel preregistration is authorized only if both tasks pass all four criteria
(2/2 tasks and 8/8 checks). Any failure stops the route without test evaluation,
parameter search, per-target selection, or further descriptor variants.
