# CliffLoss GINE strict-SCG smoke (pre-registered)

The May-2026 preprint *When Molecular Similarity Works: Property Cliffs Reveal
Hidden Errors* reports that its model-agnostic CliffLoss improves MAE across
five backbones. This is a mandatory recent baseline and a direct collision for
generic train-label-derived cliff-severity loss weighting. It is not a BPS
component or an original contribution.

The paper fixes the loss and adaptive controller defaults (`lambda_base=0.1`,
EMA alpha `0.7`, gamma `4`, scale clip `[0.25,4]`, normalization q95), but does
not report the similarity threshold, neighbor budget, aggregation budget, or
complete training settings. Its linked anonymous repository returned HTTP 401
on 2026-07-18. This run is therefore explicitly a formula-level adaptation,
not an official-code reproduction. The missing neighborhood settings are
frozen to ECFP4 Tanimoto >=0.4 with the ten nearest training neighbors and all
ten for aggregation, matching the disclosed public implementation practice
found during the audit.

Only `CHEMBL234_Ki/fold_0/seed_0` is authorized. Baseline and CliffLoss use the
same GINE initialization, batch order, MAE objective, optimizer, and validation
early stopping. CliffLoss computes training severity only from outer-train
labels. Its published adaptive controller necessarily uses outer-validation
labels to form Q1/Q4 error feedback; this is disclosed and permitted as model
selection. No outer-test label or prediction may be materialized.

The validation gate requires at least 2% overall-MAE and 3% cliff-MAE
improvement over the paired MAE-GINE, no pair-delta degradation, at least 50%
nonzero train severity coverage, and finite metrics. All checks must pass.
Passing authorizes exactly two fixed replications; failure closes the branch
without threshold, neighbor, controller, or training-horizon tuning.
