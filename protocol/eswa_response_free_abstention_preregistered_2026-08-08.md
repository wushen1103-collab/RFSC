# ESWA response-free selective-risk freeze

Frozen after the AqSolDB structure/label preparation but before any AqSolDB
outer-fold prediction was evaluated. AqSolDB responses may be used by base-model
training according to its SCG protocol; they are prohibited from fitting,
selecting, or calibrating this risk rule.

## Reviewer question

Can the known node-level calibration cost be predicted from inference-time
geometry alone, so users may abstain from applying BPS on high-risk nodes while
retaining its boundary benefit? This is a selective-decision experiment, not a
new claim that point accuracy is universally improved.

## Development and external roles

- Development: all 29 MoleculeACE strict-SCG out-of-fold SVM/RF/GINE-mixture
  and BPS predictions. MoleculeACE labels are permitted only as the supervised
  development target.
- Untouched external evaluation: AqSolDB five-fold SCG predictions. No AqSolDB
  response enters model fitting, feature normalization, risk thresholds, or
  coverage selection.

## Response-free features and frozen learner

For every query node compute only: absolute BPS correction, SVM-versus-RF/GINE
base disagreement, active degree, log active-component size, active-component
density, and active-component prediction span. Correction, disagreement, and
span are divided by the within-run standard deviation of the unscaled reference
to make the rule transferable across endpoints. No residual, true cliff flag,
response statistic, or test-set ranking may be a feature.

The development target is normalized point harm,
`(|BPS-y|-|reference-y|)/reference_run_std`. Fit one deterministic
`HistGradientBoostingRegressor` with max depth 3, learning rate 0.05, 200
iterations, L2 regularization 1.0, and seed 290813. Hyperparameters are fixed;
there is no response-based search. Risk is the predicted harm, where lower is
safer. Thresholds are the development-score quantiles corresponding to nominal
coverages 100%, 90%, 80%, 70%, and 50%.

## External reporting

- Freeze every AqSolDB risk score and ID hash before opening an outer-test
  response in the selective evaluator.
- At each frozen threshold report realized node coverage, reference and BPS
  MAE on accepted nodes, true-cliff pair coverage and pair-delta MAE when both
  endpoints are accepted, the fraction of nodes actually harmed, and a full-set
  hybrid that applies BPS only to accepted nodes.
- Aggregate seed then fold and report mean +/- standard deviation. The primary
  descriptive operating point is nominal 80% coverage, selected before external
  outcomes. The complete coverage curve is mandatory and negative results are
  retained.
- Useful risk control is supported only if at nominal 80% coverage the accepted
  BPS point-MAE relative cost is lower than the 100% cost, the accepted pair
  effect remains favorable, and realized node coverage is at least 65%.
