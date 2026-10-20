# Frozen prediction-reuse structural audits

These are post-primary analyses, not retrospectively confirmatory experiments.
No upstream predictor or selector is retrained and no external acceptance threshold
is retuned. All reportable negative results are retained.

## S8: validation-only postprocessor comparison

Use corrected earliest-year ChEMBL, its original temporal splits and MMP topology,
and the fixed mixture 0.5 SVM + 0.25 RF + 0.25 GINE. Compare identity, canonical
BPS, and validation-calibrated BPS, Laplacian unsharp, random-edge scaling, and
naive active-edge push. Each calibrated family has scales
{1.10, 1.25, 1.50, 1.75}, thresholds {0.25, 0.50, 0.75}, and identity.
Minimize validation true-cliff-pair MAE under an ordinary-MAE increase <=1%.
Tie breaks are ordinary MAE, correction norm, identity preference, candidate ID.
No validation cliff pairs means identity. Write chosen parameters and prediction
hashes before reading the corresponding test responses.

Laplacian unsharp uses prediction minus active-neighbor mean, scaled by s-1;
isolated nodes have zero correction. Naive push averages active-edge magnitudes
at each higher-prediction endpoint and scales by s-1. Random-edge scaling samples
as many MMP edges as the active set using the original deterministic seed 2917
and scope/target/model-seed/threshold hash, then scales sampled components.
Ordinary MAE has 36 eligible tasks; cliff/pair MAE has 35. First average seeds,
then bootstrap paired tasks 10,000 times. The primary estimator is the ratio of
target-macro MAEs; mean target-relative change is retained separately.

## S9: under-amplification and actual error changes

Use MoleculeACE (29 feasible tasks, including the manuscript's fixed 23-task
subset) and corrected ChEMBL (36 tasks), for mixture, SVM and GINE predictions.
Keep s=1.5 and activation threshold 0.5. Analyze bins [0,.25), [.25,.5),
[.5,.75), [.75,1), [1,infinity). Distinguish magnitude under/over/equality from
improved/worsened/unchanged absolute error in the signed molecular difference.
Report all active MMPs and evaluator-defined true cliffs separately; true-cliff
membership is never an inference input. Average seeds within folds, folds within
tasks, then tasks equally. Edge records are descriptive, not inferential units.

## S10: structural contract of selective execution

Reuse all 15 original AqSolDB fold/seed runs and the frozen nominal-80% threshold
0.008648302878998133. Audit full BPS and the candidate nodewise hybrid by
both-accepted, one-accepted, and both-rejected active edges. Report absolute
amplification residuals and original-component mean drift, separately for
nontrivial components; nonzero tolerance is 1e-10. Repeated seed records are
descriptive. The completed external risk audit requires identity for subsequent
deployment; it does not modify the frozen candidate predictions under evaluation.

## S11: threshold perturbations

Use the same 15 runs, frozen selector, s=1.5, and threshold .5. Add common nodewise
uniform noise U[-epsilon,epsilon] to SVM, RF and GINE, with epsilon {.01,.05}
and five fixed noise seeds. Recompute the mixture, active graph and selector
features. Compare reselected graphs with fixed-original-graph BPS; report edge
flips, exact component-membership changes, output changes, and selector flips.
Means average within folds and then across folds; worst-node values are maxima
over all runs. Fixed-graph Euclidean ratios must be <=1.5. The audit is not a
global continuity theorem for hard graph reselection.

## Selector provenance

The 144,297 records from 429 runs contain 35,562 unique molecules, 8,963 of which
occur across tasks. Automatic record-wise monitoring uses 14,430 records;
99.06% share molecule identities with fitting records. The model stops at 152 of
at most 200 iterations. Development association and AUROC are training-internal;
predictor-level OOF status does not establish selector-level independent validation.
The independent AqSolDB responses enter neither model fitting nor score quantiles.

## Numerical records and release adaptation

Frozen source hashes, original executed-script hashes and executor checks remain
in the result archive. Public scripts use portable root/output paths and optional
platform-aware resource limits; their hashes can therefore differ from the
original executed-script hashes without changing the scientific operators.
Library versions were Python 3.11, NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.2,
scikit-learn 1.6.1, RDKit 2025.09.6, joblib 1.4.2 and pyarrow 19.0.1.
Executor reproducibility checks are not independent integrity certification.
