# ChEMBL 36 temporal external confirmation freeze

Frozen before extracting labels from the ChEMBL 36 SQLite release and before
constructing any external-test prediction. This experiment is the independent
confirmation required for a broad external-evidence submission. The 29 MoleculeACE targets
are development/secondary evidence only and cannot select another parameter.

## Data and task definition

- Source: the official ChEMBL 36 SQLite release. The immutable archive is
  `chembl_36_sqlite.tar.gz`, size 5,611,751,319 bytes, SHA-256
  `b25820eef0f0481ad7712bdf4bac3b45f354e3cbacb76be1fdbf4205d6b48fb9`.
- Keep only human `SINGLE PROTEIN` targets, binding assays (`assay_type=B`),
  confidence score 9, exact `Ki` measurements (`standard_relation='='`) in nM,
  finite positive values, a canonical SMILES, and a non-null publication year.
- Convert activity to `pKi = 9 - log10(Ki_nM)`. Standardize molecules with the
  repository's frozen RDKit standardizer.
- For a molecule-target-year duplicate, use the median pKi. If its measurement
  range exceeds 1.0 log unit, discard that duplicate group. If a standardized
  molecule occurs in several years, assign it to its earliest publication year
  and aggregate its accepted measurements by the median.
- Temporal partitions are train through 2019, validation 2020--2021, and test
  2022 or later. Enforce molecule disjointness by removing from validation any
  standardized molecule seen in train and removing from test any molecule seen
  in train or validation.
- A target is eligible using structure/count information only when it has at
  least 200 unique train molecules, 40 validation molecules, 60 test molecules,
  and 10 internal-test MMP edges. Include every target passing this rule; target
  identity and response values cannot be used for selection. The internal-test
  MMP count is label-free.
- If fewer than eight targets pass, report the audit as underpowered and stop
  this confirmation without relaxing thresholds. Freeze a separate DockedAC
  protocol before using it as the next independent dataset.

## Frozen predictors and graph operator

- ECFP4-SVM: RBF, `C=10`, `gamma=0.01`, `epsilon=0.1`.
- ECFP4-RF and residual GINE: reuse the exact fixed configurations already
  versioned in this repository. Train per target on train only. GINE early
  stopping may read validation responses; no test response may enter training,
  selection, graph building, or calibration.
- Unscaled reference: `0.5*SVM + 0.5*(0.5*RF + 0.5*GINE)`.
- BPS-GNN+: construct internal-query MMP edges without labels, activate edges
  whose absolute unscaled predicted difference is at least 0.5, and apply the
  active-component mean-preserving projection with scale 1.5. Isolated and
  off-active nodes are unchanged.
- All weights, SVM parameters, scale, threshold, fingerprint, MMP definition,
  and training recipes are inherited unchanged. No ChEMBL 36 target-specific
  hyperparameter is authorized.

## Label firewall

1. The preparation audit may expose counts, years, structures, duplicate
   diagnostics, and MMP topology, but not validation/test response summaries.
2. Validation labels are available only to GINE early stopping and prespecified
   diagnostics. They cannot change the frozen ensemble or projection.
3. Test predictions and row identifiers must be written and SHA-256 frozen
   before the evaluator loads test labels.
4. The primary decision is evaluated once. Any later analysis is explicitly
   secondary and cannot replace the primary row.

## Endpoints and aggregation

- Overall MAE over test molecules.
- Cliff-endpoint MAE on endpoints belonging to a true internal-test MMP cliff.
- MMP cliff-pair delta MAE for internal-test MMP pairs with absolute true pKi
  difference at least 1.0.
- Aggregate seed, then target. The primary estimand is target-macro relative
  change. Targets with no true cliff pair remain in overall MAE and are reported
  transparently, but do not contribute to the pair or endpoint macro.
- Use a fixed 10,000-resample target bootstrap for confidence intervals and a
  one-sided target-level Wilcoxon test. Holm-adjust the three primary endpoint
  tests. Report raw target rows, favorable directions, and effect sizes.

## Confirmatory decision

The primary comparison is BPS-GNN+ versus the unscaled SVM/RF/GINE reference.
All conditions must hold:

1. pair-delta target-macro relative improvement is at least 3%;
2. the pair-delta 95% bootstrap upper bound is below zero;
3. at least 70% of pair-eligible targets improve;
4. overall and cliff-endpoint point costs are each at most 1%;
5. the 95% bootstrap upper bounds for both point costs are each at most 1%;
6. at least eight targets contain a true test cliff pair.

Secondary comparisons against SVM, RF, and GINE are reported regardless of the
decision. Failure does not authorize test-set tuning. Diagnose the frozen result
and use validation-only development for a v2; only a separately frozen dataset
may confirm that v2.

## Prespecified ablations after the primary freeze

The following analyses cannot select the primary row: identity (`scale=1`), all
MMP edges, prevalence-matched random edges, matched low-predicted-delta edges,
SVM-only and RF/GINE-only bases, ECFP-kNN in place of MMP, and development-only
scale/threshold sensitivity. Report active-edge coverage, component sizes,
runtime, memory, correction norm, and all negative results. Run configuration
smokes before any model retraining and stop ablations that fail their smoke.
