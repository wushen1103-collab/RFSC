# Modern-GNN temporal/OOD protocol (frozen 2026-07-29)

## Purpose

Recent broad-benchmark experimental practice makes a strong in-distribution table
insufficient by itself. This confirmatory extension tests whether conclusions
from strict structure-component splits survive two distribution shifts without
reopening architecture selection or hyperparameter tuning.

## Frozen benchmarks

1. ChEMBL36 earliest-year temporal split: all 36 previously audited eligible
   targets, with the existing train/validation/test years and MMP graph.
2. BindingDB patent latest-era split: all 42 previously audited eligible
   targets, with the existing era split and patent-only MMP graph.

The response-free split files, test-label firewall, MMP eligibility rule,
prediction hashes, robust train-only response scaling, validation-only early
stopping, and seeds `[0, 1, 2]` are inherited unchanged.

## Frozen architecture panel

Only three complementary representatives from the full strict-SCG panel are
rerun:

- GATv2: attention-based message passing;
- MPNN: edge-conditioned message passing with Set2Set readout;
- AttentiveFP: molecular attention and recurrent readout.

Their architecture and optimization settings are copied verbatim from the
strict-SCG configs. GINE, RF, SVM, the unscaled reference, and BPS predictions
already frozen under the same temporal splits are included in the final fair
comparison. No result from a different split is mixed into paired claims.

## Compute gate and completeness

- Smoke one ChEMBL and one BindingDB target per architecture before promotion.
- Full promotion requires `78 targets x 3 seeds = 234` hashed predictions per
  architecture across the two benchmarks (702 runs total).
- Missing runs remain explicitly partial; no complete-case target is silently
  substituted.
- GPU concurrency may be increased after smoke, but each job keeps four CPU
  threads and the host must retain at least 30 idle CPU cores.

## Frozen reporting

Report overall MAE, cliff-endpoint MAE, and cliff-pair delta MAE. The reporting
unit is the target: first average folds where applicable, then report the macro
mean and standard deviation across the three independent seeds. Include exact
coverage and prediction-hash audits, plus the provenance label
`Re-run (common PyG implementation)` for all three new methods.

This experiment is for robustness/completeness, not post-hoc selection. It
does not change the primary strict-SCG claim or its decision thresholds.
