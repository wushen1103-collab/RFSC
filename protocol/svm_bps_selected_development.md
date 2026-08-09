# SVM-BPS development outer-test freeze

The SVM complementarity grid was screened only on the outer-validation
partitions of the six development targets. Before reading the corresponding
outer-test labels, the selected candidate is frozen as:

- ECFP-SVM weight: 0.5;
- fixed RF/GINE weight: 0.5;
- label-free MMP component scale: 1.5;
- predicted absolute MMP-delta gate: 0.5.

The candidate is selected by robustness, not by the most favorable aggregate:
it is the only candidate containing SVM that improves all three metrics on all
six development targets in validation. The outer-test evaluation is run once.
It passes only if the target-macro relative change versus SVM is non-positive
for overall MAE, cliff MAE, and pair-delta MAE, with at least four of six target
directions improved for every metric.

The 23 Phase 4 target labels are not used by this selection. Because those
labels were already revealed in the earlier mandatory SVM addendum, any later
evaluation on them is explicitly secondary and not an independent confirmation.
