# Conservative SVM-BPS scale margin rescue

The frozen scale-1.5 candidate achieved the lowest strict-SCG pair-delta point
estimate while staying below the 1% overall/cliff point guards, but the Phase 4
hierarchical 95% upper bounds were approximately 1.30% and 1.20%. This
post-secondary branch therefore returns exclusively to the six development
outer-validation partitions. Phase 4 and development outer-test labels are
prohibited for selection.

SVM weight 0.5 and predicted MMP-delta threshold 0.5 remain fixed. Only scales
1.30, 1.35, 1.40, and 1.45 are screened. A candidate passes when validation
target-macro overall and cliff costs versus the unscaled mixture are each at
most 0.5%, pair-delta improves at least 3%, and pair direction improves on at
least five of six targets. If multiple candidates pass, choose the smallest
scale to maximize safety margin. Any later outer-test evaluation is explicitly
exploratory/secondary because no untouched MoleculeACE SCG target remains.
