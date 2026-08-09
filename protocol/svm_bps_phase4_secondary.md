# SVM-BPS Phase 4 secondary evaluation

The SVM-BPS parameters remain exactly those selected from the six-target
outer-validation screen: SVM weight 0.5, component scale 1.5, and predicted
MMP-delta threshold 0.5. No Phase 4 target or metric chooses these values.

Candidate predictions are built label-free and hash-frozen before evaluation.
The evaluation passes when target-macro overall, cliff, and pair-delta MAE are
all no worse than the global-mode ECFP-SVM and at least 12 of 23 target
directions improve for every metric.

This is secondary evidence rather than independent confirmation: Phase 4
labels were already evaluated in the mandatory SVM addendum before this
candidate existed. The result may support breadth and mechanism claims but
must not be described as a second untouched confirmatory test.
