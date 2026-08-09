# Cross-target RF/GINE utility router audit (pre-registered)

The fixed-equal RF/GINE ensemble is the strongest current point predictor, but
its component-wise oracle gap is large. Prior same-target disagreement routing
was near chance. This audit asks the narrower question of whether the component
winner transfers across assay targets when every evaluated target is excluded
from router fitting.

The inputs are the already frozen, hash-verified strict-SCG outer-fold RF and
GINE predictions on the six development targets. Three seeds are collapsed to
component mean and standard deviation. The six fixed features are target-scale
normalized ensemble level, signed and absolute disagreement, RF and GINE seed
standard deviations, and their difference. For each held target, a standard-
scaled logistic regression (`C=1`, balanced classes) is trained on the other
five targets with equal total weight per source target. No held-target label
enters fitting. Probability maps without tuning to an RF weight in `[0.25,
0.75]`; probability 0.5 exactly recovers fixed equal.

This is exploratory because the component outer-test labels and metrics were
already materialized in earlier experiments. It cannot establish a primary
claim. It is allowed only as a cheap feasibility screen before new training.

Passing requires macro AUROC >=0.55, macro AUPRC lift over prevalence >=0.03,
AUROC above chance on at least four of six targets, at least 1% target-macro
overall and cliff MAE improvement over fixed equal, no pair-delta degradation,
cliff improvement on at least four targets, and finite outputs. All criteria
must pass. Failure closes routing without feature, classifier, or amplitude
tuning. Passing authorizes only the pre-frozen fresh target
`CHEMBL204_Ki/fold_0/seed_0` outside the development panel.
