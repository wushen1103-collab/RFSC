# Fixed-half PrismNet pair expert: two unseen-fold confirmation

The three discovery runs selected PrismNet weight 0.5 and passed the exploratory
gate, but their component test metrics were already known. Freeze the next
unseen outer folds before training: CHEMBL287_Ki fold 2 seed 0 and
CHEMBL4005_Ki fold 1 seed 0. PrismNet weight is fixed at 0.5; validation weight
selection is prohibited.

PrismNet training keeps the pinned source architecture and loss, random
initialization (because the public pretrained checkpoint is network-blocked),
validation-RMSE checkpointing, minimum 75 epochs, patience 50, maximum 150
epochs, and a 1,200-second per-run wall-time budget. After each PrismNet test
prediction is frozen without test responses, the builder forms exactly
`0.5 * fixed_equal + 0.5 * PrismNet` and freezes that result before evaluation.

Confirmation requires mean overall and cliff MAE within 1% of fixed equal,
mean pair-delta improvement of at least 3%, pair improvement in both folds, and
no mean regression versus GINE on any of the three endpoints. This is a
next-fold confirmation of the pair-expert combination, not an independent
target-panel result and not a published pretrained-PrismNet reproduction.

