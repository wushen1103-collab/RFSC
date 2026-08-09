# PrismNet strict-SCG random-initialized pilot preregistration

The official pretrained checkpoint is not available in the release environment, so
this run is an architecture/source-training audit, not the published pretrained
PrismNet reproduction. The official source is pinned at commit
`4698a5bc6c3ae0e31284e84676a375f8ed10941b`.

The bounded smoke has already passed on 64 train and 32 validation molecules,
one epoch and two batches, with complete validation batching and no outer-test
prediction. The only full pilot is now frozen as CHEMBL287_Ki, outer fold 0,
seed 0, on GPU 7. It uses the released 3-layer/256-dimensional model, released
regression contrastive loss, batch size 32, 30 warmup epochs, all 150 epochs,
and the best validation-RMSE checkpoint. The source computes label mean/std but
evaluates and trains on unstandardized responses; the adapter intentionally
preserves that behavior.

The process has a 1,200-second hard wall-time budget. The builder may read
outer-train and validation labels, but must freeze its test predictions without
reading outer-test responses. Evaluation begins only after the run metadata and
prediction hash exist.

The pilot is compared with paired GINE, ECFP-RF, and their frozen equal ensemble
on the identical target/fold/seed. Scaling to any additional PrismNet run is
allowed only when overall MAE is no more than 10% worse than GINE and at least
two of overall, cliff-endpoint, and pair-delta MAE are each no more than 10%
worse. Passing this permissive feasibility gate does not establish superiority
or authorize a published-PrismNet claim.

