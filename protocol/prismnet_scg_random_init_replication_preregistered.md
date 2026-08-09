# PrismNet random-init strict-SCG two-point replication

The CHEMBL287_Ki fold-0 seed-0 pilot passed its permissive feasibility gate but
was slower and weaker on point metrics than the fixed-equal RF/GINE ensemble.
Before any matrix, freeze exactly two independent checks: CHEMBL287_Ki fold 1
seed 0 for fold transfer and CHEMBL4005_Ki fold 0 seed 0 for target transfer.

The architecture, source commit, loss, optimizer schedule, batch size, and
validation checkpoint selection are unchanged. To respect the compute budget,
replications use validation-only early stopping after at least 75 epochs and 50
epochs without a new best validation RMSE; 150 epochs remains the hard maximum.
Each run also has a 1,200-second external wall-time cap. GPU 6 and GPU 7 may run
the two tasks in parallel because both were unoccupied immediately before the
frozen launch.

No matrix is authorized unless both runs are within 10% of their paired GINE
overall MAE, at least four of the six run-by-endpoint cells (overall, cliff,
pair delta) are within 10%, and pair-delta MAE improves over GINE in at least
one run. These are architecture feasibility criteria only. Results remain a
random-initialized source audit and cannot stand in for the inaccessible public
pretrained checkpoint.

