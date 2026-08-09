# ESWA QM9 structured-missingness freeze

Frozen before any held-out QM9 response is evaluated. This supplements, rather
than repeats, the existing random connected-edge deletion experiment.

Use the first 128 nodes of each response-free QM9 evaluation cohort and the
already frozen raw direct-pair flow. At unordered observation densities 5%,
10%, and 25%, construct three label-free observation mechanisms:

1. `hub_biased`: first retain a deterministic star through the smallest-hash
   hub, then prioritize edges incident to additional smallest-hash hubs.
2. `chemistry_local`: retain a maximum-similarity spanning tree, then add pairs
   in descending ECFP4 Tanimoto similarity.
3. `disconnected_blocks`: assign nodes to two balanced structure-hash blocks,
   retain a deterministic star inside each block, and add only within-block
   pairs. Cross-block pairs are explicitly not identifiable and are excluded
   rather than silently scored.

Every selected unordered pair contributes both observed directions. Fit the
least-squares Hodge potential from observed raw flows only. Evaluate raw versus
integrated prediction on unobserved directed edges that lie in an identifiable
component. Report observed density, held-edge coverage, RMSE/MAE, all task-fold
rows, five-fold mean +/- standard deviation, favorable task directions, and
the cyclic residual energy on observed edges. The descriptive primary cell is
chemistry-local 10% density; all nine cells are mandatory. No missingness
scheme or density may be selected from responses.
