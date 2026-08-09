# Exploratory PrismNet pair-expert ensemble

This diagnostic is explicitly post-component-evaluation and therefore cannot
be confirmatory. It asks whether the three already frozen random-init PrismNet
predictions contain complementary pair-delta information that can be added to
the fixed-equal RF/GINE ensemble without retraining any model.

For each available target/fold/seed, reconstruct the fixed-equal validation
prediction from its hash-verified RF and GINE components. Candidate PrismNet
weights are 0, 0.10, 0.25, and 0.50. Validation-feasible candidates must keep
both overall and cliff-endpoint MAE within 2% of weight zero and must not worsen
pair-delta MAE. Among feasible candidates, select the lowest pair-delta MAE;
ties prefer lower cliff, lower overall, then lower PrismNet weight. Weight zero
is the mandatory fallback.

All four candidate test predictions and the validation-selected prediction are
frozen with hashes before the ensemble evaluator reads a test response. Smoke
uses only the first run and does not call the evaluator. The full diagnostic is
still limited to the three existing PrismNet runs.

A useful signal requires mean pair-delta improvement of at least 3% versus the
fixed-equal ensemble, no more than 1% mean overall/cliff regression, and pair
improvement in at least two of three runs. Passing only authorizes a future
unseen-fold confirmation; it does not establish a result on the already seen
folds and does not authorize a PrismNet matrix.

