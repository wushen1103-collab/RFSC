# DeepDelta 5x10 projection robustness protocol

Status: frozen after the first random-state-1 DeepDelta result was known, but
before parsing any response row or computing any projection metric from the
released `DeepDelta5` aggregate files. This is a preregistered robustness
extension, not an untouched independent confirmation.

## Motivation and source

The first external result used the ten individual folds released for
`random_state=1`. The pinned official DeepDelta artifact also releases five
complete 10-fold cross-validation repetitions (`random_state` 0--4) for every
one of the same ten ADMET datasets. We will apply the unchanged, zero-parameter
Hodge repair to all 50 dataset-repeat units and all 500 fold blocks. There is no
training, tuning, or new dataset selection.

The classical HodgeRank collision is acknowledged: the projection mathematics
is prior work. This experiment tests robustness of its response-free molecular
prediction-repair use.

## Information barrier and frozen candidate

Each aggregate CSV has a header, a true-delta row, and a prediction row. Before
evaluation, the candidate builder's numeric parser receives only the final
prediction line. It infers the ten official KFold block sizes from the number of
predictions, projects each block independently, freezes the concatenated node
potentials, and records a SHA-256 hash of the prediction line. It does not parse
the true-delta row.

All 50 candidate artifacts, hashes, this protocol, the config, evaluator, and
tests must be committed before the evaluator may parse the response rows.

For every held-out fold with ordered flow `F`, the fixed methods are:

1. published DeepDelta `F`;
2. antisymmetrization `A=(F-F^T)/2`; and
3. Hodge repair `G_ij=x_j-x_i`, where
   `x_k=(sum_i F_ik-sum_j F_kj)/(2n)`.

Projection is always within a fold. Pooling predictions across folds before
projection is forbidden.

## Overlap audit

The official code states that individual fold files are saved for
`random_state=1`. Before loading aggregate response rows, repeat 1 predictions
will be compared numerically with the ten individual files. The maximum
absolute discrepancy must be at most `1e-12`. The four-repeat scope
`{0,2,3,4}` is prespecified as the new-split sensitivity; all five repeats form
the full robustness estimate.

## Metrics and aggregation

- Exclude diagonal self-pairs from predictive metrics.
- Per fold: pair RMSE, pair MAE, top-quartile absolute-delta MAE, and Pearson r.
- Aggregate fold -> dataset-repeat -> dataset. Bootstrap only the ten datasets;
  never treat 50 correlated repeats as 50 independent studies.
- Report all-five-repeat and new-four-repeat contrasts, plus favorable direction
  counts over the 50 dataset-repeat units.
- Audit the exact full-matrix Pythagorean identity in every one of 500 folds.

## Frozen decision gates

The result is a strong robustness GO only if all conditions hold:

1. Hodge RMSE improves on 10/10 dataset means over all five repeats and its
   dataset-bootstrap upper bound is below zero.
2. Hodge RMSE improves on 10/10 dataset means over the four new repeats and its
   dataset-bootstrap upper bound is below zero.
3. At least 48/50 dataset-repeat RMSE directions improve.
4. Hodge MAE improves on at least 8/10 all-repeat dataset means and its
   bootstrap upper bound is below zero.
5. All-repeat high-delta MAE is not worse in macro average.
6. Hodge RMSE beats antisymmetrization with a dataset-bootstrap upper bound
   below zero.
7. All 500 Pythagorean audits pass at relative tolerance `1e-10`.
8. The response-free repeat-1 alignment audit passes at absolute tolerance
   `1e-12`.

A failed gate is retained. No alternative setting or repeat exclusion may be
tried after outcomes are exposed.

