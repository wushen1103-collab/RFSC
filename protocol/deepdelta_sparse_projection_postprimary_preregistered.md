# Frozen post-primary sparse/incomplete-flow protocol

Status: frozen before any sparse-candidate response metric. The complete-flow
Delta-LightGBM result on these same held-out responses is already known, so
this experiment is explicitly post-primary scope validation, not a pristine
independent confirmation.

## Question

Can a potential integrated from an incomplete, response-free subset of a
direct molecular pair flow improve prediction on *unobserved* pairs, while
retaining the exact least-squares guarantee on observed pairs?

## Immutable inputs and sampling

- Input flows are the ten hash-frozen direct Delta-LightGBM test matrices from
  `experiments/deepdelta_delta_lgbm_projection/predictions/`.
- Use undirected target densities 5%, 10%, 25%, and 50%, with fixed random
  states 2901--2905. Observe both directed predictions for every sampled
  undirected edge; exclude diagonal edges. Instantiate each generator with
  `SeedSequence([random_state, dataset_index])`, where `dataset_index` follows
  the frozen configuration order, so equal-sized datasets do not share masks.
- At each dataset/density/seed, first draw a response-free random spanning tree
  and then uniformly sample unselected undirected edges until reaching
  `max(n-1, ceil(density * n(n-1)/2))`. Thus small graphs may have an effective
  5% density above 5% because connectivity takes priority.
- Solve ordinary least squares for a centered node potential from only those
  observed directed flows. Freeze the potential, full upper-triangle observed
  mask, metadata, and SHA256 before computing a sparse-candidate metric.

No held-out response may affect graph sampling, the solver, densities, seeds,
or candidate construction. No model is retrained.

## Metrics

For each of the 200 dataset/density/seed graphs report:

1. connectivity and effective density;
2. raw and projected RMSE/MAE on observed directed edges;
3. the exact observed-edge squared-error Pythagorean discrepancy;
4. raw and sparse-potential RMSE/MAE on the complement (unobserved directed
   edges); and
5. sparse-potential versus complete-Hodge RMSE on unobserved edges.

For every density, average seeds within dataset, then compute the ten-dataset
macro and a 10,000-resample dataset bootstrap for sparse versus raw unobserved
RMSE/MAE. Density rows are all reported; none may be selected after outcome.

## Frozen strong decision

The primary density is 50%. `strong_sparse_projection_go=true` only if:

- unobserved-edge RMSE improves on at least 8/10 datasets and its dataset-
  bootstrap relative-change upper bound is below zero;
- unobserved-edge MAE improves on at least 8/10 datasets and its bootstrap
  upper bound is below zero;
- all 200 sampled undirected graphs are connected; and
- all 200 observed-edge Pythagorean identities pass at relative tolerance
  `1e-10`.

The theorem applies only to observed edges. A held-edge failure is a valid
scientific NO-GO and cannot be repaired by changing density, seed, sampling,
or the gate after responses. Low-density rows are dose-response evidence, not
alternative primary candidates.
