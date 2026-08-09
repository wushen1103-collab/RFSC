# DeepDelta projection mechanism audit

Status: frozen after the primary result and before computing any mechanism or
natural-alternative metric.

This is a preregistered exploratory analysis, not a new independent external
confirmation. It answers the reviewer question of whether the primary gain is
only caused by reversal antisymmetrization or whether transitivity projection
adds a distinct contribution.

## Fixed decomposition

For each published DeepDelta fold flow `F`:

- `A=(F-F^T)/2` is the reversal-consistent component;
- `G` is the already hash-frozen Hodge gradient flow;
- `S=F-A` is the symmetric/self-identity residual;
- `C=A-G` is the antisymmetric cyclic residual.

The audit will verify both orthogonal error identities on off-diagonal pairs:

- `SSE(F,Y)-SSE(A,Y)=||S||^2`;
- `SSE(A,Y)-SSE(G,Y)=||C||^2`.

It will also verify `||F-G||^2=||S||^2+||C||^2`, and report raw and projected
identity, reversal, and three-node additivity residual RMS values.

## Fixed natural alternatives

Two zero-hyperparameter one-sided potentials are evaluated without selection:

- incoming-only: column mean of `F`;
- outgoing-only: negative row mean of `F`.

Both are centered, and their average must equal the frozen Hodge potential. The
same off-diagonal RMSE, MAE, high-delta MAE, and Pearson metrics are reported for
published DeepDelta, antisymmetrization, both one-sided variants, and Hodge.
Hodge is compared to every alternative with the same dataset-level bootstrap
used in the primary protocol. All outcomes are retained; no new threshold,
scale, or dataset-specific choice is permitted.

Zeroing only the diagonal is intentionally omitted because the performance
protocol already excludes diagonal pairs, making that operation a no-op rather
than an informative ablation.
