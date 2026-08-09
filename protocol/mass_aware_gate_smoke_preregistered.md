# Mass-aware learned-gate validation smoke (pre-registered)

## Motivation

The frozen cut-only learned gate has target-macro boundary AUPRC 0.442 versus
0.310 prevalence, yet its pair-delta MAE is worse than the frozen GINE anchor.
The current normalized soft propagation always uses the full `lambda=0.5`
whenever any nonzero smooth weight remains. Thus the magnitude of a query's
boundary evidence is largely cancelled by weight normalization.

## Single change under test

Replace only the propagation operator with

`y_hat = base + lambda * sum_j s_j(1-q_j)(ref_j-base) / sum_j s_j`.

The frozen GINE checkpoints, logistic gate, edge features, training labels,
`floor=0.4`, `k=10`, `lambda=0.5`, seeds, and q-matched controls are unchanged.
This adds no trained parameter. It tests whether total surviving smooth mass,
not just the relative ordering of neighbors, must control propagation strength.

## Smoke and firewall

- Fixed task: `CHEMBL287_Ki`, SCG outer fold 0, seeds 0/1/2.
- The trainer may use outer-train labels only.
- Only outer-validation pseudo-query labels may be evaluated.
- Outer-test predictions may be frozen label-free by the shared trainer, but
  outer-test labels and metrics are prohibited before the smoke decision.
- No lambda, gate model, probability calibration, or feature sweep is allowed.

## GO criteria

After averaging random replicates within seed and then averaging the three
paired seeds, `learned_mass_aware` must satisfy every condition:

1. cliff MAE improves at least 2% over the existing normalized `learned` rule;
2. pair-delta MAE improves at least 5% over normalized `learned`;
3. overall MAE does not worsen versus normalized `learned`;
4. at least 40% of normalized learned's pair-delta gap to `base` is removed;
5. cliff MAE beats both mass-aware q-matched random and similarity controls.

Failure is a recorded NO-GO with no same-fold tuning. Passing authorizes only a
pre-registered two-unseen-fold confirmation, not the full 87-run matrix.
