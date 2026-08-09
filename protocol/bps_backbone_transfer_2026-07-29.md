# Frozen BPS transfer across modern molecular backbones (2026-07-29)

Status: frozen while the six-architecture strict-SCG matrix is still running
and before its full metrics are aggregated or inspected. The two preregistered
smoke folds were inspected only for execution validity.

## Question

Does the already selected response-free BPS rule transfer without tuning from
the SVM/RF/GINE development path to six independently trained molecular graph
architectures?

## Fixed inputs and operator

- Inputs: label-private predictions from GCN, GraphSAGE, GIN, GATv2, MPNN,
  and AttentiveFP on all 29 SCG-feasible MoleculeACE targets, all 143 outer
  folds, and seeds 0/1/2.
- Query graph: the existing eligible MMP graph restricted to test molecules.
- Gate: absolute predicted MMP delta at least 0.5.
- Operator: the existing minimum-correction component scaling with scale 1.5.
- The threshold and scale are copied exactly from the frozen SVM-BPS selection;
  there is no architecture-specific validation or test tuning.
- Builder reads predictions, structures/IDs, and MMP edges only. Test responses
  are materialized solely after every transferred prediction is frozen and
  hashed.

This is a post-primary backbone-transfer experiment. It cannot retroactively
make the original selected result independently confirmatory.

## Evaluation

For raw and BPS-transferred predictions, report overall MAE/RMSE,
cliff-endpoint MAE/RMSE, and cliff-pair delta MAE. Average folds within each
target and seed, then report three-seed target mean plus standard deviation and
target-macro mean plus standard deviation across seeds.

Paired summaries include:

- relative change per architecture and metric;
- favorable targets out of 29 per architecture;
- favorable target-architecture units out of 174 (descriptive only);
- hierarchical/dataset-level bootstrap confidence intervals, with target as
  the inferential unit after averaging folds and seeds.

## Frozen transfer gate

A broad transfer GO requires all of:

1. all 2,574 BPS predictions and hashes are complete;
2. family-macro pair-delta MAE improves by at least 1%;
3. at least 70% of the 174 target-architecture units improve pair-delta MAE;
4. family-macro overall MAE increases by no more than 1%; and
5. family-macro cliff-endpoint MAE increases by no more than 1%.

Per-architecture failures remain visible. A family-level GO does not authorize
the phrase "universal improvement."

