# ESWA QM9 direct-pair external-family freeze

Frozen on 2026-08-08 before this project reads any QM9 response. This is a new
quantum-chemistry task family, disjoint from the ten ADMET/PK DeepDelta tasks.
It tests the edge-flow claim directly rather than treating a node regressor as
an edge method.

## Source, tasks, and split

- Source: public QM9 SMILES/response table cached from the MotiL benchmark,
  expected SHA-256
  `fdcd9d922a86126a3cf23d0f3c4372e504755881e12eca019860a65096b65815`.
- Tasks: HOMO energy, LUMO energy, and HOMO-LUMO gap (`homo,lumo,gap`).
- A single response-free shuffled five-fold molecule split (seed 290808) is
  shared by all tasks. Within each fold, choose at most 25,000 training nodes
  and exactly 256 evaluation nodes by the smallest deterministic structure
  hashes. This cap is label-free and controls complete-pair memory.
- No held-out response is parsed until all five flow/potential artifacts for
  all 3 x 5 task-fold runs have been written and SHA-256 verified.

## Direct-pair predictor and repairs

- ECFP4 (1,024 bits); ordered concatenation `[fp(i),fp(j)]`.
- Delta-LightGBM uses 100 trees and at most 250,000 training ordered pairs,
  including diagonal examples and balanced forward/reverse off-diagonal pairs.
- Absolute-LightGBM with the same tree recipe is a scalar-control model.
- Freeze five outputs: raw directed flow, diagonal-zero repair, Frobenius
  antisymmetrization, complete-graph Hodge potential/flow, and absolute-model
  potential. All repairs are deterministic and require no response.
- The primary incremental comparison is Hodge versus antisymmetrization, not
  merely Hodge versus the unrepaired raw model.

## Metrics and measured-anchor recovery

- Pair RMSE, pair MAE, top-quartile absolute-delta MAE, Pearson correlation,
  diagonal/self-pair MAE, symmetric energy, and cyclic residual energy.
- Report task-fold rows, five-fold mean +/- standard deviation, task means,
  favorable directions, and a fixed 10,000-resample paired task-fold bootstrap.
- Preselect nested 1/3/5-node anchor sets by structure hash. At evaluation only,
  measured anchor labels estimate the Hodge potential offset; evaluate scalar
  MAE on non-anchor held-out nodes and compare with the absolute LightGBM.
- Strong mechanism support requires Hodge pair MAE to beat antisymmetrization
  in at least 2/3 tasks with a paired bootstrap upper bound below zero. Diagonal
  repair must reduce self-pair MAE to numerical zero. Negative high-delta or
  anchor results remain part of the report.

## Compute

Run a one-task/one-fold reduced smoke first. Full jobs may run one per task in
parallel with no more than 96 CPU threads each, keeping at least 30 host cores
free. The experiment is CPU/LightGBM bound and should not reserve GPUs.
