# Official Chemprop D-MPNN strict-SCG protocol (frozen 2026-07-29)

Status: frozen before any Chemprop strict-SCG training or evaluation.

## Purpose

Add one strong, independently maintained molecular message-passing baseline
under the exact SCG folds. This is an official software rerun, not a claim that
Chemprop was designed specifically for activity cliffs.

## Software and environment

- Chemprop 2.2.1 official package in the isolated remote `.venv_tabpfn`
  environment.
- PyTorch 2.9.1+cu128.
- Default Chemprop v2 directed-message-passing architecture and train-only
  target scaling.
- The exact package versions and commands are recorded per run.

## Split and label firewall

- All 29 SCG-feasible MoleculeACE targets and all 143 outer folds.
- Seeds 0, 1, and 2.
- Train, validation, and test IDs are identical to the common GNN matrix.
- The next outer fold is validation; the current outer fold is test.
- The Chemprop training CSV contains train and validation responses, while all
  test response cells are blank. A structure-only test CSV is passed to the
  prediction command after validation-only checkpoint selection.
- Frozen `mol_id,prediction` files and SHA256 metadata must exist before the
  common evaluator materializes test responses.

## Fixed training settings

- Task: regression; loss: MSE; reported internal metrics: MAE and RMSE.
- Maximum epochs: 100; patience: 20.
- Batch size: 64; data-loader workers: 4.
- No hyperparameter optimization, ensemble, test-time augmentation, or
  architecture-specific target selection.

## Compute gate

1. Smoke `CHEMBL287_Ki/fold_0/seed_0` and
   `CHEMBL4005_Ki/fold_0/seed_0`.
2. Promote if both complete with finite predictions, exact test IDs, valid
   hashes, and no more than 20 minutes per run. Performance is not a promotion
   criterion.
3. A main-table rerun requires all 429 runs. Partial coverage is labeled
   explicitly and cannot be used for a family-wide win.

## Reporting

The row is labeled `D-MPNN (Chemprop 2.2.1), Re-run, strict SCG`. Folds are
averaged within target/seed; report target-macro mean plus standard deviation
over three seeds. It belongs to the supervised molecular-GNN group.

