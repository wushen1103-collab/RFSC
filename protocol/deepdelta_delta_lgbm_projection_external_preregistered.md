# Direct Delta-LightGBM projection protocol

Status: frozen before any held-out response or metric is materialized.

## Purpose and claim boundary

The DeepDelta evidence currently covers one direct-pair neural architecture.
The official DeepDelta source also defines `Delta_LGBM`, which concatenates two
ordered Morgan fingerprints and directly learns `y_j-y_i`. This experiment
adapts that released model family as a second, non-neural direct-pair predictor
and applies the unchanged response-free Hodge repair.

This is not an exact reproduction of an official result table: the repository
does not release Delta-LightGBM prediction files, and we cap the quadratic
training set to avoid waste. It is a preregistered architecture-generality test
on the same ten public ADMET datasets. Classical HodgeRank remains credited for
the projection mathematics.

## Split and response firewall

- Use all ten pinned DeepDelta benchmark datasets.
- Use only fold 1 of `KFold(10, shuffle=True, random_state=29)` per dataset.
- Split from the structure-only SMILES column before selecting responses.
- Candidate construction materializes only train-fold responses via row-selective
  parsing. It may fingerprint test structures but may not parse test responses.
- Raw direct-pair predictions, Hodge potentials, absolute-LightGBM control
  potentials, metadata, and SHA-256 manifests must be committed before the
  evaluator selects held-out responses.

## Fixed predictors

1. `Delta-LightGBM`: concatenate ordered 2048-bit radius-2 Morgan fingerprints
   and directly regress the response difference.
2. `Absolute-LightGBM control`: regress one scalar property per molecule and
   subtract its predictions. This is gradient-consistent by construction and
   checks that the direct model has non-catastrophic predictive quality.

To bound memory, sample at most 250,000 ordered training pairs with fixed seed
29017. Sampling begins from unordered molecule pairs, includes both directions,
includes every diagonal pair, and is independent of structures and responses.
Both LightGBM models use 100 trees, row subsampling 0.1 every tree, deterministic
seed 29, column-wise training, and at most 64 logical CPU threads.

No model, dataset, or threshold may be selected using held-out responses.

## Frozen repair and metrics

For the complete held-out direct-pair flow `F`, compare:

- raw Delta-LightGBM;
- `A=(F-F^T)/2`;
- Hodge `G_ij=x_j-x_i`; and
- the absolute-LightGBM difference control.

Exclude diagonal pairs from RMSE, MAE, high-delta MAE, and Pearson r. The top
quartile of absolute true differences defines high delta after prediction
hashes are frozen. Bootstrap the ten datasets. Audit

`||F-Y||^2 = ||G-Y||^2 + ||F-G||^2`

on every held-out fold.

## Decision gates

A strong second-architecture GO requires all of:

1. Hodge RMSE improves on 10/10 datasets with bootstrap upper bound below zero.
2. Hodge MAE improves on at least 8/10 with bootstrap upper below zero.
3. Macro high-delta MAE is not worse.
4. Hodge RMSE beats antisymmetrization with bootstrap upper below zero.
5. All ten Pythagorean audits pass at relative tolerance `1e-10`.
6. Raw direct-model macro RMSE is no more than 25% worse than the same-split
   absolute-LightGBM control. This prevents a trivial repair claim on a
   catastrophically weak second predictor.

Every failed gate is retained. No pair-budget, LightGBM, fingerprint, split, or
repair parameter is tuned after held-out outcomes.

