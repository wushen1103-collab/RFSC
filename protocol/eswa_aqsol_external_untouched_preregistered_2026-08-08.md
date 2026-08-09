# ESWA untouched AqSolDB external confirmation freeze

Frozen on 2026-08-08 before this repository materializes any AqSolDB response.
The source had been named in the original experimental story, but no AqSolDB
data, split, prediction, or result artifact existed in the repository or on the
the frozen experiment tree at freeze time. MoleculeACE and ChEMBL/BindingDB results
are development or prior external evidence and cannot change this protocol.

## Source and response firewall

- Source file: the AqSolDB clean table distributed with the public
  ECFP-Sort-and-Slice benchmark, columns `SMILES,Solubility`.
- Expected source SHA-256:
  `1f9c4bb47c2c62b515719717899191f11a9772e7d30cc5b0443e0050bed2eae2`.
- Response: the supplied logarithmic aqueous solubility value. It is stored in
  the repository's generic private response column `y_pactivity`; the name is
  an interface convention and does not imply bioactivity.
- Stage 1 reads only the SMILES column, standardizes structures, builds the MMP
  graph, and freezes SCG folds. Stage 2 may materialize labels only after all
  label-free eligibility checks pass. Outer-test labels are read only by the
  evaluator after prediction files and hashes exist for every fold and seed.

## Structure-only eligibility and split

- Use the frozen RDKit cleanup, largest-fragment, uncharging, canonicalization,
  scaffold, connectivity-key, and one-cut MMP definitions in this repository.
- Aggregate exact standardized-structure duplicates by the median response.
- Build deterministic Structure-Component Grouped folds with seed 2901,
  requesting five folds and allowing no fewer than three.
- Eligibility requires at least 2,500 unique standardized molecules, 100
  eligible MMP edges, three feasible SCG folds, and at least 10 internal MMP
  edges in every held-out fold. If a condition fails, stop without looking at
  solubility values and report the dataset as topology-ineligible.
- For fold `k`, fold `(k+1) mod K` is validation and all remaining folds are
  training. Exact structures, scaffolds, and eligible MMP components are
  disjoint by construction.

## Frozen predictors and operator

- ECFP4-SVM: 1,024 bits, RBF, C=10, gamma=0.01, epsilon=0.1.
- ECFP4-RF: the frozen MoleculeACE recipe, validation-selected trees from
  {100, 250, 500, 1000}, three seeds.
- GINE: four 128-dimensional layers, dropout 0.2, the frozen optimizer and
  early-stopping recipe, three seeds.
- Reference: `0.5*SVM + 0.5*(0.5*RF + 0.5*GINE)`.
- BPS: internal held-out MMP edges only, activate when the absolute reference
  difference is at least 0.5 log unit, scale active-component centered signal
  by 1.5, and leave all other nodes unchanged.
- No AqSolDB target/fold-specific model or operator tuning is authorized.

## Endpoints, statistics, and decision

- Primary: true-cliff MMP pair-delta MAE, with a true cliff defined by absolute
  solubility difference at least 1.0 log unit.
- Secondary: cliff-endpoint MAE and all-molecule MAE.
- Report every seed-fold row, fold means, mean +/- standard deviation, favorable
  fold directions, and a fixed 10,000-resample paired fold bootstrap.
- Strong confirmation requires: positive primary improvement, at least four of
  five fold directions (or all folds when K<5), primary bootstrap upper bound
  below zero, and point costs no larger than 1% on either secondary endpoint.
  Results are retained and reported even if the gate fails.

## Compute and stopping

Smoke one fold/seed with shortened RF and GINE training before the full matrix.
Use only idle GPUs and no more than 352 logical CPU threads on the 384-thread
host so at least 30 cores remain available. Existing sensitivity and graph
control experiments are reused rather than rerun.
