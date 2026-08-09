# Cross-benchmark structure-overlap audit

Status: frozen before computing any overlap count.

Purpose: quantify structural dependence among the four empirical evidence
sources without changing any completed experiment or claim after the fact.

Sources:

- all standardized MoleculeACE structures;
- corrected earliest-year-only ChEMBL 36 structures and its temporal test set;
- BindingDB patents-only latest-era structures and its temporal test set;
- all ten official DeepDelta ADMET datasets, standardized with the repository's
  existing structure-only pipeline.

The audit reads only structure tables, partition membership, and the DeepDelta
`SMILES` column. It must not read any response column. Exact identity is defined
by the canonical no-stereo parent SMILES after cleanup, largest-fragment
selection, and uncharging. Report:

1. unique structure counts per source and evaluation scope;
2. all-source pairwise intersections, containment fractions, and Jaccard;
3. evaluation-cohort intersections against every full source;
4. per-target overlap for shared MoleculeACE/ChEMBL target identifiers.

Every count is descriptive. High overlap must be disclosed; it does not
authorize filtering, resplitting, model reruns, or revised test selection.
