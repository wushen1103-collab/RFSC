# BindingDB patents latest-era external confirmation freeze

Frozen after official-file checksum/header audit but before parsing an affinity
value or inspecting a target count or response distribution.

## Independent source and curation

- Use only the official BindingDB 2026-07 patents TSV, not the separately
  released ChEMBL, journal-article, PubChem, or PDSP subsets. The ZIP is
  175,001,352 bytes, has official/matched MD5
  `81e34d497cc88b2719aa9fbf13f4566c`, and contains the single
  3,806,662,313-byte `BindingDB_Patents.tsv` member. A nonblank patent number is
  required for every accepted row.
- Retain exact positive numeric Ki and Kd values in nM for human single-chain
  proteins with Swiss-Prot primary ID, SMILES, and publication year. Ki and Kd
  remain separate target-task units. Convert to
  `pAffinity=9-log10(value_nM)` with the fixed repository standardizer.
- Within target-task, standardized molecule, and publication year, retain the
  median only when the response range is at most one log unit. If a molecule
  occurs in several years, keep only the earliest accepted year's response.

## Split and response-free feasibility stop

- For each task independently, assign whole latest publication-year blocks to
  approximately 20% test, then preceding whole-year blocks to approximately
  10% validation, with all older years as train. A publication year is never
  split and standardized molecules remain disjoint.
- Keep the unchanged structure-only gates: at least 100/20/30 molecules in
  train/validation/test, at least eight internal-test MMP edges, and at least
  eight eligible tasks. Stop without changing a threshold if any gate fails.
- Report exact standardized-structure overlap with ChEMBL 36 and MoleculeACE as
  a secondary provenance diagnostic; do not silently describe source
  independence as universal chemical-space independence.

## Fixed model and decision

- Inherit the exact no-target-tuning ECFP-SVM, ECFP-RF, and GINE recipes and the
  fixed unscaled mixture `0.5 SVM + 0.25 RF + 0.25 GINE`.
- Transfer confidence-weighted BPS unchanged: internal-query MMP graph,
  predicted-delta threshold 0.5, scale 1.5, target-test median disagreement
  floor plus 1e-6, no clipping. Freeze every prediction/hash before test-label
  evaluation.
- The primary target-macro comparison requires pair improvement >=3%, pair
  bootstrap upper <0, pair improvement on >=70% of targets, overall and cliff
  endpoint point costs <=1%, both cost-bootstrap uppers <=1%, and at least eight
  pair-eligible tasks. Use 10,000 target bootstraps and Holm-corrected one-sided
  target tests. Test results cannot select another threshold, scale, model, or
  task subset.

A 5,000-row parser/standardizer/split smoke precedes full curation and has no
paper metric status.
