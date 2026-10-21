# Figure data availability

This directory contains curated CSV tables used by `scripts/draw_figures.py`.
The large `raw_sources/` export is not included in the public release.

Included tables are aggregate or plotting-level sources for:

- MoleculeACE, corrected ChEMBL, and untouched AqSolDB node-side effects.
- Frozen response-free selector diagnostics.
- DeepDelta, Delta-LightGBM, and QM9 edge-flow correction results.
- Random and structured missing-edge tests.
- Mechanism, ablation, inconsistency-energy, anchor-recovery, and runtime
  summaries.

Large case-candidate and molecule-level helper exports are intentionally
excluded from the public git history to keep the repository small and to avoid
redistributing unnecessary row-level material.
