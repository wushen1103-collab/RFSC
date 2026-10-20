# Reproducing Tables S8--S11

## Lightweight verification (no raw data or training)

From the repository root:

```bash
python scripts/audit_structural_release.py
python scripts/build_structural_audit_tables.py
```

The first command checks source-file hashes, run counts, validation-only parameter
selection, the added all-active-MMP improve/worsen statistics, and current
deployment interpretation. The second regenerates four LaTeX tables at
`reproduced/tables/structural_audits.tex`. It uses the published CSVs only and
does not change thresholds or rerun experiments. The LaTeX fragment expects
amsmath, booktabs, tabularx, array, float and a ragged-right X-type column `Y`.
Use `\renewcommand{\thetable}{S\arabic{table}}` and set the table counter to 7
before including the fragment to obtain S8--S11.

## Full prediction-reuse audits

The original base-prediction matrices, split-specific graphs, selector scores,
development records and checkpoints are not bundled. Regenerate or supply them
under the original protocols before running the full audits. Scripts retain the
established relative experiment layout, including legacy artifact-directory names;
these names are provenance keys, not current journal or deployment-status claims.
The additional selector-development frame is expected at
`results/is_revision2_selector_coverage_curve/development_selector_frame.parquet`;
it contains the development records supplied to the frozen selector, not test data.
The original fitted selector is expected under
`experiments/eswa_response_free_risk/model/`, and the original AqSolDB scored
predictions under `experiments/eswa_response_free_risk/aqsol_scores/`.

```bash
mkdir -p reproduced/structural_audits
cp protocol/structural_audits.md reproduced/structural_audits/PROTOCOL_FROZEN.md
# Only if validation predictions are missing; checkpoint inference, not training:
python scripts/infer_audit_validation.py --smoke
python scripts/infer_audit_validation.py
for stage in mechanism fair structure perturb selector; do
    python scripts/run_structural_audits.py "$stage" --smoke
    python scripts/run_structural_audits.py "$stage"
done
python scripts/verify_structural_audits.py
python scripts/finalize_structural_audits.py
```

Set `RFSC_WORKSPACE_ROOT` to an alternative regenerated artifact workspace and
`RFSC_AUDIT_OUTPUT` to an alternative output directory within that workspace if needed. Defaults are the
public repository root and `reproduced/structural_audits`, respectively. These
commands have **not** been rerun on unbundled row-level data as part of this
publication update. Published executor checks describe the original execution,
not a new independent reproduction.

## Statistical and scientific interpretation

- S8: same validation-only grid, budget and identity option across postprocessors;
  paired task bootstrap, not node/pair bootstrap. Pair superiority over Laplacian
  is not established. Canonical and calibrated results are separate.
- S9: all-active-MMP improvement/worsening is 30.85%/69.15% on the manuscript's
  23 MoleculeACE tasks and 26.74%/73.26% on corrected ChEMBL. These all-edge
  percentages differ from the true-cliff improvement rates of 74.49%/77.53%.
- S10: full-operator guarantees and both-accepted-edge amplification are verified
  numerically; nodewise mixing is a distinct operation.
- S11: the fixed-graph bound holds; graph reselection can produce localized jumps.
- The AqSolDB primary fold-paired candidate estimates remain distinct from the
  descriptive estimator in Table S6. Subsequent deployment authorization is
  NO-GO with identity, not a relabeling of the evaluated candidate predictions.

No manuscript PDFs, author details, local machine identifiers, raw private labels,
prediction matrices or checkpoints are included in this update.
