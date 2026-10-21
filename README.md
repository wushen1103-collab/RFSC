# RFSC

Response-Free Structural Correction (RFSC) is the reproducibility release for
"Response-Free Structural Correction of Molecular Machine-Learning Predictions
with Selective Risk Gating," including the MLST revision audits (Tables S8--S11).

The repository contains two correction layers:

- `BPS`: a boundary-preserving node-side correction operator for molecular
  activity prediction on query graphs.
- `Hodge`: a classical incidence/Hodge projection used as an edge-flow
  correction layer for direct molecular property-difference predictions.

Both operators are applied after base-model prediction. They do not use query
responses at deployment time and do not retrain the upstream predictor.

## What is included

- `bps_gnn/`: reusable Python package code.
- `scripts/`: experiment, evaluation, aggregation, audit, and figure scripts.
- `configs/`: frozen experiment configurations with public-path placeholders.
- `protocol/`: preregistered experiment and decision-gate notes.
- `results/`: lightweight aggregate result tables and decision summaries.
- `experiments/*/evaluation` and selected `model` files: small evaluation
  artifacts needed to audit the original benchmark results.
- `paper/tables/`: manuscript-ready CSV/Markdown tables.
- `figure_data/`: curated source data used by the released figure script.
- `figures/final/`: final PDF figures for reference.
- `reproduction/main_tables.yaml`: table-to-source lineage.
- `results/structural_audits/`: frozen validation-only controls, mechanism,
  selective-execution, stability, and selector-provenance audits (Tables S8--S11).

Large raw datasets, upstream checkouts, frozen prediction matrices, model
checkpoints, caches, local logs, and machine-specific paths are intentionally
excluded.

## Quick start

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple -r environment/requirements-final.lock
python -m pip install -e . --no-deps
pytest
```

The mirror above only changes package transport. Package identities are fixed
by the lock file.

## Reproduce the released evidence

See `REPRODUCING.md` for the ordered commands. The short version is:

```bash
python scripts/audit_release_package.py
python scripts/audit_structural_release.py
python scripts/build_expansion_tables.py
python scripts/build_structural_audit_tables.py
```

The new audit commands verify the released summaries and regenerate Tables
S8--S11 without model training. See
[`reproduction/structural_audits.md`](reproduction/structural_audits.md) for full
prediction-reuse protocols, additional artifact requirements, and evidence timing.
Optional figure reproduction uses the journal-neutral implementation
`scripts/draw_figures.py`, consistently across this README, `REPRODUCING.md`,
and the figure-data documentation. The legacy script name is retained only as a
compatibility launcher. This documentation/entry-point update does not change
reference figures, numerical results, configurations, or frozen audit evidence.

## Interpreting selective correction

The frozen AqSolDB selector has measured **candidate** boundary gains, not a
deployment risk guarantee. Predictions and thresholds are frozen before external
response access; the completed risk audit then gives **NO-GO for subsequent
deployment**, requiring identity fallback. Historical diagnostic decision fields
are retained as provenance, not reinterpreted as strict deployment authorization.
The current interpretation is recorded in
`results/structural_audits/interpretation.json`.

Some full reruns require public upstream datasets or repositories such as
MoleculeACE, ChEMBL 36, BindingDB, QM9, NOAA GHCN/HCN, and DeepDelta. The
release keeps exact configuration, split, hash, and aggregate-result records so
that those larger artifacts can be regenerated locally after the user obtains
the corresponding upstream data.

## Data and license boundaries

This repository does not vendor third-party datasets or upstream prediction
files. Review `RELEASE_AND_LICENSE.md` before redistributing any row-level
data derived from third-party resources.

No top-level reuse license has been selected in this release. A license should
be added only after author and institutional approval.
