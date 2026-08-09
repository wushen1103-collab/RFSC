# RFSC

Response-Free Structural Correction (RFSC) is a research code release for
post-prediction structural correction in molecular property prediction.

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
  artifacts needed to audit the ESWA submission results.
- `paper/tables/`: manuscript-ready CSV/Markdown tables.
- `figure_data/`: curated source data used by the released figure script.
- `figures/final/`: final PDF figures for reference.
- `reproduction/main_tables.yaml`: table-to-source lineage.

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
python scripts/build_expansion_tables.py
python scripts/draw_eswa_figures.py --input figure_data --out reproduced/figures
```

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
