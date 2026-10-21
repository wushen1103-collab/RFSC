# Reproducing the RFSC evidence package

Run commands from the repository root. Commands write to `reproduced/` so the
released aggregate artifacts remain unchanged.

## 1. Environment

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple -r environment/requirements-final.lock
python -m pip install -e . --no-deps
```

The mirror is optional and only affects package download transport.

## 2. Integrity and table audits

```bash
python scripts/audit_release_package.py
python scripts/audit_structural_release.py
python scripts/build_expansion_tables.py --output-dir reproduced/tables/benchmark_expansion
python scripts/build_structural_audit_tables.py
```

`reproduction/main_tables.yaml` maps the main tables to their source artifacts,
commands, and evidence-status labels.

## 3. Figure reproduction

```bash
python scripts/draw_figures.py --input figure_data --out reproduced/figures
```

The journal-neutral implementation `scripts/draw_figures.py` uses curated CSV
sources in `figure_data/`. The legacy filename is a compatibility launcher only.
Final reference PDFs are provided in `figures/final/`.

## 4. External upstream data

Full experiment reruns require public upstream resources that are not vendored:

- MoleculeACE benchmark data.
- ChEMBL 36 SQLite archive.
- BindingDB exports used for patent-era/source-shift evaluation.
- QM9 molecular-property data.
- NOAA GHCN/HCN records for the nonmolecular transfer check.
- DeepDelta, pinned at commit
  `cd9b131559d54be639fcb9455e4c057a3dfd2c6b`.

Example DeepDelta setup:

```bash
git clone https://github.com/RekerLab/DeepDelta external/DeepDelta
git -C external/DeepDelta checkout cd9b131559d54be639fcb9455e4c057a3dfd2c6b
```

Review `RELEASE_AND_LICENSE.md` before redistributing any upstream or row-level
derived artifact.

## 5. Representative reruns

DeepDelta cycle projection:

```bash
python scripts/build_deepdelta_cycle_predictions.py \
  --config configs/deepdelta_cycle_projection_external.yaml \
  --source-root external/DeepDelta/Results/DeepDelta_Individual_CV_Folds \
  --output-dir reproduced/deepdelta_cycle/predictions
python scripts/evaluate_deepdelta_cycle_projection.py \
  --config configs/deepdelta_cycle_projection_external.yaml \
  --source-root external/DeepDelta/Results/DeepDelta_Individual_CV_Folds \
  --data-root external/DeepDelta/Datasets/Benchmarks \
  --prediction-root reproduced/deepdelta_cycle/predictions \
  --output-dir reproduced/deepdelta_cycle/evaluation
```

Direct Delta-LightGBM architecture:

```bash
python scripts/build_deepdelta_delta_lgbm_predictions.py \
  --config configs/deepdelta_delta_lgbm_projection_external.yaml \
  --data-root external/DeepDelta/Datasets/Benchmarks \
  --output-dir reproduced/delta_lgbm/predictions
python scripts/evaluate_deepdelta_delta_lgbm_projection.py \
  --config configs/deepdelta_delta_lgbm_projection_external.yaml \
  --data-root external/DeepDelta/Datasets/Benchmarks \
  --prediction-root reproduced/delta_lgbm/predictions \
  --output-dir reproduced/delta_lgbm/evaluation
```

Projection scaling:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_projection_scaling.py \
  --output-dir reproduced/projection_scaling --repeats 5
```

The repository stores aggregate summaries and hashes, not large frozen
prediction matrices. Regenerate those matrices locally when full reruns are
needed.

## 6. Additional structural audits (Tables S8--S11)

Read `reproduction/structural_audits.md`. Released CSVs support lightweight
summary verification and table regeneration. Full audits reuse upstream frozen
predictions, graphs, selector scores, and checkpoints; these larger artifacts must
first be regenerated under the recorded protocols. No audit command is a claim
that unbundled row-level artifacts have been independently reproduced.
