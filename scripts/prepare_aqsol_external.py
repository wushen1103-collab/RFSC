#!/usr/bin/env python3
"""Two-stage AqSolDB preparation with a physical structure/response firewall."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import rdBase

try:
    from standardize_molecules import stable_mol_id, standardize_one
except ModuleNotFoundError:
    from scripts.standardize_molecules import stable_mol_id, standardize_one


TARGET = "AQSOLDB_SOLUBILITY"
EXPECTED_SHA256 = "1f9c4bb47c2c62b515719717899191f11a9772e7d30cc5b0443e0050bed2eae2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_record(item: tuple[int, str]) -> tuple[int, str, dict | None, str | None]:
    index, smiles = item
    try:
        return index, smiles, standardize_one(smiles), None
    except Exception as exc:
        return index, smiles, None, f"{type(exc).__name__}: {exc}"


def prepare_structures(source: Path, output_root: Path, workers: int) -> None:
    digest = sha256_file(source)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"AqSolDB source hash mismatch: {digest}")
    # Deliberately select only the non-response column in this stage.
    smiles = pd.read_csv(source, usecols=["SMILES"])["SMILES"].astype(str)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(standardize_record, enumerate(smiles), chunksize=64))
    valid, invalid = [], []
    for row_index, raw_smiles, structure, error in records:
        if structure is None:
            invalid.append({"source_row": row_index, "raw_smiles": raw_smiles, "reason": error})
            continue
        mol_id = stable_mol_id(TARGET, str(structure["canonical_isomeric_smiles"]))
        valid.append({"source_row": row_index, "mol_id": mol_id, **structure})
    frame = pd.DataFrame(valid)
    if frame.empty:
        raise RuntimeError("No valid AqSolDB structures")
    out = output_root / TARGET
    out.mkdir(parents=True, exist_ok=True)
    structure_columns = [
        "mol_id", "canonical_isomeric_smiles", "canonical_no_stereo_smiles",
        "full_inchikey", "connectivity_key", "scaffold_smiles", "scaffold_group",
        "heavy_atoms", "raw_formal_charge", "standardized_formal_charge", "salt_removed",
    ]
    structures = frame.drop_duplicates("mol_id", keep="first")[structure_columns].reset_index(drop=True)
    structures.to_parquet(out / "structures.parquet", index=False)
    frame[["source_row", "mol_id"]].to_parquet(out / "structure_row_map.parquet", index=False)
    pd.DataFrame(invalid).to_json(out / "invalid_rows.jsonl", orient="records", lines=True)
    manifest = {
        "dataset": "AqSolDB", "target": TARGET, "stage": "structures_only",
        "source_path": str(source), "source_sha256": digest, "raw_rows": int(len(smiles)),
        "valid_rows": int(len(frame)), "invalid_rows": int(len(invalid)),
        "unique_molecules": int(len(structures)), "response_column_read": False,
        "rdkit_version": rdBase.rdkitVersion,
    }
    (out / "structure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def prepare_labels(source: Path, output_root: Path) -> None:
    digest = sha256_file(source)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"AqSolDB source hash mismatch: {digest}")
    out = output_root / TARGET
    structure_manifest = json.loads((out / "structure_manifest.json").read_text(encoding="utf-8"))
    if structure_manifest.get("response_column_read") is not False:
        raise RuntimeError("Invalid structure-only manifest")
    row_map = pd.read_parquet(out / "structure_row_map.parquet")
    response = pd.read_csv(source, usecols=["Solubility"])["Solubility"]
    selected = pd.DataFrame({"source_row": np.arange(len(response)), "y": pd.to_numeric(response, errors="coerce")})
    selected = row_map.merge(selected, on="source_row", how="left", validate="many_to_one")
    if not np.isfinite(selected["y"]).all():
        raise RuntimeError("Non-finite AqSolDB response among valid structures")
    grouped = selected.groupby("mol_id", sort=True)["y"]
    labels = grouped.agg([("y_pactivity", "median"), ("replicate_count", "size"), ("replicate_min", "min"), ("replicate_max", "max")]).reset_index()
    labels["replicate_range"] = labels["replicate_max"] - labels["replicate_min"]
    labels.to_parquet(out / "labels_private.parquet", index=False)
    manifest = {
        "dataset": "AqSolDB", "target": TARGET, "stage": "labels_materialized_after_structure_freeze",
        "source_sha256": digest, "unique_labeled_molecules": int(len(labels)),
        "duplicate_aggregation": "median", "response_interface_column": "y_pactivity",
        "response_semantics": "supplied_log_aqueous_solubility",
    }
    (out / "label_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["structures", "labels"], required=True)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    if args.stage == "structures":
        prepare_structures(args.source, args.output_dir, args.workers)
    else:
        prepare_labels(args.source, args.output_dir)


if __name__ == "__main__":
    main()
