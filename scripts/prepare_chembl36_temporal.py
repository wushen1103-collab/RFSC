#!/usr/bin/env python3
"""Curate a label-firewalled ChEMBL 36 temporal Ki benchmark."""

from __future__ import annotations

import argparse
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import rdBase

try:
    from standardize_molecules import stable_mol_id, standardize_one
except ModuleNotFoundError:
    from scripts.standardize_molecules import stable_mol_id, standardize_one


REQUIRED_COLUMNS = {
    "activities": {
        "activity_id", "assay_id", "doc_id", "molregno", "standard_relation",
        "standard_value", "standard_units", "standard_type",
    },
    "assays": {"assay_id", "assay_type", "tid", "confidence_score"},
    "target_dictionary": {"tid", "target_type", "pref_name", "organism", "chembl_id"},
    "compound_structures": {"molregno", "canonical_smiles"},
    "molecule_dictionary": {"molregno", "chembl_id"},
    "docs": {"doc_id", "year"},
}


ACTIVITY_QUERY = """
SELECT
    td.chembl_id AS target_chembl_id,
    td.pref_name AS target_name,
    md.chembl_id AS molecule_chembl_id,
    act.molregno AS molregno,
    cs.canonical_smiles AS raw_smiles,
    d.year AS publication_year,
    act.activity_id AS activity_id,
    CAST(act.standard_value AS REAL) AS ki_nm
FROM activities AS act
JOIN assays AS ass ON ass.assay_id = act.assay_id
JOIN target_dictionary AS td ON td.tid = ass.tid
JOIN compound_structures AS cs ON cs.molregno = act.molregno
JOIN molecule_dictionary AS md ON md.molregno = act.molregno
JOIN docs AS d ON d.doc_id = act.doc_id
WHERE td.target_type = 'SINGLE PROTEIN'
  AND td.organism = 'Homo sapiens'
  AND ass.assay_type = 'B'
  AND ass.confidence_score = 9
  AND act.standard_type = 'Ki'
  AND act.standard_relation = '='
  AND act.standard_units = 'nM'
  AND act.standard_value IS NOT NULL
  AND CAST(act.standard_value AS REAL) > 0
  AND cs.canonical_smiles IS NOT NULL
  AND d.year IS NOT NULL
ORDER BY td.chembl_id, act.molregno, d.year, act.activity_id
"""


def readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def audit_schema(connection: sqlite3.Connection) -> dict[str, list[str]]:
    observed: dict[str, list[str]] = {}
    for table, required in REQUIRED_COLUMNS.items():
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        columns = [str(row[1]) for row in rows]
        missing = required - set(columns)
        if missing:
            raise RuntimeError(f"{table} lacks required columns: {sorted(missing)}")
        observed[table] = columns
    return observed


def read_activity_rows(connection: sqlite3.Connection, limit: int | None = None) -> pd.DataFrame:
    query = ACTIVITY_QUERY
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query += f"\nLIMIT {int(limit)}"
    rows = pd.read_sql_query(query, connection)
    if rows.empty:
        raise RuntimeError("ChEMBL query returned no eligible rows")
    rows["ki_nm"] = pd.to_numeric(rows["ki_nm"], errors="coerce")
    rows["publication_year"] = pd.to_numeric(rows["publication_year"], errors="coerce")
    finite = np.isfinite(rows["ki_nm"]) & (rows["ki_nm"] > 0) & np.isfinite(rows["publication_year"])
    if not bool(finite.all()):
        raise RuntimeError("SQL curation admitted invalid Ki values or years")
    return rows


def _standardize_record(record: tuple[int, str]) -> dict[str, object]:
    molregno, smiles = record
    try:
        return {"molregno": molregno, "standardization_error": None, **standardize_one(smiles)}
    except Exception as exc:
        return {
            "molregno": molregno,
            "standardization_error": f"{type(exc).__name__}: {exc}",
        }


def standardize_unique_molecules(rows: pd.DataFrame, workers: int) -> pd.DataFrame:
    unique = (
        rows.sort_values(["molregno", "activity_id"])
        .drop_duplicates("molregno", keep="first")[["molregno", "raw_smiles"]]
    )
    records = list(unique.itertuples(index=False, name=None))
    with ProcessPoolExecutor(max_workers=min(workers, len(records))) as pool:
        standardized = list(pool.map(_standardize_record, records, chunksize=64))
    return pd.DataFrame.from_records(standardized)


def assign_partition(year: int, train_max: int, validation_years: set[int], test_min: int) -> str | None:
    if year <= train_max:
        return "train"
    if year in validation_years:
        return "validation"
    if year >= test_min:
        return "test"
    return None


def curate_rows(rows: pd.DataFrame, standardized: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    valid_structures = standardized[standardized["standardization_error"].isna()].drop(
        columns="standardization_error"
    )
    merged = rows.merge(valid_structures, on="molregno", how="inner", validate="many_to_one")
    merged["y_pactivity"] = 9.0 - np.log10(merged["ki_nm"].astype(float))
    discordance = float(config["curation"]["duplicate_max_range_log_unit"])

    yearly_records: list[dict[str, object]] = []
    discordant_groups = 0
    for (target, smiles, year), group in merged.groupby(
        ["target_chembl_id", "canonical_isomeric_smiles", "publication_year"], sort=True
    ):
        spread = float(group["y_pactivity"].max() - group["y_pactivity"].min())
        if spread > discordance:
            discordant_groups += 1
            continue
        first = group.iloc[0]
        yearly_records.append(
            {
                "target_chembl_id": target,
                "target_name": first["target_name"],
                "canonical_isomeric_smiles": smiles,
                "publication_year": int(year),
                "year_median_pactivity": float(group["y_pactivity"].median()),
                "year_replicate_count": int(len(group)),
                "year_replicate_range": spread,
                "molregno": int(first["molregno"]),
                "molecule_chembl_id": first["molecule_chembl_id"],
            }
        )
    yearly = pd.DataFrame.from_records(yearly_records)
    if yearly.empty:
        raise RuntimeError("No rows remained after duplicate quality control")

    structure_lookup = valid_structures.drop_duplicates("canonical_isomeric_smiles").set_index(
        "canonical_isomeric_smiles", drop=False
    )
    molecule_records: list[dict[str, object]] = []
    for (target, smiles), group in yearly.groupby(
        ["target_chembl_id", "canonical_isomeric_smiles"], sort=True
    ):
        ordered = group.sort_values("publication_year")
        first = ordered.iloc[0]
        structure = structure_lookup.loc[smiles]
        earliest_year = int(first["publication_year"])
        molecule_records.append(
            {
                "target_chembl_id": target,
                "target_name": first["target_name"],
                "mol_id": stable_mol_id(str(target), str(smiles)),
                "molecule_chembl_id": first["molecule_chembl_id"],
                "earliest_publication_year": earliest_year,
                "y_pactivity": float(first["year_median_pactivity"]),
                "replicate_count": int(first["year_replicate_count"]),
                "accepted_year_count": 1,
                "observed_year_count": int(len(ordered)),
                "excluded_later_year_count": int(len(ordered) - 1),
                "maximum_within_year_range": float(first["year_replicate_range"]),
                **{
                    column: structure[column]
                    for column in [
                        "canonical_isomeric_smiles", "canonical_no_stereo_smiles",
                        "full_inchikey", "connectivity_key", "scaffold_smiles",
                        "scaffold_group", "heavy_atoms", "raw_formal_charge",
                        "standardized_formal_charge", "salt_removed",
                    ]
                },
            }
        )
    molecules = pd.DataFrame.from_records(molecule_records)
    split = config["time_split"]
    validation_years = {int(value) for value in split["validation_years"]}
    molecules["partition"] = molecules["earliest_publication_year"].map(
        lambda year: assign_partition(
            int(year), int(split["train_max_year"]), validation_years, int(split["test_min_year"])
        )
    )
    molecules = molecules[molecules["partition"].notna()].copy()
    diagnostics = {
        "queried_activity_rows": int(len(rows)),
        "unique_raw_molregno": int(rows["molregno"].nunique()),
        "valid_standardized_molregno": int(len(valid_structures)),
        "invalid_standardized_molregno": int(standardized["standardization_error"].notna().sum()),
        "discordant_target_molecule_year_groups": int(discordant_groups),
        "curated_target_molecules": int(len(molecules)),
        "molecules_with_later_year_measurements_excluded": int(
            (molecules["excluded_later_year_count"] > 0).sum()
        ),
        "later_year_groups_excluded": int(molecules["excluded_later_year_count"].sum()),
    }
    return molecules, diagnostics


def write_firewalled_dataset(molecules: pd.DataFrame, output_dir: Path, config: dict, smoke: bool) -> pd.DataFrame:
    eligibility = config["eligibility_structure_only"]
    counts = (
        molecules.groupby(["target_chembl_id", "target_name", "partition"])
        .size().unstack(fill_value=0).reset_index()
    )
    for partition in ["train", "validation", "test"]:
        if partition not in counts:
            counts[partition] = 0
    counts["count_eligible"] = (
        (counts["train"] >= int(eligibility["minimum_train_molecules"]))
        & (counts["validation"] >= int(eligibility["minimum_validation_molecules"]))
        & (counts["test"] >= int(eligibility["minimum_test_molecules"]))
    )
    eligible = set(counts.loc[counts["count_eligible"], "target_chembl_id"])
    if smoke:
        eligible = set(counts["target_chembl_id"])
        counts["count_eligible"] = counts["target_chembl_id"].isin(eligible)

    structure_columns = [
        "mol_id", "canonical_isomeric_smiles", "canonical_no_stereo_smiles",
        "full_inchikey", "connectivity_key", "scaffold_smiles", "scaffold_group",
        "heavy_atoms", "raw_formal_charge", "standardized_formal_charge", "salt_removed",
    ]
    for target in sorted(eligible):
        target_rows = molecules[molecules["target_chembl_id"] == target].copy()
        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        target_rows[structure_columns].sort_values("mol_id").to_parquet(
            target_dir / "structures.parquet", index=False
        )
        target_rows[["mol_id", "earliest_publication_year", "partition"]].sort_values(
            "mol_id"
        ).to_parquet(target_dir / "partitions.parquet", index=False)
        target_rows[
            [
                "mol_id", "y_pactivity", "replicate_count", "accepted_year_count",
                "observed_year_count", "excluded_later_year_count", "maximum_within_year_range",
            ]
        ].sort_values("mol_id").to_parquet(target_dir / "labels_private.parquet", index=False)
        metadata = {
            "target_chembl_id": target,
            "target_name": str(target_rows["target_name"].iloc[0]),
            "smoke": smoke,
            "response_in_structures": False,
            "response_in_partitions": False,
        }
        (target_dir / "target_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
    return counts.sort_values("target_chembl_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.workers > 320:
        raise SystemExit("workers must be in [1,320] to preserve the CPU reserve")
    if args.smoke and args.limit is None:
        raise SystemExit("--smoke requires --limit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_hash = str(config["source"]["sha256"])
    if not args.database.exists():
        raise SystemExit(f"Database not found: {args.database}")
    with readonly_connection(args.database) as connection:
        schema = audit_schema(connection)
        print(json.dumps({"schema_ok": True, "tables": {k: len(v) for k, v in schema.items()}}, indent=2))
        if args.schema_only:
            return
        rows = read_activity_rows(connection, args.limit)

    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --schema-only is used")
    standardized = standardize_unique_molecules(rows, args.workers)
    molecules, diagnostics = curate_rows(rows, standardized, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = write_firewalled_dataset(molecules, args.output_dir, config, args.smoke)
    counts.to_csv(args.output_dir / "target_count_audit.csv", index=False)
    invalid = standardized[standardized["standardization_error"].notna()][
        ["molregno", "standardization_error"]
    ]
    invalid.to_json(args.output_dir / "invalid_structures.jsonl", orient="records", lines=True)
    manifest = {
        "dataset": "ChEMBL 36 temporal Ki",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "database_path": str(args.database),
        "source_archive_sha256_expected": expected_hash,
        "database_bytes": args.database.stat().st_size,
        "rdkit_version": rdBase.rdkitVersion,
        "smoke": args.smoke,
        "row_limit": args.limit,
        "response_summary_exposed": False,
        **diagnostics,
    }
    (args.output_dir / "curation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({**diagnostics, "count_eligible_targets": int(counts["count_eligible"].sum())}, indent=2))


if __name__ == "__main__":
    main()
