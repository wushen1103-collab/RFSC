#!/usr/bin/env python3
"""Curate a source-independent BindingDB-articles temporal affinity benchmark."""

from __future__ import annotations

import argparse
import json
import re
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


COLUMNS = {
    "smiles": "Ligand SMILES",
    "monomer_id": "BindingDB MonomerID",
    "target_name": "Target Name",
    "organism": "Target Source Organism According to Curator or DataSource",
    "ki": "Ki (nM)",
    "kd": "Kd (nM)",
    "source": "Curation/DataSource",
    "doi": "Article DOI",
    "pmid": "PMID",
    "patent_number": "Patent Number",
    "publication_date": "Date of publication",
    "chain_count": "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "uniprot": "UniProt (SwissProt) Primary ID of Target Chain 1",
}
EXACT_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def parse_exact_positive_nm(value: object) -> float | None:
    text = str(value).strip()
    if not EXACT_NUMBER.fullmatch(text):
        return None
    parsed = float(text)
    return parsed if np.isfinite(parsed) and parsed > 0 else None


def publication_year(value: object) -> int | None:
    text = str(value).strip()
    match = re.match(r"^(\d{4})(?:$|[-/])", text) or re.search(r"(?:^|[-/])(\d{4})$", text)
    if not match:
        return None
    year = int(match.group(1))
    return year if 1900 <= year <= 2100 else None


def read_bindingdb_rows(
    path: Path, config: dict, row_limit: int | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=list(COLUMNS.values()),
        dtype=str,
        keep_default_na=False,
        on_bad_lines="error",
        nrows=row_limit,
    )
    source_filter = str(config["curation"].get("source_filter", "exact_value"))
    if source_filter == "exact_value":
        source_condition = frame[COLUMNS["source"]].str.strip().eq(
            str(config["curation"]["curation_source"])
        )
    elif source_filter == "official_subset_file_only":
        source_condition = pd.Series(True, index=frame.index)
    else:
        raise ValueError(f"Unknown BindingDB source filter: {source_filter}")
    organisms = {str(value) for value in config["curation"]["target_organisms"]}
    base = frame[
        source_condition
        & frame[COLUMNS["organism"]].str.strip().isin(organisms)
        & frame[COLUMNS["chain_count"]].str.strip().eq("1")
        & frame[COLUMNS["uniprot"]].str.strip().ne("")
        & frame[COLUMNS["smiles"]].str.strip().ne("")
    ].copy()
    if bool(config["curation"].get("require_patent_number", False)):
        base = base[base[COLUMNS["patent_number"]].str.strip().ne("")].copy()
    base["publication_year"] = base[COLUMNS["publication_date"]].map(publication_year)
    base = base[base["publication_year"].notna()].copy()

    records: list[dict[str, object]] = []
    requested = [str(value) for value in config["curation"]["measurement_types"]]
    measurement_columns = {"Ki": COLUMNS["ki"], "Kd": COLUMNS["kd"]}
    for measurement in requested:
        column = measurement_columns[measurement]
        for row_index, row in base.iterrows():
            affinity_nm = parse_exact_positive_nm(row[column])
            if affinity_nm is None:
                continue
            uniprot = str(row[COLUMNS["uniprot"]]).strip()
            records.append(
                {
                    "source_row": int(row_index),
                    "target_task_id": f"{uniprot}_{measurement}",
                    "target_name": str(row[COLUMNS["target_name"]]).strip(),
                    "uniprot_id": uniprot,
                    "measurement_type": measurement,
                    "bindingdb_monomer_id": str(row[COLUMNS["monomer_id"]]).strip(),
                    "raw_smiles": str(row[COLUMNS["smiles"]]).strip(),
                    "publication_year": int(row["publication_year"]),
                    "article_doi": str(row[COLUMNS["doi"]]).strip(),
                    "pmid": str(row[COLUMNS["pmid"]]).strip(),
                    "patent_number": str(row[COLUMNS["patent_number"]]).strip(),
                    "affinity_nm": affinity_nm,
                }
            )
    rows = pd.DataFrame.from_records(records)
    if rows.empty:
        raise RuntimeError("No exact positive BindingDB affinity rows passed the frozen filters")
    diagnostics = {
        "source_rows": int(len(frame)),
        "human_single_chain_rows_with_year": int(len(base)),
        "exact_affinity_task_rows": int(len(rows)),
    }
    return rows, diagnostics


def _standardize_record(record: tuple[str, str]) -> dict[str, object]:
    raw_smiles, smiles = record
    try:
        return {"raw_smiles": raw_smiles, "standardization_error": None, **standardize_one(smiles)}
    except Exception as exc:
        return {
            "raw_smiles": raw_smiles,
            "standardization_error": f"{type(exc).__name__}: {exc}",
        }


def standardize_unique_molecules(rows: pd.DataFrame, workers: int) -> pd.DataFrame:
    unique = rows.sort_values("source_row").drop_duplicates("raw_smiles", keep="first")
    records = [(str(value), str(value)) for value in unique["raw_smiles"]]
    with ProcessPoolExecutor(max_workers=min(workers, len(records))) as pool:
        standardized = list(pool.map(_standardize_record, records, chunksize=64))
    return pd.DataFrame.from_records(standardized)


def assign_partition(year: int, config: dict) -> str | None:
    split = config["time_split"]
    if year <= int(split["train_max_year"]):
        return "train"
    if year in {int(value) for value in split["validation_years"]}:
        return "validation"
    if year >= int(split["test_min_year"]):
        return "test"
    return None


def latest_year_block_partitions(molecules: pd.DataFrame, config: dict) -> pd.Series:
    """Assign whole publication-year blocks to target-specific 70/10/20 eras."""
    result = pd.Series(index=molecules.index, dtype="object")
    test_fraction = float(config["time_split"]["test_fraction"])
    validation_fraction = float(config["time_split"]["validation_fraction"])
    for _, group in molecules.groupby("target_task_id", sort=True):
        counts = group["earliest_publication_year"].value_counts().sort_index()
        years_desc = sorted((int(year) for year in counts.index), reverse=True)
        test_target = max(1, int(np.ceil(test_fraction * len(group))))
        validation_target = max(1, int(np.ceil(validation_fraction * len(group))))
        test_years: set[int] = set()
        validation_years: set[int] = set()
        accumulated = 0
        for year in years_desc:
            if accumulated >= test_target:
                break
            test_years.add(year)
            accumulated += int(counts.loc[year])
        accumulated = 0
        for year in years_desc:
            if year in test_years:
                continue
            if accumulated >= validation_target:
                break
            validation_years.add(year)
            accumulated += int(counts.loc[year])
        for index, year in group["earliest_publication_year"].items():
            numeric_year = int(year)
            if numeric_year in test_years:
                result.loc[index] = "test"
            elif numeric_year in validation_years:
                result.loc[index] = "validation"
            else:
                result.loc[index] = "train"
    return result


def curate_rows(
    rows: pd.DataFrame, standardized: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, dict[str, int]]:
    valid = standardized[standardized["standardization_error"].isna()].drop(
        columns="standardization_error"
    )
    merged = rows.merge(valid, on="raw_smiles", how="inner", validate="many_to_one")
    merged["y_pactivity"] = 9.0 - np.log10(merged["affinity_nm"].astype(float))
    max_range = float(config["curation"]["duplicate_max_range_log_unit"])

    yearly_records: list[dict[str, object]] = []
    discordant_groups = 0
    for (task, smiles, year), group in merged.groupby(
        ["target_task_id", "canonical_isomeric_smiles", "publication_year"], sort=True
    ):
        spread = float(group["y_pactivity"].max() - group["y_pactivity"].min())
        if spread > max_range:
            discordant_groups += 1
            continue
        first = group.iloc[0]
        yearly_records.append(
            {
                "target_task_id": task,
                "target_name": first["target_name"],
                "uniprot_id": first["uniprot_id"],
                "measurement_type": first["measurement_type"],
                "canonical_isomeric_smiles": smiles,
                "publication_year": int(year),
                "year_median_pactivity": float(group["y_pactivity"].median()),
                "year_replicate_count": int(len(group)),
                "year_replicate_range": spread,
                "bindingdb_monomer_id": first["bindingdb_monomer_id"],
            }
        )
    yearly = pd.DataFrame.from_records(yearly_records)
    if yearly.empty:
        raise RuntimeError("No BindingDB rows remained after duplicate quality control")

    structures = valid.drop_duplicates("canonical_isomeric_smiles").set_index(
        "canonical_isomeric_smiles", drop=False
    )
    molecule_records: list[dict[str, object]] = []
    for (task, smiles), group in yearly.groupby(
        ["target_task_id", "canonical_isomeric_smiles"], sort=True
    ):
        ordered = group.sort_values("publication_year")
        first = ordered.iloc[0]
        structure = structures.loc[smiles]
        molecule_records.append(
            {
                "target_task_id": task,
                "target_name": first["target_name"],
                "uniprot_id": first["uniprot_id"],
                "measurement_type": first["measurement_type"],
                "mol_id": stable_mol_id(str(task), str(smiles)),
                "bindingdb_monomer_id": first["bindingdb_monomer_id"],
                "earliest_publication_year": int(first["publication_year"]),
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
    split_mode = str(config["time_split"].get("mode", "fixed_calendar"))
    if split_mode == "fixed_calendar":
        molecules["partition"] = molecules["earliest_publication_year"].map(
            lambda year: assign_partition(int(year), config)
        )
    elif split_mode == "task_latest_year_blocks":
        molecules["partition"] = latest_year_block_partitions(molecules, config)
    else:
        raise ValueError(f"Unknown time split mode: {split_mode}")
    molecules = molecules[molecules["partition"].notna()].copy()
    diagnostics = {
        "valid_standardized_smiles": int(len(valid)),
        "invalid_standardized_smiles": int(standardized["standardization_error"].notna().sum()),
        "discordant_target_molecule_year_groups": int(discordant_groups),
        "curated_target_molecules": int(len(molecules)),
        "molecules_with_later_year_measurements_excluded": int(
            (molecules["excluded_later_year_count"] > 0).sum()
        ),
        "later_year_groups_excluded": int(molecules["excluded_later_year_count"].sum()),
    }
    return molecules, diagnostics


def write_firewalled_dataset(molecules: pd.DataFrame, output_dir: Path, config: dict, smoke: bool) -> pd.DataFrame:
    thresholds = config["eligibility_structure_only"]
    counts = molecules.groupby(
        ["target_task_id", "uniprot_id", "measurement_type", "partition"]
    ).size().unstack(fill_value=0).reset_index()
    names = molecules.sort_values(["target_task_id", "target_name"]).drop_duplicates(
        "target_task_id", keep="first"
    )[["target_task_id", "target_name"]]
    counts = counts.merge(names, on="target_task_id", how="left", validate="one_to_one")
    for partition in ["train", "validation", "test"]:
        if partition not in counts:
            counts[partition] = 0
    counts["count_eligible"] = (
        (counts["train"] >= int(thresholds["minimum_train_molecules"]))
        & (counts["validation"] >= int(thresholds["minimum_validation_molecules"]))
        & (counts["test"] >= int(thresholds["minimum_test_molecules"]))
    )
    eligible = set(counts.loc[counts["count_eligible"], "target_task_id"])
    if smoke:
        eligible = set(counts["target_task_id"])
        counts["count_eligible"] = counts["target_task_id"].isin(eligible)

    structure_columns = [
        "mol_id", "canonical_isomeric_smiles", "canonical_no_stereo_smiles",
        "full_inchikey", "connectivity_key", "scaffold_smiles", "scaffold_group",
        "heavy_atoms", "raw_formal_charge", "standardized_formal_charge", "salt_removed",
    ]
    label_columns = [
        "mol_id", "y_pactivity", "replicate_count", "accepted_year_count",
        "observed_year_count", "excluded_later_year_count", "maximum_within_year_range",
    ]
    for task in sorted(eligible):
        target_rows = molecules[molecules["target_task_id"] == task].copy()
        target_dir = output_dir / task
        target_dir.mkdir(parents=True, exist_ok=True)
        target_rows[structure_columns].sort_values("mol_id").to_parquet(
            target_dir / "structures.parquet", index=False
        )
        target_rows[["mol_id", "earliest_publication_year", "partition"]].sort_values(
            "mol_id"
        ).to_parquet(target_dir / "partitions.parquet", index=False)
        target_rows[label_columns].sort_values("mol_id").to_parquet(
            target_dir / "labels_private.parquet", index=False
        )
        metadata = {
            "target_task_id": task,
            "target_name": str(target_rows["target_name"].iloc[0]),
            "uniprot_id": str(target_rows["uniprot_id"].iloc[0]),
            "measurement_type": str(target_rows["measurement_type"].iloc[0]),
            "smoke": smoke,
            "response_in_structures": False,
            "response_in_partitions": False,
        }
        (target_dir / "target_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
    return counts.sort_values("target_task_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.workers > 320:
        raise SystemExit("workers must be in [1,320] to preserve the CPU reserve")
    if args.smoke and (args.row_limit is None or args.row_limit <= 0):
        raise SystemExit("--smoke requires a positive --row-limit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows, read_diagnostics = read_bindingdb_rows(args.input_tsv, config, args.row_limit)
    standardized = standardize_unique_molecules(rows, args.workers)
    molecules, curate_diagnostics = curate_rows(rows, standardized, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = write_firewalled_dataset(molecules, args.output_dir, config, args.smoke)
    counts.to_csv(args.output_dir / "target_count_audit.csv", index=False)
    standardized.loc[standardized["standardization_error"].notna(), [
        "raw_smiles", "standardization_error"
    ]].to_json(args.output_dir / "invalid_structures.jsonl", orient="records", lines=True)
    manifest = {
        "dataset": str(config["name"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input_tsv),
        "source_zip_md5_expected": str(config["source"]["md5"]),
        "input_bytes": int(args.input_tsv.stat().st_size),
        "rdkit_version": rdBase.rdkitVersion,
        "smoke": args.smoke,
        "row_limit": args.row_limit,
        "response_summary_exposed": False,
        **read_diagnostics,
        **curate_diagnostics,
    }
    (args.output_dir / "curation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        **read_diagnostics,
        **curate_diagnostics,
        "count_eligible_tasks": int(counts["count_eligible"].sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
