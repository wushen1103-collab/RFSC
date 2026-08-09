import sqlite3
from pathlib import Path

import pandas as pd

from scripts.prepare_chembl36_temporal import audit_schema, curate_rows, read_activity_rows
from scripts.standardize_molecules import standardize_one


SCHEMAS = {
    "activities": "activity_id INTEGER, assay_id INTEGER, doc_id INTEGER, molregno INTEGER, standard_relation TEXT, standard_value REAL, standard_units TEXT, standard_type TEXT",
    "assays": "assay_id INTEGER, assay_type TEXT, tid INTEGER, confidence_score INTEGER",
    "target_dictionary": "tid INTEGER, target_type TEXT, pref_name TEXT, organism TEXT, chembl_id TEXT",
    "compound_structures": "molregno INTEGER, canonical_smiles TEXT",
    "molecule_dictionary": "molregno INTEGER, chembl_id TEXT",
    "docs": "doc_id INTEGER, year INTEGER",
}


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    for table, columns in SCHEMAS.items():
        connection.execute(f"CREATE TABLE {table} ({columns})")
    connection.execute("INSERT INTO target_dictionary VALUES (1,'SINGLE PROTEIN','T','Homo sapiens','CHEMBL_T')")
    connection.execute("INSERT INTO target_dictionary VALUES (2,'PROTEIN COMPLEX','X','Homo sapiens','CHEMBL_X')")
    connection.execute("INSERT INTO assays VALUES (1,'B',1,9)")
    connection.execute("INSERT INTO assays VALUES (2,'F',1,9)")
    connection.execute("INSERT INTO docs VALUES (1,2022)")
    connection.execute("INSERT INTO compound_structures VALUES (1,'CC')")
    connection.execute("INSERT INTO molecule_dictionary VALUES (1,'CHEMBL_M1')")
    connection.execute("INSERT INTO activities VALUES (1,1,1,1,'=',10.0,'nM','Ki')")
    connection.execute("INSERT INTO activities VALUES (2,2,1,1,'=',10.0,'nM','Ki')")
    connection.commit()
    return connection


def test_schema_and_query_admit_only_frozen_activity_definition(tmp_path: Path) -> None:
    connection = _database(tmp_path / "mini.db")
    observed = audit_schema(connection)
    assert set(observed) == set(SCHEMAS)
    rows = read_activity_rows(connection)
    assert rows["activity_id"].tolist() == [1]
    assert rows["target_chembl_id"].tolist() == ["CHEMBL_T"]
    assert rows["publication_year"].tolist() == [2022]
    assert rows["ki_nm"].tolist() == [10.0]
    connection.close()


def test_schema_audit_fails_closed_on_missing_column(tmp_path: Path) -> None:
    connection = _database(tmp_path / "mini.db")
    connection.execute("ALTER TABLE docs RENAME TO docs_old")
    connection.execute("CREATE TABLE docs (doc_id INTEGER)")
    with __import__("pytest").raises(RuntimeError, match="year"):
        audit_schema(connection)
    connection.close()


def test_curation_retains_standardized_smiles_and_earliest_partition() -> None:
    rows = pd.DataFrame(
        {
            "target_chembl_id": ["CHEMBL_T", "CHEMBL_T"],
            "target_name": ["T", "T"],
            "molecule_chembl_id": ["CHEMBL_M1", "CHEMBL_M1"],
            "molregno": [1, 1],
            "raw_smiles": ["CC", "CC"],
            "publication_year": [2020, 2022],
            "activity_id": [1, 2],
            "ki_nm": [10.0, 12.0],
        }
    )
    standardized = pd.DataFrame(
        [{"molregno": 1, "standardization_error": None, **standardize_one("CC")}]
    )
    config = {
        "curation": {"duplicate_max_range_log_unit": 1.0},
        "time_split": {
            "train_max_year": 2019,
            "validation_years": [2020, 2021],
            "test_min_year": 2022,
        },
    }
    molecules, diagnostics = curate_rows(rows, standardized, config)
    assert molecules["canonical_isomeric_smiles"].tolist() == ["CC"]
    assert molecules["partition"].tolist() == ["validation"]
    assert molecules["earliest_publication_year"].tolist() == [2020]
    assert molecules["y_pactivity"].tolist() == [8.0]
    assert molecules["replicate_count"].tolist() == [1]
    assert molecules["accepted_year_count"].tolist() == [1]
    assert molecules["observed_year_count"].tolist() == [2]
    assert molecules["excluded_later_year_count"].tolist() == [1]
    assert diagnostics["curated_target_molecules"] == 1
    assert diagnostics["molecules_with_later_year_measurements_excluded"] == 1
