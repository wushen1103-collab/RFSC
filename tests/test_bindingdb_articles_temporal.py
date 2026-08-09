from pathlib import Path

import pandas as pd

from scripts.prepare_bindingdb_articles_temporal import (
    curate_rows,
    latest_year_block_partitions,
    parse_exact_positive_nm,
    publication_year,
    read_bindingdb_rows,
)
from scripts.standardize_molecules import standardize_one


def test_exact_affinity_parser_rejects_relations_and_ranges() -> None:
    assert parse_exact_positive_nm("10") == 10.0
    assert parse_exact_positive_nm("1.2e3") == 1200.0
    assert parse_exact_positive_nm(">10") is None
    assert parse_exact_positive_nm("<10") is None
    assert parse_exact_positive_nm("10-20") is None
    assert parse_exact_positive_nm("0") is None
    assert publication_year("2022-07-04") == 2022
    assert publication_year("7/4/2022") == 2022
    assert publication_year("") is None


def test_bindingdb_curation_uses_earliest_year_response_only() -> None:
    rows = pd.DataFrame(
        {
            "source_row": [0, 1],
            "target_task_id": ["P1_Ki", "P1_Ki"],
            "target_name": ["T", "T"],
            "uniprot_id": ["P1", "P1"],
            "measurement_type": ["Ki", "Ki"],
            "bindingdb_monomer_id": ["1", "1"],
            "raw_smiles": ["CC", "CC"],
            "publication_year": [2019, 2023],
            "article_doi": ["a", "b"],
            "pmid": ["1", "2"],
            "affinity_nm": [10.0, 1000.0],
        }
    )
    standardized = pd.DataFrame(
        [{"raw_smiles": "CC", "standardization_error": None, **standardize_one("CC")}]
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
    assert molecules["partition"].tolist() == ["train"]
    assert molecules["y_pactivity"].tolist() == [8.0]
    assert molecules["observed_year_count"].tolist() == [2]
    assert molecules["excluded_later_year_count"].tolist() == [1]
    assert diagnostics["molecules_with_later_year_measurements_excluded"] == 1


def test_latest_year_blocks_never_split_a_year() -> None:
    molecules = pd.DataFrame(
        {
            "target_task_id": ["P1_Ki"] * 10,
            "earliest_publication_year": [2010] * 5 + [2018] * 2 + [2020] + [2022] * 2,
        }
    )
    config = {"time_split": {"test_fraction": 0.2, "validation_fraction": 0.1}}
    partition = latest_year_block_partitions(molecules, config)
    observed = molecules.assign(partition=partition)
    assert set(observed.loc[observed.earliest_publication_year == 2022, "partition"]) == {"test"}
    assert set(observed.loc[observed.earliest_publication_year == 2020, "partition"]) == {"validation"}
    assert set(observed.loc[observed.earliest_publication_year <= 2018, "partition"]) == {"train"}


def test_official_patent_subset_does_not_require_article_source_value(tmp_path: Path) -> None:
    columns = {
        "Ligand SMILES": ["CC"],
        "BindingDB MonomerID": ["1"],
        "Target Name": ["Target"],
        "Target Source Organism According to Curator or DataSource": ["Human"],
        "Ki (nM)": ["10"],
        "Kd (nM)": [""],
        "Curation/DataSource": ["US Patent"],
        "Article DOI": [""],
        "PMID": [""],
        "Patent Number": ["US-1"],
        "Date of publication": ["1/2/2020"],
        "Number of Protein Chains in Target (>1 implies a multichain complex)": ["1"],
        "UniProt (SwissProt) Primary ID of Target Chain 1": ["P1"],
    }
    path = tmp_path / "patent.tsv"
    pd.DataFrame(columns).to_csv(path, sep="\t", index=False)
    config = {
        "curation": {
            "source_filter": "official_subset_file_only",
            "require_patent_number": True,
            "target_organisms": ["Human"],
            "measurement_types": ["Ki", "Kd"],
        }
    }
    rows, diagnostics = read_bindingdb_rows(path, config)
    assert diagnostics["exact_affinity_task_rows"] == 1
    assert rows.loc[0, "patent_number"] == "US-1"
