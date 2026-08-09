import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_chembl36_temporal_eligibility import audit_target


def _write_target(root: Path, graph_root: Path, response_in_structure: bool = False) -> None:
    target = root / "CHEMBL_T"
    graph = graph_root / "CHEMBL_T"
    target.mkdir(parents=True)
    graph.mkdir(parents=True)
    structures = pd.DataFrame({"mol_id": list("abcdef")})
    if response_in_structure:
        structures["y_pactivity"] = 1.0
    structures.to_parquet(target / "structures.parquet", index=False)
    pd.DataFrame(
        {
            "mol_id": list("abcdef"),
            "partition": ["train", "train", "validation", "validation", "test", "test"],
            "earliest_publication_year": [2018, 2019, 2020, 2021, 2022, 2023],
        }
    ).to_parquet(target / "partitions.parquet", index=False)
    pd.DataFrame(
        {"mol_id_lo": ["e", "a"], "mol_id_hi": ["f", "b"], "transform": ["x", "y"], "constant": ["c", "d"]}
    ).to_parquet(graph / "eligible_mmp_edges.parquet", index=False)
    (target / "target_manifest.json").write_text(
        json.dumps({"target_name": "T"}), encoding="utf-8"
    )


def test_eligibility_uses_counts_and_internal_test_topology_only(tmp_path: Path) -> None:
    data, graphs = tmp_path / "data", tmp_path / "graphs"
    _write_target(data, graphs)
    config = {
        "eligibility_structure_only": {
            "minimum_train_molecules": 2,
            "minimum_validation_molecules": 2,
            "minimum_test_molecules": 2,
            "minimum_internal_test_mmp_edges": 1,
        }
    }
    result = audit_target(data / "CHEMBL_T", graphs / "CHEMBL_T", config)
    assert result["eligible"] is True
    assert result["internal_test_mmp_edges"] == 1
    assert result["response_columns_read"] is False


def test_eligibility_fails_closed_if_response_leaks_into_structure(tmp_path: Path) -> None:
    data, graphs = tmp_path / "data", tmp_path / "graphs"
    _write_target(data, graphs, response_in_structure=True)
    config = {
        "eligibility_structure_only": {
            "minimum_train_molecules": 2,
            "minimum_validation_molecules": 2,
            "minimum_test_molecules": 2,
            "minimum_internal_test_mmp_edges": 1,
        }
    }
    with pytest.raises(AssertionError, match="response columns"):
        audit_target(data / "CHEMBL_T", graphs / "CHEMBL_T", config)
