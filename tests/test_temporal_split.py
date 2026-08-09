from pathlib import Path

import pandas as pd
import pytest

from bps_gnn.temporal import temporal_split_ids


def test_temporal_split_ids_are_disjoint_and_complete(tmp_path: Path) -> None:
    target = tmp_path / "CHEMBL_T"
    target.mkdir()
    pd.DataFrame(
        {
            "mol_id": ["a", "b", "c"],
            "partition": ["train", "validation", "test"],
            "earliest_publication_year": [2019, 2020, 2022],
        }
    ).to_parquet(target / "partitions.parquet", index=False)
    assert temporal_split_ids(tmp_path, "CHEMBL_T") == (["a"], ["b"], ["c"])


def test_temporal_split_rejects_duplicate_ids(tmp_path: Path) -> None:
    target = tmp_path / "CHEMBL_T"
    target.mkdir()
    pd.DataFrame(
        {
            "mol_id": ["a", "a", "c"],
            "partition": ["train", "validation", "test"],
            "earliest_publication_year": [2019, 2020, 2022],
        }
    ).to_parquet(target / "partitions.parquet", index=False)
    with pytest.raises(RuntimeError, match="duplicate"):
        temporal_split_ids(tmp_path, "CHEMBL_T")
