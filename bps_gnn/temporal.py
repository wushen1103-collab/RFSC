"""Utilities for the frozen ChEMBL temporal partition."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def temporal_split_ids(data_root: Path, target: str) -> tuple[list[str], list[str], list[str]]:
    frame = pd.read_parquet(data_root / target / "partitions.parquet")
    required = {"mol_id", "partition", "earliest_publication_year"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Temporal partition lacks {sorted(required - set(frame.columns))}")
    if frame["mol_id"].duplicated().any():
        raise RuntimeError("Temporal partition contains duplicate molecule IDs")
    if not set(frame["partition"]).issubset({"train", "validation", "test"}):
        raise RuntimeError("Temporal partition contains an unknown split")
    train = frame.loc[frame["partition"] == "train", "mol_id"].tolist()
    validation = frame.loc[frame["partition"] == "validation", "mol_id"].tolist()
    test = frame.loc[frame["partition"] == "test", "mol_id"].tolist()
    if not train or not validation or not test:
        raise RuntimeError("Temporal partition must contain train, validation, and test molecules")
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("Temporal molecule partitions overlap")
    return train, validation, test
