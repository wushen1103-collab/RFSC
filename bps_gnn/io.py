from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq


def read_structure_table(data_root: Path, target: str) -> pd.DataFrame:
    path = data_root / target / "structures.parquet"
    return pd.read_parquet(path)


def read_label_columns(data_root: Path, target: str, columns: list[str]) -> pd.DataFrame:
    """Read non-response metadata columns without materializing y."""
    path = data_root / target / "labels_private.parquet"
    return pq.read_table(path, columns=columns).to_pandas()


def read_labels_for_ids(
    data_root: Path,
    target: str,
    mol_ids: Iterable[str],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Materialize response columns only for an explicitly authorized ID set."""
    ids = sorted(set(mol_ids))
    if not ids:
        return pd.DataFrame(columns=columns or ["mol_id", "y_pactivity"])
    requested = columns or ["mol_id", "y_pactivity"]
    if "mol_id" not in requested:
        requested = ["mol_id", *requested]
    dataset = pads.dataset(data_root / target / "labels_private.parquet", format="parquet")
    table = dataset.to_table(columns=requested, filter=pads.field("mol_id").isin(ids))
    frame = table.to_pandas()
    if set(frame["mol_id"]) != set(ids):
        missing = sorted(set(ids) - set(frame["mol_id"]))
        raise RuntimeError(f"Missing authorized labels for {len(missing)} molecule IDs")
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_prediction_file(frame: pd.DataFrame, path: Path, metadata: dict) -> str:
    if any(column.lower().startswith("y_") for column in frame.columns):
        raise ValueError("Prediction firewall forbids response columns")
    required = {"mol_id", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prediction file lacks {required - set(frame.columns)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("mol_id").to_parquet(path, index=False)
    digest = sha256_file(path)
    manifest = {"prediction_sha256": digest, "metadata": metadata}
    path.with_suffix(path.suffix + ".sha256.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    os.chmod(path, 0o444)
    return digest


def read_frozen_prediction(path: Path) -> pd.Series:
    """Verify a response-free prediction artifact before returning its series."""
    manifest_path = path.with_suffix(path.suffix + ".sha256.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"mol_id", "prediction"}:
        raise RuntimeError(f"Frozen prediction schema is not response-free: {path}")
    if frame["mol_id"].duplicated().any():
        raise RuntimeError(f"Duplicate prediction IDs: {path}")
    prediction = frame.set_index("mol_id")["prediction"].sort_index().astype(float)
    if not np.isfinite(prediction.to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite prediction: {path}")
    return prediction

