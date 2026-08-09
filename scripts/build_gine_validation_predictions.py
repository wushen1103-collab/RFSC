#!/usr/bin/env python3
"""Hash-freeze response-free GINE validation predictions from a frozen checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

from bps_gnn.io import freeze_prediction_file, read_structure_table
from bps_gnn.models import GINERegressor, GraphCliffRegressor
from bps_gnn.molecular_graphs import ATOM_DIM, BOND_DIM, load_or_build_graphs
from bps_gnn.temporal import temporal_split_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    structures = read_structure_table(data_root, args.target).reset_index(drop=True)
    mol_ids = structures["mol_id"].tolist()
    id_to_index = {mol_id: index for index, mol_id in enumerate(mol_ids)}
    _, validation_ids, test_ids = temporal_split_ids(data_root, args.target)
    if set(validation_ids) & set(test_ids):
        raise RuntimeError("Validation/test molecule overlap")

    run_dir = args.output_dir / args.target / "temporal" / f"seed_{args.seed}"
    checkpoint_path = run_dir / "checkpoint.pt"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint["config"]
    model_config = dict(checkpoint_config["model"])
    architecture = model_config.pop("architecture", "gine")
    model_class = {"gine": GINERegressor, "graphcliff": GraphCliffRegressor}[architecture]
    model = model_class(ATOM_DIM, BOND_DIM, **model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    cache_path = Path(checkpoint_config["cache_root"]) / f"{args.target}.pt"
    graphs = load_or_build_graphs(structures["canonical_isomeric_smiles"].tolist(), cache_path)
    validation_graphs = [copy.copy(graphs[id_to_index[mol_id]]) for mol_id in validation_ids]
    batch_size = int(checkpoint_config["optimization"]["batch_size"])
    loader = DataLoader(validation_graphs, batch_size=batch_size, shuffle=False)
    indices: list[int] = []
    predictions: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = (
                model(batch).float().cpu().numpy() * float(checkpoint["scale"])
                + float(checkpoint["center"])
            )
            indices.extend(batch.row_id.cpu().numpy().tolist())
            predictions.extend(np.asarray(output).reshape(-1).tolist())
    predicted_ids = [mol_ids[index] for index in indices]
    if set(predicted_ids) != set(validation_ids) or len(predicted_ids) != len(validation_ids):
        raise RuntimeError("Generated GINE validation partition mismatch")
    metadata = {
        "target": args.target,
        "seed": args.seed,
        "method": str(checkpoint_config["method"]),
        "architecture": architecture,
        "source_checkpoint": str(checkpoint_path),
        "partition": "validation",
        "validation_labels_materialized": False,
        "test_labels_materialized": False,
        "n_validation": len(validation_ids),
        "n_test_not_read": len(test_ids),
    }
    digest = freeze_prediction_file(
        pd.DataFrame({"mol_id": predicted_ids, "prediction": predictions}),
        run_dir / "validation_predictions.parquet",
        metadata,
    )
    metadata["prediction_sha256"] = digest
    (run_dir / "validation_prediction_run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

