#!/usr/bin/env python3
"""Formula-level strict-SCG adaptation of the ICLR 2025 LAC regression method."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from bps_gnn.fingerprints import ecfp_matrix
from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, read_structure_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def split_ids(target: str, fold: int) -> tuple[list[str], list[str], list[str]]:
    frame = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(frame["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    train = frame.loc[
        ~frame["outer_fold"].isin([fold, validation_fold]), "mol_id"
    ].tolist()
    validation = frame.loc[frame["outer_fold"] == validation_fold, "mol_id"].tolist()
    test = frame.loc[frame["outer_fold"] == fold, "mol_id"].tolist()
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("SCG partition overlap")
    return train, validation, test


class ECFPMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(current, hidden_dim), nn.ReLU()])
            current = hidden_dim
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def curriculum_keep_mask(loss: torch.Tensor, discard_fraction: float) -> torch.Tensor:
    """Keep the largest-loss 1-R fraction, following the LAC prose/equation."""
    if loss.numel() == 0:
        return torch.zeros_like(loss, dtype=torch.bool)
    if discard_fraction <= 0:
        return torch.ones_like(loss, dtype=torch.bool)
    threshold = torch.quantile(loss.detach(), float(discard_fraction))
    return loss >= threshold


def lac_edge_losses(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    return -(truth[left] - truth[right]) * (prediction[left] - prediction[right])


def train_variant(
    *,
    variant: str,
    initial_state: dict,
    features: torch.Tensor,
    truth: torch.Tensor,
    cliff_endpoint: torch.Tensor,
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    config: dict,
    seed: int,
    device: torch.device,
) -> tuple[ECFPMLP, dict]:
    model_config = config["model"]
    optimization = config["optimization"]
    lac = config["lac"]
    model = ECFPMLP(
        int(model_config["input_dim"]),
        int(model_config["hidden_dim"]),
        int(model_config["hidden_layers"]),
    ).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        betas=(float(optimization["beta1"]), float(optimization["beta2"])),
    )
    n = len(truth)
    batch_size = int(optimization["batch_size"])
    epochs = int(optimization["epochs"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = features.to(device)
    truth = truth.to(device)
    cliff_endpoint = cliff_endpoint.to(device)
    edge_left = edge_left.to(device)
    edge_right = edge_right.to(device)
    active_edge_batches = 0
    total_batches = 0
    final_loss = float("nan")
    for epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        discard = float(lac["schedule_lambda"]) * min(
            epoch / (float(lac["schedule_gamma"]) * epochs), 1.0
        )
        epoch_loss = []
        model.train()
        for start in range(0, n, batch_size):
            batch_cpu = permutation[start : start + batch_size]
            batch = batch_cpu.to(device)
            prediction = model(features[batch])
            squared = (prediction - truth[batch]).square()
            if variant == "baseline":
                loss = squared.mean()
            elif variant == "lac":
                node_weight = torch.where(
                    cliff_endpoint[batch],
                    torch.ones_like(squared),
                    torch.full_like(squared, float(lac["non_cliff_weight"])),
                )
                weighted = node_weight * squared
                keep = curriculum_keep_mask(weighted, discard)
                node_loss = weighted[keep].mean()
                local = torch.full((n,), -1, dtype=torch.long, device=device)
                local[batch] = torch.arange(len(batch), device=device)
                edge_mask = (local[edge_left] >= 0) & (local[edge_right] >= 0)
                if edge_mask.any():
                    left = local[edge_left[edge_mask]]
                    right = local[edge_right[edge_mask]]
                    edge_loss = lac_edge_losses(
                        truth[batch], prediction, left, right
                    )
                    edge_keep = curriculum_keep_mask(edge_loss, discard)
                    loss = node_loss + float(lac["edge_alpha"]) * edge_loss[
                        edge_keep
                    ].mean()
                    active_edge_batches += 1
                else:
                    loss = node_loss
            else:
                raise ValueError(variant)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_batches += 1
            epoch_loss.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(epoch_loss))
    return model, {
        "variant": variant,
        "epochs": epochs,
        "total_batches": total_batches,
        "active_edge_batches": active_edge_batches if variant == "lac" else 0,
        "active_edge_batch_fraction": (
            active_edge_batches / total_batches if variant == "lac" else None
        ),
        "final_training_loss": final_loss,
    }


def predict(model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).float().to(device)
            values.extend(model(batch).float().cpu().numpy().tolist())
    return np.asarray(values, dtype=float)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    smoke = config["smoke"]
    target, fold, seed = smoke["target"], int(smoke["fold"]), int(smoke["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    structures = read_structure_table(Path(config["data_root"]), target).reset_index(drop=True)
    train_ids, validation_ids, test_ids = split_ids(target, fold)
    # Only outer-train responses enter this trainer.
    train_truth = read_labels_for_ids(Path(config["data_root"]), target, train_ids).set_index(
        "mol_id"
    )["y_pactivity"].loc[train_ids]
    if set(train_truth.index) & (set(validation_ids) | set(test_ids)):
        raise RuntimeError("Query response entered LAC trainer")
    fingerprint = config["fingerprint"]
    matrix = ecfp_matrix(
        structures["canonical_isomeric_smiles"].tolist(),
        radius=int(fingerprint["radius"]),
        n_bits=int(fingerprint["n_bits"]),
    )
    id_to_row = {value: index for index, value in enumerate(structures["mol_id"])}
    train_row = np.asarray([id_to_row[value] for value in train_ids])
    validation_row = np.asarray([id_to_row[value] for value in validation_ids])
    train_position = {value: index for index, value in enumerate(train_ids)}
    edges = pd.read_parquet(Path(config["mmp_root"]) / target / "eligible_mmp_edges.parquet")
    edges = edges[
        edges["mol_id_lo"].isin(train_position) & edges["mol_id_hi"].isin(train_position)
    ].copy()
    edges["delta_y"] = edges["mol_id_hi"].map(train_truth) - edges["mol_id_lo"].map(
        train_truth
    )
    cliffs = edges[edges["delta_y"].abs() >= float(config["lac"]["cliff_threshold"])]
    if cliffs.empty:
        raise RuntimeError("No outer-train MMP cliff pairs for LAC smoke")
    left = torch.tensor([train_position[value] for value in cliffs["mol_id_lo"]], dtype=torch.long)
    right = torch.tensor([train_position[value] for value in cliffs["mol_id_hi"]], dtype=torch.long)
    endpoint = torch.zeros(len(train_ids), dtype=torch.bool)
    endpoint[torch.unique(torch.cat([left, right]))] = True
    feature_tensor = torch.from_numpy(matrix[train_row]).float()
    truth_tensor = torch.from_numpy(train_truth.to_numpy(dtype=np.float32))
    initial = ECFPMLP(
        int(config["model"]["input_dim"]),
        int(config["model"]["hidden_dim"]),
        int(config["model"]["hidden_layers"]),
    )
    initial_state = copy.deepcopy(initial.state_dict())
    output = args.output_dir / target / f"fold_{fold}" / f"seed_{seed}"
    metadata = {
        "target": target,
        "fold": fold,
        "seed": seed,
        "method": config["name"],
        "adaptation_status": config["adaptation_status"],
        "train_labels_materialized": True,
        "validation_labels_materialized": False,
        "outer_test_labels_materialized": False,
        "outer_test_predictions_generated": False,
        "n_train": len(train_ids),
        "n_validation": len(validation_ids),
        "n_test": len(test_ids),
        "n_train_mmp_cliffs": len(cliffs),
        "n_train_cliff_endpoints": int(endpoint.sum()),
        "config": config,
    }
    training = []
    for variant in ["baseline", "lac"]:
        model, record = train_variant(
            variant=variant,
            initial_state=initial_state,
            features=feature_tensor,
            truth=truth_tensor,
            cliff_endpoint=endpoint,
            edge_left=left,
            edge_right=right,
            config=config,
            seed=seed,
            device=device,
        )
        training.append(record)
        variant_output = output / variant
        variant_output.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config}, variant_output / "checkpoint.pt")
        prediction = predict(
            model,
            matrix[validation_row],
            int(config["optimization"]["batch_size"]),
            device,
        )
        digest = freeze_prediction_file(
            pd.DataFrame({"mol_id": validation_ids, "prediction": prediction}),
            variant_output / "validation_predictions.parquet",
            {**metadata, "variant": variant},
        )
        metadata[f"{variant}_validation_prediction_sha256"] = digest
    metadata["training"] = training
    metadata["wall_time_seconds"] = time.perf_counter() - started
    metadata["peak_vram_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
