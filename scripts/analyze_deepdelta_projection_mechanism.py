#!/usr/bin/env python3
"""Decompose DeepDelta projection gains and evaluate fixed natural alternatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from bps_gnn.io import read_frozen_prediction
from bps_gnn.pair_flow import antisymmetric_pair_flow

try:
    from evaluate_deepdelta_cycle_projection import (
        LOWER_METRICS,
        comparison_statistics,
        flow_metrics,
        load_raw_flow,
        ordered_position_values,
    )
except ModuleNotFoundError:
    from scripts.evaluate_deepdelta_cycle_projection import (
        LOWER_METRICS,
        comparison_statistics,
        flow_metrics,
        load_raw_flow,
        ordered_position_values,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-datasets", type=int)
    return parser.parse_args()


def gradient(potential: np.ndarray) -> np.ndarray:
    values = np.asarray(potential, dtype=float)
    return values[None, :] - values[:, None]


def one_sided_potentials(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    incoming = np.asarray(flow, dtype=float).mean(axis=0)
    outgoing = -np.asarray(flow, dtype=float).mean(axis=1)
    incoming -= float(incoming.mean())
    outgoing -= float(outgoing.mean())
    return incoming, outgoing


def cycle_residual_rms(flow: np.ndarray) -> float:
    """Exact RMS of F_ij + F_jk - F_ik over all ordered triples."""
    values = np.asarray(flow, dtype=float)
    squared_sum = 0.0
    n_nodes = len(values)
    for middle in range(n_nodes):
        residual = values[:, middle][:, None] + values[middle, :][None, :] - values
        squared_sum += float(np.square(residual).sum())
    return float(np.sqrt(squared_sum / float(n_nodes**3)))


def squared_error(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(np.square(prediction[mask] - truth[mask]).sum())


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = [str(value) for value in config["datasets"]]
    if args.maximum_datasets is not None:
        datasets = datasets[: args.maximum_datasets]
    n_folds = int(config["cross_validation"]["folds"])
    quantile = float(config["evaluation"]["high_delta_quantile_within_fold"])

    mechanism_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for dataset in datasets:
        data = pd.read_csv(args.data_root / f"{dataset}.csv")
        response = data["Y"].to_numpy(dtype=float)
        splitter = KFold(
            n_splits=n_folds,
            shuffle=bool(config["cross_validation"]["shuffle"]),
            random_state=int(config["cross_validation"]["random_state"]),
        )
        for fold, (_, test_index) in enumerate(splitter.split(data), start=1):
            raw = load_raw_flow(
                args.source_root / f"{dataset}_DeepDelta_Individual_Fold_{fold}.csv"
            )
            fold_response = response[test_index]
            truth = gradient(fold_response)
            if raw.shape != truth.shape:
                raise RuntimeError(f"{dataset} fold {fold}: official split shape mismatch")
            potential = ordered_position_values(
                read_frozen_prediction(
                    args.prediction_root / dataset / f"fold_{fold}" / "potential.parquet"
                ),
                len(raw),
            )
            hodge = gradient(potential)
            anti = antisymmetric_pair_flow(raw)
            incoming_potential, outgoing_potential = one_sided_potentials(raw)
            if not np.allclose(
                potential, 0.5 * (incoming_potential + outgoing_potential), atol=1e-12
            ):
                raise RuntimeError(f"{dataset} fold {fold}: Hodge averaging identity failed")
            incoming = gradient(incoming_potential)
            outgoing = gradient(outgoing_potential)
            symmetric = raw - anti
            cyclic = anti - hodge
            off_diagonal = ~np.eye(len(raw), dtype=bool)
            cutoff = float(np.quantile(np.abs(truth[off_diagonal]), quantile))
            high_mask = off_diagonal & (np.abs(truth) >= cutoff)

            raw_sse = squared_error(raw, truth, off_diagonal)
            anti_sse = squared_error(anti, truth, off_diagonal)
            hodge_sse = squared_error(hodge, truth, off_diagonal)
            symmetric_energy = float(np.square(symmetric[off_diagonal]).sum())
            cyclic_energy = float(np.square(cyclic[off_diagonal]).sum())
            removed_energy = float(np.square((raw - hodge)[off_diagonal]).sum())
            scale = max(raw_sse, 1.0)
            mechanism_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "n_nodes": len(raw),
                    "identity_residual_rms_raw": float(np.sqrt(np.mean(np.square(np.diag(raw))))),
                    "reversal_residual_rms_raw": float(
                        np.sqrt(np.mean(np.square((raw + raw.T)[off_diagonal])))
                    ),
                    "cycle_residual_rms_raw": cycle_residual_rms(raw),
                    "cycle_residual_rms_antisymmetric": cycle_residual_rms(anti),
                    "cycle_residual_rms_hodge": cycle_residual_rms(hodge),
                    "symmetric_removed_energy": symmetric_energy,
                    "cyclic_removed_energy": cyclic_energy,
                    "total_removed_energy": removed_energy,
                    "symmetric_energy_fraction": symmetric_energy / max(removed_energy, 1e-300),
                    "cyclic_energy_fraction": cyclic_energy / max(removed_energy, 1e-300),
                    "raw_to_antisymmetric_identity_discrepancy": abs(
                        raw_sse - anti_sse - symmetric_energy
                    )
                    / scale,
                    "antisymmetric_to_hodge_identity_discrepancy": abs(
                        anti_sse - hodge_sse - cyclic_energy
                    )
                    / scale,
                    "orthogonal_residual_decomposition_discrepancy": abs(
                        removed_energy - symmetric_energy - cyclic_energy
                    )
                    / max(removed_energy, 1.0),
                }
            )
            for method, prediction in {
                "published_deepdelta": raw,
                "antisymmetric": anti,
                "incoming_only": incoming,
                "outgoing_only": outgoing,
                "hodge_cycle_consistent": hodge,
            }.items():
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": method,
                        **flow_metrics(prediction, truth, high_mask),
                    }
                )

    mechanism = pd.DataFrame.from_records(mechanism_rows)
    fold_metrics = pd.DataFrame.from_records(metric_rows)
    dataset_metrics = fold_metrics.groupby(["dataset", "method"], as_index=False)[
        [*LOWER_METRICS, "pearson_r"]
    ].mean()
    statistics: list[dict[str, object]] = []
    for comparison_index, reference in enumerate(
        ["published_deepdelta", "antisymmetric", "incoming_only", "outgoing_only"]
    ):
        statistics.extend(
            comparison_statistics(
                dataset_metrics,
                "hodge_cycle_consistent",
                reference,
                int(config["evaluation"]["bootstrap_iterations"]),
                291000 + 100 * comparison_index,
            )
        )
    statistics_frame = pd.DataFrame.from_records(statistics)
    dataset_mechanism = mechanism.groupby("dataset", as_index=False).mean(numeric_only=True)
    summary = {
        "datasets": len(datasets),
        "folds": len(mechanism),
        "maximum_raw_to_antisymmetric_identity_discrepancy": float(
            mechanism["raw_to_antisymmetric_identity_discrepancy"].max()
        ),
        "maximum_antisymmetric_to_hodge_identity_discrepancy": float(
            mechanism["antisymmetric_to_hodge_identity_discrepancy"].max()
        ),
        "maximum_orthogonal_residual_decomposition_discrepancy": float(
            mechanism["orthogonal_residual_decomposition_discrepancy"].max()
        ),
        "maximum_hodge_cycle_residual_rms": float(
            mechanism["cycle_residual_rms_hodge"].max()
        ),
        "macro_symmetric_energy_fraction": float(
            dataset_mechanism["symmetric_energy_fraction"].mean()
        ),
        "macro_cyclic_energy_fraction": float(dataset_mechanism["cyclic_energy_fraction"].mean()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mechanism.to_csv(args.output_dir / "fold_mechanism.csv", index=False)
    dataset_mechanism.to_csv(args.output_dir / "dataset_mechanism.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "fold_variant_metrics.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "dataset_variant_metrics.csv", index=False)
    statistics_frame.to_csv(args.output_dir / "variant_statistics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
