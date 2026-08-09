#!/usr/bin/env python3
"""Response-free reversal/cycle energy audit for frozen Delta-LightGBM flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction
from bps_gnn.pair_flow import antisymmetric_pair_flow

try:
    from analyze_deepdelta_projection_mechanism import cycle_residual_rms
    from evaluate_deepdelta_delta_lgbm_projection import ordered_values
except ModuleNotFoundError:
    from scripts.analyze_deepdelta_projection_mechanism import cycle_residual_rms
    from scripts.evaluate_deepdelta_delta_lgbm_projection import ordered_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for dataset in [str(value) for value in config["datasets"]]:
        run_dir = args.prediction_root / dataset
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if metadata["held_out_response_materialized"] is not False:
            raise RuntimeError(f"{dataset}: response-free metadata failed")
        n_nodes = int(metadata["test_molecules"])
        edge_ids = [
            f"edge_{i:04d}_{j:04d}" for i in range(n_nodes) for j in range(n_nodes)
        ]
        node_ids = [f"position_{i:04d}" for i in range(n_nodes)]
        raw = ordered_values(
            read_frozen_prediction(run_dir / "direct_flow.parquet"), edge_ids, "direct_flow"
        ).reshape(n_nodes, n_nodes)
        potential = ordered_values(
            read_frozen_prediction(run_dir / "hodge_potential.parquet"),
            node_ids,
            "hodge_potential",
        )
        hodge = potential[None, :] - potential[:, None]
        anti = antisymmetric_pair_flow(raw)
        symmetric = raw - anti
        cyclic = anti - hodge
        off_diagonal = ~np.eye(n_nodes, dtype=bool)
        symmetric_energy = float(np.square(symmetric[off_diagonal]).sum())
        cyclic_energy = float(np.square(cyclic[off_diagonal]).sum())
        removed_energy = float(np.square((raw - hodge)[off_diagonal]).sum())
        if removed_energy <= 0.0:
            raise RuntimeError(f"{dataset}: projection removed no measurable energy")
        rows.append(
            {
                "dataset": dataset,
                "n_nodes": n_nodes,
                "identity_residual_rms_raw": float(
                    np.sqrt(np.mean(np.square(np.diag(raw))))
                ),
                "reversal_residual_rms_raw": float(
                    np.sqrt(np.mean(np.square((raw + raw.T)[off_diagonal])))
                ),
                "cycle_residual_rms_raw": cycle_residual_rms(raw),
                "cycle_residual_rms_antisymmetric": cycle_residual_rms(anti),
                "cycle_residual_rms_hodge": cycle_residual_rms(hodge),
                "symmetric_removed_energy": symmetric_energy,
                "cyclic_removed_energy": cyclic_energy,
                "total_removed_energy": removed_energy,
                "symmetric_energy_fraction": symmetric_energy / removed_energy,
                "cyclic_energy_fraction": cyclic_energy / removed_energy,
                "orthogonal_decomposition_discrepancy": abs(
                    removed_energy - symmetric_energy - cyclic_energy
                )
                / max(removed_energy, 1.0),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    summary = {
        "datasets": int(len(frame)),
        "response_columns_read": False,
        "macro_symmetric_energy_fraction": float(frame["symmetric_energy_fraction"].mean()),
        "macro_cyclic_energy_fraction": float(frame["cyclic_energy_fraction"].mean()),
        "maximum_orthogonal_decomposition_discrepancy": float(
            frame["orthogonal_decomposition_discrepancy"].max()
        ),
        "maximum_hodge_cycle_residual_rms": float(frame["cycle_residual_rms_hodge"].max()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "dataset_mechanism.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
