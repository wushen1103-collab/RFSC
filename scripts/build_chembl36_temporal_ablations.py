#!/usr/bin/env python3
"""Build the frozen response-free temporal ablation prediction matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from bps_gnn.io import freeze_prediction_file, read_frozen_prediction, read_structure_table
from bps_gnn.projection import active_components, boundary_projection, select_active_edges
from bps_gnn.temporal import temporal_split_ids


def ecfp_knn_edges(smiles: list[str], mol_ids: list[str], radius: int, bits: int, k: int, floor: float) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fingerprints = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid standardized test SMILES: {value}")
        fingerprints.append(generator.GetFingerprint(molecule))
    pairs: set[tuple[str, str]] = set()
    for index, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
        neighbors = sorted(
            (
                (float(similarity), mol_ids[other])
                for other, similarity in enumerate(similarities)
                if other != index and float(similarity) >= floor
            ),
            key=lambda item: (-item[0], item[1]),
        )[:k]
        for _, other_id in neighbors:
            pairs.add(tuple(sorted((mol_ids[index], other_id))))
    return pd.DataFrame(sorted(pairs), columns=["mol_id_lo", "mol_id_hi"])


def stable_rng(base_seed: int, target: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{base_seed}:{target}:{seed}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def component_diagnostics(index: pd.Index, edges: pd.DataFrame) -> dict[str, int]:
    components = active_components(index, edges)
    sizes = [len(members) for members in components.values() if len(members) > 1]
    return {
        "active_nodes": int(sum(sizes)),
        "nontrivial_components": int(len(sizes)),
        "largest_component": int(max(sizes, default=1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ablation-config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--primary-prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ablation = yaml.safe_load(args.ablation_config.read_text(encoding="utf-8"))
    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["seeds"]]
    main_threshold = float(config["operator"]["predicted_delta_threshold"])
    main_scale = float(config["operator"]["scale"])
    knn_config = ablation["graph"]["ecfp_knn"]
    records: list[dict[str, object]] = []

    for target in targets:
        structures = read_structure_table(args.data_root, target).set_index("mol_id")
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        test_structures = structures.loc[test_ids]
        knn_edges = ecfp_knn_edges(
            test_structures["canonical_isomeric_smiles"].tolist(),
            test_ids,
            int(knn_config["fingerprint_radius"]),
            int(knn_config["fingerprint_bits"]),
            int(knn_config["k"]),
            float(knn_config["minimum_similarity"]),
        )
        all_mmp = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        test_set = set(test_ids)
        mmp_edges = all_mmp[
            all_mmp["mol_id_lo"].isin(test_set) & all_mmp["mol_id_hi"].isin(test_set)
        ].copy()
        for seed in seeds:
            primary_dir = args.primary_prediction_root / target / f"seed_{seed}"
            svm = read_frozen_prediction(primary_dir / "svm" / "predictions.parquet")
            rf_gine = read_frozen_prediction(primary_dir / "rf_gine" / "predictions.parquet")
            unscaled = read_frozen_prediction(primary_dir / "unscaled_reference" / "predictions.parquet")
            main_edges = select_active_edges(unscaled, mmp_edges, main_threshold)
            q = len(main_edges)
            rng = stable_rng(int(ablation["random_seed"]), target, seed)
            random_edges = (
                mmp_edges.iloc[np.sort(rng.choice(len(mmp_edges), size=q, replace=False))].copy()
                if q else mmp_edges.iloc[:0].copy()
            )
            delta = (mmp_edges["mol_id_hi"].map(unscaled) - mmp_edges["mol_id_lo"].map(unscaled)).abs()
            low_edges = mmp_edges.assign(_delta=delta).sort_values(
                ["_delta", "mol_id_lo", "mol_id_hi"]
            ).head(q).drop(columns="_delta")
            svm_edges = select_active_edges(svm, mmp_edges, main_threshold)
            rf_gine_edges = select_active_edges(rf_gine, mmp_edges, main_threshold)
            knn_active = select_active_edges(unscaled, knn_edges, float(knn_config["predicted_delta_threshold"]))

            specifications: list[tuple[str, pd.Series, pd.DataFrame, float]] = [
                ("identity", unscaled, main_edges, 1.0),
                ("all_mmp", unscaled, mmp_edges, main_scale),
                ("random_q_mmp", unscaled, random_edges, main_scale),
                ("low_delta_q_mmp", unscaled, low_edges, main_scale),
                ("svm_main_mmp", svm, svm_edges, main_scale),
                ("rf_gine_main_mmp", rf_gine, rf_gine_edges, main_scale),
                ("ecfp_knn_main", unscaled, knn_active, float(knn_config["scale"])),
            ]
            for scale in ablation["sensitivity"]["scales"]:
                for threshold in ablation["sensitivity"]["predicted_delta_thresholds"]:
                    active = select_active_edges(unscaled, mmp_edges, float(threshold))
                    name = f"sensitivity_s{str(scale).replace('.', 'p')}_t{str(threshold).replace('.', 'p')}"
                    specifications.append((name, unscaled, active, float(scale)))

            run_dir = args.output_dir / target / f"seed_{seed}"
            for method, base, active_edges, scale in specifications:
                started = time.perf_counter()
                prediction = boundary_projection(base, active_edges, scale)
                elapsed = time.perf_counter() - started
                correction_norm = float(np.linalg.norm((prediction - base).to_numpy(dtype=float)))
                diagnostics = component_diagnostics(base.index, active_edges)
                metadata = {
                    "target": target,
                    "seed": seed,
                    "method": method,
                    "scale": scale,
                    "active_edges": int(len(active_edges)),
                    "base_prediction": "svm" if method == "svm_main_mmp" else (
                        "rf_gine" if method == "rf_gine_main_mmp" else "unscaled_reference"
                    ),
                    "correction_l2": correction_norm,
                    "projection_seconds": elapsed,
                    "test_labels_materialized": False,
                    **diagnostics,
                }
                freeze_prediction_file(
                    pd.DataFrame({"mol_id": prediction.index, "prediction": prediction.to_numpy()}),
                    run_dir / method / "predictions.parquet",
                    metadata,
                )
                records.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame.from_records(records)
    summary.to_csv(args.output_dir / "build_summary.csv", index=False)
    print(
        json.dumps(
            {
                "targets": len(targets),
                "seeds": len(seeds),
                "methods": int(summary["method"].nunique()),
                "predictions": len(summary),
                "test_labels_materialized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
