#!/usr/bin/env python3
"""Post-hoc fixed-method robustness by prior-structure novelty in patent test data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids
from bps_gnn.temporal import temporal_split_ids

try:
    from evaluate_chembl36_temporal import bootstrap_relative
except ModuleNotFoundError:
    from scripts.evaluate_chembl36_temporal import bootstrap_relative


METHODS = ["unscaled_reference", "confidence_weighted_bps"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-targets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--primary-prediction-root", type=Path, required=True)
    parser.add_argument("--weighted-prediction-root", type=Path, required=True)
    parser.add_argument("--prior-structure-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-targets", type=int)
    return parser.parse_args()


def structure_identity_set(roots: list[Path]) -> set[str]:
    identities: set[str] = set()
    for root in roots:
        paths = sorted(root.glob("*/structures.parquet"))
        if not paths:
            raise RuntimeError(f"no structures under prior root {root}")
        for path in paths:
            frame = pd.read_parquet(path, columns=["canonical_no_stereo_smiles"])
            identities.update(frame["canonical_no_stereo_smiles"].dropna().astype(str))
    return identities


def classify_cliff_edges(
    edges: pd.DataFrame, truth: pd.Series, novel_by_id: pd.Series
) -> dict[str, pd.DataFrame]:
    delta = edges["mol_id_hi"].map(truth) - edges["mol_id_lo"].map(truth)
    cliff = edges.loc[delta.abs() >= 1.0].copy()
    cliff["delta_y"] = delta.loc[cliff.index]
    lo_novel = cliff["mol_id_lo"].map(novel_by_id).astype(bool)
    hi_novel = cliff["mol_id_hi"].map(novel_by_id).astype(bool)
    return {
        "all": cliff,
        "both_novel": cliff.loc[lo_novel & hi_novel],
        "mixed": cliff.loc[lo_novel ^ hi_novel],
        "both_prior_overlap": cliff.loc[~lo_novel & ~hi_novel],
    }


def pair_mae(prediction: pd.Series, edges: pd.DataFrame) -> float:
    if edges.empty:
        return float("nan")
    predicted_delta = edges["mol_id_hi"].map(prediction) - edges["mol_id_lo"].map(prediction)
    return float(np.mean(np.abs(predicted_delta - edges["delta_y"])))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    import json

    targets = json.loads(args.eligible_targets.read_text(encoding="utf-8"))
    if args.maximum_targets is not None:
        targets = targets[: args.maximum_targets]
    seeds = [int(value) for value in config["seeds"]]
    prior_structures = structure_identity_set(args.prior_structure_roots)

    # Recheck hashes and test partitions before reading already-exposed responses.
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        expected = set(test_ids)
        for seed in seeds:
            paths = {
                "unscaled_reference": args.primary_prediction_root
                / target
                / f"seed_{seed}"
                / "unscaled_reference"
                / "predictions.parquet",
                "confidence_weighted_bps": args.weighted_prediction_root
                / target
                / f"seed_{seed}"
                / "predictions.parquet",
            }
            for method, path in paths.items():
                prediction = read_frozen_prediction(path)
                if set(prediction.index) != expected:
                    raise RuntimeError(f"{target} seed {seed} {method}: partition mismatch")

    pair_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for target in targets:
        _, _, test_ids = temporal_split_ids(args.data_root, target)
        truth = read_labels_for_ids(args.data_root, target, test_ids).set_index("mol_id")[
            "y_pactivity"
        ]
        structures = pd.read_parquet(
            args.data_root / target / "structures.parquet",
            columns=["mol_id", "canonical_no_stereo_smiles"],
        ).set_index("mol_id")
        test_structures = structures.loc[truth.index]
        novel_by_id = ~test_structures["canonical_no_stereo_smiles"].astype(str).isin(
            prior_structures
        )
        all_edges = pd.read_parquet(args.mmp_root / target / "eligible_mmp_edges.parquet")
        test_edges = all_edges[
            all_edges["mol_id_lo"].isin(truth.index)
            & all_edges["mol_id_hi"].isin(truth.index)
        ].copy()
        edge_strata = classify_cliff_edges(test_edges, truth, novel_by_id)
        point_strata = {
            "all": truth.index,
            "novel": novel_by_id.index[novel_by_id],
            "prior_overlap": novel_by_id.index[~novel_by_id],
        }
        for seed in seeds:
            primary = args.primary_prediction_root / target / f"seed_{seed}"
            predictions = {
                "unscaled_reference": read_frozen_prediction(
                    primary / "unscaled_reference" / "predictions.parquet"
                ),
                "confidence_weighted_bps": read_frozen_prediction(
                    args.weighted_prediction_root / target / f"seed_{seed}" / "predictions.parquet"
                ),
            }
            for method, prediction in predictions.items():
                for stratum, stratum_edges in edge_strata.items():
                    pair_rows.append(
                        {
                            "target": target,
                            "seed": seed,
                            "method": method,
                            "stratum": stratum,
                            "n_cliff_pairs": len(stratum_edges),
                            "pair_delta_mae": pair_mae(prediction, stratum_edges),
                        }
                    )
                for stratum, ids in point_strata.items():
                    point_rows.append(
                        {
                            "target": target,
                            "seed": seed,
                            "method": method,
                            "stratum": stratum,
                            "n_molecules": len(ids),
                            "overall_mae": float(
                                np.mean(np.abs(prediction.loc[ids] - truth.loc[ids]))
                            )
                            if len(ids)
                            else float("nan"),
                        }
                    )

    pair_seed = pd.DataFrame.from_records(pair_rows)
    point_seed = pd.DataFrame.from_records(point_rows)
    pair_target = pair_seed.groupby(["target", "method", "stratum"], as_index=False).agg(
        n_cliff_pairs=("n_cliff_pairs", "first"), pair_delta_mae=("pair_delta_mae", "mean")
    )
    point_target = point_seed.groupby(["target", "method", "stratum"], as_index=False).agg(
        n_molecules=("n_molecules", "first"), overall_mae=("overall_mae", "mean")
    )

    statistics: list[dict[str, object]] = []
    iterations = int(config["evaluation"]["bootstrap_iterations"])
    for table, metric, count_column, family_index in [
        (pair_target, "pair_delta_mae", "n_cliff_pairs", 0),
        (point_target, "overall_mae", "n_molecules", 1),
    ]:
        for stratum_index, stratum in enumerate(sorted(table["stratum"].unique())):
            candidate = table[
                (table["method"] == "confidence_weighted_bps")
                & (table["stratum"] == stratum)
            ].set_index("target")
            reference = table[
                (table["method"] == "unscaled_reference") & (table["stratum"] == stratum)
            ].set_index("target")
            paired = pd.concat(
                [candidate[metric].rename("candidate"), reference[metric].rename("reference")],
                axis=1,
            ).dropna()
            if paired.empty:
                continue
            lower, upper = bootstrap_relative(
                paired["candidate"].to_numpy(),
                paired["reference"].to_numpy(),
                iterations,
                292000 + 100 * family_index + stratum_index,
            )
            count_by_target = candidate.loc[paired.index, count_column]
            statistics.append(
                {
                    "metric": metric,
                    "stratum": stratum,
                    "targets": len(paired),
                    "observations": int(count_by_target.sum()),
                    "reference_macro": float(paired["reference"].mean()),
                    "candidate_macro": float(paired["candidate"].mean()),
                    "relative_change": float(
                        paired["candidate"].mean() / paired["reference"].mean() - 1.0
                    ),
                    "bootstrap95_lower": lower,
                    "bootstrap95_upper": upper,
                    "favorable_targets": int((paired["candidate"] < paired["reference"]).sum()),
                    "favorable_fraction": float((paired["candidate"] < paired["reference"]).mean()),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_seed.to_csv(args.output_dir / "pair_seed_metrics.csv", index=False)
    pair_target.to_csv(args.output_dir / "pair_target_metrics.csv", index=False)
    point_seed.to_csv(args.output_dir / "point_seed_metrics.csv", index=False)
    point_target.to_csv(args.output_dir / "point_target_metrics.csv", index=False)
    statistics_frame = pd.DataFrame.from_records(statistics)
    statistics_frame.to_csv(args.output_dir / "statistics.csv", index=False)
    manifest = {
        "status": "post_hoc_fixed_method_structure_novelty_diagnostic",
        "candidate": "confidence_weighted_bps_threshold0.5_scale1.5",
        "reference": "unscaled_reference",
        "prior_structure_count": len(prior_structures),
        "targets": len(targets),
        "prediction_hash_preflight": True,
        "setting_selection_from_subgroups": False,
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8"
    )
    print(statistics_frame.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
