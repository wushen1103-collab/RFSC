#!/usr/bin/env python3
"""Matched-risk and matched-strength diagnostic postprocessor controls.

This is a post-hoc evaluation over frozen predictions and query graphs.  It
does not train models and does not select a new deployment rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from bps_gnn.io import read_frozen_prediction, read_labels_for_ids, read_structure_table
from bps_gnn.projection import active_components, boundary_projection, select_active_edges
from bps_gnn.temporal import temporal_split_ids
from scripts.evaluate_chembl36_temporal import METRICS, metric_row


SCOPE_DEFAULTS = {
    "primary": {
        "config": Path("configs/chembl36_temporal_external.yaml"),
        "eligible": Path("experiments/chembl36_temporal/eligibility/eligible_targets.json"),
        "data_root": Path("data/processed/chembl36_temporal"),
        "mmp_root": Path("graphs/chembl36_temporal_mmp"),
        "prediction_root": Path("experiments/chembl36_temporal/predictions"),
        "label": "Original temporal aggregation (pre-correction)",
    },
    "earliest": {
        "config": Path("configs/chembl36_temporal_external_earliest_year.yaml"),
        "eligible": Path("experiments/chembl36_temporal_earliest/eligibility/eligible_targets.json"),
        "data_root": Path("data/processed/chembl36_temporal_earliest"),
        "mmp_root": Path("graphs/chembl36_temporal_mmp"),
        "prediction_root": Path("experiments/chembl36_temporal_earliest/predictions"),
        "label": "Corrected earliest-year temporal",
    },
}


METHOD_LABELS = {
    "identity": "Identity",
    "all_mmp": "All-MMP scaling",
    "random_q_mmp": "Random-edge scaling",
    "low_delta_q_mmp": "Low-predicted-delta scaling",
    "ecfp_knn": "ECFP-neighborhood scaling",
    "proposed_active_mmp": "Proposed active-MMP BPS",
    "laplacian_unsharp": "Laplacian unsharp control",
    "naive_active_edge": "Naive active-edge push",
}


def ecfp_knn_edges(smiles: list[str], mol_ids: list[str], radius: int, bits: int, k: int, floor: float) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fingerprints = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES: {value}")
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


def stable_rng(base_seed: int, scope: str, target: str, seed: int, threshold: float) -> np.random.Generator:
    token = f"{base_seed}:{scope}:{target}:{seed}:{threshold:.3f}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def component_diagnostics(index: pd.Index, edges: pd.DataFrame) -> dict[str, int]:
    components = active_components(index, edges)
    sizes = [len(members) for members in components.values() if len(members) > 1]
    return {
        "active_edges": int(len(edges)),
        "active_nodes": int(sum(sizes)),
        "nontrivial_components": int(len(sizes)),
        "largest_component": int(max(sizes, default=1)),
    }


def laplacian_unsharp(base: pd.Series, edges: pd.DataFrame, alpha: float) -> pd.Series:
    if edges.empty or alpha == 0:
        return base.copy()
    neighbors: dict[str, list[str]] = {mol_id: [] for mol_id in base.index}
    for lo, hi in edges[["mol_id_lo", "mol_id_hi"]].itertuples(index=False):
        if lo in neighbors and hi in neighbors:
            neighbors[lo].append(hi)
            neighbors[hi].append(lo)
    corrected = base.copy()
    for mol_id, values in neighbors.items():
        if values:
            corrected.loc[mol_id] = base.loc[mol_id] + alpha * (
                base.loc[mol_id] - float(base.reindex(values).mean())
            )
    return corrected


def naive_active_edge_push(base: pd.Series, edges: pd.DataFrame, scale: float) -> pd.Series:
    if edges.empty or scale == 1.0:
        return base.copy()
    shift = pd.Series(0.0, index=base.index)
    count = pd.Series(0.0, index=base.index)
    strength = scale - 1.0
    for lo, hi in edges[["mol_id_lo", "mol_id_hi"]].itertuples(index=False):
        if lo not in base.index or hi not in base.index:
            continue
        delta = float(base.loc[hi] - base.loc[lo])
        if delta >= 0:
            shift.loc[hi] += strength * abs(delta)
            count.loc[hi] += 1.0
        else:
            shift.loc[lo] += strength * abs(delta)
            count.loc[lo] += 1.0
    mask = count > 0
    corrected = base.copy()
    corrected.loc[mask] = corrected.loc[mask] + shift.loc[mask] / count.loc[mask]
    return corrected


def correction_norm(candidate: pd.Series, base: pd.Series) -> float:
    return float(np.linalg.norm((candidate.reindex(base.index) - base).to_numpy(dtype=float)))


def build_scope_rows(scope_key: str, scope: dict, scales: list[float], thresholds: list[float], seed_value: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = yaml.safe_load(scope["config"].read_text(encoding="utf-8"))
    if scope["eligible"].exists():
        targets = json.loads(scope["eligible"].read_text(encoding="utf-8"))
    else:
        targets = sorted(path.name for path in scope["prediction_root"].iterdir() if path.is_dir())
    seeds = [int(value) for value in config["seeds"]]
    knn_config = {
        "fingerprint_radius": 2,
        "fingerprint_bits": 1024,
        "k": 10,
        "minimum_similarity": 0.4,
    }
    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for target in targets:
        structures = read_structure_table(scope["data_root"], target).set_index("mol_id")
        _, _, test_ids = temporal_split_ids(scope["data_root"], target)
        test_set = set(test_ids)
        all_mmp = pd.read_parquet(scope["mmp_root"] / target / "eligible_mmp_edges.parquet")
        mmp_edges = all_mmp[all_mmp.mol_id_lo.isin(test_set) & all_mmp.mol_id_hi.isin(test_set)].copy()
        test_structures = structures.loc[test_ids]
        knn_edges = ecfp_knn_edges(
            test_structures["canonical_isomeric_smiles"].tolist(),
            test_ids,
            int(knn_config["fingerprint_radius"]),
            int(knn_config["fingerprint_bits"]),
            int(knn_config["k"]),
            float(knn_config["minimum_similarity"]),
        )
        truth = read_labels_for_ids(scope["data_root"], target, test_ids).set_index("mol_id")["y_pactivity"]
        for seed in seeds:
            primary_dir = scope["prediction_root"] / target / f"seed_{seed}"
            base = read_frozen_prediction(primary_dir / "unscaled_reference" / "predictions.parquet")
            if set(base.index) != test_set:
                raise RuntimeError(f"Partition mismatch: {scope_key} {target} seed {seed}")
            q_by_threshold = {
                threshold: int(len(select_active_edges(base, mmp_edges, threshold))) for threshold in thresholds
            }
            specifications: list[tuple[str, str, float, float, pd.DataFrame, pd.Series]] = [
                ("identity", "identity", 1.0, 0.0, mmp_edges.iloc[:0], base),
            ]
            for scale in scales:
                specifications.append(("all_mmp", "all_mmp", scale, float("nan"), mmp_edges, boundary_projection(base, mmp_edges, scale)))
                for threshold in thresholds:
                    active = select_active_edges(base, mmp_edges, threshold)
                    rng = stable_rng(seed_value, scope_key, target, seed, threshold)
                    q = q_by_threshold[threshold]
                    random_edges = (
                        mmp_edges.iloc[np.sort(rng.choice(len(mmp_edges), size=q, replace=False))].copy()
                        if q and len(mmp_edges)
                        else mmp_edges.iloc[:0].copy()
                    )
                    delta = (mmp_edges.mol_id_hi.map(base) - mmp_edges.mol_id_lo.map(base)).abs()
                    low_edges = mmp_edges.assign(_delta=delta).sort_values(
                        ["_delta", "mol_id_lo", "mol_id_hi"], kind="stable"
                    ).head(q).drop(columns="_delta")
                    knn_active = select_active_edges(base, knn_edges, threshold)
                    specifications.extend(
                        [
                            (
                                "proposed_active_mmp",
                                "active_mmp",
                                scale,
                                threshold,
                                active,
                                boundary_projection(base, active, scale),
                            ),
                            (
                                "random_q_mmp",
                                "random_q",
                                scale,
                                threshold,
                                random_edges,
                                boundary_projection(base, random_edges, scale),
                            ),
                            (
                                "low_delta_q_mmp",
                                "low_delta_q",
                                scale,
                                threshold,
                                low_edges,
                                boundary_projection(base, low_edges, scale),
                            ),
                            (
                                "ecfp_knn",
                                "ecfp_knn",
                                scale,
                                threshold,
                                knn_active,
                                boundary_projection(base, knn_active, scale),
                            ),
                            (
                                "laplacian_unsharp",
                                "active_mmp_laplacian",
                                scale,
                                threshold,
                                active,
                                laplacian_unsharp(base, active, alpha=scale - 1.0),
                            ),
                            (
                                "naive_active_edge",
                                "active_mmp_naive",
                                scale,
                                threshold,
                                active,
                                naive_active_edge_push(base, active, scale),
                            ),
                        ]
                    )
            for family, graph_rule, scale, threshold, active_edges, prediction in specifications:
                method = family if family == "identity" else f"{family}_s{scale:.2f}_t{threshold:.2f}"
                metric_rows.append(
                    {
                        "scope": scope["label"],
                        "scope_key": scope_key,
                        "target": target,
                        "seed": seed,
                        "method": method,
                        "method_family": family,
                        "method_label": METHOD_LABELS[family],
                        "graph_rule": graph_rule,
                        "scale": scale,
                        "threshold": threshold,
                        **metric_row(prediction, truth, mmp_edges),
                    }
                )
                diagnostic_rows.append(
                    {
                        "scope": scope["label"],
                        "scope_key": scope_key,
                        "target": target,
                        "seed": seed,
                        "method": method,
                        "method_family": family,
                        "scale": scale,
                        "threshold": threshold,
                        "correction_l2": correction_norm(prediction, base),
                        **component_diagnostics(base.index, active_edges),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows)


def summarize(metric_frame: pd.DataFrame, diagnostic_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = metric_frame.groupby(
        ["scope", "scope_key", "target", "method", "method_family", "method_label", "graph_rule", "scale", "threshold"],
        as_index=False,
    )[METRICS].mean()
    diag = diagnostic_frame.groupby(
        ["scope", "scope_key", "method", "method_family", "scale", "threshold"], as_index=False
    )[["correction_l2", "active_edges", "active_nodes", "nontrivial_components", "largest_component"]].mean()
    rows: list[dict[str, object]] = []
    for keys, group in target.groupby(["scope", "scope_key", "method", "method_family", "method_label", "graph_rule", "scale", "threshold"], dropna=False):
        scope, scope_key, method, family, label, graph_rule, scale, threshold = keys
        current = group.set_index("target")
        identity = target[(target.scope_key == scope_key) & (target.method_family == "identity")].set_index("target")
        row: dict[str, object] = {
            "scope": scope,
            "scope_key": scope_key,
            "method": method,
            "method_family": family,
            "method_label": label,
            "graph_rule": graph_rule,
            "scale": scale,
            "threshold": threshold,
            "targets": int(len(current)),
        }
        for metric in METRICS:
            row[metric] = float(current[metric].mean())
            relative = current[metric] / identity.loc[current.index, metric] - 1.0
            row[f"{metric}_relative_change"] = float(relative.mean())
            row[f"{metric}_favorable_targets"] = int((relative < 0).sum())
        rows.append(row)
    summary = pd.DataFrame(rows).merge(
        diag,
        on=["scope", "scope_key", "method", "method_family", "scale", "threshold"],
        how="left",
    )
    return target, summary


def matched_tables(summary: pd.DataFrame, risk_budget: float, cliff_budget: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk_rows = []
    norm_rows = []
    for scope_key, scope_group in summary.groupby("scope_key"):
        proposed_main = scope_group[
            (scope_group.method_family == "proposed_active_mmp")
            & np.isclose(scope_group.scale.astype(float), 1.5)
            & np.isclose(scope_group.threshold.astype(float), 0.5)
        ]
        reference_norm = float(proposed_main["correction_l2"].mean()) if not proposed_main.empty else float("nan")
        for family, family_group in scope_group.groupby("method_family", sort=False):
            if family == "identity":
                continue
            eligible = family_group[
                (family_group.overall_mae_relative_change <= risk_budget)
                & (family_group.cliff_endpoint_mae_relative_change <= cliff_budget)
            ].copy()
            if eligible.empty:
                pool = family_group.assign(
                    risk_violation=np.maximum(0.0, family_group.overall_mae_relative_change - risk_budget)
                    + np.maximum(0.0, family_group.cliff_endpoint_mae_relative_change - cliff_budget)
                ).sort_values(["risk_violation", "pair_delta_mae_relative_change"], kind="stable")
                chosen = pool.iloc[0]
                satisfied = False
            else:
                chosen = eligible.sort_values(
                    ["pair_delta_mae_relative_change", "overall_mae_relative_change", "cliff_endpoint_mae_relative_change"],
                    kind="stable",
                ).iloc[0]
                satisfied = True
            risk_rows.append(
                {
                    "scope": chosen.scope,
                    "method_label": chosen.method_label,
                    "chosen_method": chosen.method,
                    "scale": float(chosen.scale),
                    "threshold": float(chosen.threshold) if pd.notna(chosen.threshold) else float("nan"),
                    "risk_budget_satisfied": bool(satisfied),
                    "overall_relative_change": float(chosen.overall_mae_relative_change),
                    "cliff_relative_change": float(chosen.cliff_endpoint_mae_relative_change),
                    "pair_relative_change": float(chosen.pair_delta_mae_relative_change),
                    "pair_favorable_targets": int(chosen.pair_delta_mae_favorable_targets),
                    "correction_l2": float(chosen.correction_l2),
                }
            )
            norm_chosen = family_group.assign(norm_distance=(family_group.correction_l2 - reference_norm).abs()).sort_values(
                ["norm_distance", "pair_delta_mae_relative_change"], kind="stable"
            ).iloc[0]
            norm_rows.append(
                {
                    "scope": norm_chosen.scope,
                    "reference": "Proposed active-MMP BPS at scale=1.50, threshold=0.50",
                    "reference_correction_l2": reference_norm,
                    "method_label": norm_chosen.method_label,
                    "chosen_method": norm_chosen.method,
                    "scale": float(norm_chosen.scale),
                    "threshold": float(norm_chosen.threshold) if pd.notna(norm_chosen.threshold) else float("nan"),
                    "correction_l2": float(norm_chosen.correction_l2),
                    "correction_l2_distance": float(abs(norm_chosen.correction_l2 - reference_norm)),
                    "overall_relative_change": float(norm_chosen.overall_mae_relative_change),
                    "cliff_relative_change": float(norm_chosen.cliff_endpoint_mae_relative_change),
                    "pair_relative_change": float(norm_chosen.pair_delta_mae_relative_change),
                    "pair_favorable_targets": int(norm_chosen.pair_delta_mae_favorable_targets),
                }
            )
    return pd.DataFrame(risk_rows), pd.DataFrame(norm_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scopes", nargs="*", default=["primary", "earliest"])
    parser.add_argument("--scales", nargs="*", type=float, default=[1.10, 1.25, 1.50, 1.75])
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--risk-budget", type=float, default=0.01)
    parser.add_argument("--cliff-budget", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=2917)
    parser.add_argument("--limit-targets", type=int)
    args = parser.parse_args()

    all_metrics = []
    all_diagnostics = []
    for scope_key in args.scopes:
        scope = dict(SCOPE_DEFAULTS[scope_key])
        if args.limit_targets:
            if scope["eligible"].exists():
                targets = json.loads(scope["eligible"].read_text(encoding="utf-8"))[: args.limit_targets]
            else:
                targets = sorted(path.name for path in scope["prediction_root"].iterdir() if path.is_dir())[: args.limit_targets]
            limited = args.output_dir / f"{scope_key}_limited_targets.json"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            limited.write_text(json.dumps(targets, indent=2), encoding="utf-8")
            scope["eligible"] = limited
        metrics, diagnostics = build_scope_rows(scope_key, scope, args.scales, args.thresholds, args.random_seed)
        all_metrics.append(metrics)
        all_diagnostics.append(diagnostics)
    metric_frame = pd.concat(all_metrics, ignore_index=True)
    diagnostic_frame = pd.concat(all_diagnostics, ignore_index=True)
    target_metrics, summary = summarize(metric_frame, diagnostic_frame)
    matched_risk, matched_norm = matched_tables(summary, args.risk_budget, args.cliff_budget)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(args.output_dir / "postprocessor_grid_run_metrics.csv", index=False)
    diagnostic_frame.to_csv(args.output_dir / "postprocessor_grid_diagnostics.csv", index=False)
    target_metrics.to_csv(args.output_dir / "postprocessor_grid_target_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "postprocessor_grid_method_summary.csv", index=False)
    matched_risk.to_csv(args.output_dir / "table_is_postprocessor_matched_point_risk.csv", index=False)
    matched_norm.to_csv(args.output_dir / "table_is_postprocessor_matched_correction_norm.csv", index=False)
    manifest = {
        "scopes": args.scopes,
        "scales": args.scales,
        "thresholds": args.thresholds,
        "risk_budget": args.risk_budget,
        "cliff_budget": args.cliff_budget,
        "new_training": False,
        "test_labels_used_for_posthoc_evaluation_only": True,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(matched_risk.to_string(index=False))
    print(matched_norm.to_string(index=False))


if __name__ == "__main__":
    main()
