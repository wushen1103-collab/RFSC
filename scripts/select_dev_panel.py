#!/usr/bin/env python3
"""Select a structure/audit-diverse development panel before model results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


FEATURES = [
    "log_n",
    "log_mmp_edges_per_node",
    "official_cliff_prevalence",
    "largest_scg_component_fraction",
]


def robust_scale(frame: pd.DataFrame) -> pd.DataFrame:
    median = frame.median()
    iqr = frame.quantile(0.75) - frame.quantile(0.25)
    return (frame - median) / iqr.mask(iqr == 0, 1.0)


def select_anchor_panel(eligible: pd.DataFrame, scaled: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Cover six distinct audit regimes without looking at model performance."""

    selected: list[str] = []
    trace: list[dict] = []

    def add(target: str, rule: str) -> None:
        if target in selected:
            raise RuntimeError(f"Anchor rule selected duplicate target: {target}")
        selected.append(target)
        trace.append({"step": len(selected), "target": target, "rule": rule})

    center_distance = np.square(scaled).sum(axis=1)
    add(
        sorted(center_distance.index, key=lambda name: (center_distance.loc[name], name))[0],
        "closest_to_robust_center",
    )
    add(
        eligible.sort_values(["mmp_edges_per_node", "target"], ascending=[False, True]).iloc[0]["target"],
        "maximum_mmp_edges_per_node",
    )
    add(
        eligible.sort_values(["mmp_edges_per_node", "target"], ascending=[True, True]).iloc[0]["target"],
        "minimum_mmp_edges_per_node",
    )
    add(
        eligible.sort_values(["unique_isomeric_smiles", "target"], ascending=[False, True]).iloc[0]["target"],
        "maximum_sample_size",
    )
    five_fold = eligible[(eligible["scg_folds"] == eligible["scg_folds"].max()) & (~eligible["target"].isin(selected))]
    add(
        five_fold.sort_values(["largest_scg_component_fraction", "target"], ascending=[False, True]).iloc[0]["target"],
        "maximum_component_concentration_among_five_fold_targets",
    )
    ec50 = eligible[eligible["target"].str.endswith("_EC50") & (~eligible["target"].isin(selected))]
    add(
        ec50.sort_values(["minimum_cliff_endpoints_per_fold", "target"], ascending=[True, True]).iloc[0]["target"],
        "minimum_strict_scg_cliff_power_among_ec50_targets",
    )
    return selected, trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--power-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.audit)
    power = pd.read_csv(args.power_audit)
    minimum_power = (
        power.groupby("target")["test_mmp_cliff_endpoints_dy_ge_1"]
        .min()
        .rename("minimum_cliff_endpoints_per_fold")
    )
    eligible = audit[audit["scg_folds"].notna()].copy()
    eligible = eligible.merge(minimum_power, on="target", how="left", validate="one_to_one")
    eligible["log_n"] = np.log1p(eligible["unique_isomeric_smiles"])
    eligible["mmp_edges_per_node"] = eligible["eligible_mmp_edges"] / eligible["unique_isomeric_smiles"]
    eligible["log_mmp_edges_per_node"] = np.log1p(eligible["mmp_edges_per_node"])
    eligible["official_cliff_prevalence"] = (
        eligible["official_cliff_rows"] / eligible["unique_isomeric_smiles"]
    )
    scaled = robust_scale(eligible.set_index("target")[FEATURES])
    panel, trace = select_anchor_panel(eligible, scaled)
    feature_records = (
        eligible.set_index("target")
        .loc[panel, ["unique_isomeric_smiles", "eligible_mmp_edges", "mmp_edges_per_node", "scg_folds", "minimum_cliff_endpoints_per_fold", *FEATURES[2:]]]
        .reset_index()
        .to_dict(orient="records")
    )
    payload = {
        "status": "frozen_before_model_results",
        "selection": {
            "algorithm": "deterministic six-anchor audit coverage",
            "anchors": [item["rule"] for item in trace],
            "tie_break": "lexicographic target identifier",
            "eligible_pool": "SCG-feasible MoleculeACE targets only",
            "features": FEATURES,
            "panel_size": len(panel),
        },
        "development_panel": panel,
        "selection_trace": trace,
        "audit_values": feature_records,
        "prohibited_selection_inputs": [
            "BPS predictions",
            "baseline prediction errors",
            "test-set method comparisons",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(json.dumps({"development_panel": panel, "trace": trace}, indent=2))


if __name__ == "__main__":
    main()
