#!/usr/bin/env python3
"""Create paired H1 contrast tables and an aggregate GO/NO-GO verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def paired_t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2 or np.isclose(values.std(ddof=1), 0.0):
        return mean, mean
    sem = stats.sem(values)
    low, high = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
    return float(low), float(high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    criteria = config["go_no_go"]
    evaluation = pd.read_csv(args.input_dir / "evaluation_raw.csv")
    selected = pd.read_csv(args.input_dir / "selected_lambdas.csv")
    index = ["seed", "jump", "noise_std"]
    metrics = ["boundary_mse", "overall_mse", "jump_delta_mae"]
    wide = evaluation.pivot(index=index, columns="method", values=metrics)
    selected_wide = selected.pivot(index=["jump", "noise_std"], columns="method", values="lambda")

    rows: list[dict] = []
    for (jump, noise), group in wide.groupby(level=["jump", "noise_std"]):
        global_boundary = group[("boundary_mse", "global")].to_numpy()
        oracle_boundary = group[("boundary_mse", "oracle")].to_numpy()
        anti_boundary = group[("boundary_mse", "anti_oracle")].to_numpy()
        global_overall = group[("overall_mse", "global")].to_numpy()
        oracle_overall = group[("overall_mse", "oracle")].to_numpy()
        boundary_gain = (global_boundary - oracle_boundary) / global_boundary
        overall_change = (oracle_overall - global_overall) / global_overall
        boundary_ci = paired_t_interval(boundary_gain)
        overall_ci = paired_t_interval(overall_change)
        relative_boundary_gain = float(
            (global_boundary.mean() - oracle_boundary.mean()) / global_boundary.mean()
        )
        relative_overall_change = float(
            (oracle_overall.mean() - global_overall.mean()) / global_overall.mean()
        )
        is_material = jump >= criteria["material_jump_min"]
        is_no_cliff = np.isclose(jump, 0.0)
        material_boundary_pass = (
            relative_boundary_gain
            >= criteria["oracle_boundary_mse_relative_improvement_min"]
        )
        overall_pass = (
            relative_overall_change <= criteria["oracle_overall_degradation_max"]
        )
        anti_pass = bool(oracle_boundary.mean() < anti_boundary.mean())
        no_cliff_pass = bool(
            abs(relative_boundary_gain)
            <= criteria["no_cliff_oracle_vs_global_relative_tolerance"]
            and abs(relative_overall_change)
            <= criteria["no_cliff_oracle_vs_global_relative_tolerance"]
        )
        condition_pass = bool(
            (is_material and material_boundary_pass and overall_pass and anti_pass)
            or (is_no_cliff and no_cliff_pass)
        )
        global_jump = group[("jump_delta_mae", "global")].mean()
        oracle_jump = group[("jump_delta_mae", "oracle")].mean()
        rows.append(
            {
                "jump": float(jump),
                "noise_std": float(noise),
                "n_seeds": int(len(group)),
                "lambda_global": float(selected_wide.loc[(jump, noise), "global"]),
                "lambda_oracle": float(selected_wide.loc[(jump, noise), "oracle"]),
                "global_boundary_mse": float(global_boundary.mean()),
                "oracle_boundary_mse": float(oracle_boundary.mean()),
                "oracle_boundary_relative_improvement": relative_boundary_gain,
                "paired_boundary_improvement_mean": float(boundary_gain.mean()),
                "paired_boundary_improvement_ci95_low": boundary_ci[0],
                "paired_boundary_improvement_ci95_high": boundary_ci[1],
                "oracle_overall_relative_change": relative_overall_change,
                "paired_overall_change_ci95_low": overall_ci[0],
                "paired_overall_change_ci95_high": overall_ci[1],
                "oracle_vs_anti_boundary_relative_improvement": float(
                    (anti_boundary.mean() - oracle_boundary.mean()) / anti_boundary.mean()
                ),
                "oracle_jump_delta_relative_improvement": float(
                    (global_jump - oracle_jump) / global_jump
                )
                if not np.isnan(global_jump)
                else np.nan,
                "material_jump": bool(is_material),
                "no_cliff": bool(is_no_cliff),
                "condition_pass": condition_pass,
            }
        )

    contrasts = pd.DataFrame(rows).sort_values(["jump", "noise_std"])
    contrasts.to_csv(args.input_dir / "table_synthetic_contrasts.csv", index=False)
    material = contrasts[contrasts["material_jump"]]
    no_cliff = contrasts[contrasts["no_cliff"]]
    verdict = {
        "hypothesis": "H1 synthetic oracle boundary gate",
        "status": "supported"
        if material["condition_pass"].all() and no_cliff["condition_pass"].all()
        else "not_supported",
        "material_conditions_passed": int(material["condition_pass"].sum()),
        "material_conditions_total": int(len(material)),
        "no_cliff_conditions_passed": int(no_cliff["condition_pass"].sum()),
        "no_cliff_conditions_total": int(len(no_cliff)),
        "material_boundary_relative_improvement_min": float(
            material["oracle_boundary_relative_improvement"].min()
        ),
        "material_boundary_relative_improvement_max": float(
            material["oracle_boundary_relative_improvement"].max()
        ),
        "material_overall_relative_change_min": float(
            material["oracle_overall_relative_change"].min()
        ),
        "material_overall_relative_change_max": float(
            material["oracle_overall_relative_change"].max()
        ),
        "weak_jump_failed_conditions": contrasts.loc[
            (~contrasts["material_jump"])
            & (~contrasts["no_cliff"])
            & (contrasts["oracle_boundary_relative_improvement"] < 0),
            ["jump", "noise_std"],
        ].to_dict(orient="records"),
        "interpretation": (
            "Oracle cuts validate the boundary-protection mechanism for material "
            "jumps. Weak jumps at medium/high noise require a conservative learned "
            "gate with fallback rather than unconditional cutting."
        ),
    }
    (args.input_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
