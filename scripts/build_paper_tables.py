#!/usr/bin/env python3
"""Build paper-ready CSV/Markdown tables from frozen validation artifacts.

This script only reads already frozen, hash-verified result tables; it never
opens private label files and never produces figures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"


def build_scg_panel() -> None:
    source = pd.read_csv(ROOT / "experiments/rf_gine_ensemble/results/ensemble_target_metrics.csv")
    metrics = ["overall_mae", "cliff_mae", "pair_delta_mae"]
    selected = source[source["method"].isin(["gine", "rf_ecfp", "fixed_equal"])].copy()
    selected = selected.drop(columns=["fold", "seed"])
    wide = selected.pivot(index="target", columns="method", values=metrics)
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    for metric in metrics:
        wide[f"{metric}_fixed_equal_relative_vs_gine_pct"] = (
            100.0 * (wide[f"{metric}_fixed_equal"] / wide[f"{metric}_gine"] - 1.0)
        )
    wide.to_csv(OUT / "table_scg_development_panel.csv", index=False)


def build_oracle_panel() -> None:
    source = pd.read_csv(ROOT / "experiments/h1_oracle/results/real_oracle_target_metrics.csv")
    selected = source[source["method"].isin(["base", "global", "oracle", "random", "similarity"])].copy()
    selected.to_csv(OUT / "table_scg_oracle_controls.csv", index=False)


def build_method_screening() -> None:
    rows = [
        {"method": "GINE backbone", "evidence": "6-target SCG panel, 87 paired runs", "status": "anchor"},
        {"method": "Fixed equal RF/GINE", "evidence": "overall/cliff -8.51%/-5.91% vs GINE; pair +1.22%", "status": "strong control; not promoted as BPS"},
        {"method": "BPS-GNN gated component scale (global 1.5/0.5)", "evidence": "23-target confirmation vs fixed equal +1.013%/+1.041%/-2.746%; pair 22/23, bootstrap95 [-3.681%,-1.890%]", "status": "confirmatory Pareto result; locked point GO narrowly false"},
        {"method": "Component-preserving RF/GINE (scale 1.25)", "evidence": "development vs fixed equal overall/cliff/pair +0.41%/+0.22%/-1.73%; pair improved 5/6 targets", "status": "development fixed candidate"},
        {"method": "Pair-aligned RF/GINE", "evidence": "outer-test pair -1.15% vs fixed equal, but overall/cliff +4.66%/+3.06%", "status": "closed trade-off"},
        {"method": "Guarded pair-aligned RF/GINE", "evidence": "validation safety guard; test overall +0.73%, pair -0.34% vs fixed equal", "status": "closed; no pair gain"},
        {"method": "Three-seed fold mean", "evidence": "overall/cliff -4.62%/-2.52% vs seed-mean GINE; pair +1.48%", "status": "closed; variance reduction insufficient"},
        {"method": "Oracle boundary floor 0.4", "evidence": "SCG oracle cliff -7.23% vs global; 6/6", "status": "mechanism supported"},
        {"method": "Learned gate / mass-aware / dual", "evidence": "replications failed frozen gates", "status": "closed"},
        {"method": "PrismNet random-init", "evidence": "pair gain did not transfer; point/cliff worsened", "status": "closed"},
        {"method": "GraphCliff / LAC / CliffLoss", "evidence": "formula-level strict-SCG smokes failed", "status": "closed"},
        {"method": "ExtraTrees descriptors", "evidence": "0/2 replication; cliff pair anti-correlated on CHEMBL4616", "status": "closed"},
        {"method": "TabPFN-CheMeleonFP", "evidence": "0/2 replication; cliff amplitude collapse", "status": "closed"},
        {"method": "Frozen MoLFormer Ridge", "evidence": "CHEMBL287 +116.59%/+133.04%/+103.75% overall/cliff/pair", "status": "closed"},
        {"method": "Context-local MMP transformation expert", "evidence": "train-only gate passed, but validation smoke pair worsened in 3/3 seeds", "status": "closed"},
        {"method": "MTPNet / SemiMol / MaskMol", "evidence": "audit only: leakage, missing protocol, or legacy assets", "status": "hold"},
        {"method": "CamS-LLaMA", "evidence": "reports 0.624 RMSE after Enamine675M pretraining under KPGT/official split; no linked checkpoint/code", "status": "different-protocol literature SOTA anchor"},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "table_method_screening.csv", index=False)
    header = "| " + " | ".join(frame.columns) + " |\n"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |\n"
    body = "".join(
        "| " + " | ".join(str(row[column]).replace("|", "\\|") for column in frame.columns) + " |\n"
        for _, row in frame.iterrows()
    )
    (OUT / "table_method_screening.md").write_text(header + divider + body, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_scg_panel()
    build_oracle_panel()
    build_method_screening()
    print(f"wrote tables to {OUT}")


if __name__ == "__main__":
    main()
