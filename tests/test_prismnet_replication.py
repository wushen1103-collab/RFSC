import pandas as pd

from scripts.evaluate_prismnet_replication import replication_gate
from scripts.train_prismnet_scg import should_stop_early


def test_validation_only_early_stop() -> None:
    assert not should_stop_early(74, 20, patience=50, min_epochs=75)
    assert not should_stop_early(90, 50, patience=50, min_epochs=75)
    assert should_stop_early(100, 50, patience=50, min_epochs=75)
    assert not should_stop_early(150, 1, patience=None, min_epochs=75)


def test_replication_gate_requires_both_overall_cells() -> None:
    decision = {
        "maximum_endpoint_relative_change": 0.10,
        "all_runs_overall_must_pass": True,
        "minimum_passed_endpoint_cells": 4,
        "minimum_pair_improved_runs": 1,
    }
    comparison = pd.DataFrame(
        {"overall": [0.05, 0.08], "cliff": [0.09, 0.20], "pair_delta": [-0.01, 0.09]}
    )
    summary, advance = replication_gate(comparison, decision)
    assert summary["passed_endpoint_cells"] == 5
    assert advance
    comparison.loc[1, "overall"] = 0.11
    _, advance = replication_gate(comparison, decision)
    assert not advance

