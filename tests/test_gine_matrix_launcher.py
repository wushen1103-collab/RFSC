from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_gine_matrix import select_subset, select_targets


def _audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["feasible_b", "infeasible", "feasible_a"],
            "scg_folds": [5.0, float("nan"), 4.0],
        }
    )


def test_scg_default_excludes_infeasible_targets() -> None:
    assert select_targets(None, _audit(), "scg") == ["feasible_a", "feasible_b"]


def test_official_default_retains_all_targets() -> None:
    assert select_targets(None, _audit(), "official") == [
        "feasible_a",
        "feasible_b",
        "infeasible",
    ]


def test_explicit_unknown_target_fails() -> None:
    with pytest.raises(ValueError, match="missing from audit"):
        select_targets(["missing"], _audit(), "scg")


def test_smoke_fold_and_seed_subsets_are_validated() -> None:
    assert select_subset([2, 0, 0], [0, 1, 2, 3, 4], "folds") == [0, 2]
    with pytest.raises(ValueError, match="Invalid seeds"):
        select_subset([4], [0, 1, 2], "seeds")
