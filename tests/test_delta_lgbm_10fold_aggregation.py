from __future__ import annotations

import pandas as pd

from scripts.aggregate_delta_lgbm_10fold import comparison_statistics


def test_tenfold_statistics_average_folds_before_dataset_bootstrap() -> None:
    rows = []
    for dataset_index, dataset in enumerate(["A", "B", "C"]):
        for fold in [1, 2]:
            reference = 2.0 + dataset_index + 0.1 * fold
            for method, factor in [
                ("direct_delta_lgbm", 1.0),
                ("hodge_cycle_consistent", 0.9),
            ]:
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": method,
                        "pair_rmse": reference * factor,
                        "pair_mae": reference * factor,
                        "high_delta_pair_mae": reference * factor,
                    }
                )
    statistics = comparison_statistics(
        pd.DataFrame(rows),
        "hodge_cycle_consistent",
        "direct_delta_lgbm",
        iterations=200,
        seed=7,
    )
    assert len(statistics) == 3
    assert all(row["datasets"] == 3 for row in statistics)
    assert all(row["dataset_folds"] == 6 for row in statistics)
    assert all(row["favorable_datasets"] == 3 for row in statistics)
    assert all(row["favorable_dataset_folds"] == 6 for row in statistics)
    assert all(abs(row["relative_change"] + 0.1) < 1e-12 for row in statistics)

