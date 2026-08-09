from __future__ import annotations

import pandas as pd

from scripts.aggregate_modern_baselines import METRICS, aggregate_runs, completeness_table


def _runs() -> pd.DataFrame:
    rows = []
    for method in ["gcn", "gin"]:
        for target in ["A", "B"]:
            for fold in [0, 1]:
                for seed in [0, 1, 2]:
                    value = float(seed + fold + (target == "B"))
                    row = {
                        "method": method,
                        "target": target,
                        "fold": fold,
                        "seed": seed,
                        "wall_time_seconds": 2.0,
                        "peak_vram_bytes": 1024,
                    }
                    row.update({metric: value for metric in METRICS})
                    rows.append(row)
    return pd.DataFrame(rows)


def test_aggregate_runs_uses_seed_then_target_macro() -> None:
    targets, macro = aggregate_runs(_runs())
    assert len(targets) == 4
    assert set(targets.n_seeds) == {3}
    assert len(macro) == 2
    assert set(macro.n_runs) == {12}
    assert set(macro.n_targets) == {2}
    assert set(macro.overall_mae_mean.round(6)) == {2.0}


def test_completeness_requires_three_seeds_per_fold() -> None:
    audit = pd.DataFrame({"target": ["A", "B"], "scg_folds": [2, 2]})
    complete = completeness_table(_runs(), audit)
    assert complete.complete.all()
    partial = completeness_table(_runs().iloc[:-1], audit)
    assert not partial.loc[partial.method == "gin", "complete"].item()

