import pandas as pd

from scripts.build_expansion_tables import METRICS, aggregate_raw, markdown_table


def test_raw_aggregation_uses_fold_then_target_then_seed_macro() -> None:
    rows = []
    for target, offset in (("A", 0.0), ("B", 2.0)):
        for fold in (0, 1):
            for seed in (0, 1, 2):
                row = {
                    "target": target,
                    "fold": fold,
                    "seed": seed,
                    "method": "m",
                }
                row.update({metric: offset + fold + seed for metric in METRICS})
                rows.append(row)
    result = aggregate_raw(pd.DataFrame(rows)).iloc[0]
    assert result.n_targets == 2
    assert result.n_runs == 12
    assert result.n_seeds == 3
    assert result.overall_mae_mean == 2.5
    assert result.overall_mae_std == 1.0


def test_markdown_table_has_no_optional_dependency() -> None:
    rendered = markdown_table(pd.DataFrame({"name": ["a|b"], "value": [1.23456]}))
    assert "a\\|b" in rendered
    assert "1.2346" in rendered
