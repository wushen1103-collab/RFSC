import pandas as pd

from scripts.evaluate_temporal_modern_baselines import parse_source, summarize
from scripts.run_temporal_gine_matrix import select_targets


def test_source_requires_target_and_seed_placeholders() -> None:
    assert parse_source("gatv2=root/{target}/seed_{seed}/predictions.parquet") == (
        "gatv2",
        "root/{target}/seed_{seed}/predictions.parquet",
    )


def test_temporal_smoke_subset_must_be_eligible() -> None:
    assert select_targets(["B", "A", "A"], ["A", "B", "C"]) == ["A", "B"]


def test_summary_uses_target_macro_and_seed_std() -> None:
    rows = []
    for target, offset in (("A", 0.0), ("B", 2.0)):
        for seed in (0, 1, 2):
            rows.append(
                {
                    "target": target,
                    "method": "m",
                    "seed": seed,
                    "overall_mae": 1.0 + offset + seed,
                    "cliff_endpoint_mae": 2.0 + offset + seed,
                    "pair_delta_mae": 3.0 + offset + seed,
                }
            )
    target, seed_macro, macro = summarize(pd.DataFrame(rows))
    assert len(target) == 2
    assert len(seed_macro) == 3
    assert macro.loc[0, "overall_mae_mean"] == 3.0
    assert macro.loc[0, "overall_mae_std"] == 1.0
    assert macro.loc[0, "overall_mae_count"] == 3
