from pathlib import Path

import numpy as np

from scripts.build_deepdelta_delta_lgbm_predictions import (
    balanced_ordered_pairs,
    selected_response_values,
)


def test_balanced_pair_sampler_is_bounded_bidirectional_and_keeps_diagonal() -> None:
    first, second = balanced_ordered_pairs(20, maximum=100, seed=29)
    assert len(first) == len(second) <= 100
    pairs = set(zip(first.tolist(), second.tolist()))
    assert all((index, index) in pairs for index in range(20))
    assert all((j, i) in pairs for i, j in pairs)


def test_row_selective_response_loader_preserves_selected_order(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text(
        "SMILES,Y\nC,1.0\nCC,2.0\nCCC,3.0\nCCCC,4.0\n", encoding="utf-8"
    )
    values = selected_response_values(path, np.asarray([0, 2]))
    np.testing.assert_allclose(values, [1.0, 3.0])

