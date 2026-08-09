from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_deepdelta_5x10_predictions import (
    expected_ids,
    infer_kfold_sizes,
    load_prediction_row_only,
    split_square_blocks,
)
from scripts.evaluate_deepdelta_5x10_projection import ordered_potentials


def test_kfold_sizes_are_uniquely_inferred_from_concatenated_squares() -> None:
    expected = [11, 11, 11, 10, 10, 10, 10, 10, 10, 10]
    count = sum(size * size for size in expected)
    assert infer_kfold_sizes(count, 10) == expected


def test_last_row_loader_returns_prediction_not_response(tmp_path: Path) -> None:
    path = tmp_path / "official.csv"
    path.write_text("0,1,2,3\n9,8,7,6\n1.5,2.5,3.5,4.5\n", encoding="utf-8")
    values, digest = load_prediction_row_only(path)
    np.testing.assert_allclose(values, [1.5, 2.5, 3.5, 4.5])
    assert len(digest) == 64


def test_square_blocks_and_frozen_ids_restore_fold_order() -> None:
    sizes = [2, 1]
    values = np.arange(5, dtype=float)
    blocks = split_square_blocks(values, sizes)
    np.testing.assert_allclose(blocks[0], [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(blocks[1], [[4.0]])
    ids = expected_ids(sizes)
    series = pd.Series([30.0, 10.0, 20.0], index=[ids[2], ids[0], ids[1]])
    restored = ordered_potentials(series, sizes)
    np.testing.assert_allclose(restored[0], [10.0, 20.0])
    np.testing.assert_allclose(restored[1], [30.0])

