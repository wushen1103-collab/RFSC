import numpy as np
import pandas as pd

from scripts.evaluate_deepdelta_delta_lgbm_projection import ordered_values


def test_ordered_values_restore_zero_padded_edge_order() -> None:
    ids = ["edge_0000_0000", "edge_0000_0001", "edge_0001_0000", "edge_0001_0001"]
    series = pd.Series([3.0, 1.0, 4.0, 2.0], index=[ids[2], ids[0], ids[3], ids[1]])
    np.testing.assert_allclose(ordered_values(series, ids, "test"), [1.0, 2.0, 3.0, 4.0])

