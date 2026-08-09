from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bps_gnn.io import freeze_prediction_file
from scripts.evaluate_predictions import errors


def test_metric_units() -> None:
    observed = np.asarray([1.0, 2.0, 4.0])
    predicted = np.asarray([2.0, 2.0, 2.0])
    result = errors(observed, predicted)
    assert result["mae"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))


def test_prediction_firewall_rejects_labels(tmp_path) -> None:
    frame = pd.DataFrame(
        {"mol_id": ["a"], "prediction": [1.0], "y_pactivity": [1.0]}
    )
    with pytest.raises(ValueError, match="response columns"):
        freeze_prediction_file(frame, tmp_path / "bad.parquet", {})
