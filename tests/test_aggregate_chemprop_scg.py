from __future__ import annotations

import json

from bps_gnn.io import freeze_prediction_file
from scripts.aggregate_chemprop_scg import read_runs


def test_read_runs_verifies_all_prediction_hashes(tmp_path) -> None:
    import pandas as pd

    run_dir = tmp_path / "TARGET" / "fold_0" / "seed_0"
    prediction = run_dir / "predictions.parquet"
    digest = freeze_prediction_file(
        pd.DataFrame({"mol_id": ["a"], "prediction": [1.0]}), prediction, {}
    )
    evaluation = {metric: 1.0 for metric in (
        "overall_mae", "overall_rmse", "cliff_mae", "cliff_rmse", "pair_delta_mae"
    )}
    evaluation["prediction_sha256"] = digest
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "prediction_sha256": digest,
                "architecture": "chemprop_dmpnn_v2",
                "wall_time_seconds": 2.0,
            }
        ),
        encoding="utf-8",
    )
    runs = read_runs(tmp_path)
    assert len(runs) == 1
    assert runs.loc[0, "method"] == "chemprop_dmpnn"
    assert runs.loc[0, "prediction_sha256"] == digest
