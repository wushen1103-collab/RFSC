from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_chemprop_scg_matrix import task_environment
from scripts.train_chemprop_scg import select_checkpoint, select_prediction_column


def test_project_import_is_available_to_isolated_runner() -> None:
    environment = task_environment("7")
    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    project_root = Path(environment["PYTHONPATH"].split(os.pathsep)[0])
    assert project_root.joinpath("bps_gnn").is_dir()


def test_select_single_chemprop_prediction_column() -> None:
    frame = pd.DataFrame({"mol_id": ["a"], "smiles": ["C"], "y": [1.2]})
    assert select_prediction_column(frame) == "y"


def test_reject_ambiguous_prediction_columns() -> None:
    frame = pd.DataFrame({"mol_id": ["a"], "smiles": ["C"], "a": [1.0], "b": [2.0]})
    with pytest.raises(RuntimeError, match="one numeric"):
        select_prediction_column(frame)


def test_selects_validation_best_over_last_checkpoint(tmp_path) -> None:
    checkpoint_dir = tmp_path / "fold_0" / "model_0" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best = checkpoint_dir / "best-epoch=7-val_loss=0.42.ckpt"
    best.touch()
    (checkpoint_dir / "last.ckpt").touch()
    assert select_checkpoint(tmp_path) == best


def test_rejects_ambiguous_best_checkpoints(tmp_path) -> None:
    (tmp_path / "best-epoch=1-val_loss=1.0.ckpt").touch()
    (tmp_path / "best-epoch=2-val_loss=0.9.ckpt").touch()
    with pytest.raises(RuntimeError, match="validation-best"):
        select_checkpoint(tmp_path)
