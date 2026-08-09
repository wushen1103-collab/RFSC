import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.train_cliffloss_gine_scg import (
    adaptive_weight,
    cliffloss_objective,
    normalize_severity,
)


def test_cliffloss_objective_matches_paper_formula():
    error = torch.tensor([1.0, 2.0])
    severity = torch.tensor([0.0, 1.0])
    value = cliffloss_objective(error, severity, 0.1)
    assert torch.isclose(value, torch.tensor(1.6))


def test_cliffloss_adaptive_controller_raises_weight_for_q4_gap():
    weight, ema, gap = adaptive_weight(
        q1_mae=2.0,
        q4_mae=4.0,
        previous_ema=0.0,
        base_weight=0.1,
        ema_alpha=0.7,
        gamma=4.0,
        scale_min=0.25,
        scale_max=4.0,
    )
    assert gap > 0
    assert ema > 0
    assert 0.1 < weight <= 0.4


def test_cliffloss_severity_uses_training_quantile_normalization():
    severity, normalizer = normalize_severity(
        np.asarray([0.0, 1.0, 2.0]), [1.0, 2.0, 3.0, 4.0], 0.95
    )
    assert 3.0 < normalizer < 4.0
    assert severity[0] == 0
    assert severity[2] > severity[1] > 0


def test_cliffloss_smoke_config_is_json_serializable():
    config = yaml.safe_load(
        Path("configs/baselines/cliffloss_gine_scg_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    json.dumps(config)
