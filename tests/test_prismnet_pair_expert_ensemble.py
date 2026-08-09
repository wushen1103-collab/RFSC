import pandas as pd

from scripts.build_prismnet_pair_expert_ensemble import select_weight, weight_name


def test_pair_expert_selection_respects_point_constraints() -> None:
    frame = pd.DataFrame(
        {
            "prism_weight": [0.0, 0.1, 0.25],
            "validation_overall_mae": [1.0, 1.01, 1.03],
            "validation_cliff_mae": [1.0, 1.02, 0.95],
            "validation_pair_delta_mae": [1.0, 0.9, 0.7],
        }
    )
    assert select_weight(frame, 0.02, 0.02) == 0.1


def test_pair_expert_identity_is_fallback() -> None:
    frame = pd.DataFrame(
        {
            "prism_weight": [0.0, 0.1],
            "validation_overall_mae": [1.0, 1.1],
            "validation_cliff_mae": [1.0, 1.1],
            "validation_pair_delta_mae": [1.0, 0.8],
        }
    )
    assert select_weight(frame, 0.02, 0.02) == 0.0
    assert weight_name(0.1) == "prism_weight_0p10"

