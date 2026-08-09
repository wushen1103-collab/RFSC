import pandas as pd

from scripts.evaluate_prismnet_pair_expert_confirmation import confirmation_gate


def test_confirmation_gate_requires_both_pair_directions() -> None:
    decision = {
        "maximum_overall_relative_change_vs_fixed_equal": 0.01,
        "maximum_cliff_relative_change_vs_fixed_equal": 0.01,
        "minimum_pair_delta_improvement_vs_fixed_equal": 0.03,
        "require_pair_improvement_in_both_runs": True,
        "maximum_overall_relative_change_vs_gine": 0.0,
        "maximum_cliff_relative_change_vs_gine": 0.0,
        "maximum_pair_delta_relative_change_vs_gine": 0.0,
    }
    frame = pd.DataFrame({
        "overall_vs_fixed_equal": [0.0, 0.0], "cliff_vs_fixed_equal": [0.0, 0.0],
        "pair_delta_vs_fixed_equal": [-0.08, -0.02], "overall_vs_gine": [-0.1, -0.1],
        "cliff_vs_gine": [-0.1, -0.1], "pair_delta_vs_gine": [-0.1, -0.1],
    })
    _, passed = confirmation_gate(frame, decision)
    assert passed
    frame.loc[1, "pair_delta_vs_fixed_equal"] = 0.01
    _, passed = confirmation_gate(frame, decision)
    assert not passed

