import numpy as np
import pandas as pd

from scripts.evaluate_mass_aware_gate_smoke import decide, summarize


def test_mass_aware_smoke_averages_random_within_seed_and_passes_all_gates():
    rows = []
    values = {
        "base": (1.00, 1.00, 1.00),
        "learned": (0.95, 0.95, 1.20),
        "learned_mass_aware": (0.90, 0.90, 1.08),
        "similarity_mass_aware": (0.92, 0.93, 1.10),
    }
    for seed in range(3):
        for method, metrics in values.items():
            rows.append(
                {
                    "method": method,
                    "replicate": -1,
                    "seed": seed,
                    "partition": "validation",
                    "overall_mae": metrics[0],
                    "cliff_mae": metrics[1],
                    "pair_delta_mae": metrics[2],
                }
            )
        for replicate, cliff in enumerate([0.94, 0.96]):
            rows.append(
                {
                    "method": "random_mass_aware",
                    "replicate": replicate,
                    "seed": seed,
                    "partition": "validation",
                    "overall_mae": 0.93,
                    "cliff_mae": cliff,
                    "pair_delta_mae": 1.11,
                }
            )
    table = summarize(pd.DataFrame(rows))
    random = table.set_index("method").loc["random_mass_aware"]
    assert np.isclose(random.cliff_mae, 0.95)
    result = decide(
        table,
        {
            "minimum_cliff_improvement_vs_normalized_soft": 0.02,
            "minimum_pair_improvement_vs_normalized_soft": 0.05,
            "maximum_overall_change_vs_normalized_soft": 0.0,
            "minimum_pair_gap_reduction_toward_base": 0.40,
        },
    )
    assert result["advance_to_unseen_replication"]
