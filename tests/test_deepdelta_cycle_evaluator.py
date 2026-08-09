import pandas as pd
import pytest

from scripts.evaluate_deepdelta_cycle_projection import ordered_position_values


def test_position_ids_are_restored_numerically_after_lexical_freeze_sort() -> None:
    lexical_ids = sorted(f"position_{index}" for index in range(12))
    lexical = pd.Series(
        [float(mol_id.removeprefix("position_")) for mol_id in lexical_ids],
        index=lexical_ids,
    )
    values = ordered_position_values(lexical, 12)
    assert values.tolist() == [float(index) for index in range(12)]


def test_position_id_mismatch_is_rejected() -> None:
    malformed = pd.Series([0.0, 1.0], index=["position_0", "wrong_1"])
    with pytest.raises(RuntimeError, match="frozen position IDs mismatch"):
        ordered_position_values(malformed, 2)
