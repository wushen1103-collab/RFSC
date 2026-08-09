from scripts.evaluate_chembl36_temporal_ablations import baseline_for


def test_ablation_baselines_are_prespecified_by_base_family() -> None:
    assert baseline_for("svm_main_mmp") == "svm"
    assert baseline_for("rf_gine_main_mmp") == "rf_gine"
    assert baseline_for("all_mmp") == "identity"
