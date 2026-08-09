from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location


def test_split_and_mmp_scripts_never_open_private_label_table():
    root = Path(__file__).parents[1]
    for name in ("build_mmp_graph.py", "make_scg_splits.py"):
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "labels_private.parquet" not in source


def _load_script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = spec_from_file_location(name.removesuffix(".py"), path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_label_firewall_allows_legitimate_structure_column_names():
    columns = [
        "mol_id",
        "canonical_isomeric_smiles",
        "connectivity_key",
        "heavy_atoms",
    ]
    _load_script("build_mmp_graph.py").assert_label_free_columns(columns)
    _load_script("make_scg_splits.py").assert_label_free_columns(columns)


def test_label_firewall_rejects_private_labels():
    module = _load_script("build_mmp_graph.py")
    try:
        module.assert_label_free_columns(["mol_id", "y_pactivity"])
    except AssertionError:
        pass
    else:
        raise AssertionError("private label column was accepted")


def test_variable_fragment_score_prefers_smaller_transform():
    module = _load_script("build_mmp_graph.py")
    small = module.variable_heavy_atom_score("[*:1]C>>[*:1]Cl")
    large = module.variable_heavy_atom_score("[*:1]CCCC>>[*:1]c1ccccc1")
    assert small < large
