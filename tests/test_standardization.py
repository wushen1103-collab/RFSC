from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "standardize_molecules.py"
SPEC = spec_from_file_location("standardize_molecules", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_label_shift_relation_is_exact_for_known_values():
    assert abs((-2.6989700043360187 + 9.0) - 6.301029995663981) < 1e-12


def test_standardization_preserves_stereochemistry():
    left = MODULE.standardize_one("F[C@H](Cl)Br")
    right = MODULE.standardize_one("F[C@@H](Cl)Br")
    assert left["canonical_isomeric_smiles"] != right["canonical_isomeric_smiles"]
    assert left["canonical_no_stereo_smiles"] == right["canonical_no_stereo_smiles"]


def test_acyclic_scaffolds_do_not_form_one_giant_group():
    first = MODULE.standardize_one("CCO")
    second = MODULE.standardize_one("CCN")
    assert first["scaffold_group"].startswith("ACYCLIC::")
    assert second["scaffold_group"].startswith("ACYCLIC::")
    assert first["scaffold_group"] != second["scaffold_group"]

