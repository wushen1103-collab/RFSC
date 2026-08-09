from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_scg_splits.py"
SPEC = spec_from_file_location("make_scg_splits", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_component_assignment_never_splits_component():
    components = {"a": ["a1", "a2"], "b": ["b1"], "c": ["c1", "c2"]}
    assignment, sizes = MODULE.assign_components(components, 2, 2901)
    assert set(assignment) == set(components)
    assert sum(sizes) == 5


def test_assignment_is_deterministic():
    components = {str(i): [f"{i}_{j}" for j in range(i % 4 + 1)] for i in range(20)}
    first = MODULE.assign_components(components, 5, 2901)
    second = MODULE.assign_components(components, 5, 2901)
    assert first == second

