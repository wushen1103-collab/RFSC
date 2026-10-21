"""Keep the documented figure command and compatibility launcher consistent."""
from pathlib import Path
import ast
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_figure_documentation_uses_canonical_entrypoint():
    for relative in (
        'README.md', 'REPRODUCING.md',
        'figure_data/README.md', 'figure_data/DATA_AVAILABILITY.md',
    ):
        text = (ROOT / relative).read_text(encoding='utf-8')
        assert 'draw_figures.py' in text, relative
        assert 'draw_eswa_figures.py' not in text, relative
        assert 'ESWA submission results' not in text, relative


def test_canonical_entrypoint_contains_the_implementation():
    tree = ast.parse((ROOT/'scripts/draw_figures.py').read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(functions) > 10
    assert any(n.name == 'main' for n in functions)


def test_legacy_and_canonical_help_match_without_rendering():
    outputs = []
    for filename in ('draw_figures.py', 'draw_eswa_figures.py'):
        result = subprocess.run(
            [sys.executable, str(ROOT/'scripts'/filename), '--help'],
            cwd=ROOT, check=True, capture_output=True, text=True,
            env={**os.environ, 'MPLBACKEND': 'Agg'},
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert '--input' in outputs[0] and '--out' in outputs[0]
