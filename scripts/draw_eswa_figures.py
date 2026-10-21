#!/usr/bin/env python3
"""Compatibility launcher; use draw_figures.py for current reproduction."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('draw_figures.py')), run_name='__main__')
