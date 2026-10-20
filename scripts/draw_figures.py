#!/usr/bin/env python3
"""Journal-neutral entry point retaining the original figure implementation."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('draw_eswa_figures.py')), run_name='__main__')
