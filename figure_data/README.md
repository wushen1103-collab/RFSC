# Figure data

Run:

```bash
python scripts/draw_figures.py --input figure_data --out reproduced/figures
```

The script reads this directory directly. `figure_render_sources/` contains the
small panel-level CSV files used by the final figure layouts.

`draw_figures.py` is the canonical journal-neutral implementation; the former
entry point remains available only for backward compatibility. These plotting
sources are separate from the frozen MLST revision audits (Tables S8--S11),
which are documented in `../reproduction/structural_audits.md`.
