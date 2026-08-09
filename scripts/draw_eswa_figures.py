#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the released RFSC figure package from a curated figure_data directory.

Outputs
-------
Main text:
    Figure_2_final.png / .pdf
    Figure_3_final.png / .pdf
    Figure_4_final.png / .pdf
    Figure_5_final.png / .pdf

Supplementary:
    Figure_S1_final.png / .pdf
    Figure_S2_final.png / .pdf
    Figure_S3_final.png / .pdf
    Figure_S4_final.png / .pdf
    Figure_S5_final.png / .pdf
    Figure_S6_final.png / .pdf

Style
-----
- White background
- Thin full box frame
- Light grid
- Fixed manuscript-wide palette
- No large in-panel titles
- (a), (b), ... panel labels at upper left
- 600 dpi PNG + PDF
- Times New Roman if available; Tinos fallback otherwise
"""

from pathlib import Path
import argparse
import zipfile
import shutil
import math
import re

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


# ============================================================
# Fixed manuscript-wide visual system
# ============================================================
PALETTE = {
    "blue": "#4C78A8",       # improvement / Hodge
    "teal": "#2A9D8F",       # selector / hybrid
    "orange": "#E39C35",     # antisym
    "red": "#C94F4F",        # harm / positive penalty
    "purple": "#7B6FB2",     # uncertain
    "gray": "#777F86",       # neutral / absolute / reference
    "dark": "#333333",
    "grid": "#E5E9ED",
    "spine": "#596168",
    "white": "#FFFFFF",
}

for _font_file in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _font_path = Path("C:/Windows/Fonts") / _font_file
    if _font_path.exists():
        font_manager.fontManager.addfont(str(_font_path))

FONT_FAMILY = ["Times New Roman", "serif"]

matplotlib.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 10,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.45,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.32,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 5/72,   # ≈ 5 pt
    "xtick.major.width": 0.45,
    "ytick.major.width": 0.45,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style_axes(ax, xgrid=False, ygrid=False):
    """Publication frame: white background, thin four-sided border, faint grid."""
    ax.set_facecolor("white")
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.45)
        ax.spines[side].set_color(PALETTE["spine"])
    ax.tick_params(direction="out", width=0.45, colors=PALETTE["dark"])
    if xgrid:
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.32, zorder=0)
    if ygrid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.32, zorder=0)


def panel_label(ax, letter):
    ax.text(
        0.012, 0.985, f"({letter})",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10.5, fontweight="bold",
        color=PALETTE["dark"],
    )


def save_figure(fig, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=600)
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)


def mean_se(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x.iloc[0]), 0.0
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def status_color(status):
    s = str(status).upper()
    if s == "GO":
        return PALETTE["blue"]
    if s == "NO-GO":
        return PALETTE["red"]
    return PALETTE["purple"]


# ============================================================
# Data preparation
# ============================================================
def locate_data_root(input_path: Path, workdir: Path):
    """Accept either a figure_data directory or a ZIP with that directory."""
    input_path = input_path.resolve()
    if input_path.is_dir():
        if (input_path / "figure_render_sources").exists():
            return input_path
        if (input_path / "figure_data" / "figure_render_sources").exists():
            return input_path / "figure_data"
        raise FileNotFoundError("Could not find figure_render_sources under input directory.")

    if input_path.suffix.lower() != ".zip":
        raise ValueError("Input must be a figure_data directory or a ZIP archive.")

    extract_dir = workdir / "_figure_data_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(extract_dir)

    candidates = [
        extract_dir / "figure_data",
        extract_dir,
    ]
    for c in candidates:
        if (c / "figure_render_sources").exists():
            return c
    raise FileNotFoundError("figure_data root not found after extracting zip.")


def load_all(base):
    render = base / "figure_render_sources"
    data = {
        "fig2": pd.read_csv(render / "fig2_domain_shift_selector_source.csv"),
        "fig3": pd.read_csv(render / "fig3_effect_forest_source.csv"),
        "fig4": pd.read_csv(render / "fig4_selective_curves_source.csv"),
        "fig5": pd.read_csv(render / "fig5_mechanism_evidence_source.csv"),
        "moleculeace": pd.read_csv(base / "node_effects_moleculeace.csv"),
        "chembl": pd.read_csv(base / "node_effects_chembl.csv"),
        "aqsoldb": pd.read_csv(base / "node_effects_aqsoldb.csv"),
        "selector_features": pd.read_csv(base / "selector_features.csv"),
        "sensitivity": pd.read_csv(base / "sensitivity_3x3.csv"),
        "ablation": pd.read_csv(base / "mechanism_ablation.csv"),
        "inconsistency": pd.read_csv(base / "inconsistency_energy.csv"),
        "anchors": pd.read_csv(base / "anchor_recovery.csv"),
        "random_missingness": pd.read_csv(base / "random_missingness.csv"),
        "structured_missingness": pd.read_csv(base / "structured_missingness.csv"),
        "runtime": pd.read_csv(base / "runtime_scaling.csv"),
        "qm9": pd.read_csv(base / "qm9_pair_metrics.csv"),
        "deepdelta": pd.read_csv(base / "deepdelta_pair_metrics.csv"),
        "delta_lgbm": pd.read_csv(base / "delta_lightgbm_pair_metrics.csv"),
    }
    return data


# ============================================================
# Figure 2
# ============================================================
def draw_figure_2(d, outdir):
    """
    Four-panel external-cohort and selector diagnostic using the frozen
    source table: cohort scale, within-fold MMP support, development
    risk--harm association, and accepted-node harm by nominal coverage.
    """
    src = d["fig2"]
    p_a = src[src["panel"] == "a"].copy()
    p_b = src[src["panel"] == "b"].copy()
    p_c = src[src["panel"] == "c"].copy()
    p_d = src[src["panel"] == "d"].copy()

    fig, axes = plt.subplots(
        2, 2, figsize=(7.25, 6.0),
        gridspec_kw={"wspace": 0.34, "hspace": 0.34}
    )
    axes = axes.ravel()

    # (a) internal median/IQR vs untouched external: paired horizontal bars
    ax = axes[0]
    y = np.arange(len(p_a))[::-1]
    internal = p_a["internal_median"].to_numpy(float)
    internal_err = np.vstack([
        internal - p_a["internal_q25"].to_numpy(float),
        p_a["internal_q75"].to_numpy(float) - internal,
    ])
    ax.barh(
        y + 0.17, internal, xerr=internal_err, height=0.28,
        color=PALETTE["gray"], edgecolor=PALETTE["spine"], linewidth=0.25,
        alpha=0.72, label="MoleculeACE median (IQR)",
        error_kw={"ecolor": PALETTE["spine"], "elinewidth": 0.65,
                  "capsize": 2.0, "capthick": 0.65}, zorder=3,
    )
    ax.barh(
        y - 0.17, p_a["external"].to_numpy(float), height=0.28,
        color=PALETTE["blue"], edgecolor=PALETTE["spine"], linewidth=0.25,
        alpha=0.82, label="AqSolDB", zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(p_a["metric"].tolist())
    ax.set_xlabel("Count")
    style_axes(ax, xgrid=True)
    panel_label(ax, "a")
    ax.legend(loc="upper right", frameon=False, handlelength=1.2)

    # (b) fold-level MMP support in the untouched external cohort
    ax = axes[1]
    p_b = p_b.sort_values("fold")
    yy = np.arange(len(p_b))[::-1]
    ax.barh(
        yy, p_b["internal_mmp_edges"].to_numpy(float), height=0.42,
        color=PALETTE["blue"], edgecolor=PALETTE["spine"], linewidth=0.25,
        alpha=0.82, zorder=3,
    )
    ax.set_yticks(yy)
    ax.set_yticklabels([f"fold {int(v)}" for v in p_b["fold"]])
    ax.set_xlabel("Within-fold MMP edges")
    style_axes(ax, xgrid=True)
    panel_label(ax, "b")

    # (c) development-fold risk--harm association
    ax = axes[2]
    p_c = p_c.sort_values("group")
    yy = np.arange(len(p_c))[::-1]
    means = p_c["mean"].to_numpy(float)
    ses = p_c["se"].fillna(0).to_numpy(float)
    ax.barh(
        yy, means, xerr=ses, height=0.42,
        color=PALETTE["teal"], edgecolor=PALETTE["spine"], linewidth=0.25,
        alpha=0.82,
        error_kw={"ecolor": PALETTE["spine"], "elinewidth": 0.65,
                  "capsize": 2.0, "capthick": 0.65}, zorder=3,
    )
    ax.axvline(0, color=PALETTE["spine"], linestyle="--", linewidth=0.65)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"fold {int(v)}" for v in p_c["group"]])
    ax.set_xlabel("Risk--harm correlation")
    style_axes(ax, xgrid=True)
    panel_label(ax, "c")

    # (d) accepted-node harm across frozen nominal coverage levels
    ax = axes[3]
    p_d = p_d.sort_values("group")
    yy = np.arange(len(p_d))[::-1]
    means = p_d["mean"].to_numpy(float) * 100
    ses = p_d["se"].fillna(0).to_numpy(float) * 100
    ax.barh(
        yy, means, xerr=ses, height=0.42,
        color=PALETTE["red"], edgecolor=PALETTE["spine"], linewidth=0.25,
        alpha=0.78,
        error_kw={"ecolor": PALETTE["spine"], "elinewidth": 0.65,
                  "capsize": 2.0, "capthick": 0.65}, zorder=3,
    )
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{int(float(v) * 100)}%" for v in p_d["group"]])
    ax.set_xlabel("Harmed accepted nodes (%)")
    ax.set_ylabel("Nominal coverage")
    style_axes(ax, xgrid=True)
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.10, top=0.975)
    save_figure(fig, outdir, "Figure_2_final")


# ============================================================
# Figure 3
# ============================================================
def forest_panel(ax, frame, letter, xlabel, color_mode="status"):
    frame = frame.copy()
    y = np.arange(len(frame))[::-1]
    for yi, row in zip(y, frame.itertuples(index=False)):
        c = status_color(row.status) if color_mode == "status" else PALETTE["blue"]
        est = row.relative_change * 100
        lo = row.bootstrap95_lower * 100
        hi = row.bootstrap95_upper * 100
        ax.hlines(yi, lo, hi, color=c, linewidth=0.95, zorder=2)
        ax.vlines([lo, hi], yi - 0.08, yi + 0.08, color=c, linewidth=0.65, zorder=2)
        ax.scatter(est, yi, s=18, color=c, edgecolor=PALETTE["spine"], linewidth=0.22, zorder=3)
    ax.axvline(0, color=PALETTE["spine"], linestyle="--", linewidth=0.65, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"].tolist())
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_xlabel(xlabel, fontsize=11.5)
    style_axes(ax, xgrid=True)
    panel_label(ax, letter)


def draw_figure_3(d, outdir):
    """Horizontal 1x3 evidence map."""
    src = d["fig3"]

    p_a = src[src["panel"].isin(["a", "b"])].copy()
    desired = [
        "overall MAE", "cliff MAE", "pair-delta MAE",
        "accepted point", "accepted cliff", "accepted pair",
        "hybrid overall", "hybrid cliff", "hybrid pair",
    ]
    p_a["label"] = pd.Categorical(p_a["label"], desired, ordered=True)
    p_a = p_a.sort_values("label")

    p_b = src[src["panel"] == "c"].copy()
    p_c = src[src["panel"] == "d"].copy()
    p_b["label"] = p_b["label"].replace({
        "Delta-LGBM pair RMSE": "Delta-LGBM RMSE",
        "Delta-LGBM pair MAE": "Delta-LGBM MAE",
        "Delta-LGBM high-delta MAE": "Delta-LGBM high-delta MAE",
        "QM9 vs raw pair RMSE": "QM9/raw RMSE",
        "QM9 vs raw pair MAE": "QM9/raw MAE",
        "QM9 vs raw high-delta MAE": "QM9/raw high-delta MAE",
        "QM9 vs antisym pair RMSE": "QM9/antisym RMSE",
        "QM9 vs antisym pair MAE": "QM9/antisym MAE",
        "QM9 vs antisym high-delta MAE": "QM9/antisym high-delta MAE",
    })
    p_c["label"] = p_c["label"].replace({
        "chemistry local 0.05": "chemistry-local 5%",
        "chemistry local 0.1": "chemistry-local 10%",
        "chemistry local 0.25": "chemistry-local 25%",
        "disconnected blocks 0.05": "disconnected 5%",
        "disconnected blocks 0.1": "disconnected 10%",
        "disconnected blocks 0.25": "disconnected 25%",
        "hub biased 0.05": "hub-biased 5%",
        "hub biased 0.1": "hub-biased 10%",
        "hub biased 0.25": "hub-biased 25%",
    })

    fig = plt.figure(figsize=(7.25, 5.2))
    grid = fig.add_gridspec(2, 2, wspace=0.95, hspace=0.38)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]
    forest_panel(axes[0], p_a, "a", "Relative change (%)", color_mode="status")
    forest_panel(axes[1], p_b, "b", "Relative change (%)", color_mode="status")
    forest_panel(axes[2], p_c, "c", "Held-edge RMSE change (%)", color_mode="status")

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["blue"],
                   markeredgecolor=PALETTE["spine"], markersize=5, label="Improved"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["purple"],
                   markeredgecolor=PALETTE["spine"], markersize=5, label="Uncertain"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["red"],
                   markeredgecolor=PALETTE["spine"], markersize=5, label="Harm"),
        ],
        loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=3,
        frameon=False, fontsize=8
    )

    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.10, top=0.92)
    save_figure(fig, outdir, "Figure_3_final")


# ============================================================
# Figure 4
# ============================================================
def draw_figure_4(d, outdir):
    """2x2 selective coverage-risk curves with frozen 80% operating point."""
    src = d["fig4"]
    specs = [
        ("realized_node_coverage", "Realized node coverage (%)", "a", PALETTE["gray"], False),
        ("accepted_pair_relative_change", "Accepted-set pair change (%)", "b", PALETTE["teal"], True),
        ("accepted_point_relative_change", "Accepted-set point change (%)", "c", PALETTE["red"], True),
        ("accepted_harmed_fraction", "Accepted harmed fraction", "d", PALETTE["purple"], False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.0))
    axes = axes.ravel()

    for ax, (metric, ylabel, letter, color, pct) in zip(axes, specs):
        p = src[src["metric"] == metric].sort_values("coverage")
        x = p["coverage"].to_numpy(float) * 100
        y = p["mean"].to_numpy(float)
        se = p["se"].to_numpy(float)
        if metric == "realized_node_coverage":
            y = y * 100
            se = se * 100
        elif pct:
            y = y * 100
            se = se * 100

        ax.plot(
            x, y, marker="o", markersize=3.8,
            color=color, linewidth=1.0,
            markerfacecolor="white", markeredgewidth=0.9
        )
        ax.fill_between(x, y-se, y+se, color=color, alpha=0.14, linewidth=0)
        ax.axvline(80, color=PALETTE["gray"], linestyle="--", linewidth=0.65)
        if metric in ("accepted_pair_relative_change", "accepted_point_relative_change"):
            ax.axhline(0, color=PALETTE["spine"], linestyle="--", linewidth=0.55)

        ax.set_xlabel("Nominal coverage (%)")
        ax.set_ylabel(ylabel)
        style_axes(ax, ygrid=True)
        panel_label(ax, letter)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.10, top=0.975, wspace=0.34, hspace=0.34)
    save_figure(fig, outdir, "Figure_4_final")


# ============================================================
# Figure 5
# ============================================================
def parse_sensitivity_levels(method_name):
    s = str(method_name)
    # Handle 1.25 / 1.50 / 1.75 / 2.00 robustly
    scale = np.nan
    threshold = np.nan

    m = re.search(r"s(\d+)p(\d+)", s)
    if m:
        scale = float(f"{m.group(1)}.{m.group(2)}")
    else:
        m2 = re.search(r"s(\d+)(?:_|$)", s)
        if m2:
            scale = float(m2.group(1))

    t = re.search(r"t0p(\d+)", s)
    if t:
        threshold = float(f"0.{t.group(1)}")

    return scale, threshold


def draw_heatmap(ax, pivot, letter, vcenter=0):
    rows = list(pivot.index)
    cols = list(pivot.columns)
    vals = pivot.to_numpy(float)

    vmax = np.nanmax(np.abs(vals))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    cmap = LinearSegmentedColormap.from_list(
        "paper_diverging",
        [PALETTE["blue"], PALETTE["white"], PALETTE["red"]]
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                txt = ""
            else:
                txt = f"{v:.2f}"
            # adaptive annotation color
            rgba = cmap(norm(v)) if np.isfinite(v) else (1,1,1,1)
            luminance = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
            text_color = "white" if luminance < 0.48 else PALETTE["dark"]
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.2, color=text_color)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([f"{float(c):.2f}" for c in cols])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{float(r):.2f}" for r in rows])
    ax.set_xlabel("Activation threshold")
    ax.set_ylabel("Scale")
    style_axes(ax)
    panel_label(ax, letter)
    return im


def draw_figure_5(d, outdir):
    """
    Mechanism summary:
      (a) BPS pair-delta sensitivity
      (b) BPS overall-point penalty sensitivity
      (c) cyclic energy vs incremental Hodge gain
    """
    sens = d["sensitivity"].copy()
    if "evidence_family" in sens.columns:
        sens = sens[sens["evidence_family"] == "ChEMBL36_temporal_earliest"].copy()
    fig5_src = d["fig5"].copy()
    qm9 = d["qm9"].copy()
    inc = d["inconsistency"].copy()

    parsed = sens["method"].apply(parse_sensitivity_levels)
    sens["scale_plot"] = [x[0] for x in parsed]
    sens["thr_plot"] = [x[1] for x in parsed]
    sens = sens[sens["scale_plot"].notna() & sens["thr_plot"].notna()].copy()

    # Baselines from curated source table
    base_pair = fig5_src[
        (fig5_src["metric"] == "pair_delta_mae") &
        (fig5_src["baseline"] == "identity")
    ]["baseline_macro"].median()

    base_overall = fig5_src[
        (fig5_src["metric"] == "overall_mae") &
        (fig5_src["baseline"] == "identity")
    ]["baseline_macro"].median()

    pair_p = sens.groupby(["scale_plot", "thr_plot"])["pair_delta_mae"].mean().unstack()
    overall_p = sens.groupby(["scale_plot", "thr_plot"])["overall_mae"].mean().unstack()

    pair_rel = (pair_p - base_pair) / base_pair * 100
    overall_rel = (overall_p - base_overall) / base_overall * 100

    # Keep all available rows/cols (normally 3 x 3)
    pair_rel = pair_rel.sort_index().sort_index(axis=1)
    overall_rel = overall_rel.sort_index().sort_index(axis=1)

    # QM9 task-fold incremental Hodge gain
    q = qm9.pivot_table(
        index=["task", "fold"],
        columns="method",
        values="pair_mae",
        aggfunc="mean"
    ).reset_index()

    if {"hodge_flow", "antisymmetric_flow"}.issubset(q.columns):
        q["gain_vs_antisym"] = (
            (q["hodge_flow"] - q["antisymmetric_flow"]) /
            q["antisymmetric_flow"] * 100
        )
    else:
        q["gain_vs_antisym"] = np.nan

    energy = inc[
        inc["task"].notna() &
        inc["fold"].notna() &
        inc["cyclic_energy_fraction"].notna()
    ].groupby(["task", "fold"], as_index=False)["cyclic_energy_fraction"].mean()

    scatter = pd.merge(q, energy, on=["task", "fold"], how="inner").dropna(
        subset=["gain_vs_antisym", "cyclic_energy_fraction"]
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(7.25, 3.25),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.10], "wspace": 0.48}
    )

    im1 = draw_heatmap(axes[0], pair_rel, "a")
    im2 = draw_heatmap(axes[1], overall_rel, "b")

    # Compact color bars without titles; meaning belongs in caption
    cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.025)
    cb1.ax.tick_params(labelsize=8, width=0.35)
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.025)
    cb2.ax.tick_params(labelsize=8, width=0.35)

    ax = axes[2]
    ax.scatter(
        scatter["cyclic_energy_fraction"] * 100,
        scatter["gain_vs_antisym"],
        s=20, color=PALETTE["blue"],
        edgecolor=PALETTE["spine"], linewidth=0.25, zorder=3
    )
    # fitted trend if enough unique x values
    if len(scatter) >= 4 and scatter["cyclic_energy_fraction"].nunique() >= 3:
        x = scatter["cyclic_energy_fraction"].to_numpy(float) * 100
        y = scatter["gain_vs_antisym"].to_numpy(float)
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = np.polyval(coef, xx)
        ax.plot(xx, yy, color=PALETTE["blue"], linewidth=0.85, alpha=0.85)
    ax.axhline(0, color=PALETTE["spine"], linestyle="--", linewidth=0.65)
    ax.set_xlabel("Cyclic energy fraction (%)")
    ax.set_ylabel("Hodge vs antisym pair MAE change (%)")
    style_axes(ax, ygrid=True)
    panel_label(ax, "c")

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.965)
    save_figure(fig, outdir, "Figure_5_final")


# ============================================================
# Supplementary Figure S1
# ============================================================
def draw_s1(d, outdir):
    abl = d["ablation"].copy()
    if "evidence_family" in abl.columns:
        abl = abl[abl["evidence_family"] == "ChEMBL36_temporal_earliest"].copy()
    node = abl[abl["pair_delta_mae"].notna()].copy()

    # Compare macro pair MAE by mechanism/control
    g = node.groupby("method", as_index=False)["pair_delta_mae"].mean()
    g = g.sort_values("pair_delta_mae")

    readable = {
        "identity": "Identity",
        "all_mmp": "All MMP edges",
        "low_delta_q_mmp": "Low-delta MMP control",
        "random_q_mmp": "Random MMP control",
        "ecfp_knn_main": "ECFP4 kNN graph",
        "rf_gine_main_mmp": "RF/GINE active MMP",
        "svm_main_mmp": "SVM active MMP",
    }
    for scale in (1.25, 1.50, 1.75):
        for threshold in (0.25, 0.50, 0.75):
            key = f"sensitivity_s{str(scale).replace('.', 'p').rstrip('0').rstrip('p')}_t{str(threshold).replace('.', 'p').rstrip('0').rstrip('p')}"
            readable[key] = f"Scale {scale:.2f}, threshold {threshold:.2f}"
    labels = [readable.get(x, x.replace("_", " ")) for x in g["method"]]

    fig, ax = plt.subplots(figsize=(6.85, 4.25))
    y = np.arange(len(g))
    ax.barh(
        y, g["pair_delta_mae"],
        color=PALETTE["gray"],
        edgecolor=PALETTE["spine"], linewidth=0.25,
        alpha=0.72, height=0.56
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=8.2)
    ax.set_xlabel("Pair-delta MAE")
    style_axes(ax, xgrid=True)
    fig.subplots_adjust(left=0.29, right=0.985, bottom=0.15, top=0.98)
    save_figure(fig, outdir, "Figure_S1_final")


# ============================================================
# Supplementary Figure S2
# ============================================================
def paired_target_change(df, unit_col, ref_method, candidate_method):
    sub = df[df["method"].isin([ref_method, candidate_method])].copy()
    wide = sub.pivot_table(
        index=unit_col, columns="method",
        values="pair_mae", aggfunc="mean"
    )
    if ref_method not in wide or candidate_method not in wide:
        return pd.Series(dtype=float)
    out = (wide[candidate_method] - wide[ref_method]) / wide[ref_method] * 100
    return out.dropna().sort_values()


def draw_s2(d, outdir):
    panels = [
        (d["moleculeace"], "target", "svm_fixed_mixture", "svm_bps_selected", PALETTE["red"]),
        (d["chembl"][d["chembl"]["dataset"] == "ChEMBL36_temporal_earliest"],
         "target", "unscaled_reference", "bps", PALETTE["red"]),
    ]

    aq = d["aqsoldb"]
    panels.append((aq, "fold", "unscaled_reference", "bps", PALETTE["red"]))

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.4), gridspec_kw={"wspace": 0.34})
    for k, (ax, spec) in enumerate(zip(axes, panels)):
        df, unit, ref_m, cand_m, color = spec
        diff = paired_target_change(df, unit, ref_m, cand_m)
        ax.scatter(
            np.arange(len(diff)), diff.to_numpy(),
            s=12, color=color, edgecolor=color, linewidth=0.18, alpha=0.92
        )
        ax.axhline(0, color=PALETTE["spine"], linestyle="--", linewidth=0.65)
        ax.set_xlabel("Evaluation unit (sorted)")
        ax.set_ylabel("Pair MAE change (%)")
        style_axes(ax, ygrid=True)
        panel_label(ax, chr(ord("a")+k))

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.97)
    save_figure(fig, outdir, "Figure_S2_final")


# ============================================================
# Supplementary Figure S3
# ============================================================
def draw_s3(d, outdir):
    rm = d["random_missingness"].copy()
    sm = d["structured_missingness"].copy()

    # random missingness macro relative effects per density
    rm["rmse_change"] = (
        (rm["heldout_sparse_hodge_rmse"] - rm["heldout_direct_rmse"]) /
        rm["heldout_direct_rmse"] * 100
    )
    rm["mae_change"] = (
        (rm["heldout_sparse_hodge_mae"] - rm["heldout_direct_mae"]) /
        rm["heldout_direct_mae"] * 100
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.3), gridspec_kw={"wspace": 0.34})

    for ax, metric, ylabel, letter in [
        (axes[0], "rmse_change", "Held-edge RMSE change (%)", "a"),
        (axes[1], "mae_change", "Held-edge MAE change (%)", "b"),
    ]:
        rows = []
        for density, sub in rm.groupby("density"):
            m, se = mean_se(sub[metric])
            rows.append((density*100, m, se))
        rows = pd.DataFrame(rows, columns=["density", "mean", "se"]).sort_values("density")
        ax.plot(
            rows["density"], rows["mean"],
            marker="o", markersize=3.3,
            color=PALETTE["blue"], linewidth=0.9
        )
        ax.fill_between(
            rows["density"],
            rows["mean"]-rows["se"],
            rows["mean"]+rows["se"],
            color=PALETTE["blue"], alpha=0.14, linewidth=0
        )
        ax.axhline(0, color=PALETTE["spine"], linestyle="--", linewidth=0.65)
        ax.set_xlabel("Observed density (%)")
        ax.set_ylabel(ylabel)
        style_axes(ax, ygrid=True)
        panel_label(ax, letter)

    ax = axes[2]
    scenario_colors = {
        "chemistry_local": PALETTE["red"],
        "disconnected_blocks": PALETTE["blue"],
        "hub_biased": PALETTE["teal"],
    }
    for scen, sub in sm.groupby("scenario"):
        rows = []
        for density, ssub in sub.groupby("nominal_density"):
            m, se = mean_se(ssub["rmse_relative_change"] * 100)
            rows.append((density*100, m, se))
        rows = pd.DataFrame(rows, columns=["density", "mean", "se"]).sort_values("density")
        c = scenario_colors.get(scen, PALETTE["gray"])
        ax.plot(
            rows["density"], rows["mean"],
            marker="o", markersize=3.3,
            color=c, linewidth=0.9, label=scen.replace("_", " ")
        )
        ax.fill_between(
            rows["density"],
            rows["mean"]-rows["se"],
            rows["mean"]+rows["se"],
            color=c, alpha=0.10, linewidth=0
        )
    ax.axhline(0, color=PALETTE["spine"], linestyle="--", linewidth=0.65)
    ax.set_xlabel("Observed density (%)")
    ax.set_ylabel("Held-edge RMSE change (%)")
    style_axes(ax, ygrid=True)
    panel_label(ax, "c")
    ax.legend(loc="best", frameon=False, handlelength=1.4, markerscale=0.85)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.97)
    save_figure(fig, outdir, "Figure_S3_final")


# ============================================================
# Supplementary Figure S4
# ============================================================
def draw_s4(d, outdir):
    anc = d["anchors"].copy()
    g = anc.groupby(["method", "n_anchors"]).agg(
        mean=("scalar_mae", "mean"),
        se=("scalar_mae", lambda x: x.std(ddof=1)/np.sqrt(len(x)))
    ).reset_index()

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for method, color, label in [
        ("hodge_measured_anchor", PALETTE["blue"], "Hodge recovery"),
        ("absolute_lgbm", PALETTE["gray"], "Absolute control"),
    ]:
        p = g[g["method"] == method].sort_values("n_anchors")
        if p.empty:
            continue
        ax.plot(
            p["n_anchors"], p["mean"],
            marker="o", markersize=3.3,
            color=color, linewidth=0.9, label=label
        )
        ax.fill_between(
            p["n_anchors"], p["mean"]-p["se"], p["mean"]+p["se"],
            color=color, alpha=0.10, linewidth=0
        )
    ax.set_xlabel("Number of anchors")
    ax.set_ylabel("Scalar MAE")
    style_axes(ax, ygrid=True)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4, markerscale=0.85)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.19, top=0.98)
    save_figure(fig, outdir, "Figure_S4_final")


# ============================================================
# Supplementary Figure S5
# ============================================================
def draw_s5(d, outdir):
    runtime = d["runtime"].copy()
    node = runtime[runtime["operator"].str.contains("node_boundary_projection", na=False)]
    edge = runtime[~runtime["operator"].str.contains("node_boundary_projection", na=False)]

    fig, axes = plt.subplots(
        2, 2, figsize=(7.25, 6.0),
        gridspec_kw={"wspace": 0.34, "hspace": 0.34},
    )
    axes = axes.ravel()

    ax = axes[0]
    if not node.empty:
        node = node.sort_values("n_nodes")
        ax.plot(
            node["n_nodes"], node["runtime_ms_median"],
            marker="o", markersize=3.0,
            color=PALETTE["gray"], linewidth=0.85
        )
        ax.fill_between(
            node["n_nodes"], node["runtime_ms_median"], node["runtime_ms_p95"],
            color=PALETTE["gray"], alpha=0.12, linewidth=0
        )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Runtime (ms)")
    style_axes(ax, ygrid=True)
    panel_label(ax, "a")

    ax = axes[1]
    if not edge.empty:
        for op, sub in edge.groupby("operator"):
            sub = sub.sort_values("n_nodes")
            label = {
                "complete_edge_hodge": "Complete Hodge",
                "incomplete_edge_hodge_dense_lstsq": "Incomplete Hodge",
            }.get(str(op), str(op))
            ax.plot(
                sub["n_nodes"], sub["runtime_ms_median"],
                marker="o", markersize=2.9,
                linewidth=0.85, label=label
            )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Runtime (ms)")
    style_axes(ax, ygrid=True)
    panel_label(ax, "b")
    if edge["operator"].nunique() > 1:
        ax.legend(loc="upper left", frameon=False, handlelength=1.4, markerscale=0.85)

    ax = axes[2]
    if not node.empty:
        node = node.sort_values("n_nodes")
        ax.plot(
            node["n_nodes"], node["peak_traced_allocation_mib"],
            marker="o", markersize=3.0,
            color=PALETTE["gray"], linewidth=0.85,
        )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Peak traced allocation (MiB)")
    style_axes(ax, ygrid=True)
    panel_label(ax, "c")

    ax = axes[3]
    if not edge.empty:
        for op, sub in edge.groupby("operator"):
            sub = sub.sort_values("n_nodes")
            label = {
                "complete_edge_hodge": "Complete Hodge",
                "incomplete_edge_hodge_dense_lstsq": "Incomplete Hodge",
            }.get(str(op), str(op))
            ax.plot(
                sub["n_nodes"], sub["peak_traced_allocation_mib"],
                marker="o", markersize=2.9,
                linewidth=0.85, label=label,
            )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Peak traced allocation (MiB)")
    style_axes(ax, ygrid=True)
    panel_label(ax, "d")
    if edge["operator"].nunique() > 1:
        ax.legend(loc="upper left", frameon=False, handlelength=1.4, markerscale=0.85)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.10, top=0.975)
    save_figure(fig, outdir, "Figure_S5_final")


# ============================================================
# Supplementary Figure S6
# ============================================================
def draw_s6(d, outdir):
    feat = d["selector_features"].copy()
    vals = feat["risk_harm_correlation"].dropna()

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.hist(
        vals, bins=14,
        color=PALETTE["teal"],
        edgecolor=PALETTE["spine"], linewidth=0.25,
        alpha=0.68
    )
    ax.axvline(0, color=PALETTE["spine"], linestyle="--", linewidth=0.65)
    ax.set_xlabel("Risk–harm correlation")
    ax.set_ylabel("Count")
    style_axes(ax, ygrid=True)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.19, top=0.98)
    save_figure(fig, outdir, "Figure_S6_final")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("figure_data"),
        help="Path to the curated figure_data directory or a ZIP archive."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/rendered"),
        help="Output directory."
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=Path(".eswa_figure_work"),
        help="Temporary extraction directory."
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help=(
            "Optional comma-separated subset to draw, e.g. "
            "'s1,s2,s3,s4,s5,s6' or 'fig2,fig5'. "
            "Empty draws all figures."
        ),
    )
    args = parser.parse_args()

    base = locate_data_root(args.input, args.work)
    d = load_all(base)

    args.out.mkdir(parents=True, exist_ok=True)

    draw_steps = {
        "fig2": draw_figure_2,
        "figure2": draw_figure_2,
        "fig3": draw_figure_3,
        "figure3": draw_figure_3,
        "fig4": draw_figure_4,
        "figure4": draw_figure_4,
        "fig5": draw_figure_5,
        "figure5": draw_figure_5,
        "s1": draw_s1,
        "figs1": draw_s1,
        "s2": draw_s2,
        "figs2": draw_s2,
        "s3": draw_s3,
        "figs3": draw_s3,
        "s4": draw_s4,
        "figs4": draw_s4,
        "s5": draw_s5,
        "figs5": draw_s5,
        "s6": draw_s6,
        "figs6": draw_s6,
    }
    default_order = ["fig2", "fig3", "fig4", "fig5", "s1", "s2", "s3", "s4", "s5", "s6"]
    if args.only.strip():
        requested = [x.strip().lower() for x in args.only.split(",") if x.strip()]
    else:
        requested = default_order
    unknown = [x for x in requested if x not in draw_steps]
    if unknown:
        raise ValueError(f"Unknown --only entries: {unknown}")
    for key in requested:
        draw_steps[key](d, args.out)

    # Manifest
    files = sorted([p for p in args.out.iterdir() if p.is_file()])
    manifest = pd.DataFrame({
        "file": [p.name for p in files],
        "size_kb": [round(p.stat().st_size/1024, 1) for p in files],
    })
    manifest.to_csv(args.out / "figure_manifest.csv", index=False)

    print(f"Generated {len(files)} figure files in: {args.out.resolve()}")


if __name__ == "__main__":
    main()
