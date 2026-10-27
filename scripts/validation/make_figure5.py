#!/usr/bin/env python3
"""
Manuscript Figure 5, redrawn for the second revision.

Reviewer 1 asked for: no grid lines, consistent fonts and line widths across
panels, and the legend moved away from the data.

What changed
    - grid removed entirely
    - 2 rows x 4 columns instead of a tall stack, so the figure fits the
      double-column width without shrinking the labels
    - one shared legend under the figure, so no legend sits on the data
    - metrics printed top-left, where a parity plot is always empty
    - every panel uses the same font, tick size and line widths, matching
      make_figure6.py so the two figures look like a pair

Data
    extra_trees/extra_trees_cv_predictions.csv
    mlp/mlp_cv_predictions.csv
        both inside the production run directory; 2,582 out-of-fold
        predictions per target per model

Usage
    python make_figure5.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE / "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211"
FONT_DIR = (RUN / "supporting_analyses"
                / "Publication_Top7_FeatureImportance_FODFT" / "fonts")
OUT = HERE / "figure5_revised"

ET_CSV = RUN / "extra_trees" / "extra_trees_cv_predictions.csv"
MLP_CSV = RUN / "mlp" / "mlp_cv_predictions.csv"

TARGETS = [
    ("target_fodft_coupling_abs_ev",                    r"$|t_{FO}|$",                        "eV"),
    ("target_frontier_gap_ev",                          r"$\Delta E_{\mathrm{frontier}}$",    "eV"),
    ("target_interaction_energy_kcal_mol",              r"$\Delta E_{\mathrm{int}}$",         r"kcal mol$^{-1}$"),
    ("target_somo_somo_elst_corrected_signed_kcal_mol", r"$\Delta E^{corr}_{SOMO-SOMO}$",     r"kcal mol$^{-1}$"),
]

# same family palette as the Supporting Information distribution figures
FAMILIES = [
    ("phenalenyl",            "Phenalenyl",   "#008C95"),
    ("olympicenyl",           "Olympicenyl",  "#EF6C3A"),
    ("fluorenyl",             "Fluorenyl",    "#36A269"),
    ("CPBP",                  "CPBP",         "#173B57"),
    ("phenalenyl_olympicenyl", "PLY–OLY", "#B84545"),
]

ROWS = [("ExtRa Trees", ET_CSV, "abcd"), ("MLP", MLP_CSV, "efgh")]


def configure_style() -> None:
    for path in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(path))
    fam = ("Calibri" if "Calibri" in {f.name for f in font_manager.fontManager.ttflist}
           else "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": fam,
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "text.color": "#000000",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def metrics(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return (1.0 - ss_res / ss_tot,
            float(np.mean(np.abs(y - p))),
            float(np.sqrt(np.mean((y - p) ** 2))))


def main() -> None:
    configure_style()
    OUT.mkdir(exist_ok=True)

    data = {label: pd.read_csv(path) for label, path, _ in ROWS}
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0))
    rows_out = []

    for r, (model_label, _, letters) in enumerate(ROWS):
        d = data[model_label]
        for c, (target, tlabel, unit) in enumerate(TARGETS):
            ax = axes[r, c]
            sub = d[d["target"] == target]

            lo = min(sub["true"].min(), sub["cv_prediction"].min())
            hi = max(sub["true"].max(), sub["cv_prediction"].max())
            pad = 0.06 * (hi - lo)
            lo, hi = lo - pad, hi + pad

            ax.plot([lo, hi], [lo, hi], color="#555555", lw=0.8, zorder=1)

            for key, _, colour in FAMILIES:
                f = sub[sub["system"] == key]
                ax.scatter(f["true"], f["cv_prediction"], s=2.2, c=colour,
                           alpha=0.55, linewidths=0, zorder=2, rasterized=True)

            r2, mae, rmse = metrics(sub["true"].to_numpy(float),
                                    sub["cv_prediction"].to_numpy(float))
            ax.text(0.04, 0.96,
                    f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.2, linespacing=1.35, zorder=3)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=7.5, pad=2)
            ax.locator_params(nbins=5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            ax.text(-0.08, 1.06, f"({letters[c]})", transform=ax.transAxes,
                    fontsize=9.5, fontweight="bold", va="bottom", ha="left")
            if r == 0:
                ax.set_title(tlabel, fontsize=10.5, pad=12)
            if r == 1:
                ax.set_xlabel(f"DFT ({unit})", fontsize=8.5, labelpad=2)
            if c == 0:
                ax.set_ylabel(f"{model_label}\npredicted", fontsize=8.5, labelpad=2)

            rows_out.append({"model": model_label, "target": target, "n": len(sub),
                             "r2": r2, "mae": mae, "rmse": rmse, "unit": unit})

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=4.2,
                      markerfacecolor=col, markeredgecolor="none", label=name)
               for _, name, col in FAMILIES]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, handletextpad=0.35, columnspacing=1.5,
               bbox_to_anchor=(0.5, -0.015))

    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.165,
                        hspace=0.38, wspace=0.34)

    stem = "figure5_parity_revised"
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.svg")
    plt.close(fig)

    pd.DataFrame(rows_out).to_csv(OUT / f"{stem}_metrics.csv", index=False)
    print(f"wrote {OUT}/{stem}.[png|pdf|svg] and {stem}_metrics.csv")
    print(pd.DataFrame(rows_out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
