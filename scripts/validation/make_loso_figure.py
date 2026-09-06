#!/usr/bin/env python3
"""
Redraw the leave-one-system-out figure for the Supporting Information.

The first version of this plot was unreadable: two MLP failures at R2 = -428 and
-1338 set the axis range, so every other bar collapsed onto the zero line. Here
the axis is fixed to the range that matters (-3 to 1) and the bars that fall
below it are drawn to the axis floor with their true value printed underneath,
so nothing is hidden.

Usage
    python make_loso_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "revision_analyses" / "loso_results.csv"
OUT = HERE / "revision_analyses"

TARGETS = [
    ("target_fodft_coupling_abs_ev",                    r"$|t_{FO}|$"),
    ("target_frontier_gap_ev",                          r"$\Delta E_{frontier}$"),
    ("target_interaction_energy_kcal_mol",              r"$\Delta E_{int}$"),
    ("target_somo_somo_elst_corrected_signed_kcal_mol", r"$\Delta E^{corr}_{SOMO-SOMO}$"),
]
FAMILIES = [
    ("CPBP",                   "CPBP"),
    ("fluorenyl",              "fluorenyl"),
    ("olympicenyl",            "olympicenyl"),
    ("phenalenyl",             "phenalenyl"),
    ("phenalenyl_olympicenyl", "PLY-OLY"),
]

FLOOR = -3.0          # bottom of the plotted range
CEIL = 1.0
ET_C = "#2b6cb0"
ML_C = "#68a357"


def main() -> None:
    d = pd.read_csv(RES)

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.4), sharey=True)
    x = np.arange(len(FAMILIES))
    w = 0.38

    for ax, (tgt, label) in zip(axes, TARGETS):
        sub = d[d["target"] == tgt].set_index(["held_out_system", "model_family"])["r2"]

        # band marking the range a chemist would call a usable prediction
        ax.axhspan(0.5, CEIL, color="0.90", zorder=0)
        ax.axhline(0.0, color="black", lw=1.0, zorder=1)

        for off, fam_key, colour, name in [(-w / 2, "extra_trees", ET_C, "ExtRa Trees"),
                                           (+w / 2, "mlp", ML_C, "MLP")]:
            vals = np.array([sub.loc[(f, fam_key)] for f, _ in FAMILIES])
            drawn = np.clip(vals, FLOOR, CEIL)
            ax.bar(x + off, drawn, w, color=colour, edgecolor="black", lw=0.6,
                   label=name, zorder=2)
            # print the true value for anything that ran off the bottom
            for xi, v in zip(x + off, vals):
                if v < FLOOR:
                    ax.text(xi, FLOOR + 0.10, f"{v:.0f}", ha="center", va="bottom",
                            fontsize=7.5, rotation=90, color="black", zorder=3)

        ax.set_title(label, fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in FAMILIES], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(FLOOR, CEIL)
        ax.set_yticks([-3, -2, -1, 0, 0.5, 1])
        ax.grid(axis="y", color="0.85", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"$R^2$ on the held-out family", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle("Prediction for a radical family that was left out of training",
                 fontsize=13.5, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_loso_si.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "fig_loso_si.png", "and .pdf")

    # values that fell off the bottom, for the caption
    off = d[d["r2"] < FLOOR][["held_out_system", "target", "model_family", "r2"]]
    print("\noff-scale bars:")
    print(off.to_string(index=False))


if __name__ == "__main__":
    main()
