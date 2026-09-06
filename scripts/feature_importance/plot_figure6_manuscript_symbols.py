#!/usr/bin/env python3
"""
Manuscript Figure 6, redrawn with permutation importance for BOTH models and
with the descriptor symbols used in the paper.

Reviewer 2, comment 3: the published figure used impurity importance in the
ExtRa Trees panels and permutation importance in the MLP panels, so a difference
between the two columns could come from the model or from the measure. Here both
columns are permutation importance.

Layout follows the published Figure 6: ExtRa Trees down the left column with
panel letters (a)-(d), MLP down the right with (e)-(h), target label inside each
panel, one x-axis label per column.

Inputs
    revision_analyses/extra_trees_permutation_importance.csv          (new, ET)
    .../Publication_Top7_FeatureImportance_FODFT/inputs/
        mlp_feature_importance.csv                                    (unchanged)

Usage
    python plot_figure6_manuscript_symbols.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

HERE = Path(__file__).resolve().parent
PUB = (HERE / "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211"
            / "supporting_analyses" / "Publication_Top7_FeatureImportance_FODFT")
FONT_DIR = PUB / "fonts"

ET_CSV = HERE / "revision_analyses" / "extra_trees_permutation_importance.csv"
MLP_CSV = PUB / "inputs" / "mlp_feature_importance.csv"
OUT = HERE / "figure6_revised"

# row order as printed in the manuscript
TARGETS = [
    ("target_fodft_coupling_abs_ev",                    r"$|t_{FO}|$"),
    ("target_frontier_gap_ev",                          r"$\Delta E_{\mathrm{frontier}}$"),
    ("target_interaction_energy_kcal_mol",              r"$\Delta E_{\mathrm{int}}$"),
    ("target_somo_somo_elst_corrected_signed_kcal_mol", r"$\Delta E^{corr}_{SOMO-SOMO}$"),
]

# descriptor symbols exactly as defined in the manuscript
SYMBOLS = {
    "mean_top10_closest_c_distances":       r"$\bar{d}_{10}$",
    "contact_min_cc_distance":              r"$d_{min}$",
    "contact_interfragment_distance_std":   r"$\sigma_{\mathrm{C}\cdots\mathrm{C}}$",
    "n_interfragment_c_contacts_3p4":       r"$N_{3.4}$",
    "n_interfragment_c_contacts_3p6":       r"$N_{3.6}$",
    "n_interfragment_c_contacts_4p0":       r"$N_{4.0}$",
    "contact_density_3p6_per_overlap_area": r"$\rho_{contact}$",
    "delta_contact_atoms_3p4":              r"$\Delta N_{atom}$",
    "projected_pi_overlap_area":            r"$S_{ov}$",
    "projected_pi_overlap_fraction_min":    r"$f_{ov,min}$",
    "pi_projected_area_mean":               r"$\bar{S}_{\pi}$",
    "pi_projected_area_ratio_min_over_max": r"$R_{\mathrm{S}}$",
    "delta_overlap_fraction":               r"$\Delta f_{ov}$",
    "stack_interplane_distance":            r"$d_{\perp}$",
    "stack_lateral_slip":                   r"$d_{slip}$",
    "stack_plane_normal_angle_deg":         r"$\theta_{plane}$",
    "bla_fragment_mean":                    r"$\overline{BLA}_{frag}$",
    "contact_region_bla_mean":              r"$\overline{BLA}_{contact}$",
    "delta_contact_region_bla":             r"$\Delta BLA_{contact}$",
}

COLUMNS = [
    ("ExtRa Trees",          ET_CSV,  "#356FA8", "abcd"),
    ("MLP",                  MLP_CSV, "#D55E00", "efgh"),
]


def configure_style() -> None:
    for path in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(path))
    if "Calibri" not in {f.name for f in font_manager.fontManager.ttflist}:
        raise RuntimeError(f"Calibri not registered from {FONT_DIR}")
    mpl.rcParams.update({
        "font.family": "Calibri",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#000000",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def top_seven(path: Path, target: str) -> pd.DataFrame:
    f = pd.read_csv(path)
    cols = ["feature", "importance"] + (["importance_std"]
                                        if "importance_std" in f.columns else [])
    s = (f.loc[f["target"].eq(target), cols].dropna(subset=["importance"])
         .sort_values("importance", ascending=False).head(7).copy())
    if len(s) != 7:
        raise ValueError(f"expected 7 features for {target} in {path.name}")
    top = s["importance"].iloc[0]
    s["rel"] = s["importance"] / top
    s["rel_std"] = (s["importance_std"] / top
                    if "importance_std" in s.columns else 0.0)
    s["sym"] = s["feature"].map(SYMBOLS)
    if s["sym"].isna().any():
        raise ValueError(f"no symbol for {list(s.loc[s['sym'].isna(),'feature'])}")
    return s


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(8.6, 9.4))
    exported = []

    for col, (model_label, path, colour, letters) in enumerate(COLUMNS):
        for row, (target, target_label) in enumerate(TARGETS):
            ax = axes[row, col]
            s = top_seven(path, target).iloc[::-1].reset_index(drop=True)

            ax.barh(s["sym"], s["rel"], color=colour, edgecolor="none", height=0.62,
                    xerr=s["rel_std"],
                    error_kw={"ecolor": "#444444", "elinewidth": 0.7,
                              "capsize": 1.6, "capthick": 0.7})
            ax.set_xlim(0.0, 1.10)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.tick_params(axis="y", length=0, pad=3, labelsize=11)
            ax.tick_params(axis="x", labelsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            # panel letter, outside the axes at top left
            ax.text(-0.30, 1.02, f"({letters[row]})", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom", ha="left")
            # target label inside the panel
            ax.text(0.66, 0.40, target_label, transform=ax.transAxes,
                    fontsize=13, fontweight="bold", va="center", ha="left")
            # column heading
            if row == 0:
                ax.text(0.5, 1.16, model_label, transform=ax.transAxes,
                        fontsize=15, fontweight="bold", ha="center", va="bottom")
            # x label only on the bottom row
            if row == 3:
                ax.set_xlabel("Normalized feature importance",
                              fontsize=11, fontweight="bold", labelpad=6)

            e = s.iloc[::-1].copy()
            e.insert(0, "model", model_label)
            e.insert(1, "target", target)
            e.insert(2, "rank", range(1, 8))
            exported.append(e)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.935, bottom=0.065,
                        hspace=0.52, wspace=0.40)

    stem = "figure6_permutation_manuscript_symbols"
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.svg")
    plt.close(fig)

    (pd.concat(exported, ignore_index=True)
       [["model", "target", "rank", "feature", "sym", "importance",
         "importance_std", "rel", "rel_std"]]
       .to_csv(OUT / f"{stem}_values.csv", index=False))

    print(f"wrote {OUT}/{stem}.[png|pdf|svg] and {stem}_values.csv")


if __name__ == "__main__":
    main()
