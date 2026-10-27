#!/usr/bin/env python3
"""
Manuscript Figure 6, redrawn for the second revision.

Reviewer 1 asked for a more compact layout: the previous version was a tall
4 x 2 stack with a lot of empty space, which would leave the labels unreadable
once scaled to the column width.

What changed
    - 2 rows x 4 columns instead of 4 x 2, so the figure is wide rather than
      tall and the labels keep their size at the printed width
    - target name as a column title and model name as a row label, instead of
      a large floating label inside every panel
    - margins and panel spacing tightened
    - same font, tick sizes and line widths as make_figure5.py, so the two
      figures match

Content is unchanged: permutation importance for both models, top seven
descriptors per target, normalized to the highest-ranked descriptor, error
bars from the standard deviation over 30 permutation repeats.

Inputs
    revision_analyses/extra_trees_permutation_importance.csv
    .../Publication_Top7_FeatureImportance_FODFT/inputs/mlp_feature_importance.csv

Usage
    python make_figure6.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
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

TARGETS = [
    ("target_fodft_coupling_abs_ev",                    r"$|t_{FO}|$"),
    ("target_frontier_gap_ev",                          r"$\Delta E_{\mathrm{frontier}}$"),
    ("target_interaction_energy_kcal_mol",              r"$\Delta E_{\mathrm{int}}$"),
    ("target_somo_somo_elst_corrected_signed_kcal_mol", r"$\Delta E^{corr}_{SOMO-SOMO}$"),
]

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

ROWS = [("ExtRa Trees", ET_CSV, "#356FA8", "abcd"),
        ("MLP",         MLP_CSV, "#D55E00", "efgh")]


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
        raise ValueError(f"no symbol for {list(s.loc[s['sym'].isna(), 'feature'])}")
    return s


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.3))
    exported = []

    for r, (model_label, path, colour, letters) in enumerate(ROWS):
        for c, (target, tlabel) in enumerate(TARGETS):
            ax = axes[r, c]
            s = top_seven(path, target).iloc[::-1].reset_index(drop=True)

            ax.barh(s["sym"], s["rel"], color=colour, edgecolor="none", height=0.64,
                    xerr=s["rel_std"],
                    error_kw={"ecolor": "#444444", "elinewidth": 0.7,
                              "capsize": 1.4, "capthick": 0.7})
            ax.set_xlim(0.0, 1.14)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.tick_params(axis="y", length=0, pad=2, labelsize=9.5)
            ax.tick_params(axis="x", labelsize=7.5, pad=2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            ax.text(-0.30, 1.05, f"({letters[c]})", transform=ax.transAxes,
                    fontsize=9.5, fontweight="bold", va="bottom", ha="left")
            if r == 0:
                ax.set_title(tlabel, fontsize=10.5, pad=10)
            if r == 1:
                ax.set_xlabel("Normalized importance", fontsize=8.5, labelpad=2)
            if c == 0:
                ax.set_ylabel(model_label, fontsize=9.5, labelpad=14)

            e = s.iloc[::-1].copy()
            e.insert(0, "model", model_label)
            e.insert(1, "target", target)
            e.insert(2, "rank", range(1, 8))
            exported.append(e)

    fig.subplots_adjust(left=0.125, right=0.995, top=0.885, bottom=0.115,
                        hspace=0.30, wspace=0.50)

    stem = "figure6_permutation_compact"
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
