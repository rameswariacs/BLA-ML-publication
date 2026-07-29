#!/usr/bin/env python3
"""Create the publication feature-importance figure for four primary targets."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "inputs"
OUTPUT_DIR = HERE / "outputs"
FONT_DIR = HERE / "fonts"

EXTRA_TREES_CSV = INPUT_DIR / "extra_trees_feature_importance.csv"
MLP_CSV = INPUT_DIR / "mlp_feature_importance.csv"

TARGETS = [
    ("target_fodft_coupling_abs_ev", "|t_FO|"),
    ("target_interaction_energy_kcal_mol", "Interaction energy"),
    ("target_frontier_gap_ev", "Spin-conserving frontier gap"),
    (
        "target_somo_somo_elst_corrected_signed_kcal_mol",
        "Corrected SOMO–SOMO energy",
    ),
]

FEATURE_LABELS = {
    "bla_fragment_mean": "Mean BLA",
    "contact_region_bla_mean": "Contact BLA",
    "contact_min_cc_distance": "Minimum C···C distance",
    "stack_interplane_distance": "Interplane distance",
    "stack_lateral_slip": "Lateral slip",
    "stack_plane_normal_angle_deg": "Plane-normal angle",
    "contact_interfragment_distance_std": "Contact-distance spread",
    "projected_pi_overlap_area": "Projected π-overlap area",
    "projected_pi_overlap_fraction_min": "Minimum overlap fraction",
    "pi_projected_area_mean": "Mean projected π-area",
    "n_interfragment_c_contacts_3p4": "C···C contacts ≤ 3.4 Å",
    "n_interfragment_c_contacts_3p6": "C···C contacts ≤ 3.6 Å",
    "n_interfragment_c_contacts_4p0": "C···C contacts ≤ 4.0 Å",
    "mean_top10_closest_c_distances": "Mean ten shortest C···C distances",
    "contact_density_3p6_per_overlap_area": "Short-contact density",
    "pi_projected_area_ratio_min_over_max": "Projected π-area ratio",
    "delta_overlap_fraction": "Overlap-fraction asymmetry",
    "delta_contact_atoms_3p4": "Contact-atom asymmetry",
    "delta_contact_region_bla": "Contact-BLA asymmetry",
}

MODEL_CONFIG = {
    "Extra Trees": {
        "path": EXTRA_TREES_CSV,
        "color": "#356FA8",
    },
    "Descriptor-based MLP": {
        "path": MLP_CSV,
        "color": "#D55E00",
    },
}


def configure_style() -> None:
    for path in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(path))

    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Calibri" not in available:
        raise RuntimeError(
            f"Calibri was not registered from {FONT_DIR}. "
            "Expected Calibri.ttf and related font files."
        )

    mpl.rcParams.update(
        {
            "font.family": "Calibri",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def select_top_seven(path: Path, target: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"target", "feature", "importance"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    selected = (
        frame.loc[frame["target"].eq(target), ["feature", "importance"]]
        .dropna()
        .sort_values("importance", ascending=False)
        .head(7)
        .copy()
    )
    if len(selected) != 7:
        raise ValueError(f"Expected seven features for {target} in {path}")

    selected["relative_importance"] = (
        selected["importance"] / selected["importance"].iloc[0]
    )
    selected["display_label"] = selected["feature"].map(FEATURE_LABELS)
    selected["display_label"] = selected["display_label"].fillna(selected["feature"])
    return selected


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(13.2, 15.2),
        constrained_layout=True,
    )
    exported_rows: list[pd.DataFrame] = []
    panel_index = 0

    for row_index, (target, target_label) in enumerate(TARGETS):
        for column_index, (model_label, config) in enumerate(MODEL_CONFIG.items()):
            ax = axes[row_index, column_index]
            selected = select_top_seven(config["path"], target)
            selected = selected.iloc[::-1].reset_index(drop=True)

            ax.barh(
                selected["display_label"],
                selected["relative_importance"],
                color=config["color"],
                edgecolor="none",
                height=0.68,
            )
            ax.set_xlim(0.0, 1.04)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xlabel("Relative predictive importance")
            ax.set_title(target_label, pad=8)
            ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=4)
            ax.text(
                -0.13,
                1.04,
                f"({chr(ord('a') + panel_index)})",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="bottom",
            )
            if row_index == 0:
                ax.text(
                    0.5,
                    1.20,
                    model_label,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=15,
                    fontweight="bold",
                )

            export = selected.iloc[::-1].copy()
            export.insert(0, "model", model_label)
            export.insert(1, "target", target)
            export.insert(2, "rank", range(1, 8))
            exported_rows.append(export)
            panel_index += 1

    figure_stem = "top7_feature_importance_four_targets_extra_trees_mlp"
    fig.savefig(OUTPUT_DIR / f"{figure_stem}.png", dpi=600)
    fig.savefig(OUTPUT_DIR / f"{figure_stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{figure_stem}.svg")
    plt.close(fig)

    pd.concat(exported_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "top7_feature_importance_values.csv",
        index=False,
    )

    print(f"Created publication figure in {OUTPUT_DIR}")
    print(f"Font: {mpl.rcParams['font.family']}")
    print("Outputs: PNG (600 dpi), PDF, SVG, and plotted-value CSV")


if __name__ == "__main__":
    main()
