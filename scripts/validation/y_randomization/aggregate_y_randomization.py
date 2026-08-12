#!/usr/bin/env python3
"""Aggregate Y-randomization tasks and create summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path("/home/rb1820/BLA-ML")
RUNSCRIPTS = ROOT / "RunScripts_plus_CPBP"
SOURCE = ROOT / "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211"
OUT = ROOT / "/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/YRandomization_500x_FODFT_4Targets"
RESULTS = OUT / "results_by_task"
PLOTS = OUT / "plots"

if str(RUNSCRIPTS) not in sys.path:
    sys.path.insert(0, str(RUNSCRIPTS))

import unified_19descriptor_bs_filtered_elstfix_pipeline as core


TARGETS = [
    "target_fodft_coupling_abs_ev",
    "target_interaction_energy_kcal_mol",
    "target_frontier_gap_ev",
    "target_somo_somo_elst_corrected_signed_kcal_mol",
]
FAMILIES = ["extra_trees", "mlp"]
TARGET_LABELS = {
    "target_fodft_coupling_abs_ev": "|t_FO|",
    "target_interaction_energy_kcal_mol": "Interaction energy",
    "target_frontier_gap_ev": "Spin-conserving frontier gap",
    "target_somo_somo_elst_corrected_signed_kcal_mol": "Corrected SOMO–SOMO energy",
}
FAMILY_LABELS = {"extra_trees": "Extra Trees", "mlp": "Descriptor-based MLP"}
COLORS = {"extra_trees": "#356FA8", "mlp": "#D55E00"}


def selected_real_metrics(family: str, target: str) -> pd.Series:
    path = (
        SOURCE
        / "train_validate_test"
        / family
        / f"{family}_train_validate_test_metrics.csv"
    )
    frame = pd.read_csv(path)
    selection = frame[
        frame["target"].eq(target)
        & frame["stage"].eq("validation_selection")
    ].sort_values(["mae", "rmse"])
    selected_name = str(selection.iloc[0]["model"])
    test = frame[
        frame["target"].eq(target)
        & frame["stage"].eq("selected_test")
        & frame["model"].eq(selected_name)
    ]
    if len(test) != 1:
        raise ValueError(f"Missing real test row for {family}, {target}")
    return test.iloc[0]


def configure_font() -> None:
    font_dir = (
        ROOT
        / "Publication_Top7_FeatureImportance_BSFiltered"
        / "fonts"
    )
    for path in font_dir.glob("*.ttf"):
        font_manager.fontManager.addfont(str(path))
    available = {item.name for item in font_manager.fontManager.ttflist}
    family = "Calibri" if "Calibri" in available else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def create_plot(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    configure_font()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(12.6, 14.0), constrained_layout=True)
    panel = 0
    for row, target in enumerate(TARGETS):
        for col, family in enumerate(FAMILIES):
            ax = axes[row, col]
            values = frame[
                frame["target"].eq(target) & frame["family"].eq(family)
            ]["r2"].to_numpy()
            info = summary[
                summary["target"].eq(target) & summary["family"].eq(family)
            ].iloc[0]
            ax.hist(
                values,
                bins=30,
                color=COLORS[family],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.axvline(
                info["real_r2"],
                color="#111111",
                linestyle="--",
                linewidth=2.0,
                label=f"Real target $R^2$ = {info['real_r2']:.3f}",
            )
            ax.axvline(0.0, color="#777777", linestyle=":", linewidth=1.1)
            ax.set_title(TARGET_LABELS[target])
            ax.set_xlabel("Test-set $R^2$ after target permutation")
            ax.set_ylabel("Count")
            ax.legend(frameon=False, fontsize=9, loc="upper left")
            ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                -0.12,
                1.04,
                f"({chr(ord('a') + panel)})",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
            )
            if row == 0:
                ax.text(
                    0.5,
                    1.20,
                    FAMILY_LABELS[family],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                )
            panel += 1
    stem = "y_randomization_r2_histograms_4targets_2models"
    fig.savefig(PLOTS / f"{stem}.png", dpi=600)
    fig.savefig(PLOTS / f"{stem}.pdf")
    fig.savefig(PLOTS / f"{stem}.svg")
    plt.close(fig)

    for target in TARGETS:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
        for ax, family in zip(axes, FAMILIES):
            values = frame[
                frame["target"].eq(target) & frame["family"].eq(family)
            ]["r2"].to_numpy()
            info = summary[
                summary["target"].eq(target) & summary["family"].eq(family)
            ].iloc[0]
            ax.hist(values, bins=30, color=COLORS[family], alpha=0.82, edgecolor="white")
            ax.axvline(info["real_r2"], color="#111111", linestyle="--", linewidth=2)
            ax.axvline(0.0, color="#777777", linestyle=":", linewidth=1.1)
            ax.set_title(FAMILY_LABELS[family])
            ax.set_xlabel("Test-set $R^2$ after target permutation")
            ax.set_ylabel("Count")
            ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(TARGET_LABELS[target], fontsize=13, fontweight="bold")
        safe = target.removeprefix("target_")
        fig.savefig(PLOTS / f"{safe}_y_randomization_histogram.png", dpi=600)
        fig.savefig(PLOTS / f"{safe}_y_randomization_histogram.pdf")
        plt.close(fig)


def main() -> None:
    expected = 500
    files = sorted(RESULTS.glob("*.csv"))
    if len(files) != 8:
        raise RuntimeError(f"Expected 8 task CSV files, found {len(files)}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    counts = frame.groupby(["family", "target"]).size()
    if not bool((counts == expected).all()):
        raise RuntimeError(f"Incomplete permutation counts:\n{counts}")

    rows = []
    for target in TARGETS:
        for family in FAMILIES:
            shuffled = frame[
                frame["target"].eq(target) & frame["family"].eq(family)
            ].copy()
            real = selected_real_metrics(family, target)
            exceedances = int((shuffled["r2"] >= float(real["r2"])).sum())
            rows.append(
                {
                    "family": family,
                    "model_label": FAMILY_LABELS[family],
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "selected_model": shuffled["selected_model"].iloc[0],
                    "n_permutations": len(shuffled),
                    "real_r2": float(real["r2"]),
                    "real_mae": float(real["mae"]),
                    "real_rmse": float(real["rmse"]),
                    "mean_shuffled_r2": float(shuffled["r2"].mean()),
                    "std_shuffled_r2": float(shuffled["r2"].std(ddof=1)),
                    "median_shuffled_r2": float(shuffled["r2"].median()),
                    "best_shuffled_r2": float(shuffled["r2"].max()),
                    "worst_shuffled_r2": float(shuffled["r2"].min()),
                    "mean_shuffled_mae": float(shuffled["mae"].mean()),
                    "mean_shuffled_rmse": float(shuffled["rmse"].mean()),
                    "n_shuffled_r2_ge_real": exceedances,
                    "empirical_p_value": (exceedances + 1) / (len(shuffled) + 1),
                }
            )
    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "Y_RANDOMIZATION_ALL_500_RUNS.csv", index=False)
    summary.to_csv(OUT / "Y_RANDOMIZATION_SUMMARY.csv", index=False)

    report = [
        "# Y-Randomization Control",
        "",
        "The 19-descriptor matrix and original manuscript split were kept unchanged.",
        "For each target and model, the target vector was globally permuted 500 times.",
        "The manuscript-selected estimator was refitted on the original combined",
        "train+validation rows and evaluated on the original test rows.",
        "",
        "| Model | Target | Real R2 | Mean shuffled R2 | SD | Best shuffled R2 | Empirical p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        report.append(
            f"| {row['model_label']} | {row['target_label']} | {row['real_r2']:.4f} | "
            f"{row['mean_shuffled_r2']:.4f} | {row['std_shuffled_r2']:.4f} | "
            f"{row['best_shuffled_r2']:.4f} | {row['empirical_p_value']:.6f} |"
        )
    report.extend(
        [
            "",
            "Empirical p = (number of shuffled R2 values greater than or equal to the",
            "real-target R2 + 1) / (500 + 1). The minimum attainable value is 1/501.",
            "",
        ]
    )
    (OUT / "Y_RANDOMIZATION_REPORT.md").write_text("\n".join(report))
    create_plot(frame, summary)

    manifest = {
        "source_workflow": str(SOURCE),
        "n_permutations_per_target_model": expected,
        "n_total_runs": int(len(frame)),
        "targets": TARGETS,
        "families": FAMILIES,
        "split_seed": core.RANDOM_STATE,
        "permutation_seed": 42,
        "descriptor_matrix": "unchanged",
        "outputs": [
            "Y_RANDOMIZATION_ALL_500_RUNS.csv",
            "Y_RANDOMIZATION_SUMMARY.csv",
            "Y_RANDOMIZATION_REPORT.md",
            "plots/",
        ],
    }
    (OUT / "Y_RANDOMIZATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
