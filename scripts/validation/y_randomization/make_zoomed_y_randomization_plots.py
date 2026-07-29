from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
ALL = ROOT / "Y_RANDOMIZATION_ALL_500_RUNS.csv"
SUMMARY = ROOT / "Y_RANDOMIZATION_SUMMARY.csv"
OUT = ROOT / "plots_zoomed_shuffled_r2"


TARGET_ORDER = [
    "target_fodft_coupling_abs_ev",
    "target_interaction_energy_kcal_mol",
    "target_frontier_gap_ev",
    "target_somo_somo_elst_corrected_signed_kcal_mol",
]

TARGET_LABELS = {
    "target_fodft_coupling_abs_ev": "|t_FO|",
    "target_interaction_energy_kcal_mol": "Interaction energy",
    "target_frontier_gap_ev": "Spin-conserving frontier gap",
    "target_somo_somo_elst_corrected_signed_kcal_mol": "Corrected SOMO-SOMO energy",
}

MODEL_LABELS = {
    "extra_trees": "Extra Trees",
    "mlp": "Descriptor-based MLP",
}

COLORS = {
    "extra_trees": "#4C78A8",
    "mlp": "#F58518",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs = pd.read_csv(ALL)
    summary = pd.read_csv(SUMMARY)

    fig, axes = plt.subplots(4, 2, figsize=(8.2, 9.6), sharex=True)
    bins = 28
    xlim = (-0.30, 0.08)

    for row, target in enumerate(TARGET_ORDER):
        for col, family in enumerate(["extra_trees", "mlp"]):
            ax = axes[row, col]
            data = runs[(runs["target"] == target) & (runs["family"] == family)]["r2"].astype(float)
            info = summary[(summary["target"] == target) & (summary["family"] == family)].iloc[0]

            ax.hist(
                data,
                bins=bins,
                range=xlim,
                color=COLORS[family],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.6,
            )
            ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
            ax.axvline(float(info["best_shuffled_r2"]), color="#222222", linestyle=":", linewidth=1.2)
            ax.set_xlim(*xlim)
            ax.grid(axis="y", color="#E6E8EB", linewidth=0.7)
            ax.tick_params(axis="both", labelsize=8)
            ax.text(
                0.98,
                0.90,
                f"real R$^2$ = {float(info['real_r2']):.3f}\n"
                f"best shuffled = {float(info['best_shuffled_r2']):.3f}\n"
                f"p = {float(info['empirical_p_value']):.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "#C9CDD3", "boxstyle": "round,pad=0.25"},
            )
            if row == 0:
                ax.set_title(MODEL_LABELS[family], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{TARGET_LABELS[target]}\nCount", fontsize=8.5)
            else:
                ax.set_ylabel("Count", fontsize=8.5)
            if row == len(TARGET_ORDER) - 1:
                ax.set_xlabel("Test-set R$^2$ after target shuffling", fontsize=9)

    fig.suptitle(
        "Y-randomization control: shuffled-target models remain near zero or negative R$^2$",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Histograms are zoomed to the shuffled-target distributions. Real-target R$^2$ values lie far to the right of this range and are annotated in each panel.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"y_randomization_zoomed_shuffled_r2_4targets_2models.{ext}", dpi=500)
    plt.close(fig)

    # Individual target plots: one panel with both models overlaid.
    for target in TARGET_ORDER:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        for family in ["extra_trees", "mlp"]:
            data = runs[(runs["target"] == target) & (runs["family"] == family)]["r2"].astype(float)
            info = summary[(summary["target"] == target) & (summary["family"] == family)].iloc[0]
            ax.hist(
                data,
                bins=26,
                range=xlim,
                color=COLORS[family],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.5,
                label=f"{MODEL_LABELS[family]} shuffled; real R$^2$={float(info['real_r2']):.3f}",
            )
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.0, label="R$^2$ = 0")
        ax.set_xlim(*xlim)
        ax.set_xlabel("Test-set R$^2$ after target shuffling")
        ax.set_ylabel("Count")
        ax.set_title(TARGET_LABELS[target], fontsize=10, fontweight="bold")
        ax.grid(axis="y", color="#E6E8EB", linewidth=0.7)
        ax.legend(frameon=False, fontsize=7.2, loc="upper left")
        fig.tight_layout()
        safe = target.replace("target_", "")
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(OUT / f"{safe}_zoomed_shuffled_r2_histogram.{ext}", dpi=500)
        plt.close(fig)

    print(f"Wrote zoomed Y-randomization plots to {OUT.resolve()}")


if __name__ == "__main__":
    main()
