from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA = Path(
    "/Users/ram/Documents/BLA-ML/LATEST-FOLDERS/WITHOUT-BS-FILTER/"
    "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/"
    "ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv"
)
OUTDIR = Path(
    "/Users/ram/Documents/Pancake-Bond-Search/Script/"
    "outputs/manuscript_v15_review/s2_distributions"
)

SYSTEM_ORDER = [
    "phenalenyl",
    "olympicenyl",
    "fluorenyl",
    "CPBP",
    "phenalenyl_olympicenyl",
]
SYSTEM_LABELS = {
    "phenalenyl": "Phenalenyl",
    "olympicenyl": "Olympicenyl",
    "fluorenyl": "Fluorenyl",
    "CPBP": "CPBP",
    "phenalenyl_olympicenyl": "Phen-Oly",
}
PALETTE = {
    "phenalenyl": "#008C95",
    "olympicenyl": "#EF6C3A",
    "fluorenyl": "#36A269",
    "CPBP": "#173B57",
    "phenalenyl_olympicenyl": "#B84545",
}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df = df[df["system"].isin(SYSTEM_ORDER)].copy()
    df["system_label"] = df["system"].map(SYSTEM_LABELS)
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "font.family": "Arial",
            "axes.edgecolor": "#2F3542",
            "axes.linewidth": 0.8,
            "grid.color": "#E2E5E8",
            "grid.linewidth": 0.6,
            "xtick.color": "#1F2933",
            "ytick.color": "#1F2933",
            "axes.labelcolor": "#1F2933",
            "text.color": "#1F2933",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    sns.violinplot(
        data=df,
        x="system",
        y="bs_spin_s2_audit",
        order=SYSTEM_ORDER,
        palette=PALETTE,
        inner=None,
        cut=0,
        linewidth=0.8,
        saturation=0.95,
        ax=ax,
    )
    sns.boxplot(
        data=df,
        x="system",
        y="bs_spin_s2_audit",
        order=SYSTEM_ORDER,
        width=0.22,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#1F2933", "linewidth": 0.8},
        medianprops={"color": "#111827", "linewidth": 1.2},
        whiskerprops={"color": "#1F2933", "linewidth": 0.8},
        capprops={"color": "#1F2933", "linewidth": 0.8},
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="system",
        y="bs_spin_s2_audit",
        order=SYSTEM_ORDER,
        palette=PALETTE,
        size=1.35,
        alpha=0.22,
        jitter=0.24,
        linewidth=0,
        ax=ax,
    )

    ax.axhline(0.0, color="#7A7F87", linestyle=":", linewidth=0.9)
    ax.axhline(1.0, color="#007C89", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.axhline(2.0, color="#7A7F87", linestyle=":", linewidth=0.9)

    ax.text(
        0.01,
        1.0,
        r"ideal broken-symmetry diradical $\approx 1$",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8,
        color="#007C89",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.text(
        0.99,
        2.0,
        r"triplet reference $\approx 2$",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#5D6670",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    ax.set_title(
        r"Distribution of final broken-symmetry $\langle S^2\rangle$ values by dimer family",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel(r"Final $\langle S^2\rangle$ from BS calculation", fontsize=10)
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in SYSTEM_ORDER], rotation=25, ha="right")
    ax.set_ylim(-0.08, 2.12)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    summary = (
        f"n = {len(df):,}; range: "
        f"{df['bs_spin_s2_audit'].min():.2f}-{df['bs_spin_s2_audit'].max():.2f}"
    )
    ax.text(
        0.98,
        0.05,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
    )

    fig.tight_layout()

    png = OUTDIR / "bs_s2_distribution_by_system_no_filter_no_threshold.png"
    pdf = OUTDIR / "bs_s2_distribution_by_system_no_filter_no_threshold.pdf"
    csv = OUTDIR / "bs_s2_distribution_by_system_no_filter_statistics.csv"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    stats = (
        df.groupby("system")["bs_spin_s2_audit"]
        .agg(n="count", min="min", p5=lambda s: s.quantile(0.05), median="median", mean="mean", p95=lambda s: s.quantile(0.95), max="max")
        .reset_index()
    )
    stats.to_csv(csv, index=False)

    print(png)
    print(pdf)
    print(csv)


if __name__ == "__main__":
    main()
