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
    "outputs/manuscript_v15_review/target_distributions"
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

TARGETS = [
    (
        "target_fodft_coupling_abs_ev",
        r"$|t_{\mathrm{FO}}|$",
        "eV",
        "Fragment-orbital SOMO coupling",
    ),
    (
        "target_frontier_gap_ev",
        r"$\Delta E_{\mathrm{frontier}}$",
        "eV",
        "Spin-conserving frontier gap",
    ),
    (
        "target_interaction_energy_kcal_mol",
        r"$\Delta E_{\mathrm{int}}$",
        r"kcal mol$^{-1}$",
        "Dimer interaction energy",
    ),
    (
        "target_somo_somo_elst_corrected_signed_kcal_mol",
        r"$\Delta E_{\mathrm{SOMO-SOMO}}^{\mathrm{corr}}$",
        r"kcal mol$^{-1}$",
        "Corrected SOMO-SOMO energy",
    ),
]


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

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (col, symbol, unit, title) in zip(axes, TARGETS):
        stats = df[col].dropna()
        p5 = stats.quantile(0.05)
        p50 = stats.median()
        p95 = stats.quantile(0.95)

        sns.violinplot(
            data=df,
            x="system",
            y=col,
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
            y=col,
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
            y=col,
            order=SYSTEM_ORDER,
            palette=PALETTE,
            size=1.15,
            alpha=0.18,
            jitter=0.22,
            linewidth=0,
            ax=ax,
        )

        ax.axhline(p50, color="#111827", linestyle="-", linewidth=0.8, alpha=0.55)
        ax.axhline(p5, color="#6B7280", linestyle="--", linewidth=0.7, alpha=0.65)
        ax.axhline(p95, color="#6B7280", linestyle="--", linewidth=0.7, alpha=0.65)

        ax.set_title(f"{title}\n{symbol}", fontsize=10, fontweight="bold", pad=7)
        ax.set_xlabel("")
        ax.set_ylabel(unit, fontsize=9)
        ax.set_xticklabels([SYSTEM_LABELS[s] for s in SYSTEM_ORDER], rotation=28, ha="right")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        label = f"central 90%: {p5:.3g} to {p95:.3g}\nmedian: {p50:.3g}"
        ax.text(
            0.02,
            0.97,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "linewidth": 0.5,
                "alpha": 0.88,
            },
        )

    fig.suptitle(
        "Distribution of DFT-derived target properties across 2,582 dimer geometries",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    png = OUTDIR / "four_target_property_distributions_by_system.png"
    pdf = OUTDIR / "four_target_property_distributions_by_system.pdf"
    csv = OUTDIR / "four_target_property_distribution_statistics.csv"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    rows = []
    for col, symbol, unit, title in TARGETS:
        s = df[col].dropna()
        rows.append(
            {
                "target": title,
                "script_column": col,
                "n": len(s),
                "p5": s.quantile(0.05),
                "median": s.median(),
                "mean": s.mean(),
                "p95": s.quantile(0.95),
                "unit": unit.replace("$", ""),
            }
        )
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(png)
    print(pdf)
    print(csv)


if __name__ == "__main__":
    main()
