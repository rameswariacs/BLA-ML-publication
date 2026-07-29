#!/usr/bin/env python3
"""Publication-oriented numerical and chemical QC for FO-DFT SOMO couplings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HARTREE_TO_EV = 27.211386245988
SYSTEM_ORDER = ["phenalenyl", "olympicenyl", "fluorenyl", "CPBP", "phenalenyl_olympicenyl"]
SYSTEM_LABELS = {
    "phenalenyl": "Phenalenyl",
    "olympicenyl": "Olympicenyl",
    "fluorenyl": "Fluorenyl",
    "CPBP": "CPBP",
    "phenalenyl_olympicenyl": "Phenalenyl-olympicenyl",
}
COLORS = {
    "phenalenyl": "#0072B2",
    "olympicenyl": "#E69F00",
    "fluorenyl": "#009E73",
    "CPBP": "#7A7A7A",
    "phenalenyl_olympicenyl": "#CC79A7",
}
GEOMETRY = {
    "stack_interplane_distance": ("Interplane distance", r"$d_\mathrm{plane}$ ($\mathrm{\AA}$)"),
    "stack_lateral_slip": ("Lateral displacement", r"$d_\mathrm{slip}$ ($\mathrm{\AA}$)"),
    "projected_pi_overlap_area": ("Projected pi-overlap area", r"$A_{\pi,\mathrm{ov}}$ ($\mathrm{\AA}^2$)"),
    "projected_pi_overlap_fraction_min": ("Minimum overlap fraction", r"$f_{\pi,\mathrm{ov}}^{\min}$"),
    "n_interfragment_c_contacts_3p4": ("Short C...C contacts", r"$N_\mathrm{C\cdots C}^{3.4}$"),
    "contact_min_cc_distance": ("Shortest C...C distance", r"$d_\mathrm{C\cdots C}^{\min}$ ($\mathrm{\AA}$)"),
}


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def normalized_id(df: pd.DataFrame) -> pd.Series:
    if {"system", "orca_index"}.issubset(df.columns):
        return df["system"].astype(str) + "_orca_" + df["orca_index"].astype(int).astype(str)
    return df["id"].astype(str)


def lowdin_diagnostics(row: pd.Series) -> tuple[float, float, float, float, float]:
    s = float(row["raw_fragment_somo_overlap_s_ab"])
    overlap = np.array([[1.0, s], [s, 1.0]])
    hamiltonian = np.array(
        [
            [float(row["h_aa_hartree"]), float(row["h_ab_hartree"])],
            [float(row["h_ab_hartree"]), float(row["h_bb_hartree"])],
        ]
    )
    values, vectors = np.linalg.eigh(overlap)
    if values.min() <= 0.0:
        return np.nan, np.nan, np.nan, np.nan, float(values.min())
    inv_sqrt = vectors @ np.diag(values ** -0.5) @ vectors.T
    h_orth = inv_sqrt @ hamiltonian @ inv_sqrt
    eigenvalues = np.linalg.eigvalsh(h_orth)
    return (
        abs(h_orth[0, 1]) * HARTREE_TO_EV,
        (eigenvalues[1] - eigenvalues[0]) * HARTREE_TO_EV,
        h_orth[0, 0] * HARTREE_TO_EV,
        h_orth[1, 1] * HARTREE_TO_EV,
        float(values.min()),
    )


def correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("All systems", data)] + [
        (SYSTEM_LABELS[system], data[data["system"] == system]) for system in SYSTEM_ORDER
    ]
    for label, group in groups:
        for column, (descriptor, _) in GEOMETRY.items():
            subset = group[[column, "target_fodft_coupling_abs_ev"]].dropna()
            rho = subset.corr(method="spearman").iloc[0, 1] if len(subset) >= 3 else np.nan
            rows.append({"system": label, "descriptor": descriptor, "script_column": column, "n": len(subset), "spearman_rho": rho})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fodft-csv", type=Path, required=True)
    parser.add_argument("--descriptor-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    plots = args.out_dir / "plots"
    tables = args.out_dir / "tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
        }
    )

    data = pd.read_csv(args.fodft_csv)
    data["merge_id"] = normalized_id(data)
    diagnostics = np.array([lowdin_diagnostics(row) for _, row in data.iterrows()])
    data[
        [
            "tfo_reconstructed_ev",
            "splitting_reconstructed_ev",
            "orthogonalized_site_a_ev",
            "orthogonalized_site_b_ev",
            "overlap_matrix_min_eigenvalue",
        ]
    ] = diagnostics
    data["tfo_reconstruction_error_ev"] = (
        data["target_fodft_coupling_abs_ev"] - data["tfo_reconstructed_ev"]
    ).abs()
    data["splitting_reconstruction_error_ev"] = (
        data["generalized_two_state_splitting_ev"] - data["splitting_reconstructed_ev"]
    ).abs()
    data["overlap_denominator"] = 1.0 - data["raw_fragment_somo_overlap_s_ab"] ** 2
    data["max_norm_deviation"] = np.maximum(
        (data["fragment_somo_a_norm"] - 1.0).abs(), (data["fragment_somo_b_norm"] - 1.0).abs()
    )
    expected_order = np.where(
        data["system"].eq("phenalenyl_olympicenyl"), "B_then_A", "A_then_B"
    )
    order_ok = data.get("projection_fragment_order", pd.Series("MISSING", index=data.index)).eq(expected_order)
    data["flag_orbital_norm"] = data["max_norm_deviation"] > 1.0e-6
    data["flag_overlap_conditioning"] = data["overlap_denominator"] < 0.10
    data["flag_fock_diagonal"] = data[["h_aa_hartree", "h_bb_hartree"]].abs().max(axis=1) > 2.0
    data["flag_reconstruction"] = (
        (data["tfo_reconstruction_error_ev"] > 1.0e-6)
        | (data["splitting_reconstruction_error_ev"] > 1.0e-6)
    )
    data["flag_projection_order"] = ~order_ok
    data["review_high_tfo_gt_5ev"] = data["target_fodft_coupling_abs_ev"] > 5.0
    hard_flags = [
        "flag_orbital_norm",
        "flag_overlap_conditioning",
        "flag_fock_diagonal",
        "flag_reconstruction",
        "flag_projection_order",
    ]
    data["numerical_qc_status"] = np.where(data[hard_flags].any(axis=1), "FLAG", "PASS")
    data.to_csv(tables / "fodft_numerical_diagnostics_all_structures.csv", index=False)
    data.loc[data[hard_flags + ["review_high_tfo_gt_5ev"]].any(axis=1)].to_csv(
        tables / "fodft_flagged_or_review_structures.csv", index=False
    )

    metrics = {
        "target_fodft_coupling_abs_ev": "|t_FO| (eV)",
        "raw_fragment_somo_overlap_s_ab": "S_AB",
        "site_energy_mismatch_abs_ev": "Site-energy mismatch (eV)",
        "generalized_two_state_splitting_ev": "Two-state splitting (eV)",
    }
    summary_rows = []
    for system in SYSTEM_ORDER:
        group = data[data["system"] == system]
        for column, label in metrics.items():
            values = group[column].dropna()
            summary_rows.append(
                {
                    "system": SYSTEM_LABELS[system],
                    "quantity": label,
                    "n": len(values),
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "minimum": values.min(),
                    "q05": values.quantile(0.05),
                    "median": values.median(),
                    "q95": values.quantile(0.95),
                    "maximum": values.max(),
                }
            )
    pd.DataFrame(summary_rows).to_csv(tables / "fodft_summary_by_system.csv", index=False)

    extremes = []
    for system in SYSTEM_ORDER:
        group = data[data["system"] == system]
        for direction, selected in [("lowest", group.nsmallest(10, "target_fodft_coupling_abs_ev")), ("highest", group.nlargest(10, "target_fodft_coupling_abs_ev"))]:
            part = selected.copy()
            part.insert(0, "extreme_group", direction)
            extremes.append(part)
    pd.concat(extremes, ignore_index=True).to_csv(tables / "fodft_extreme_structures_top_bottom_10.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    values = [data.loc[data.system == system, "target_fodft_coupling_abs_ev"].dropna() for system in SYSTEM_ORDER]
    box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.62)
    for patch, system in zip(box["boxes"], SYSTEM_ORDER):
        patch.set_facecolor(COLORS[system])
        patch.set_alpha(0.75)
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in SYSTEM_ORDER], rotation=18, ha="right")
    ax.set_ylabel(r"$|t_{\mathrm{FO}}|$ (eV)")
    ax.set_title("Fragment-orbital SOMO coupling by dimer family")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, plots / "01_tfo_distribution_by_system")

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for system in SYSTEM_ORDER:
        group = data[data.system == system]
        ax.scatter(
            group.raw_fragment_somo_overlap_s_ab.abs(),
            group.target_fodft_coupling_abs_ev,
            s=12,
            alpha=0.50,
            color=COLORS[system],
            edgecolors="none",
            label=SYSTEM_LABELS[system],
        )
    ax.set_xlabel(r"Fragment SOMO overlap, $|S_{AB}|$")
    ax.set_ylabel(r"$|t_{\mathrm{FO}}|$ (eV)")
    ax.set_title("Coupling and fragment-orbital overlap")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.22)
    save_figure(fig, plots / "02_tfo_vs_fragment_somo_overlap")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    axes[0].scatter(data.target_fodft_coupling_abs_ev, data.tfo_reconstructed_ev, s=10, alpha=0.45, color="#0072B2")
    axes[1].scatter(data.generalized_two_state_splitting_ev, data.splitting_reconstructed_ev, s=10, alpha=0.45, color="#009E73")
    for ax, label in zip(axes, [r"$|t_{\mathrm{FO}}|$ (eV)", "Two-state splitting (eV)"]):
        bounds = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(bounds, bounds, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_xlabel("Stored value")
        ax.set_ylabel("Reconstructed value")
        ax.set_title(label)
        ax.grid(alpha=0.22)
    fig.suptitle("Internal consistency of the two-orbital projection", y=1.02)
    save_figure(fig, plots / "03_two_state_internal_consistency")

    if args.descriptor_csv and args.descriptor_csv.exists():
        descriptors = pd.read_csv(args.descriptor_csv)
        descriptors["merge_id"] = normalized_id(descriptors)
        keep = ["merge_id", "bs_s2_filter_status"] + list(GEOMETRY)
        keep = [column for column in keep if column in descriptors.columns]
        merged = data.merge(descriptors[keep], on="merge_id", how="inner", validate="one_to_one")
        merged.to_csv(tables / "fodft_qc_with_geometry_retained_structures.csv", index=False)
        correlations = correlation_table(merged)
        correlations.to_csv(tables / "fodft_geometry_spearman_correlations.csv", index=False)

        fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.2), constrained_layout=True)
        for ax, (column, (title, xlabel)) in zip(axes.flat, GEOMETRY.items()):
            for system in SYSTEM_ORDER:
                group = merged[merged.system == system]
                ax.scatter(
                    group[column], group.target_fodft_coupling_abs_ev,
                    s=9, alpha=0.35, color=COLORS[system], edgecolors="none",
                )
            rho = merged[[column, "target_fodft_coupling_abs_ev"]].corr(method="spearman").iloc[0, 1]
            ax.set_title(f"{title}  ($\\rho_s$ = {rho:.2f})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$|t_{\mathrm{FO}}|$ (eV)")
            ax.grid(alpha=0.20)
        save_figure(fig, plots / "04_tfo_geometry_relationships_retained_structures")

    qc = {
        "n_total": int(len(data)),
        "n_numerical_pass": int(data.numerical_qc_status.eq("PASS").sum()),
        "n_numerical_flag": int(data.numerical_qc_status.eq("FLAG").sum()),
        "n_high_tfo_review_gt_5ev": int(data.review_high_tfo_gt_5ev.sum()),
        "maximum_norm_deviation": float(data.max_norm_deviation.max()),
        "minimum_overlap_denominator": float(data.overlap_denominator.min()),
        "maximum_tfo_reconstruction_error_ev": float(data.tfo_reconstruction_error_ev.max()),
        "maximum_splitting_reconstruction_error_ev": float(data.splitting_reconstruction_error_ev.max()),
        "thresholds": {
            "orbital_norm_deviation": 1.0e-6,
            "minimum_1_minus_s_squared": 0.10,
            "maximum_abs_fock_diagonal_hartree": 2.0,
            "maximum_reconstruction_error_ev": 1.0e-6,
            "high_tfo_manual_review_ev_not_exclusion": 5.0,
        },
    }
    (args.out_dir / "fodft_qc_summary.json").write_text(json.dumps(qc, indent=2) + "\n")
    (args.out_dir / "SI_FODFT_QC_README.md").write_text(
        "# FO-DFT coupling quality control\n\n"
        "Numerical validity was assessed from fragment-orbital normalization, overlap-matrix conditioning, "
        "the magnitudes of the projected diagonal Fock elements, the recorded fragment order, and independent "
        "reconstruction of both the Loewdin-corrected coupling and generalized two-state splitting. A coupling "
        "above 5 eV was marked for geometric inspection but was not automatically excluded. Geometry-coupling "
        "relationships were evaluated for the BS-filtered structures using Spearman rank correlations.\n\n"
        "Suggested SI figures: `01_tfo_distribution_by_system`, `02_tfo_vs_fragment_somo_overlap`, "
        "`03_two_state_internal_consistency`, and `04_tfo_geometry_relationships_retained_structures`.\n"
    )
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
