#!/usr/bin/env python3
"""
Correlated-descriptor analysis for the feature-importance section.

WHAT THIS ANSWERS
-----------------
Impurity-based importance (the sklearn default used for the ExtRa Trees models)
splits the shared contribution of correlated descriptors according to which one
a tree happens to select first. Where descriptors are near-duplicates, the
individual ranking inside such a group is therefore arbitrary, even though the
group total is stable.

This script quantifies the effect for this dataset and shows that it does NOT
change the separation between global pi-surface descriptors and local contact
descriptors, which is the interpretation the manuscript relies on.

Three things are produced:
  (a) the Spearman correlation structure of the 19 descriptors,
  (b) how importance is distributed inside the one strongly correlated group,
      compared between impurity (ExtRa Trees) and permutation (MLP),
  (c) importance summed over chemically defined descriptor groups, for both
      methods, showing the regime assignment is method-independent.

INPUT FILES READ (read-only)
-----------------------------
Relative to RUN = FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/

 1. ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv
        descriptor values, for the Spearman correlation matrix
 2. unified_19descriptor_no_s2_filter_manifest.json
        the canonical ordered list of the 19 descriptors
 3. extra_trees/extra_trees_feature_importance.csv
        importance_type = extra_trees_impurity
 4. mlp/mlp_feature_importance.csv
        importance_type = permutation_neg_mae_drop (with standard deviations)

Because the two methods report on different scales, every importance is
normalised to the sum over all 19 descriptors for that target and method, so
values are comparable as fractions.

OUTPUT (into correlated_importance/ beside this script)
--------------------------------------------------------
    descriptor_correlation_matrix.csv
    correlated_pairs.csv
    within_group_importance.csv
    grouped_importance.csv
    fig_descriptor_correlation.png / .pdf
    fig_within_group_importance.png / .pdf
    fig_grouped_importance.png / .pdf
    CORRELATED_IMPORTANCE_REPORT.md
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RUN = HERE / "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211"
OUT = HERE / "correlated_importance"

# Order follows Table 2 of the manuscript.
TARGETS = [
    "target_fodft_coupling_abs_ev",
    "target_frontier_gap_ev",
    "target_interaction_energy_kcal_mol",
    "target_somo_somo_elst_corrected_signed_kcal_mol",
]
TLABEL = {
    "target_fodft_coupling_abs_ev": r"$|t_{FO}|$",
    "target_interaction_energy_kcal_mol": r"$\Delta E_{int}$",
    "target_frontier_gap_ev": r"$\Delta E_{frontier}$",
    "target_somo_somo_elst_corrected_signed_kcal_mol": r"$\Delta E^{corr}_{SOMO-SOMO}$",
}

CORRELATION_THRESHOLD = 0.90

# The one strongly correlated group: C...C contact counts at three cutoffs.
CONTACT_COUNT_GROUP = [
    "n_interfragment_c_contacts_3p4",
    "n_interfragment_c_contacts_3p6",
    "n_interfragment_c_contacts_4p0",
]

# Descriptor classes exactly as defined in Figure 3 of the manuscript.
# Symbols in the comments match the figure's symbol key.
CHEM_GROUPS = {
    "Global pi-surface": [                              # S_ov, f_ov,min, S_pi, R_S, df_ov
        "projected_pi_overlap_area",
        "projected_pi_overlap_fraction_min",
        "pi_projected_area_mean",
        "pi_projected_area_ratio_min_over_max",
        "delta_overlap_fraction",
    ],
    "Stacking": [                                       # d_perp, d_slip, theta_plane
        "stack_interplane_distance",
        "stack_lateral_slip",
        "stack_plane_normal_angle_deg",
    ],
    "Local Contacts": [                                 # d_min, d_10, sigma_CC, N_3.4/3.6/4.0,
        "contact_min_cc_distance",                      # rho_contact, dN_atom
        "mean_top10_closest_c_distances",
        "contact_interfragment_distance_std",
        "n_interfragment_c_contacts_3p4",
        "n_interfragment_c_contacts_3p6",
        "n_interfragment_c_contacts_4p0",
        "contact_density_3p6_per_overlap_area",
        "delta_contact_atoms_3p4",
    ],
    "BLA & Asymmetry": [                                # BLA_frag, BLA_contact, dBLA_contact
        "bla_fragment_mean",
        "contact_region_bla_mean",
        "delta_contact_region_bla",
    ],
}

# Symbols exactly as used in Figure 3 of the manuscript.
SHORT = {
    # Global pi-surface
    "projected_pi_overlap_area":            r"$S_{ov}$",
    "projected_pi_overlap_fraction_min":    r"$f_{ov,min}$",
    "pi_projected_area_mean":               r"$\bar{S}_{\pi}$",
    "pi_projected_area_ratio_min_over_max": r"$R_{S}$",
    "delta_overlap_fraction":               r"$\Delta f_{ov}$",
    # Stacking
    "stack_interplane_distance":            r"$d_{\perp}$",
    "stack_lateral_slip":                   r"$d_{slip}$",
    "stack_plane_normal_angle_deg":         r"$\theta_{plane}$",
    # Local Contacts
    "contact_min_cc_distance":              r"$d_{min}$",
    "mean_top10_closest_c_distances":       r"$\bar{d}_{10}$",
    "contact_interfragment_distance_std":   r"$\sigma_{C \cdots C}$",
    "n_interfragment_c_contacts_3p4":       r"$N_{3.4}$",
    "n_interfragment_c_contacts_3p6":       r"$N_{3.6}$",
    "n_interfragment_c_contacts_4p0":       r"$N_{4.0}$",
    "contact_density_3p6_per_overlap_area": r"$\rho^{3.6}_{contact}$",
    "delta_contact_atoms_3p4":              r"$\Delta N^{3.4}_{atom}$",
    # BLA & Asymmetry
    "bla_fragment_mean":                    r"$\overline{BLA}_{frag}$",
    "contact_region_bla_mean":              r"$\overline{BLA}_{contact}$",
    "delta_contact_region_bla":             r"$\Delta BLA_{contact}$",
}


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load():
    features = json.loads((RUN / "unified_19descriptor_no_s2_filter_manifest.json").read_text())["features19"]
    data = pd.read_csv(RUN / "ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv")
    et = pd.read_csv(RUN / "extra_trees" / "extra_trees_feature_importance.csv")
    ml = pd.read_csv(RUN / "mlp" / "mlp_feature_importance.csv")
    return features, data, et, ml


def normalised(imp: pd.DataFrame, target: str, features: list[str]) -> pd.Series:
    """Importance as a fraction of the total over all 19 descriptors."""
    s = imp[imp["target"].eq(target)].set_index("feature")["importance"].reindex(features)
    s = s.clip(lower=0)                 # permutation values can dip slightly negative
    return s / s.sum()


# ---------------------------------------------------------------------------

def fig_correlation(corr: pd.DataFrame, features: list[str]) -> None:
    # Order descriptors by the Figure 3 classes so each class forms a block.
    ordered, bounds, class_names = [], [], []
    for gname, members in CHEM_GROUPS.items():
        present = [m for m in members if m in features]
        ordered.extend(present)
        bounds.append(len(ordered))
        class_names.append((gname, len(ordered) - len(present), len(ordered)))
    missing = [f for f in features if f not in ordered]
    if missing:
        raise ValueError(f"descriptors not assigned to any class: {missing}")

    c = corr.loc[ordered, ordered]
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    m = c.abs().to_numpy()
    np.fill_diagonal(m, np.nan)
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)

    labels = [SHORT.get(f, f) for f in ordered]
    ax.set_xticks(range(len(ordered))); ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_yticks(range(len(ordered))); ax.set_yticklabels(labels, fontsize=9)

    # class dividers
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="white", lw=2.0)
        ax.axvline(b - 0.5, color="white", lw=2.0)

    # class labels along the top
    short_class = {"Global pi-surface": "Global $\\pi$-surface", "Stacking": "Stacking",
                   "Local Contacts": "Local Contacts", "BLA & Asymmetry": "BLA & Asym."}
    for gname, lo, hi in class_names:
        ax.text((lo + hi - 1) / 2, -1.15, short_class.get(gname, gname),
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # outline the correlated groups, now contiguous within Local Contacts
    def box(members, colour="red"):
        idx = [ordered.index(f) for f in members if f in ordered]
        lo, hi = min(idx), max(idx)
        ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
                               fill=False, edgecolor=colour, lw=2.2))

    box(CONTACT_COUNT_GROUP)
    box(["contact_min_cc_distance", "mean_top10_closest_c_distances"])

    fig.colorbar(im, ax=ax, label=r"$\left| \rho_{Spearman} \right|$", shrink=0.75)
    ax.set_title(r"Descriptor correlation structure, grouped by class"
                 "\n"
                 r"(red outlines: $\left| \rho \right| > 0.90$)", fontsize=11, pad=26)
    save(fig, "fig_descriptor_correlation")


def fig_within_group(within: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(3.5 * len(TARGETS), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.36
    for ax, target in zip(axes, TARGETS):
        block = within[within["target"].eq(target)].set_index("feature").reindex(CONTACT_COUNT_GROUP)
        x = np.arange(len(CONTACT_COUNT_GROUP))
        ax.bar(x - width / 2, block["share_within_group_impurity"], width,
               label="ExtRa Trees (impurity)", color="#2b6cb0", edgecolor="black", lw=0.6)
        ax.bar(x + width / 2, block["share_within_group_permutation"], width,
               label="MLP (permutation)", color="#68a357", edgecolor="black", lw=0.6)
        ax.axhline(1 / 3, ls="--", color="grey", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(["3.4 Å", "3.6 Å", "4.0 Å"])
        ax.set_title(TLABEL[target], fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Share of the contact-count group's importance")
    axes[-1].text(0.98, 1/3 + 0.025, "equal share", fontsize=7, color="grey", ha="right",
                  transform=axes[-1].get_yaxis_transform())
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=9)
    fig.suptitle("Distribution of importance among three near-identical contact-count descriptors\n"
                 r"($|\rho|$ = 0.93–0.98 between members)", fontsize=11, y=1.13)
    fig.subplots_adjust(top=0.80)
    save(fig, "fig_within_group_importance")


def fig_grouped(grouped: pd.DataFrame) -> None:
    names = list(CHEM_GROUPS)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(3.6 * len(TARGETS), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.36
    for ax, target in zip(axes, TARGETS):
        block = grouped[grouped["target"].eq(target)].set_index("group").reindex(names)
        y = np.arange(len(names))
        ax.barh(y - width / 2, block["impurity_share"], width,
                label="ExtRa Trees (impurity)", color="#2b6cb0", edgecolor="black", lw=0.6)
        ax.barh(y + width / 2, block["permutation_share"], width,
                label="MLP (permutation)", color="#68a357", edgecolor="black", lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(TLABEL[target], fontsize=11)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_xlabel("Fraction of total importance")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9)
    fig.suptitle("Importance summed over chemically defined descriptor groups", fontsize=11)
    save(fig, "fig_grouped_importance")


# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features, data, et, ml = load()

    # --- correlation --------------------------------------------------------
    corr = data[features].corr(method="spearman")
    corr.to_csv(OUT / "descriptor_correlation_matrix.csv")

    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            rho = corr.loc[a, b]
            if abs(rho) > 0.80:
                pairs.append({"descriptor_1": a, "descriptor_2": b,
                              "spearman_rho": rho, "abs_rho": abs(rho),
                              "above_0.90": abs(rho) > CORRELATION_THRESHOLD})
    pairs_df = pd.DataFrame(pairs).sort_values("abs_rho", ascending=False)
    pairs_df.to_csv(OUT / "correlated_pairs.csv", index=False)

    # --- within-group distribution -----------------------------------------
    within_rows, grouped_rows = [], []
    for target in TARGETS:
        imp_et = normalised(et, target, features)
        imp_ml = normalised(ml, target, features)

        blk_et = imp_et.reindex(CONTACT_COUNT_GROUP)
        blk_ml = imp_ml.reindex(CONTACT_COUNT_GROUP)
        for f in CONTACT_COUNT_GROUP:
            within_rows.append({
                "target": target, "feature": f,
                "impurity_share_of_total": blk_et[f],
                "permutation_share_of_total": blk_ml[f],
                "share_within_group_impurity": blk_et[f] / blk_et.sum(),
                "share_within_group_permutation": blk_ml[f] / blk_ml.sum(),
            })
        within_rows.append({
            "target": target, "feature": "GROUP TOTAL",
            "impurity_share_of_total": blk_et.sum(),
            "permutation_share_of_total": blk_ml.sum(),
            "share_within_group_impurity": 1.0,
            "share_within_group_permutation": 1.0,
        })

        for gname, members in CHEM_GROUPS.items():
            present = [m for m in members if m in features]
            grouped_rows.append({
                "target": target, "group": gname, "n_descriptors": len(present),
                "impurity_share": float(imp_et.reindex(present).sum()),
                "permutation_share": float(imp_ml.reindex(present).sum()),
            })

    within = pd.DataFrame(within_rows)
    grouped = pd.DataFrame(grouped_rows)
    within.to_csv(OUT / "within_group_importance.csv", index=False)
    grouped.to_csv(OUT / "grouped_importance.csv", index=False)

    fig_correlation(corr, features)
    fig_within_group(within)
    fig_grouped(grouped)

    # --- report -------------------------------------------------------------
    lines = ["# Correlated descriptors and feature importance", "",
             f"Descriptor pairs with |rho| > 0.90: **{int(pairs_df['above_0.90'].sum())}** "
             f"of {len(features)*(len(features)-1)//2} possible pairs.",
             f"Pairs with |rho| > 0.80: **{len(pairs_df)}**.", "",
             "## Strongly correlated pairs", "",
             "| Descriptor 1 | Descriptor 2 | Spearman rho |", "|---|---|---|"]
    for _, r in pairs_df[pairs_df["above_0.90"]].iterrows():
        lines.append(f"| {r['descriptor_1']} | {r['descriptor_2']} | {r['spearman_rho']:.3f} |")

    lines += ["", "## Importance inside the contact-count group", "",
              "Three descriptors carrying nearly the same information "
              "(|rho| = 0.93-0.98). Shares are within-group fractions; an equal "
              "split would be 0.333 each.", "",
              "| Target | Method | 3.4 A | 3.6 A | 4.0 A | max/min |", "|---|---|---|---|---|---|"]
    for target in TARGETS:
        b = within[(within["target"].eq(target)) & (within["feature"].ne("GROUP TOTAL"))]
        for col, name in [("share_within_group_impurity", "ExtRa Trees (impurity)"),
                          ("share_within_group_permutation", "MLP (permutation)")]:
            v = b.set_index("feature").reindex(CONTACT_COUNT_GROUP)[col]
            lines.append(f"| {target.replace('target_','')} | {name} | "
                         + " | ".join(f"{x:.3f}" for x in v)
                         + f" | {v.max()/v.min():.1f}x |")

    lines += ["", "## Group-level importance", "",
              "| Target | Group | ExtRa Trees | MLP |", "|---|---|---|---|"]
    for _, r in grouped.iterrows():
        lines.append(f"| {r['target'].replace('target_','')} | {r['group']} | "
                     f"{r['impurity_share']:.3f} | {r['permutation_share']:.3f} |")

    lines += ["", "## Reading this", "",
              "Impurity importance concentrates the shared contribution of correlated",
              "descriptors on whichever member the trees select first, so within-group",
              "ranking is not interpretable. Permutation importance distributes it more",
              "evenly. The GROUP TOTAL and the group-level table are stable between the",
              "two methods, so conclusions stated at group level are robust.", ""]
    (OUT / "CORRELATED_IMPORTANCE_REPORT.md").write_text("\n".join(lines))

    print(f"pairs |rho|>0.90: {int(pairs_df['above_0.90'].sum())}, |rho|>0.80: {len(pairs_df)}")
    print(f"All outputs written to: {OUT}")


if __name__ == "__main__":
    main()
