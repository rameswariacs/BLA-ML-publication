#!/usr/bin/env python3
"""
System-mean baseline analysis for the FODFT 4-target manuscript.

WHAT THIS ANSWERS
-----------------
A model's R^2 can be high for two very different reasons:

  (a) the descriptors genuinely predict the property from geometry, or
  (b) the property is mostly set by which radical family the dimer belongs to,
      and the model is really just recognising the family.

To tell these apart, compare against a baseline that predicts the MEAN VALUE OF
THAT MOLECULAR SYSTEM and uses none of the 19 descriptors. If the baseline
already scores well, most of the model's R^2 was available without any geometry
information. If the baseline scores near zero, the predictive power is genuinely
geometric.

The baseline mean is always computed from TRAINING data only and applied to
held-out data, so it is a fair out-of-sample comparison and not a cheat.

NO MODEL IS REFITTED. Everything is computed from prediction files already
written by the manuscript pipeline, so results correspond exactly to the
published models. Requires only numpy, pandas, matplotlib -- no scikit-learn.

INPUT FILES READ (all read-only, nothing is modified)
-----------------------------------------------------
Relative to RUN = FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/

 1. extra_trees/extra_trees_cv_predictions.csv
       5-fold out-of-fold predictions, ExtRa Trees.
       Columns used: id, system, target, true, cv_prediction

 2. mlp/mlp_cv_predictions.csv
       Same for MLP.

 3. train_validate_test/extra_trees/extra_trees_train_validate_test_predictions.csv
 4. train_validate_test/mlp/mlp_train_validate_test_predictions.csv
       70/15/15 split predictions. Columns used: split, target, id, system,
       true, prediction. The 'train' rows supply the baseline means; the
       'test' rows are scored.

 5. GeometryClustered_ExternalTest_FODFT_4Targets/external_test_predictions.csv
       Test-set predictions under BOTH the random and geometry-clustered
       schemes. Columns used: scheme, model_family, target, id, system,
       true, prediction

 6. GeometryClustered_ExternalTest_FODFT_4Targets/
        geometry_cluster_assignments_and_splits.csv
       Split labels used to identify development rows for the baseline means.
       Columns used: id, system, clustered_split, random_split

 7. ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv
       Used only to supply true target values for development rows.

OUTPUT FILES WRITTEN (into baseline_analysis/ beside this script)
-----------------------------------------------------------------
    baseline_vs_model_metrics.csv     main table: baseline vs model R^2
    variance_decomposition.csv        between- vs within-system variance
    per_system_metrics.csv            model R^2 within each radical family
    fig_baseline_vs_model.png / .pdf
    fig_variance_decomposition.png / .pdf
    fig_per_system_r2.png / .pdf
    BASELINE_REPORT.md                text summary

Usage:
    python baseline_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE / "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211"
GEO = RUN / "GeometryClustered_ExternalTest_FODFT_4Targets"
OUT = HERE / "baseline_analysis"

# Schemes shown in the figures and the report. All schemes are still computed
# and written to the CSVs, so nothing is lost from the deposited data; this only
# controls what is displayed. The geometry-clustered partition is excluded from
# the SI presentation, and the "random" partition duplicates the 70/15/15 split.
DISPLAY_SCHEMES = ["five_fold_cv", "train_test_70_15_15"]

# Partition whose split defines the family means and variance decomposition
# quoted in the SI text.
REFERENCE_SCHEME = "train_test_70_15_15"

# Order follows Table 2 of the manuscript.
TARGETS = [
    "target_fodft_coupling_abs_ev",
    "target_frontier_gap_ev",
    "target_interaction_energy_kcal_mol",
    "target_somo_somo_elst_corrected_signed_kcal_mol",
]
LABEL = {
    "target_fodft_coupling_abs_ev": r"$|t_{FO}|$",
    "target_interaction_energy_kcal_mol": r"$\Delta E_{int}$",
    "target_frontier_gap_ev": r"$\Delta E_{frontier}$",
    "target_somo_somo_elst_corrected_signed_kcal_mol": r"$\Delta E^{corr}_{SOMO-SOMO}$",
}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def r2(y_true, y_pred) -> float:
    """R^2 against the mean of y_true. Same convention as sklearn.r2_score."""
    y = np.asarray(y_true, float)
    f = np.asarray(y_pred, float)
    ss_res = ((y - f) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def mae(y_true, y_pred) -> float:
    return float(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)).mean())


def within_system_r2(frame: pd.DataFrame) -> float:
    """
    R^2 after removing each system's own mean from both truth and prediction.

    This asks: once you already know which radical family the dimer belongs to,
    how much of the REMAINING variation does the model explain? A model that
    only recognises the family scores ~0 here regardless of its pooled R^2.
    """
    y_parts, f_parts = [], []
    for _, sub in frame.groupby("system"):
        mean = sub["true"].mean()
        y_parts.append(sub["true"].to_numpy() - mean)
        f_parts.append(sub["pred"].to_numpy() - mean)
    y = np.concatenate(y_parts)
    f = np.concatenate(f_parts)
    return float(1.0 - ((y - f) ** 2).sum() / (y ** 2).sum())


def variance_split(frame: pd.DataFrame) -> float:
    """Fraction of the total variance of the TRUE values lying between systems."""
    y = frame["true"].to_numpy(float)
    grand = y.mean()
    total = ((y - grand) ** 2).sum()
    between = sum(len(s) * (s["true"].mean() - grand) ** 2 for _, s in frame.groupby("system"))
    return float(between / total)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found:\n  {path}")
    return path


def load_evaluations() -> list[dict]:
    """
    Assemble every (scheme, model_family, target) evaluation into a common shape:
        {'scheme','family','target','frame'(id,system,true,pred),'baseline'(Series)}
    'baseline' maps system -> mean of TRAINING truth for that system.
    """
    records = []

    # --- 1 & 2. five-fold cross-validation, out-of-fold predictions ----------
    # Fold membership is not stored, so the honest baseline analogue is a
    # leave-one-out system mean: for each structure, the mean of its system
    # EXCLUDING that structure. This never lets a structure inform its own
    # baseline prediction.
    for family, path in [
        ("extra_trees", RUN / "extra_trees" / "extra_trees_cv_predictions.csv"),
        ("mlp", RUN / "mlp" / "mlp_cv_predictions.csv"),
    ]:
        cv = pd.read_csv(require(path))
        for target in TARGETS:
            sub = cv[cv["target"].eq(target)][["id", "system", "true", "cv_prediction"]].copy()
            sub = sub.rename(columns={"cv_prediction": "pred"})
            group_sum = sub.groupby("system")["true"].transform("sum")
            group_n = sub.groupby("system")["true"].transform("size")
            sub["baseline_pred"] = (group_sum - sub["true"]) / (group_n - 1)
            records.append({
                "scheme": "five_fold_cv", "family": family, "target": target,
                "frame": sub, "baseline_from": "leave-one-out system mean",
            })

    # --- 3 & 4. 70/15/15 train/validation/test ------------------------------
    for family, path in [
        ("extra_trees", RUN / "train_validate_test" / "extra_trees"
         / "extra_trees_train_validate_test_predictions.csv"),
        ("mlp", RUN / "train_validate_test" / "mlp"
         / "mlp_train_validate_test_predictions.csv"),
    ]:
        tvt = pd.read_csv(require(path))
        for target in TARGETS:
            block = tvt[tvt["target"].eq(target)]
            train_means = block[block["split"].eq("train")].groupby("system")["true"].mean()
            test = block[block["split"].eq("test")][["id", "system", "true", "prediction"]].copy()
            test = test.rename(columns={"prediction": "pred"})
            test["baseline_pred"] = test["system"].map(train_means)
            records.append({
                "scheme": "train_test_70_15_15", "family": family, "target": target,
                "frame": test, "baseline_from": "training-split system mean",
            })

    # --- 5 & 6. random and geometry-clustered external test -----------------
    ext = pd.read_csv(require(GEO / "external_test_predictions.csv"))
    splits = pd.read_csv(require(GEO / "geometry_cluster_assignments_and_splits.csv"))
    data = pd.read_csv(require(RUN / "ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv"))
    truth = data[["id", "system"] + TARGETS].merge(
        splits[["id", "clustered_split", "random_split"]], on="id", validate="one_to_one"
    )

    for scheme, split_col in [("random", "random_split"),
                              ("geometry_clustered", "clustered_split")]:
        dev = truth[truth[split_col].isin(["train", "validation"])]
        for family in ["extra_trees", "mlp"]:
            for target in TARGETS:
                dev_means = dev.groupby("system")[target].mean()
                block = ext[(ext["scheme"].eq(scheme))
                            & (ext["model_family"].eq(family))
                            & (ext["target"].eq(target))]
                frame = block[["id", "system", "true", "prediction"]].copy()
                frame = frame.rename(columns={"prediction": "pred"})
                frame["baseline_pred"] = frame["system"].map(dev_means)
                records.append({
                    "scheme": scheme, "family": family, "target": target,
                    "frame": frame, "baseline_from": "development-split system mean",
                })

    return records


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyse(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main, variance, per_system = [], [], []

    for rec in records:
        f = rec["frame"].dropna(subset=["true", "pred", "baseline_pred"])
        if f.empty:
            continue
        model_r2 = r2(f["true"], f["pred"])
        base_r2 = r2(f["true"], f["baseline_pred"])
        main.append({
            "scheme": rec["scheme"], "model_family": rec["family"], "target": rec["target"],
            "n": len(f),
            "model_r2": model_r2,
            "system_mean_baseline_r2": base_r2,
            "improvement_over_baseline": model_r2 - base_r2,
            "within_system_r2": within_system_r2(f),
            "model_mae": mae(f["true"], f["pred"]),
            "baseline_mae": mae(f["true"], f["baseline_pred"]),
            "baseline_definition": rec["baseline_from"],
        })

        if rec["family"] == "extra_trees":
            between = variance_split(f)
            variance.append({
                "scheme": rec["scheme"], "target": rec["target"], "n": len(f),
                "between_system_variance_fraction": between,
                "within_system_variance_fraction": 1.0 - between,
            })

        for system, sub in f.groupby("system"):
            if len(sub) < 5:
                continue
            per_system.append({
                "scheme": rec["scheme"], "model_family": rec["family"],
                "target": rec["target"], "system": system, "n": len(sub),
                "model_r2": r2(sub["true"], sub["pred"]),
                "model_mae": mae(sub["true"], sub["pred"]),
            })

    return (pd.DataFrame(main).sort_values(["scheme", "model_family", "target"]),
            pd.DataFrame(variance).sort_values(["scheme", "target"]),
            pd.DataFrame(per_system).sort_values(["scheme", "model_family", "target", "system"]))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_baseline_vs_model(main: pd.DataFrame) -> None:
    schemes = [s for s in DISPLAY_SCHEMES if s in set(main["scheme"])]
    fig, axes = plt.subplots(1, len(schemes), figsize=(4.2 * len(schemes), 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, scheme in zip(axes, schemes):
        et = (main[(main["scheme"].eq(scheme)) & (main["model_family"].eq("extra_trees"))]
              .set_index("target").reindex(TARGETS))
        ml = (main[(main["scheme"].eq(scheme)) & (main["model_family"].eq("mlp"))]
              .set_index("target").reindex(TARGETS))
        keep = et["model_r2"].notna()
        et, ml = et[keep], ml[keep]
        x = np.arange(len(et))
        # Baseline is model-independent by construction, so it is drawn once.
        ax.bar(x - 0.27, et["system_mean_baseline_r2"], 0.27,
               label="System-mean baseline\n(no descriptors)", color="#c0c0c0", edgecolor="black", lw=0.6)
        ax.bar(x, et["model_r2"], 0.27,
               label="ExtRa Trees", color="#2b6cb0", edgecolor="black", lw=0.6)
        ax.bar(x + 0.27, ml["model_r2"], 0.27,
               label="MLP", color="#68a357", edgecolor="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL.get(t, t) for t in et.index], rotation=30, ha="right")
        ax.set_title(scheme.replace("_", " "), fontsize=11)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(r"$R^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.suptitle("Model vs. system-mean baseline: how much comes from geometry?", fontsize=12)
    save(fig, "fig_baseline_vs_model")


def fig_variance(variance: pd.DataFrame) -> None:
    block = variance[variance["scheme"].eq(REFERENCE_SCHEME)]
    if block.empty:
        block = variance[variance["scheme"].eq(variance["scheme"].iloc[0])]
    block = block.set_index("target").reindex(TARGETS).dropna(subset=["between_system_variance_fraction"])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(block))
    b = block["between_system_variance_fraction"].to_numpy()
    ax.bar(x, b, 0.55, label="Between radical families", color="#e07b39", edgecolor="black", lw=0.6)
    ax.bar(x, 1 - b, 0.55, bottom=b, label="Within a family (geometry-driven)",
           color="#2b6cb0", edgecolor="black", lw=0.6)
    for xi, bi in zip(x, b):
        ax.text(xi, bi / 2, f"{100*bi:.0f}%", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(t, t) for t in block.index], rotation=20, ha="right")
    ax.set_ylabel("Fraction of total variance")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title("Where the variation in each target actually lives")
    save(fig, "fig_variance_decomposition")


def fig_per_system(per_system: pd.DataFrame) -> None:
    block = per_system[(per_system["scheme"].eq(REFERENCE_SCHEME))
                       & (per_system["model_family"].eq("extra_trees"))]
    if block.empty:
        return
    pivot = block.pivot_table(index="system", columns="target", values="model_r2")
    pivot = pivot.reindex(columns=[t for t in TARGETS if t in pivot.columns])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlBu", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([LABEL.get(c, c) for c in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if v > 0.2 else "white")
    fig.colorbar(im, ax=ax, label=r"$R^2$ within system")
    ax.set_title("Per-family model $R^2$")
    save(fig, "fig_per_system_r2")


# ---------------------------------------------------------------------------

def write_report(main: pd.DataFrame, variance: pd.DataFrame) -> None:
    lines = ["# System-mean baseline analysis", ""]
    lines.append("The baseline predicts each structure's **molecular-system mean** and uses")
    lines.append("none of the 19 geometric descriptors. Means are taken from training data only.")
    lines.append("")
    lines.append("The baseline is identical for both model families because it never uses a model.")
    lines.append("")
    for scheme in DISPLAY_SCHEMES:
        et = (main[(main["scheme"].eq(scheme)) & (main["model_family"].eq("extra_trees"))]
              .set_index("target").reindex(TARGETS))
        ml = (main[(main["scheme"].eq(scheme)) & (main["model_family"].eq("mlp"))]
              .set_index("target").reindex(TARGETS))
        if et["model_r2"].isna().all():
            continue
        lines += [f"## {scheme.replace('_', ' ')}", "",
                  "| Target | Baseline R² | ExtRa Trees R² | ET gain | MLP R² | MLP gain |",
                  "|---|---|---|---|---|---|"]
        for target in TARGETS:
            if target not in et.index or pd.isna(et.loc[target, "model_r2"]):
                continue
            e, m = et.loc[target], ml.loc[target]
            lines.append(f"| {target.replace('target_','')} | {e['system_mean_baseline_r2']:.3f} "
                         f"| {e['model_r2']:.3f} | {e['improvement_over_baseline']:+.3f} "
                         f"| {m['model_r2']:.3f} | {m['improvement_over_baseline']:+.3f} |")
        lines.append("")
    lines += ["## How to read this", "",
              "A **low baseline** means the property is not determined by which radical family",
              "the dimer belongs to, so the model's accuracy comes from packing geometry.",
              "A **high baseline** means most of the model's R² was available without using any",
              "descriptor, and the remaining gain is the true geometric contribution.", ""]
    (OUT / "BASELINE_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Reading prediction files (no models are refitted)...")
    records = load_evaluations()
    print(f"  loaded {len(records)} scheme x model x target evaluations")

    main_df, variance_df, per_system_df = analyse(records)
    main_df.to_csv(OUT / "baseline_vs_model_metrics.csv", index=False)
    variance_df.to_csv(OUT / "variance_decomposition.csv", index=False)
    per_system_df.to_csv(OUT / "per_system_metrics.csv", index=False)

    fig_baseline_vs_model(main_df)
    fig_variance(variance_df)
    fig_per_system(per_system_df)
    write_report(main_df, variance_df)

    print(f"\n--- ExtRa Trees, {REFERENCE_SCHEME} ---")
    show = main_df[(main_df["scheme"].eq(REFERENCE_SCHEME))
                   & (main_df["model_family"].eq("extra_trees"))]
    for _, r in show.iterrows():
        print(f"  {r['target'].replace('target_',''):<46} "
              f"baseline {r['system_mean_baseline_r2']:>6.3f} | "
              f"model {r['model_r2']:>6.3f} | within-system {r['within_system_r2']:>6.3f}")
    print(f"\nAll outputs written to: {OUT}")


if __name__ == "__main__":
    main()
