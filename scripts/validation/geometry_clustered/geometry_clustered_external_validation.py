#!/usr/bin/env python3
"""Geometry-clustered external validation for the four manuscript targets.

The 19-descriptor no-S2-filter target table is read without modification.
Structures are clustered separately within each represented molecular system using
only median-imputed, standardized descriptors. Entire geometry clusters are assigned
to train, validation, or test, preventing close members of one descriptor-space
region from being divided among subsets.

The fixed manuscript models are evaluated without target-specific retuning:
  * Extra Trees: 500 trees, min_samples_leaf=2
  * MLP: 19-128-64-32-1, alpha=1e-3
"""

from __future__ import annotations

import json
import math
import pickle
import shutil
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/rb1820/BLA-ML")
SOURCE = ROOT / "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211"
INPUT = SOURCE / "ALL_DIMERS_19descriptors_7targets_BS_filtered.csv"
OUT = ROOT / "/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/GeometryClustered_ExternalTest_FODFT_4Targets"
RANDOM_STATE = 42
BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 2000
TARGET_CLUSTER_SIZE = 25
N_JOBS = 16

FEATURES = [
    "bla_fragment_mean",
    "contact_region_bla_mean",
    "contact_min_cc_distance",
    "stack_interplane_distance",
    "stack_lateral_slip",
    "stack_plane_normal_angle_deg",
    "contact_interfragment_distance_std",
    "projected_pi_overlap_area",
    "projected_pi_overlap_fraction_min",
    "pi_projected_area_mean",
    "n_interfragment_c_contacts_3p4",
    "n_interfragment_c_contacts_3p6",
    "n_interfragment_c_contacts_4p0",
    "mean_top10_closest_c_distances",
    "contact_density_3p6_per_overlap_area",
    "pi_projected_area_ratio_min_over_max",
    "delta_overlap_fraction",
    "delta_contact_atoms_3p4",
    "delta_contact_region_bla",
]

TARGETS = [
    "target_fodft_coupling_abs_ev",
    "target_interaction_energy_kcal_mol",
    "target_frontier_gap_ev",
    "target_somo_somo_elst_corrected_signed_kcal_mol",
]

TARGET_LABELS = {
    "target_fodft_coupling_abs_ev": "|t_FO| (eV)",
    "target_interaction_energy_kcal_mol": "Interaction energy (kcal mol$^{-1}$)",
    "target_frontier_gap_ev": "Spin-conserving frontier gap (eV)",
    "target_somo_somo_elst_corrected_signed_kcal_mol": "Corrected SOMO-SOMO energy (kcal mol$^{-1}$)",
}

SHORT_LABELS = {
    "target_fodft_coupling_abs_ev": "|t_FO|",
    "target_interaction_energy_kcal_mol": "Interaction energy",
    "target_frontier_gap_ev": "Frontier gap",
    "target_somo_somo_elst_corrected_signed_kcal_mol": "Corrected SOMO-SOMO",
}

SYSTEM_COLORS = {
    "phenalenyl": "#0072B2",
    "olympicenyl": "#D55E00",
    "fluorenyl": "#009E73",
    "CPBP": "#CC79A7",
    "phenalenyl_olympicenyl": "#6A3D9A",
}

SPLIT_COLORS = {"train": "#4C78A8", "validation": "#F2A541", "test": "#D1495B"}


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else math.nan,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


def bootstrap_ci(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {"r2": [], "mae": [], "rmse": []}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        row = metrics(y[idx], pred[idx])
        for key in values:
            values[key].append(row[key])
    out = {}
    for key, vals in values.items():
        out[f"{key}_ci95_low"] = float(np.percentile(vals, 2.5))
        out[f"{key}_ci95_high"] = float(np.percentile(vals, 97.5))
    return out


def extra_trees() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    criterion="squared_error",
                    max_depth=None,
                    max_features=1.0,
                    bootstrap=False,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )


def mlp() -> TransformedTargetRegressor:
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("x_scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    batch_size=64,
                    learning_rate_init=1e-3,
                    max_iter=900,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    tol=1e-5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


MODELS = {"extra_trees": extra_trees, "mlp": mlp}


def model_label(family: str) -> str:
    return "Extra Trees" if family == "extra_trees" else "MLP"


def subset_closest_to_target(items: list[tuple[str, int]], target: int) -> set[str]:
    """Deterministic subset-sum selection of cluster IDs closest to target size."""
    states: dict[int, tuple[str, ...]] = {0: ()}
    for cluster_id, size in items:
        additions = {}
        for total, chosen in list(states.items()):
            new_total = total + size
            candidate = chosen + (cluster_id,)
            if new_total not in states and new_total not in additions:
                additions[new_total] = candidate
        states.update(additions)
    eligible = [(abs(total - target), len(chosen), total, chosen) for total, chosen in states.items() if chosen]
    _, _, _, best = min(eligible)
    return set(best)


def make_geometry_clusters(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    cluster_summary = []
    for system, sub in df.groupby("system", sort=True):
        sub = sub.copy()
        x = SimpleImputer(strategy="median").fit_transform(sub[FEATURES])
        x = StandardScaler().fit_transform(x)
        n_clusters = max(6, int(math.ceil(len(sub) / TARGET_CLUSTER_SIZE)))
        n_clusters = min(n_clusters, len(sub))
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(x)
        sub["cluster_within_system"] = labels
        sub["geometry_cluster"] = [f"{system}__g{value:03d}" for value in labels]
        sizes = sub["geometry_cluster"].value_counts().sort_index()

        rng = np.random.default_rng(RANDOM_STATE + sum(map(ord, str(system))))
        ids = sizes.index.to_numpy().copy()
        rng.shuffle(ids)
        items = [(cluster_id, int(sizes[cluster_id])) for cluster_id in ids]
        test_ids = subset_closest_to_target(items, round(0.15 * len(sub)))
        remaining = [(cid, size) for cid, size in items if cid not in test_ids]
        validation_ids = subset_closest_to_target(remaining, round(0.15 * len(sub)))
        sub["clustered_split"] = "train"
        sub.loc[sub["geometry_cluster"].isin(validation_ids), "clustered_split"] = "validation"
        sub.loc[sub["geometry_cluster"].isin(test_ids), "clustered_split"] = "test"
        frames.append(sub)

        for cluster_id, size in sizes.items():
            split = sub.loc[sub["geometry_cluster"] == cluster_id, "clustered_split"].iloc[0]
            cluster_summary.append(
                {
                    "system": system,
                    "geometry_cluster": cluster_id,
                    "n_structures": int(size),
                    "split": split,
                }
            )
    assigned = pd.concat(frames).sort_index()
    pd.DataFrame(cluster_summary).to_csv(OUT / "geometry_cluster_summary.csv", index=False)
    return assigned


def make_random_split(df: pd.DataFrame) -> pd.Series:
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=RANDOM_STATE, shuffle=True, stratify=df["system"]
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE + 1,
        shuffle=True,
        stratify=df.iloc[temp_idx]["system"],
    )
    out = pd.Series("train", index=df.index, dtype=object)
    out.iloc[val_idx] = "validation"
    out.iloc[test_idx] = "test"
    return out


def split_balance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme, column in [("random", "random_split"), ("geometry_clustered", "clustered_split")]:
        for (system, split), frame in df.groupby(["system", column]):
            rows.append(
                {
                    "scheme": scheme,
                    "system": system,
                    "split": split,
                    "n": len(frame),
                    "fraction_within_system": len(frame) / int((df["system"] == system).sum()),
                    "n_clusters": frame["geometry_cluster"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def descriptor_separation(df: pd.DataFrame, split_column: str, scheme: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_val = df[split_column].isin(["train", "validation"])
    test = df[split_column].eq("test")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_dev = scaler.fit_transform(imputer.fit_transform(df.loc[train_val, FEATURES]))
    x_test = scaler.transform(imputer.transform(df.loc[test, FEATURES]))

    nn = NearestNeighbors(n_neighbors=1).fit(x_dev)
    distance, neighbor = nn.kneighbors(x_test)
    distance = distance[:, 0]
    neighbor = neighbor[:, 0]

    dev_nn = NearestNeighbors(n_neighbors=2).fit(x_dev)
    dev_distance = dev_nn.kneighbors(x_dev)[0][:, 1]
    threshold = float(np.percentile(dev_distance, 5))

    dev_rows = df.loc[train_val].reset_index()
    test_rows = df.loc[test].reset_index()
    detail = pd.DataFrame(
        {
            "scheme": scheme,
            "test_id": test_rows["id"],
            "test_system": test_rows["system"],
            "nearest_development_id": dev_rows.iloc[neighbor]["id"].to_numpy(),
            "nearest_development_system": dev_rows.iloc[neighbor]["system"].to_numpy(),
            "standardized_descriptor_distance": distance,
            "below_development_5pct_nn_threshold": distance <= threshold,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "scheme": scheme,
                "n_test": len(distance),
                "mean_nearest_development_distance": float(np.mean(distance)),
                "median_nearest_development_distance": float(np.median(distance)),
                "q05_nearest_development_distance": float(np.percentile(distance, 5)),
                "q95_nearest_development_distance": float(np.percentile(distance, 95)),
                "development_internal_nn_5pct_threshold": threshold,
                "fraction_test_below_threshold": float(np.mean(distance <= threshold)),
            }
        ]
    )
    return detail, summary


def grouped_cv_predictions(
    df: pd.DataFrame, development_mask: pd.Series, target: str, model_factory
) -> np.ndarray:
    dev = df.loc[development_mask].copy()
    groups = dev["geometry_cluster"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    pred = np.full(len(dev), np.nan)
    for train_pos, valid_pos in splitter.split(dev[FEATURES], dev[target], groups):
        fitted = model_factory()
        fitted.fit(dev.iloc[train_pos][FEATURES], dev.iloc[train_pos][target])
        pred[valid_pos] = fitted.predict(dev.iloc[valid_pos][FEATURES])
    return pred


def evaluate_scheme(
    df: pd.DataFrame, split_column: str, scheme: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metric_rows = []
    prediction_rows = []
    saved_models = {}
    for family, model_factory in MODELS.items():
        saved_models[family] = {}
        for target in TARGETS:
            development = df[split_column].isin(["train", "validation"])
            train = df[split_column].eq("train")
            validation = df[split_column].eq("validation")
            test = df[split_column].eq("test")

            if scheme == "geometry_clustered":
                cv_pred = grouped_cv_predictions(df, development, target, model_factory)
                y_dev = df.loc[development, target].to_numpy()
                cv_metrics = metrics(y_dev, cv_pred)
                metric_rows.append(
                    {
                        "scheme": scheme,
                        "model_family": family,
                        "target": target,
                        "stage": "cluster_grouped_5fold_cv",
                        "n": int(development.sum()),
                        **cv_metrics,
                    }
                )

            validation_fit = model_factory()
            validation_fit.fit(df.loc[train, FEATURES], df.loc[train, target])
            validation_pred = validation_fit.predict(df.loc[validation, FEATURES])
            metric_rows.append(
                {
                    "scheme": scheme,
                    "model_family": family,
                    "target": target,
                    "stage": "validation_fixed_model",
                    "n": int(validation.sum()),
                    **metrics(df.loc[validation, target].to_numpy(), validation_pred),
                }
            )

            final = model_factory()
            final.fit(df.loc[development, FEATURES], df.loc[development, target])
            test_pred = final.predict(df.loc[test, FEATURES])
            y_test = df.loc[test, target].to_numpy()
            test_metrics = metrics(y_test, test_pred)
            metric_rows.append(
                {
                    "scheme": scheme,
                    "model_family": family,
                    "target": target,
                    "stage": "external_test",
                    "n": int(test.sum()),
                    **test_metrics,
                    **bootstrap_ci(y_test, test_pred),
                }
            )
            saved_models[family][target] = final
            for (_, row), value in zip(df.loc[test].iterrows(), test_pred):
                prediction_rows.append(
                    {
                        "scheme": scheme,
                        "model_family": family,
                        "target": target,
                        "id": row["id"],
                        "system": row["system"],
                        "dimer_class": row["dimer_class"],
                        "geometry_cluster": row["geometry_cluster"],
                        "true": float(row[target]),
                        "prediction": float(value),
                        "residual": float(row[target] - value),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), saved_models


def plot_split_pca(df: pd.DataFrame) -> None:
    imputed = SimpleImputer(strategy="median").fit_transform(df[FEATURES])
    scaled = StandardScaler().fit_transform(imputed)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    plot = df[["system", "clustered_split"]].copy()
    plot["PC1"] = coords[:, 0]
    plot["PC2"] = coords[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for system, frame in plot.groupby("system"):
        axes[0].scatter(
            frame["PC1"], frame["PC2"], s=13, alpha=0.65,
            color=SYSTEM_COLORS.get(system, "#777777"), label=system,
        )
    for split, frame in plot.groupby("clustered_split"):
        axes[1].scatter(
            frame["PC1"], frame["PC2"], s=14, alpha=0.72,
            color=SPLIT_COLORS[split], label=split,
        )
    axes[0].set_title("Represented molecular systems")
    axes[1].set_title("Geometry-clustered partition")
    for ax in axes:
        ax.set_xlabel("Descriptor PC1")
        ax.set_ylabel("Descriptor PC2")
        ax.grid(color="#E6E6E6", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plots/geometry_clustered_split_pca.png", dpi=500)
    fig.savefig(OUT / "plots/geometry_clustered_split_pca.pdf")
    plt.close(fig)


def plot_separation(detail: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.linspace(0, np.percentile(detail["standardized_descriptor_distance"], 99), 35)
    for scheme, color in [("random", "#4C78A8"), ("geometry_clustered", "#D1495B")]:
        values = detail.loc[detail["scheme"] == scheme, "standardized_descriptor_distance"]
        ax.hist(values, bins=bins, density=True, alpha=0.48, color=color, label=scheme.replace("_", " "))
        ax.axvline(values.median(), color=color, linewidth=2)
    ax.set_xlabel("Nearest development-set distance in standardized descriptor space")
    ax.set_ylabel("Density")
    ax.set_title("Structural separation of the external test set")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "plots/test_descriptor_separation.png", dpi=500)
    fig.savefig(OUT / "plots/test_descriptor_separation.pdf")
    plt.close(fig)


def plot_correlations(predictions: pd.DataFrame) -> None:
    clustered = predictions[predictions["scheme"] == "geometry_clustered"]
    for family, family_frame in clustered.groupby("model_family"):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.3))
        for ax, target in zip(axes.flat, TARGETS):
            frame = family_frame[family_frame["target"] == target]
            for system, group in frame.groupby("system"):
                ax.scatter(
                    group["true"], group["prediction"], s=18, alpha=0.75,
                    color=SYSTEM_COLORS.get(system, "#777777"), edgecolor="white",
                    linewidth=0.25, label=system,
                )
            lower = min(frame["true"].min(), frame["prediction"].min())
            upper = max(frame["true"].max(), frame["prediction"].max())
            pad = 0.05 * (upper - lower) if upper > lower else 1
            ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], color="#333333", linewidth=1)
            ax.set_xlim(lower - pad, upper + pad)
            ax.set_ylim(lower - pad, upper + pad)
            met = metrics(frame["true"].to_numpy(), frame["prediction"].to_numpy())
            ax.set_title(SHORT_LABELS[target])
            ax.set_xlabel("DFT")
            ax.set_ylabel("Predicted")
            ax.text(
                0.04, 0.96,
                f"$R^2$ = {met['r2']:.3f}\nMAE = {met['mae']:.3f}\nRMSE = {met['rmse']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
            ax.grid(color="#E6E6E6", linewidth=0.6)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
        fig.suptitle(f"{model_label(family)}: geometry-clustered external test", fontsize=14)
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        fig.savefig(OUT / "plots" / f"{family}_clustered_external_test_correlations.png", dpi=500)
        fig.savefig(OUT / "plots" / f"{family}_clustered_external_test_correlations.pdf")
        plt.close(fig)


def plot_metric_comparison(metrics_table: pd.DataFrame) -> None:
    test = metrics_table[metrics_table["stage"] == "external_test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = np.arange(len(TARGETS))
    width = 0.35
    for ax, family in zip(axes, ["extra_trees", "mlp"]):
        frame = test[test["model_family"] == family]
        random_values = [
            frame[(frame["scheme"] == "random") & (frame["target"] == target)]["r2"].iloc[0]
            for target in TARGETS
        ]
        cluster_values = [
            frame[(frame["scheme"] == "geometry_clustered") & (frame["target"] == target)]["r2"].iloc[0]
            for target in TARGETS
        ]
        ax.bar(x - width / 2, random_values, width, label="random test", color="#4C78A8")
        ax.bar(x + width / 2, cluster_values, width, label="clustered test", color="#D1495B")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x, [SHORT_LABELS[t] for t in TARGETS], rotation=22, ha="right")
        ax.set_title(model_label(family))
        ax.set_ylabel("External-test $R^2$")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "plots/random_vs_clustered_test_r2.png", dpi=500)
    fig.savefig(OUT / "plots/random_vs_clustered_test_r2.pdf")
    plt.close(fig)


def write_report(
    df: pd.DataFrame, balance: pd.DataFrame, separation: pd.DataFrame, metric_table: pd.DataFrame
) -> None:
    test = metric_table[metric_table["stage"] == "external_test"].copy()
    grouped = metric_table[metric_table["stage"] == "cluster_grouped_5fold_cv"].copy()
    lines = [
        "# Geometry-Clustered External Validation",
        "",
        "## Purpose",
        "",
        "This independent analysis tests whether the reported performance survives a split in which structurally similar geometries are kept together. The authoritative dataset, 19 descriptors, BS-state filter, target definitions, and manuscript model settings were not modified.",
        "",
        "## Split construction",
        "",
        f"- Input structures: {len(df)}.",
        "- Clustering variables: the same 19 structural descriptors used by the regression models.",
        "- Clustering was performed separately within each molecular system after median imputation and descriptor standardization.",
        f"- Ward agglomerative clustering used approximately one cluster per {TARGET_CLUSTER_SIZE} structures.",
        "- Complete clusters were assigned to train, validation, or test; no geometry cluster occurs in more than one subset.",
        "- Assignment targeted 70/15/15 within every represented system.",
        "- Target values were not used to construct clusters or assign subsets.",
        "",
        "## Model protocol",
        "",
        "- Extra Trees: 500 trees, minimum leaf size 2, all 19 descriptors, random seed 42.",
        "- MLP: 19-128-64-32-1, alpha = 1e-3, ReLU, Adam, standardized X and y, early stopping, random seed 42.",
        "- The model settings were fixed before this analysis and were not retuned against the clustered test set.",
        "- A cluster-grouped five-fold CV was performed within the combined training-validation pool.",
        "- Final models were fitted to the combined training-validation pool and evaluated once on the cluster-held-out test set.",
        "- A random 70/15/15 split using the same fixed models was rerun as an apples-to-apples reference.",
        "",
        "## Split balance",
        "",
        "| System | Train | Validation | Test |",
        "|---|---:|---:|---:|",
    ]
    clustered_balance = balance[balance["scheme"] == "geometry_clustered"]
    for system in sorted(df["system"].unique()):
        values = {
            split: int(clustered_balance[(clustered_balance["system"] == system) & (clustered_balance["split"] == split)]["n"].iloc[0])
            for split in ["train", "validation", "test"]
        }
        lines.append(f"| {system} | {values['train']} | {values['validation']} | {values['test']} |")
    lines.extend(
        [
            "",
            "## Descriptor-space separation",
            "",
            "| Scheme | Mean nearest-development distance | Median distance | Fraction below development 5th-percentile threshold |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in separation.iterrows():
        lines.append(
            f"| {row['scheme']} | {row['mean_nearest_development_distance']:.4f} | "
            f"{row['median_nearest_development_distance']:.4f} | {row['fraction_test_below_threshold']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Cluster-grouped cross-validation",
            "",
            "| Model | Target | R2 | MAE | RMSE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in grouped.sort_values(["model_family", "target"]).iterrows():
        lines.append(
            f"| {row['model_family']} | {SHORT_LABELS[row['target']]} | {row['r2']:.4f} | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Final external-test comparison",
            "",
            "| Model | Target | Split | R2 | MAE | RMSE | 95% CI for R2 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in test.sort_values(["model_family", "target", "scheme"]).iterrows():
        lines.append(
            f"| {row['model_family']} | {SHORT_LABELS[row['target']]} | {row['scheme']} | "
            f"{row['r2']:.4f} | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"[{row['r2_ci95_low']:.4f}, {row['r2_ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "The clustered split is intentionally more demanding than a random structure-level split. A reduction in R2 is expected if neighboring scan geometries previously appeared across subsets. The central questions are whether performance remains chemically useful, whether both model families show the same qualitative behavior, and whether the cluster-held-out test points are demonstrably farther from the development set in descriptor space.",
            "",
            "This validation addresses interpolation across withheld geometric regions within the represented molecular systems. It is not a substitute for leave-one-system-out transfer and does not establish prediction for an entirely new molecular family.",
            "",
            "Aggregate R2 values combine systems that occupy different target ranges. The accompanying `individual_system_external_test_metrics.csv` should therefore be consulted together with MAE and RMSE. Within-system R2 can be unstable when a held-out cluster spans only a narrow range of DFT values. The `worst_external_test_residuals.csv` file identifies localized extrapolation failures rather than concealing them in the aggregate correlation.",
            "",
        ]
    )
    (OUT / "GEOMETRY_CLUSTERED_EXTERNAL_VALIDATION_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)

    df = pd.read_csv(INPUT)
    missing = sorted(set(FEATURES + TARGETS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != 2582:
        raise ValueError(f"Expected authoritative 2582-row no-S2-filter dataset, found {len(df)}")

    df = make_geometry_clusters(df)
    df["random_split"] = make_random_split(df)
    df.to_csv(OUT / "geometry_cluster_assignments_and_splits.csv", index=False)

    balance = split_balance(df)
    balance.to_csv(OUT / "split_balance_by_system.csv", index=False)

    random_detail, random_summary = descriptor_separation(df, "random_split", "random")
    cluster_detail, cluster_summary = descriptor_separation(df, "clustered_split", "geometry_clustered")
    separation_detail = pd.concat([random_detail, cluster_detail], ignore_index=True)
    separation_summary = pd.concat([random_summary, cluster_summary], ignore_index=True)
    separation_detail.to_csv(OUT / "test_nearest_development_descriptor_distances.csv", index=False)
    separation_summary.to_csv(OUT / "descriptor_space_separation_summary.csv", index=False)

    all_metrics = []
    all_predictions = []
    model_bundles = {}
    for scheme, split_column in [("random", "random_split"), ("geometry_clustered", "clustered_split")]:
        metric_table, predictions, saved_models = evaluate_scheme(df, split_column, scheme)
        all_metrics.append(metric_table)
        all_predictions.append(predictions)
        model_bundles[scheme] = saved_models

    metric_table = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metric_table.to_csv(OUT / "validation_and_external_test_metrics.csv", index=False)
    predictions.to_csv(OUT / "external_test_predictions.csv", index=False)

    individual_rows = []
    for keys, frame in predictions.groupby(["scheme", "model_family", "target", "system"]):
        scheme, family, target, system = keys
        row = {
            "scheme": scheme,
            "model_family": family,
            "target": target,
            "system": system,
            "n": len(frame),
            **metrics(frame["true"].to_numpy(), frame["prediction"].to_numpy()),
            "mean_residual": float(frame["residual"].mean()),
        }
        individual_rows.append(row)
    pd.DataFrame(individual_rows).to_csv(OUT / "individual_system_external_test_metrics.csv", index=False)

    worst = predictions.assign(abs_residual=predictions["residual"].abs()).sort_values(
        ["scheme", "model_family", "target", "abs_residual"], ascending=[True, True, True, False]
    )
    worst.groupby(["scheme", "model_family", "target"], group_keys=False).head(20).to_csv(
        OUT / "worst_external_test_residuals.csv", index=False
    )

    distribution_rows = []
    for split_column, scheme in [("random_split", "random"), ("clustered_split", "geometry_clustered")]:
        for (system, split), frame in df.groupby(["system", split_column]):
            for target in TARGETS:
                distribution_rows.append(
                    {
                        "scheme": scheme,
                        "system": system,
                        "split": split,
                        "target": target,
                        "n": len(frame),
                        "mean": float(frame[target].mean()),
                        "sd": float(frame[target].std(ddof=1)),
                        "minimum": float(frame[target].min()),
                        "maximum": float(frame[target].max()),
                    }
                )
    pd.DataFrame(distribution_rows).to_csv(OUT / "target_distribution_by_system_and_split.csv", index=False)

    with (OUT / "models/fixed_models_random_and_clustered.pkl").open("wb") as handle:
        pickle.dump(model_bundles, handle)

    test = metric_table[metric_table["stage"] == "external_test"]
    random_test = test[test["scheme"] == "random"].set_index(["model_family", "target"])
    clustered_test = test[test["scheme"] == "geometry_clustered"].set_index(["model_family", "target"])
    comparison = clustered_test[["n", "r2", "mae", "rmse"]].rename(
        columns={"n": "clustered_n", "r2": "clustered_r2", "mae": "clustered_mae", "rmse": "clustered_rmse"}
    ).join(
        random_test[["n", "r2", "mae", "rmse"]].rename(
            columns={"n": "random_n", "r2": "random_r2", "mae": "random_mae", "rmse": "random_rmse"}
        )
    )
    comparison["delta_r2_clustered_minus_random"] = comparison["clustered_r2"] - comparison["random_r2"]
    comparison["mae_ratio_clustered_over_random"] = comparison["clustered_mae"] / comparison["random_mae"]
    comparison.reset_index().to_csv(OUT / "random_vs_geometry_clustered_test_comparison.csv", index=False)

    plot_split_pca(df)
    plot_separation(separation_detail)
    plot_correlations(predictions)
    plot_metric_comparison(metric_table)
    write_report(df, balance, separation_summary, metric_table)

    manifest = {
        "input": str(INPUT),
        "output": str(OUT),
        "n_structures": len(df),
        "features": FEATURES,
        "targets": TARGETS,
        "systems": df["system"].value_counts().sort_index().to_dict(),
        "clustering": {
            "method": "Ward agglomerative clustering",
            "performed_within_system": True,
            "target_cluster_size": TARGET_CLUSTER_SIZE,
            "uses_target_values": False,
        },
        "split": {
            "target_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "cluster_exclusive": True,
            "random_seed": RANDOM_STATE,
        },
        "models": {
            "extra_trees": {"n_estimators": 500, "min_samples_leaf": 2},
            "mlp": {"architecture": [19, 128, 64, 32, 1], "alpha": 1e-3},
        },
        "test_set_not_used_for_tuning": True,
    }
    (OUT / "geometry_clustered_validation_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(__file__, OUT / "geometry_clustered_external_validation.py")
    print("\nGeometry-clustered external validation complete.")
    print(comparison.reset_index().round(4).to_string(index=False))
    print("\nDescriptor separation:")
    print(separation_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
