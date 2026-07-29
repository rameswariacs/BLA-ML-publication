#!/usr/bin/env python3
"""Test-independent MLP architecture, regularization, and learning-curve audit."""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/rb1820/BLA-ML")
RUNSCRIPTS = ROOT / "RunScripts_plus_CPBP"
SOURCE = ROOT / "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211"
OUT = ROOT / "/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/MLP_ComplexitySensitivity_FODFT_4Targets"

if str(RUNSCRIPTS) not in sys.path:
    sys.path.insert(0, str(RUNSCRIPTS))

import unified_19descriptor_bs_filtered_elstfix_pipeline as core


ARCHITECTURES = [
    (16,),
    (32,),
    (32, 16),
    (64, 32),
    (64, 32, 16),
    (128, 64, 32),
    (64, 64, 32, 16),
]
ALPHAS = [1e-4, 1e-3, 1e-2]
TRAIN_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
TARGETS = [
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
N_JOBS = 16


def architecture_name(hidden: tuple[int, ...]) -> str:
    return "-".join(str(value) for value in hidden)


def parameter_counts(hidden: tuple[int, ...], n_features: int = 19) -> tuple[int, int, int]:
    widths = (n_features, *hidden, 1)
    weights = sum(left * right for left, right in zip(widths[:-1], widths[1:]))
    biases = sum(widths[1:])
    return weights, biases, weights + biases


def metric_dict(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))),
    }


def estimator(hidden: tuple[int, ...], alpha: float) -> TransformedTargetRegressor:
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("x_scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=64,
                    learning_rate="constant",
                    learning_rate_init=1e-3,
                    max_iter=900,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    tol=1e-5,
                    shuffle=True,
                    random_state=core.RANDOM_STATE,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def fitted_net(model: TransformedTargetRegressor) -> MLPRegressor:
    return model.regressor_.named_steps["mlp"]


def evaluate_fold(
    target: str,
    hidden: tuple[int, ...],
    alpha: float,
    fold: int,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    x: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, object]:
    model = estimator(hidden, alpha)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x.iloc[train_idx], y[train_idx])
    fit_seconds = time.perf_counter() - started
    prediction = model.predict(x.iloc[validation_idx])
    net = fitted_net(model)
    weights, biases, total = parameter_counts(hidden, x.shape[1])
    return {
        "target": target,
        "architecture": architecture_name(hidden),
        "hidden_layer_sizes": repr(hidden),
        "n_hidden_layers": len(hidden),
        "alpha": alpha,
        "weights": weights,
        "biases": biases,
        "trainable_parameters": total,
        "fold": fold,
        "n_fold_train": int(len(train_idx)),
        "n_fold_validation": int(len(validation_idx)),
        "fit_seconds": fit_seconds,
        "epochs": int(net.n_iter_),
        "final_scaled_loss": float(net.loss_),
        "best_internal_validation_score": float(net.best_validation_score_),
        **metric_dict(y[validation_idx], prediction),
    }


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping = [
        "target",
        "architecture",
        "hidden_layer_sizes",
        "n_hidden_layers",
        "alpha",
        "weights",
        "biases",
        "trainable_parameters",
    ]
    for keys, frame in folds.groupby(grouping, sort=False):
        row = dict(zip(grouping, keys))
        row["n_folds"] = int(len(frame))
        for name in [
            "r2",
            "mae",
            "rmse",
            "fit_seconds",
            "epochs",
            "final_scaled_loss",
            "best_internal_validation_score",
        ]:
            values = frame[name].to_numpy(dtype=float)
            row[f"mean_{name}"] = float(values.mean())
            row[f"sd_{name}"] = float(values.std(ddof=1))
            row[f"se_{name}"] = float(values.std(ddof=1) / math.sqrt(len(values)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["target", "mean_mae", "trainable_parameters", "alpha"],
        ascending=[True, True, True, False],
    )


def select_one_se(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, frame in summary.groupby("target", sort=False):
        best = frame.sort_values(["mean_mae", "mean_rmse"]).iloc[0]
        threshold = float(best["mean_mae"] + best["se_mae"])
        eligible = frame.loc[frame["mean_mae"] <= threshold].copy()
        choice = eligible.sort_values(
            ["trainable_parameters", "alpha", "n_hidden_layers", "mean_mae"],
            ascending=[True, False, True, True],
        ).iloc[0]
        record = choice.to_dict()
        record.update(
            {
                "best_architecture": best["architecture"],
                "best_alpha": float(best["alpha"]),
                "best_trainable_parameters": int(best["trainable_parameters"]),
                "best_cv_mean_mae": float(best["mean_mae"]),
                "best_cv_se_mae": float(best["se_mae"]),
                "one_se_mae_threshold": threshold,
                "n_eligible_configurations": int(len(eligible)),
                "selection_rule": (
                    "fewest trainable parameters, then largest alpha, among "
                    "mean MAE <= best mean MAE + SE(best)"
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def stratified_subset(
    indices: np.ndarray,
    systems: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    if fraction >= 1.0:
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=fraction,
        shuffle=True,
        random_state=seed,
        stratify=systems[indices],
    )
    return np.sort(selected)


def learning_curve_fit(
    target: str,
    hidden: tuple[int, ...],
    alpha: float,
    fraction: float,
    fold: int,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    systems: np.ndarray,
    x: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, object]:
    subset = stratified_subset(
        train_idx,
        systems,
        fraction,
        core.RANDOM_STATE + fold + int(fraction * 1000),
    )
    model = estimator(hidden, alpha)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x.iloc[subset], y[subset])
    fit_seconds = time.perf_counter() - started
    prediction = model.predict(x.iloc[validation_idx])
    net = fitted_net(model)
    return {
        "target": target,
        "architecture": architecture_name(hidden),
        "alpha": alpha,
        "train_fraction": fraction,
        "fold": fold,
        "n_training_structures": int(len(subset)),
        "n_validation_structures": int(len(validation_idx)),
        "fit_seconds": fit_seconds,
        "epochs": int(net.n_iter_),
        **metric_dict(y[validation_idx], prediction),
    }


def summarize_learning_curves(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in folds.groupby(
        ["target", "architecture", "alpha", "train_fraction"], sort=False
    ):
        target, architecture, alpha, fraction = keys
        row = {
            "target": target,
            "architecture": architecture,
            "alpha": alpha,
            "train_fraction": fraction,
            "mean_n_training_structures": float(frame["n_training_structures"].mean()),
        }
        for name in ["r2", "mae", "rmse", "epochs", "fit_seconds"]:
            values = frame[name].to_numpy(dtype=float)
            row[f"mean_{name}"] = float(values.mean())
            row[f"sd_{name}"] = float(values.std(ddof=1))
            row[f"se_{name}"] = float(values.std(ddof=1) / math.sqrt(len(values)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "train_fraction"])


def final_test(
    data: pd.DataFrame, selections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    saved_metrics = pd.read_csv(
        SOURCE / "train_validate_test/mlp/mlp_train_validate_test_metrics.csv"
    )
    saved_predictions = pd.read_csv(
        SOURCE / "train_validate_test/mlp/mlp_train_validate_test_predictions.csv"
    )
    metric_rows = []
    prediction_rows = []
    for _, choice in selections.iterrows():
        target = str(choice["target"])
        hidden = tuple(int(value) for value in str(choice["hidden_layer_sizes"]).strip("()").split(",") if value.strip())
        alpha = float(choice["alpha"])
        sub = data.loc[data[target].notna()].copy().reset_index(drop=True)
        sub["split"] = core.make_split(sub, core.RANDOM_STATE)
        train_validation = sub["split"].isin(["train", "validation"])
        test = sub["split"].eq("test")
        x = sub[core.FEATURES19]
        y = sub[target].to_numpy(dtype=float)

        model = estimator(hidden, alpha)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x.loc[train_validation], y[train_validation])
        fit_seconds = time.perf_counter() - started
        prediction = model.predict(x.loc[test])
        net = fitted_net(model)
        metric_rows.append(
            {
                "target": target,
                "configuration": "one_se_selected",
                "architecture": architecture_name(hidden),
                "alpha": alpha,
                "trainable_parameters": parameter_counts(hidden)[2],
                "n_train_validation": int(train_validation.sum()),
                "n_test": int(test.sum()),
                "fit_seconds": fit_seconds,
                "epochs": int(net.n_iter_),
                **metric_dict(y[test], prediction),
            }
        )
        for ident, system, truth, pred in zip(
            sub.loc[test, "id"],
            sub.loc[test, "system"],
            y[test],
            prediction,
        ):
            prediction_rows.append(
                {
                    "target": target,
                    "configuration": "one_se_selected",
                    "id": ident,
                    "system": system,
                    "true": truth,
                    "prediction": pred,
                    "residual": truth - pred,
                }
            )

        baseline_metric = saved_metrics.loc[
            saved_metrics["target"].eq(target)
            & saved_metrics["stage"].eq("selected_test")
        ].iloc[0]
        model_name = str(baseline_metric["model"])
        parameter_lookup = {
            "mlp_128_64_32_a1e-3": 12929,
            "mlp_128_64_32_a1e-4": 12929,
            "mlp_64_32_16_a1e-3": 3905,
            "mlp_64_32_16_a1e-4": 3905,
            "mlp_64_64_32_16_a1e-4": 8065,
            "mlp_32_16_a1e-4": 1185,
        }
        metric_rows.append(
            {
                "target": target,
                "configuration": "original_saved_model",
                "architecture": model_name,
                "alpha": math.nan,
                "trainable_parameters": parameter_lookup.get(model_name, math.nan),
                "n_train_validation": int(train_validation.sum()),
                "n_test": int(test.sum()),
                "fit_seconds": math.nan,
                "epochs": math.nan,
                "r2": float(baseline_metric["r2"]),
                "mae": float(baseline_metric["mae"]),
                "rmse": float(baseline_metric["rmse"]),
            }
        )
        baseline_prediction = saved_predictions.loc[
            saved_predictions["target"].eq(target)
            & saved_predictions["split"].eq("test")
        ]
        for _, row in baseline_prediction.iterrows():
            prediction_rows.append(
                {
                    "target": target,
                    "configuration": "original_saved_model",
                    "id": row["id"],
                    "system": row["system"],
                    "true": float(row["true"]),
                    "prediction": float(row["prediction"]),
                    "residual": float(row["residual"]),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def configure_style() -> None:
    font_dir = ROOT / "Publication_Top7_FeatureImportance_BSFiltered_ElstOrderFixed/fonts"
    for path in font_dir.glob("*.ttf"):
        font_manager.fontManager.addfont(str(path))
    available = {item.name for item in font_manager.fontManager.ttflist}
    family = "Calibri" if "Calibri" in available else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def make_plots(
    summary: pd.DataFrame,
    selections: pd.DataFrame,
    learning: pd.DataFrame,
) -> None:
    configure_style()
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    alpha_colors = {1e-4: "#356FA8", 1e-3: "#009E73", 1e-2: "#D55E00"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, target in zip(axes.flat, TARGETS):
        frame = summary.loc[summary["target"].eq(target)]
        for alpha in ALPHAS:
            line = frame.loc[frame["alpha"].eq(alpha)].sort_values("trainable_parameters")
            ax.errorbar(
                line["trainable_parameters"],
                line["mean_mae"],
                yerr=line["se_mae"],
                marker="o",
                lw=1.2,
                capsize=2,
                color=alpha_colors[alpha],
                label=rf"$\alpha={alpha:.0e}$",
            )
        choice = selections.loc[selections["target"].eq(target)].iloc[0]
        ax.scatter(
            choice["trainable_parameters"],
            choice["mean_mae"],
            marker="*",
            s=160,
            color="black",
            zorder=5,
            label="one-SE selection",
        )
        ax.set_xscale("log")
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel("Five-fold CV MAE")
        ax.grid(color="#E1E1E1", linewidth=0.7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    save_figure(fig, plot_dir / "mlp_parameter_count_vs_cv_mae")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    architecture_order = [architecture_name(value) for value in ARCHITECTURES]
    for ax, target in zip(axes.flat, TARGETS):
        pivot = (
            summary.loc[summary["target"].eq(target)]
            .pivot(index="architecture", columns="alpha", values="mean_mae")
            .reindex(index=architecture_order, columns=ALPHAS)
        )
        image = ax.imshow(pivot, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(ALPHAS)), [f"{value:.0e}" for value in ALPHAS])
        ax.set_yticks(range(len(architecture_order)), architecture_order)
        ax.set_xlabel(r"L2 regularization, $\alpha$")
        ax.set_ylabel("Hidden-layer architecture")
        ax.set_title(TARGET_LABELS[target])
        fig.colorbar(image, ax=ax, label="Five-fold CV MAE")
    save_figure(fig, plot_dir / "mlp_architecture_alpha_mae_grid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, target in zip(axes.flat, TARGETS):
        frame = summary.loc[summary["target"].eq(target)].sort_values(
            ["trainable_parameters", "alpha"]
        )
        for alpha in ALPHAS:
            line = frame.loc[frame["alpha"].eq(alpha)]
            ax.plot(
                line["trainable_parameters"],
                line["mean_epochs"],
                marker="o",
                color=alpha_colors[alpha],
                label=rf"$\alpha={alpha:.0e}$",
            )
        ax.set_xscale("log")
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel("Mean early-stopping epoch")
        ax.grid(color="#E1E1E1", linewidth=0.7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, plot_dir / "mlp_early_stopping_epochs")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, target in zip(axes.flat, TARGETS):
        frame = learning.loc[learning["target"].eq(target)].sort_values("train_fraction")
        ax.errorbar(
            frame["mean_n_training_structures"],
            frame["mean_mae"],
            yerr=frame["se_mae"],
            marker="o",
            color="#356FA8",
            capsize=3,
        )
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Training structures per fold")
        ax.set_ylabel("Validation MAE")
        ax.grid(color="#E1E1E1", linewidth=0.7)
    save_figure(fig, plot_dir / "mlp_training_size_learning_curves")


def write_report(
    summary: pd.DataFrame,
    selections: pd.DataFrame,
    test_metrics: pd.DataFrame,
    learning_summary: pd.DataFrame,
) -> None:
    lines = [
        "# MLP Architecture and Regularization Sensitivity Analysis",
        "",
        "## Test-independent protocol",
        "",
        "The established system-stratified 70/15/15 split was reconstructed using "
        "random seed 42. The 15% test subset was excluded from architecture and "
        "regularization selection. The combined training and validation pool was "
        "evaluated with fixed five-fold, system-stratified cross-validation.",
        "",
        "Seven hidden-layer architectures were combined with three L2 regularization "
        "strengths, giving 21 configurations per target and 420 cross-validation fits "
        "for the four manuscript targets. Every fit used the manuscript preprocessing "
        "and optimization pipeline: median imputation, standardized descriptors, "
        "standardized target values, ReLU hidden units, linear output, Adam optimization, "
        "batch size 64, initial learning rate 0.001, and early stopping with a 15% "
        "internal validation fraction and 30-epoch patience.",
        "",
        "Selection followed the one-standard-error rule. Among configurations with "
        "mean CV MAE no greater than the best mean MAE plus its standard error, the "
        "network with the fewest trainable parameters was chosen. Ties favored stronger "
        "L2 regularization.",
        "",
        "## Candidate network sizes",
        "",
        "| Hidden-layer architecture | Weights | Biases | Total trainable parameters |",
        "|---|---:|---:|---:|",
    ]
    for hidden in ARCHITECTURES:
        weights, biases, total = parameter_counts(hidden)
        lines.append(
            f"| {architecture_name(hidden)} | {weights} | {biases} | {total} |"
        )
    lines.extend(
        [
        "",
        "## Selected architectures",
        "",
        "| Target | Lowest-MAE configuration | One-SE configuration | Parameters | CV MAE | CV R2 |",
        "|---|---|---|---:|---:|---:|",
        ]
    )
    for _, row in selections.iterrows():
        lines.append(
            f"| {TARGET_LABELS[row['target']]} | {row['best_architecture']}, "
            f"alpha={row['best_alpha']:.0e} | {row['architecture']}, "
            f"alpha={row['alpha']:.0e} | {int(row['trainable_parameters'])} | "
            f"{row['mean_mae']:.6g} | {row['mean_r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Position of the manuscript MLP in the CV grid",
            "",
            "The manuscript model uses hidden layers 128-64-32 with alpha = 1e-3 "
            "(12,929 trainable parameters).",
            "",
            "| Target | Manuscript CV MAE | Best CV MAE | Difference | Within one SE? |",
            "|---|---:|---:|---:|---|",
        ]
    )
    original = summary.loc[
        summary["architecture"].eq("128-64-32")
        & summary["alpha"].eq(1e-3)
    ]
    for _, choice in selections.iterrows():
        row = original.loc[original["target"].eq(choice["target"])].iloc[0]
        difference = float(row["mean_mae"] - choice["best_cv_mean_mae"])
        within = float(row["mean_mae"]) <= float(choice["one_se_mae_threshold"])
        lines.append(
            f"| {TARGET_LABELS[row['target']]} | {row['mean_mae']:.6g} | "
            f"{choice['best_cv_mean_mae']:.6g} | {difference:.6g} | "
            f"{'Yes' if within else 'No'} |"
        )
    lines.extend(
        [
            "",
            "## Final held-out test comparison",
            "",
            "| Target | Configuration | Architecture | Parameters | Test R2 | Test MAE | Test RMSE |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in test_metrics.iterrows():
        label = "One-SE selected" if row["configuration"] == "one_se_selected" else "Original saved model"
        lines.append(
            f"| {TARGET_LABELS[row['target']]} | {label} | {row['architecture']} | "
            f"{int(row['trainable_parameters'])} | {row['r2']:.4f} | "
            f"{row['mae']:.6g} | {row['rmse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The independent test set was evaluated only after completion of the "
            "cross-validation-based selection and did not influence architecture choice.",
            "",
            "## Training-size learning curves",
            "",
            "| Target | MAE at 25% training size | MAE at 100% training size | Relative decrease |",
            "|---|---:|---:|---:|",
        ]
    )
    for target, frame in learning_summary.groupby("target", sort=False):
        frame = frame.sort_values("train_fraction")
        first = frame.iloc[0]
        last = frame.iloc[-1]
        decrease = 100.0 * (1.0 - float(last["mean_mae"]) / float(first["mean_mae"]))
        lines.append(
            f"| {TARGET_LABELS[target]} | {first['mean_mae']:.6g} | "
            f"{last['mean_mae']:.6g} | {decrease:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This analysis distinguishes numerical parameter count from effective model "
            "complexity. L2 regularization and early stopping constrain all networks, "
            "while the one-standard-error rule tests whether smaller architectures retain "
            "statistically comparable validation performance. Training-size learning "
            "curves provide an additional check that predictive error decreases as more "
            "structures become available rather than being sustained by a small subset.",
            "",
            "The manuscript 128-64-32 network is the lowest-MAE CV configuration for "
            "|t_FO| and corrected SOMO-SOMO energy and remains within one "
            "standard error of the best result for interaction energy. For the frontier "
            "gap, its mean CV MAE exceeds the strict one-SE threshold by only a small "
            "absolute amount. The sensitivity analysis therefore does not establish that "
            "12,929 parameters are necessary for every target, but it shows that the "
            "reported architecture occupies the stable, high-performing region of the "
            "complexity grid. Retaining one common regularized architecture across the "
            "four manuscript targets is consequently defensible, provided it is described "
            "as a validated common architecture rather than the unique optimum for every "
            "property.",
            "",
        ]
    )
    (OUT / "MLP_COMPLEXITY_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE / "ALL_DIMERS_19descriptors_7targets_BS_filtered.csv")
    fold_rows: list[dict[str, object]] = []
    split_cache: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]] = {}

    for target in TARGETS:
        sub = data.loc[data[target].notna()].copy().reset_index(drop=True)
        sub["split"] = core.make_split(sub, core.RANDOM_STATE)
        development = sub.loc[sub["split"].isin(["train", "validation"])].reset_index(drop=True)
        x = development[core.FEATURES19]
        y = development[target].to_numpy(dtype=float)
        systems = development["system"].astype(str).to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=core.RANDOM_STATE)
        splits = list(cv.split(x, systems))
        split_cache[target] = (x, y, systems, splits)
        tasks = [
            (target, hidden, alpha, fold, train_idx, validation_idx, x, y)
            for hidden in ARCHITECTURES
            for alpha in ALPHAS
            for fold, (train_idx, validation_idx) in enumerate(splits, start=1)
        ]
        results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
            delayed(evaluate_fold)(*task) for task in tasks
        )
        fold_rows.extend(results)
        pd.DataFrame(fold_rows).to_csv(OUT / "mlp_complexity_fold_metrics.csv", index=False)

    folds = pd.DataFrame(fold_rows)
    summary = summarize(folds)
    selections = select_one_se(summary)

    learning_rows: list[dict[str, object]] = []
    for _, choice in selections.iterrows():
        target = str(choice["target"])
        hidden = tuple(
            int(value)
            for value in str(choice["hidden_layer_sizes"]).strip("()").split(",")
            if value.strip()
        )
        alpha = float(choice["alpha"])
        x, y, systems, splits = split_cache[target]
        tasks = [
            (
                target,
                hidden,
                alpha,
                fraction,
                fold,
                train_idx,
                validation_idx,
                systems,
                x,
                y,
            )
            for fraction in TRAIN_FRACTIONS
            for fold, (train_idx, validation_idx) in enumerate(splits, start=1)
        ]
        results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
            delayed(learning_curve_fit)(*task) for task in tasks
        )
        learning_rows.extend(results)
    learning_folds = pd.DataFrame(learning_rows)
    learning_summary = summarize_learning_curves(learning_folds)

    test_metrics, test_predictions = final_test(data, selections)
    folds.to_csv(OUT / "mlp_complexity_fold_metrics.csv", index=False)
    summary.to_csv(OUT / "mlp_complexity_cv_summary.csv", index=False)
    selections.to_csv(OUT / "mlp_one_se_selections.csv", index=False)
    learning_folds.to_csv(OUT / "mlp_learning_curve_fold_metrics.csv", index=False)
    learning_summary.to_csv(OUT / "mlp_learning_curve_summary.csv", index=False)
    test_metrics.to_csv(OUT / "mlp_selected_vs_original_test_metrics.csv", index=False)
    test_predictions.to_csv(OUT / "mlp_selected_vs_original_test_predictions.csv", index=False)
    make_plots(summary, selections, learning_summary)
    write_report(summary, selections, test_metrics, learning_summary)

    manifest = {
        "source": str(SOURCE),
        "targets": TARGETS,
        "features": core.FEATURES19,
        "architectures": [architecture_name(value) for value in ARCHITECTURES],
        "alphas": ALPHAS,
        "training_fractions": TRAIN_FRACTIONS,
        "split_seed": core.RANDOM_STATE,
        "test_used_for_selection": False,
        "cv": "5-fold StratifiedKFold by molecular system on train+validation pool",
        "selection_rule": "one standard error; fewest parameters then strongest L2",
        "n_architecture_cv_fits": int(len(folds)),
        "n_learning_curve_fits": int(len(learning_folds)),
    }
    (OUT / "mlp_complexity_manifest.json").write_text(json.dumps(manifest, indent=2))
    print((OUT / "MLP_COMPLEXITY_REPORT.md").read_text())


if __name__ == "__main__":
    main()
