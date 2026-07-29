#!/usr/bin/env python3
"""Independent Y-randomization control for the no-S2-filter 19-descriptor models.

The descriptor matrix and saved manuscript split are not modified. For each
target/model task, the target vector is globally permuted 500 times, the
manuscript-selected estimator is refitted on the original train+validation
rows, and performance is measured on the original test rows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path("/home/rb1820/BLA-ML")
RUNSCRIPTS = ROOT / "RunScripts_plus_CPBP"
SOURCE = ROOT / "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211"
OUT = ROOT / "/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/YRandomization_500x_FODFT_4Targets"
RESULTS = OUT / "results_by_task"

if str(RUNSCRIPTS) not in sys.path:
    sys.path.insert(0, str(RUNSCRIPTS))

import unified_19descriptor_bs_filtered_elstfix_pipeline as core


N_PERMUTATIONS = 500
PERMUTATION_SEED = 42

TARGETS = [
    "target_fodft_coupling_abs_ev",
    "target_interaction_energy_kcal_mol",
    "target_frontier_gap_ev",
    "target_somo_somo_elst_corrected_signed_kcal_mol",
]
FAMILIES = ["extra_trees", "mlp"]
TASKS = [(family, target) for target in TARGETS for family in FAMILIES]


def metric_dict(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))),
    }


def task_slug(family: str, target: str) -> str:
    return f"{family}__{target.removeprefix('target_')}"


def original_files(family: str) -> tuple[Path, Path]:
    base = SOURCE / "train_validate_test" / family
    return (
        base / f"{family}_train_validate_test_metrics.csv",
        base / f"{family}_train_validate_test_predictions.csv",
    )


def load_selected_model(family: str, target: str):
    metrics_path, _ = original_files(family)
    metrics = pd.read_csv(metrics_path)
    selection = metrics[
        metrics["target"].eq(target)
        & metrics["stage"].eq("validation_selection")
    ].sort_values(["mae", "rmse"])
    if selection.empty:
        raise ValueError(f"No validation-selection rows for {family}, {target}")
    selected_name = str(selection.iloc[0]["model"])

    real_test = metrics[
        metrics["target"].eq(target)
        & metrics["stage"].eq("selected_test")
        & metrics["model"].eq(selected_name)
    ]
    if len(real_test) != 1:
        raise ValueError(
            f"Expected one selected-test row for {family}, {target}, {selected_name}; "
            f"found {len(real_test)}"
        )

    candidates = core.models_for_family(family)
    if selected_name not in candidates:
        raise KeyError(f"Selected model {selected_name} is absent from estimator factory")
    return selected_name, candidates[selected_name], real_test.iloc[0]


def prepare_data(family: str, target: str):
    data_path = SOURCE / "ALL_DIMERS_19descriptors_7targets_BS_filtered.csv"
    frame = pd.read_csv(data_path)
    missing_features = sorted(set(core.FEATURES19).difference(frame.columns))
    if missing_features:
        raise ValueError(f"Missing descriptors: {missing_features}")

    valid = frame[target].notna()
    sub = (
        frame.loc[valid]
        .copy()
        .reset_index(drop=False)
        .rename(columns={"index": "source_index"})
    )
    sub["split"] = core.make_split(sub, core.RANDOM_STATE)

    _, predictions_path = original_files(family)
    saved = pd.read_csv(predictions_path)
    saved = saved[saved["target"].eq(target)].copy()
    for split_name in ("train", "validation", "test"):
        reconstructed = set(sub.loc[sub["split"].eq(split_name), "id"].astype(str))
        recorded = set(saved.loc[saved["split"].eq(split_name), "id"].astype(str))
        if reconstructed != recorded:
            raise RuntimeError(
                f"Reconstructed {split_name} IDs differ from saved manuscript split "
                f"for {family}, {target}"
            )

    x = sub[core.FEATURES19].copy()
    y = sub[target].to_numpy(dtype=float)
    train_validation = sub["split"].isin(["train", "validation"]).to_numpy()
    test = sub["split"].eq("test").to_numpy()
    return sub, x, y, train_validation, test


def write_checkpoint(rows: list[dict[str, object]], path: Path) -> None:
    temp = path.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def run_task(task_index: int, n_permutations: int) -> Path:
    if task_index < 0 or task_index >= len(TASKS):
        raise ValueError(f"task-index must be between 0 and {len(TASKS) - 1}")

    family, target = TASKS[task_index]
    slug = task_slug(family, target)
    RESULTS.mkdir(parents=True, exist_ok=True)
    output_csv = RESULTS / f"{slug}.csv"
    metadata_path = RESULTS / f"{slug}.json"

    selected_name, estimator, real_test = load_selected_model(family, target)
    sub, x, y, train_validation, test = prepare_data(family, target)
    rng = np.random.default_rng(PERMUTATION_SEED)

    rows: list[dict[str, object]] = []
    started = time.time()
    for permutation in range(1, n_permutations + 1):
        y_permuted = rng.permutation(y)
        fitted = clone(estimator)
        fitted.fit(x.loc[train_validation], y_permuted[train_validation])
        prediction = fitted.predict(x.loc[test])
        metrics = metric_dict(y_permuted[test], prediction)
        rows.append(
            {
                "task_index": task_index,
                "permutation": permutation,
                "family": family,
                "target": target,
                "selected_model": selected_name,
                "n_total": int(len(sub)),
                "n_train_validation": int(train_validation.sum()),
                "n_test": int(test.sum()),
                "n_features": len(core.FEATURES19),
                "split_seed": core.RANDOM_STATE,
                "permutation_seed": PERMUTATION_SEED,
                "model_random_state": core.RANDOM_STATE,
                **metrics,
            }
        )
        if permutation % 10 == 0 or permutation == n_permutations:
            write_checkpoint(rows, output_csv)
            elapsed = time.time() - started
            print(
                f"{slug}: {permutation}/{n_permutations} permutations; "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    metadata = {
        "task_index": task_index,
        "family": family,
        "target": target,
        "selected_model": selected_name,
        "n_permutations": n_permutations,
        "permutation_protocol": (
            "Global permutation of the complete target vector before applying the "
            "unchanged manuscript train/validation/test row masks. The selected "
            "manuscript estimator is fitted on the original train+validation rows "
            "and evaluated on the original test rows."
        ),
        "descriptor_matrix": "unchanged",
        "features": core.FEATURES19,
        "n_total": int(len(sub)),
        "n_train": int(sub["split"].eq("train").sum()),
        "n_validation": int(sub["split"].eq("validation").sum()),
        "n_train_validation": int(train_validation.sum()),
        "n_test": int(test.sum()),
        "split_seed": core.RANDOM_STATE,
        "permutation_seed": PERMUTATION_SEED,
        "model_random_state": core.RANDOM_STATE,
        "real_test_metrics": {
            "r2": float(real_test["r2"]),
            "mae": float(real_test["mae"]),
            "rmse": float(real_test["rmse"]),
        },
        "output_csv": str(output_csv),
        "elapsed_seconds": time.time() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    path = run_task(args.task_index, args.n_permutations)
    print(f"Completed: {path}")


if __name__ == "__main__":
    main()
