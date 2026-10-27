#!/usr/bin/env python3
"""Triplet-aware target extraction and descriptor-model benchmarks."""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/rb1820/BLA-ML")
INPUT = ROOT / "Curated_OpenShellGap_Definition_Inputs_plus_CPBP" / (
    "ALL_SYSTEMS_curated_inputs_contactlocal_bla_3p4_counts_3p4_3p6_4p0_plus_CPBP_spin_min_frontier_gap.csv"
)
OUT = ROOT / "TripletAware_EstSomO_Targets_plus_CPBP"
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
COULOMB_KCAL_PER_MOL = 332.06371

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
]

FEATURE_GROUPS = {
    "BLA_global_only": ["bla_fragment_mean"],
    "BLA_global_plus_contact_region": ["bla_fragment_mean", "contact_region_bla_mean"],
    "stacking_area": [
        "contact_min_cc_distance",
        "stack_interplane_distance",
        "stack_lateral_slip",
        "stack_plane_normal_angle_deg",
        "contact_interfragment_distance_std",
        "projected_pi_overlap_area",
        "projected_pi_overlap_fraction_min",
        "pi_projected_area_mean",
    ],
    "stacking_area_contact_dispersion": [
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
    ],
    "BLA_contact_plus_stacking_area_contact_dispersion": FEATURES,
}

NEW_TARGETS = [
    "target_singlet_triplet_gap_kcal_mol",
    "target_somo_somo_elst_corrected_signed_kcal_mol",
]

TARGET_LABELS = {
    "target_singlet_triplet_gap_kcal_mol": "E_ST (kcal/mol)",
    "target_singlet_triplet_gap_ev": "E_ST (eV)",
    "target_somo_somo_raw_signed_kcal_mol": "Raw E_int^BS - E_int^T (kcal/mol)",
    "target_somo_somo_elst_corrected_signed_kcal_mol": "Electrostatic-corrected E_SOMO-SOMO (kcal/mol)",
}

COLORS = {
    "phenalenyl": "#0072B2",
    "olympicenyl": "#D55E00",
    "fluorenyl": "#009E73",
    "CPBP": "#CC79A7",
}


@dataclass
class TripletTargets:
    bs_energy_hartree: float
    triplet_energy_hartree: float
    monomer_energy_sum_hartree: float
    bs_interaction_energy_kcal_mol: float
    triplet_interaction_energy_kcal_mol: float
    bs_elst_kcal_mol: float
    triplet_elst_kcal_mol: float


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(errors="ignore").splitlines()
    n = int(lines[0].strip())
    symbols: list[str] = []
    coords = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad xyz line in {path}: {line}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=float)


def final_single_point_energy(path: Path) -> float:
    pattern = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
    found: float | None = None
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            found = float(match.group(1))
    if found is None:
        raise ValueError(f"No FINAL SINGLE POINT ENERGY found in {path}")
    return found


def terminated_normally(path: Path) -> bool:
    return "ORCA TERMINATED NORMALLY" in path.read_text(errors="ignore")


def final_mayer_mulliken_charges(path: Path, n_atoms: int) -> np.ndarray:
    lines = path.read_text(errors="ignore").splitlines()
    tables: list[list[float]] = []
    i = 0
    row_re = re.compile(
        r"^\s*(\d+)\s+[A-Za-z]+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+([-+]?\d+\.\d+)\s+"
    )
    while i < len(lines):
        if "MAYER POPULATION ANALYSIS" not in lines[i]:
            i += 1
            continue
        charges: list[float] = []
        i += 1
        while i < len(lines):
            match = row_re.match(lines[i])
            if match:
                atom_idx = int(match.group(1))
                charge = float(match.group(2))
                if atom_idx == len(charges):
                    charges.append(charge)
                i += 1
                continue
            if charges and len(charges) >= n_atoms:
                break
            i += 1
        if len(charges) == n_atoms:
            tables.append(charges)
    if not tables:
        raise ValueError(f"No complete MAYER POPULATION ANALYSIS QA table found in {path}")
    return np.asarray(tables[-1], dtype=float)


def fragment_indices(n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    if n_atoms % 2 != 0:
        raise ValueError(f"Cannot split odd atom count homodimer automatically: {n_atoms}")
    half = n_atoms // 2
    return np.arange(half), np.arange(half, n_atoms)


def electrostatic_energy_kcal(coords: np.ndarray, charges: np.ndarray) -> float:
    frag_a, frag_b = fragment_indices(len(charges))
    total = 0.0
    for i in frag_a:
        rij = np.linalg.norm(coords[frag_b] - coords[i], axis=1)
        total += float(np.sum(charges[i] * charges[frag_b] / rij))
    return COULOMB_KCAL_PER_MOL * total


def paths_from_row(row: pd.Series) -> tuple[Path, Path, Path]:
    xyz = Path(row["xyz_file"])
    if not xyz.is_absolute():
        xyz = ROOT / xyz
    idx = int(row["orca_index"])
    orca_dir = xyz.parent
    bs_out = orca_dir / f"orca_{idx}.out"
    triplet_out = orca_dir / "triplet" / f"orca_{idx}_triplet.out"
    return xyz, bs_out, triplet_out

def monomer_output_candidates(monomer_dir: Path) -> list[Path]:
    return [
        monomer_dir / "Monomer.out",
        monomer_dir / "monomer.out",
        *sorted(monomer_dir.glob("*.out")),
    ]


def first_existing_monomer_out(monomer_dir: Path) -> Path:
    seen: set[Path] = set()
    for candidate in monomer_output_candidates(monomer_dir):
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No monomer ORCA output found in {monomer_dir}")


def monomer_energy_sum_from_row(row: pd.Series) -> float:
    xyz = Path(row["xyz_file"])
    if not xyz.is_absolute():
        xyz = ROOT / xyz
    case_dir = xyz.parent.parent
    monomer_a = case_dir / "Monomer_A"
    monomer_b = case_dir / "Monomer_B"
    if monomer_a.is_dir() and monomer_b.is_dir():
        return final_single_point_energy(first_existing_monomer_out(monomer_a)) + final_single_point_energy(
            first_existing_monomer_out(monomer_b)
        )
    monomer = case_dir / "Monomer"
    if monomer.is_dir():
        return 2.0 * final_single_point_energy(first_existing_monomer_out(monomer))
    raise FileNotFoundError(f"No Monomer or Monomer_A/Monomer_B reference found in {case_dir}")



def compute_triplet_targets(row: pd.Series) -> TripletTargets | None:
    xyz, bs_out, triplet_out = paths_from_row(row)
    if not triplet_out.exists() or not bs_out.exists():
        return None
    if not terminated_normally(triplet_out) or not terminated_normally(bs_out):
        return None
    _, coords = read_xyz(xyz)
    bs_energy = final_single_point_energy(bs_out)
    triplet_energy = final_single_point_energy(triplet_out)
    monomer_sum = monomer_energy_sum_from_row(row)
    bs_interaction = (bs_energy - monomer_sum) * HARTREE_TO_KCAL
    triplet_interaction = (triplet_energy - monomer_sum) * HARTREE_TO_KCAL
    bs_q = final_mayer_mulliken_charges(bs_out, len(coords))
    triplet_q = final_mayer_mulliken_charges(triplet_out, len(coords))
    return TripletTargets(
        bs_energy_hartree=bs_energy,
        triplet_energy_hartree=triplet_energy,
        monomer_energy_sum_hartree=monomer_sum,
        bs_interaction_energy_kcal_mol=bs_interaction,
        triplet_interaction_energy_kcal_mol=triplet_interaction,
        bs_elst_kcal_mol=electrostatic_energy_kcal(coords, bs_q),
        triplet_elst_kcal_mol=electrostatic_energy_kcal(coords, triplet_q),
    )


def augment_table() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    rows = []
    failures: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        values: dict[str, Any] = {}
        try:
            tt = compute_triplet_targets(row)
        except Exception as exc:
            failures.append({"row": int(idx), "id": row.get("id", ""), "error": repr(exc)})
            tt = None
        if tt is not None:
            st_h = tt.triplet_energy_hartree - tt.bs_energy_hartree
            raw_signed = tt.bs_interaction_energy_kcal_mol - tt.triplet_interaction_energy_kcal_mol
            delta_elst = tt.bs_elst_kcal_mol - tt.triplet_elst_kcal_mol
            values.update(
                {
                    "bs_energy_hartree": tt.bs_energy_hartree,
                    "triplet_energy_hartree": tt.triplet_energy_hartree,
                    "monomer_energy_sum_hartree": tt.monomer_energy_sum_hartree,
                    "target_bs_interaction_energy_kcal_mol": tt.bs_interaction_energy_kcal_mol,
                    "target_triplet_interaction_energy_kcal_mol": tt.triplet_interaction_energy_kcal_mol,
                    "target_singlet_triplet_gap_kcal_mol": st_h * HARTREE_TO_KCAL,
                    "target_singlet_triplet_gap_ev": st_h * HARTREE_TO_EV,
                    "target_somo_somo_raw_signed_kcal_mol": raw_signed,
                    "target_elst_bs_kcal_mol": tt.bs_elst_kcal_mol,
                    "target_elst_triplet_kcal_mol": tt.triplet_elst_kcal_mol,
                    "target_delta_elst_bs_minus_triplet_kcal_mol": delta_elst,
                    "target_somo_somo_elst_corrected_signed_kcal_mol": raw_signed - delta_elst,
                    "target_somo_somo_elst_corrected_stabilization_kcal_mol": -(raw_signed - delta_elst),
                }
            )
        rows.append(values)
    aug = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    aug_path = OUT / "ALL_SYSTEMS_triplet_aware_augmented_targets.csv"
    aug.to_csv(aug_path, index=False)
    manifest = {
        "source_input": str(INPUT),
        "augmented_input": str(aug_path),
        "n_rows": int(len(aug)),
        "new_target_non_null_counts": {target: int(aug[target].notna().sum()) for target in NEW_TARGETS},
        "triplet_rows_by_system": {
            str(k): int(v)
            for k, v in aug.dropna(subset=["target_singlet_triplet_gap_kcal_mol"])["system"].value_counts().sort_index().items()
        },
        "failures": failures[:50],
        "n_failures_recorded": len(failures),
        "equations": {
            "E_ST_kcal_mol": "(E_triplet - E_BS) * 627.5094740631",
            "E_elst_kcal_mol": "332.06371 * sum_{i in fragA} sum_{j in fragB} q_i*q_j/r_ij",
            "raw_signed_somo_somo_kcal_mol": "E_int_BS - E_int_triplet, where E_int_state = (E_dimer_state - E_monomer_A - E_monomer_B) * 627.5094740631",
            "elst_corrected_signed_somo_somo_kcal_mol": "(E_int_BS - E_elst_BS) - (E_int_triplet - E_elst_triplet)",
        },
    }
    (OUT / "triplet_target_extraction_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return aug


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def extra_trees_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def mlp_models(random_state: int) -> dict[str, TransformedTargetRegressor]:
    specs = [
        ("mlp_64_32_16_a1e-4", (64, 32, 16), 1e-4),
        ("mlp_64_32_16_a1e-3", (64, 32, 16), 1e-3),
        ("mlp_128_64_32_a1e-4", (128, 64, 32), 1e-4),
        ("mlp_128_64_32_a1e-3", (128, 64, 32), 1e-3),
        ("mlp_64_64_32_16_a1e-4", (64, 64, 32, 16), 1e-4),
        ("mlp_32_16_a1e-4", (32, 16), 1e-4),
    ]
    out = {}
    for name, hidden, alpha in specs:
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
                        learning_rate_init=1e-3,
                        max_iter=900,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        tol=1e-5,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        out[name] = TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
    return out


def make_model_set(model_family: str, random_state: int) -> dict[str, Any]:
    if model_family == "extra_trees":
        return {"extra_trees": extra_trees_model(random_state)}
    if model_family == "mlp":
        return mlp_models(random_state)
    raise ValueError(model_family)


def plot_predictions(pred: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    plot_dir = out_dir / "correlation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, frame in pred.groupby("target"):
        frame = frame.copy()
        y = frame["true"].to_numpy()
        yp = frame["cv_prediction"].to_numpy()
        metrics = metric_dict(y, yp)
        rows.append({"target": target, "n": int(len(frame)), **metrics})
        lower = float(min(np.min(y), np.min(yp)))
        upper = float(max(np.max(y), np.max(yp)))
        pad = 0.06 * (upper - lower) if upper > lower else 1.0
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for system, group in frame.groupby("system"):
            ax.scatter(
                group["true"],
                group["cv_prediction"],
                s=22,
                color=COLORS.get(system, "#666666"),
                edgecolor="white",
                linewidth=0.25,
                alpha=0.82,
                label=system,
            )
        ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], color="#444444", lw=1.0)
        ax.set_xlim(lower - pad, upper + pad)
        ax.set_ylim(lower - pad, upper + pad)
        label = TARGET_LABELS.get(target, target)
        ax.set_xlabel(f"DFT {label}")
        ax.set_ylabel(f"{prefix} predicted {label}")
        ax.text(
            1.03,
            0.98,
            f"$R^2$ = {metrics['r2']:.2f}\nMAE = {metrics['mae']:.2f}\nRMSE = {metrics['rmse']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        ax.grid(color="#E8EAED", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(1.03, 0.02), borderaxespad=0)
        fig.subplots_adjust(left=0.14, right=0.68, bottom=0.16, top=0.94)
        safe = target.replace("target_", "")
        fig.savefig(plot_dir / f"{safe}_predicted_vs_dft.png", dpi=500)
        fig.savefig(plot_dir / f"{safe}_predicted_vs_dft.pdf")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(plot_dir / "plot_metrics.csv", index=False)


def train_family(df: pd.DataFrame, model_family: str, random_state: int = 42) -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    out_dir = OUT / model_family
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    feature_groups = {name: [f for f in feats if f in df.columns] for name, feats in FEATURE_GROUPS.items()}
    models = make_model_set(model_family, random_state)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    summary_rows = []
    pred_rows = []
    individual_rows = []
    bundle: dict[str, Any] = {"models": {}, "feature_groups": feature_groups, "targets": NEW_TARGETS}

    for target in NEW_TARGETS:
        keep = df[target].notna()
        if int(keep.sum()) < 20:
            continue
        y = df.loc[keep, target].astype(float)
        best: dict[str, Any] | None = None
        best_mae = float("inf")
        for group_name, features in feature_groups.items():
            if not features:
                continue
            x = df.loc[keep, features]
            for model_name, estimator in models.items():
                pred = cross_val_predict(clone(estimator), x, y, cv=cv)
                metrics = metric_dict(y.to_numpy(), pred)
                row = {
                    "target": target,
                    "model_family": model_family,
                    "model": model_name,
                    "feature_group": group_name,
                    "n_samples": int(len(y)),
                    "n_features": int(len(features)),
                    **metrics,
                }
                summary_rows.append(row)
                if metrics["mae"] < best_mae:
                    best_mae = metrics["mae"]
                    best = {"row": row, "pred": pred, "features": features, "estimator": estimator}
        assert best is not None
        final = clone(best["estimator"])
        final.fit(df.loc[keep, best["features"]], y)
        bundle["models"][target] = {
            "feature_group": best["row"]["feature_group"],
            "model_name": best["row"]["model"],
            "features": best["features"],
            "cv_metrics": best["row"],
            "model": final,
        }
        for idx, pred_value in zip(df.loc[keep].index, best["pred"]):
            pred_rows.append(
                {
                    "id": df.loc[idx, "id"],
                    "system": df.loc[idx, "system"],
                    "orca_index": int(df.loc[idx, "orca_index"]),
                    "target": target,
                    "model_family": model_family,
                    "model": best["row"]["model"],
                    "feature_group": best["row"]["feature_group"],
                    "true": float(df.loc[idx, target]),
                    "cv_prediction": float(pred_value),
                    "residual": float(df.loc[idx, target] - pred_value),
                }
            )
        best_pred = pd.DataFrame([r for r in pred_rows if r["target"] == target])
        for system, group in best_pred.groupby("system"):
            if len(group) < 5:
                continue
            individual_rows.append(
                {
                    "system": system,
                    "target": target,
                    "model_family": model_family,
                    "model": best["row"]["model"],
                    "feature_group": best["row"]["feature_group"],
                    "n_samples": int(len(group)),
                    **metric_dict(group["true"].to_numpy(), group["cv_prediction"].to_numpy()),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["target", "mae", "rmse"])
    predictions = pd.DataFrame(pred_rows)
    summary.to_csv(out_dir / f"{model_family}_cv_summary.csv", index=False)
    predictions.to_csv(out_dir / f"{model_family}_cv_predictions.csv", index=False)
    pd.DataFrame(individual_rows).to_csv(out_dir / f"{model_family}_individual_system_metrics.csv", index=False)
    with (model_dir / f"{model_family}_triplet_models.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)
    (out_dir / f"{model_family}_metadata.json").write_text(
        json.dumps(
            {
                "input": str(OUT / "ALL_SYSTEMS_triplet_aware_augmented_targets.csv"),
                "targets": NEW_TARGETS,
                "best_models": {
                    target: {
                        "model": bundle["models"][target]["model_name"],
                        "feature_group": bundle["models"][target]["feature_group"],
                        "cv_metrics": bundle["models"][target]["cv_metrics"],
                    }
                    for target in bundle["models"]
                },
            },
            indent=2,
        )
    )
    plot_predictions(predictions, out_dir, "Extra Trees" if model_family == "extra_trees" else "MLP")
    print(f"\nBest {model_family} models:")
    if not summary.empty:
        print(summary.groupby("target", as_index=False).first()[["target", "feature_group", "model", "n_samples", "mae", "rmse", "r2"]].round(4).to_string(index=False))


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    df = augment_table()
    train_family(df, "extra_trees")
    train_family(df, "mlp")
    print(f"\nTriplet-aware workflow complete: {OUT}")


if __name__ == "__main__":
    main()
