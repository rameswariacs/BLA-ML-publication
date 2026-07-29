#!/usr/bin/env python3
"""Unified 19-descriptor homo+cross dimer workflow without S2 filtering.

All flat dimers are represented by the same 19 descriptors:
15 core stacking/contact/BLA descriptors plus 4 fragment-asymmetry/contact-imbalance
descriptors.  Triplet-derived targets are added only for rows where the triplet
output exists and terminated normally. The final <S^2> value is parsed directly
from each original broken-symmetry ORCA output and retained as a diagnostic.
No structures are removed by an <S^2> criterion in this no-filter workflow.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
import warnings
from collections import Counter
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/rb1820/BLA-ML")
RUNSCRIPTS = ROOT / "RunScripts_plus_CPBP"
if str(RUNSCRIPTS) not in sys.path:
    sys.path.insert(0, str(RUNSCRIPTS))

import triplet_aware_new_targets_pipeline as triplet_core

OUT = ROOT / "Unified_19Descriptor_NoS2Filter_FODFT_plus_CPBP"
RANDOM_STATE = 42
BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 2000
BS_S2_MIN = 0.10
BS_S2_MAX = 1.80
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
COULOMB_KCAL_PER_MOL = 332.06371

CORE15 = [
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

ASYM4 = [
    "pi_projected_area_ratio_min_over_max",
    "delta_overlap_fraction",
    "delta_contact_atoms_3p4",
    "delta_contact_region_bla",
]

FEATURES19 = CORE15 + ASYM4
FEATURE_GROUPS = {"core15_plus_asymmetry19": FEATURES19}

TARGETS = [
    "target_somo_stabilization_mev",
    "target_interaction_energy_kcal_mol",
    "target_frontier_gap_ev",
    "target_somo_somo_coupling_proxy_mev",
    "target_singlet_triplet_gap_kcal_mol",
    "target_somo_somo_raw_signed_kcal_mol",
    "target_somo_somo_elst_corrected_signed_kcal_mol",
]

TARGET_LABELS = {
    "target_somo_stabilization_mev": "SOMO stabilization (meV)",
    "target_interaction_energy_kcal_mol": "Interaction energy (kcal/mol)",
    "target_frontier_gap_ev": "Spin-min frontier gap (eV)",
    "target_somo_somo_coupling_proxy_mev": "SOMO splitting / 2 (meV)",
    "target_singlet_triplet_gap_kcal_mol": "E_ST (kcal/mol)",
    "target_somo_somo_raw_signed_kcal_mol": "Raw E_SOMO-SOMO (kcal/mol)",
    "target_somo_somo_elst_corrected_signed_kcal_mol": "Electrostatic-corrected E_SOMO-SOMO (kcal/mol)",
}

COLORS = {
    "phenalenyl": "#0072B2",
    "olympicenyl": "#D55E00",
    "fluorenyl": "#009E73",
    "CPBP": "#CC79A7",
    "phenalenyl_olympicenyl": "#6A3D9A",
}
SYSTEM_ORDER = ["phenalenyl", "olympicenyl", "fluorenyl", "CPBP", "phenalenyl_olympicenyl"]


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else math.nan,
    }


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
    return triplet_core.final_single_point_energy(path)


def final_spin_squared(path: Path) -> float:
    """Read the last printed <S^2> value from a completed ORCA output."""
    pattern = re.compile(r"Expectation value of <S\*\*2>\s*:\s*([-+0-9.Ee]+)")
    matches = pattern.findall(path.read_text(errors="ignore"))
    if not matches:
        raise ValueError(f"No <S^2> value found in {path}")
    return float(matches[-1])


def terminated_normally(path: Path) -> bool:
    return triplet_core.terminated_normally(path)


def final_mayer_mulliken_charges(path: Path, n_atoms: int) -> np.ndarray:
    return triplet_core.final_mayer_mulliken_charges(path, n_atoms)


def monomer_xyz_symbols(monomer_dir: Path) -> list[str]:
    for name in ["Monomer.xyz", "monomer.xyz", "Monomer_trj.xyz"]:
        p = monomer_dir / name
        if p.exists():
            symbols, _ = read_xyz(p)
            return symbols
    raise FileNotFoundError(f"No monomer xyz found in {monomer_dir}")


def fragment_indices_for_case(case_dir: Path, dimer_symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = len(dimer_symbols)
    monomer_a = case_dir / "Monomer_A"
    monomer_b = case_dir / "Monomer_B"
    if monomer_a.is_dir() and monomer_b.is_dir():
        symbols_a = monomer_xyz_symbols(monomer_a)
        symbols_b = monomer_xyz_symbols(monomer_b)
        n_a = len(symbols_a)
        n_b = len(symbols_b)
        if n_a + n_b != n_atoms:
            raise ValueError(f"Monomer_A/Monomer_B atom counts {n_a}+{n_b} do not match dimer atom count {n_atoms}")
        composition_a = Counter(symbols_a)
        composition_b = Counter(symbols_b)
        if Counter(dimer_symbols[:n_a]) == composition_a and Counter(dimer_symbols[n_a:]) == composition_b:
            return np.arange(n_a), np.arange(n_a, n_a + n_b)
        if Counter(dimer_symbols[:n_b]) == composition_b and Counter(dimer_symbols[n_b:]) == composition_a:
            return np.arange(n_b, n_b + n_a), np.arange(n_b)
        raise ValueError(
            f"Dimer atom order in {case_dir} matches neither Monomer_A--Monomer_B "
            "nor Monomer_B--Monomer_A"
        )
    if n_atoms % 2 != 0:
        raise ValueError(f"Cannot split odd atom count homodimer automatically: {n_atoms}")
    half = n_atoms // 2
    return np.arange(half), np.arange(half, n_atoms)


def electrostatic_energy_kcal(
    symbols: list[str], coords: np.ndarray, charges: np.ndarray, case_dir: Path
) -> float:
    frag_a, frag_b = fragment_indices_for_case(case_dir, symbols)
    total = 0.0
    for i in frag_a:
        rij = np.linalg.norm(coords[frag_b] - coords[i], axis=1)
        total += float(np.sum(charges[i] * charges[frag_b] / rij))
    return COULOMB_KCAL_PER_MOL * total


def first_existing_monomer_out(monomer_dir: Path) -> Path:
    return triplet_core.first_existing_monomer_out(monomer_dir)


def monomer_energy_sum(case_dir: Path) -> float:
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


def compute_optional_triplet_targets(row: pd.Series) -> tuple[dict[str, Any], dict[str, Any] | None]:
    xyz = Path(row["xyz_file"])
    if not xyz.is_absolute():
        xyz = ROOT / xyz
    idx = int(row["orca_index"])
    case_dir = xyz.parent.parent
    bs_out = xyz.parent / f"orca_{idx}.out"
    triplet_dir = xyz.parent / "triplet"
    triplet_candidates = [
        triplet_dir / f"orca_{idx}_triplet.out",
        triplet_dir / f"orca_{idx}.out",
    ]
    triplet_out = next((path for path in triplet_candidates if path.exists()), triplet_candidates[0])
    if not triplet_out.exists():
        return {
        }, {
            "id": row["id"],
            "reason": "missing_triplet_output",
            "triplet_candidates": ";".join(str(path) for path in triplet_candidates),
        }
    if not bs_out.exists():
        return {}, {"id": row["id"], "reason": "missing_bs_output", "bs_out": str(bs_out)}
    if not terminated_normally(triplet_out):
        return {}, {"id": row["id"], "reason": "triplet_not_normal", "triplet_out": str(triplet_out)}
    if not terminated_normally(bs_out):
        return {}, {"id": row["id"], "reason": "bs_not_normal", "bs_out": str(bs_out)}

    symbols, coords = read_xyz(xyz)
    bs_energy = final_single_point_energy(bs_out)
    triplet_energy = final_single_point_energy(triplet_out)
    mon_sum = monomer_energy_sum(case_dir)
    bs_int = (bs_energy - mon_sum) * HARTREE_TO_KCAL
    triplet_int = (triplet_energy - mon_sum) * HARTREE_TO_KCAL
    bs_q = final_mayer_mulliken_charges(bs_out, len(coords))
    triplet_q = final_mayer_mulliken_charges(triplet_out, len(coords))
    bs_elst = electrostatic_energy_kcal(symbols, coords, bs_q, case_dir)
    triplet_elst = electrostatic_energy_kcal(symbols, coords, triplet_q, case_dir)
    st_h = triplet_energy - bs_energy
    raw = bs_int - triplet_int
    delta_elst = bs_elst - triplet_elst
    return (
        {
            "bs_spin_s2_audit": final_spin_squared(bs_out),
            "bs_energy_hartree": bs_energy,
            "triplet_energy_hartree": triplet_energy,
            "monomer_energy_sum_hartree": mon_sum,
            "target_bs_interaction_energy_kcal_mol": bs_int,
            "target_triplet_interaction_energy_kcal_mol": triplet_int,
            "target_singlet_triplet_gap_kcal_mol": st_h * HARTREE_TO_KCAL,
            "target_singlet_triplet_gap_ev": st_h * HARTREE_TO_EV,
            "target_somo_somo_raw_signed_kcal_mol": raw,
            "target_elst_bs_kcal_mol": bs_elst,
            "target_elst_triplet_kcal_mol": triplet_elst,
            "target_delta_elst_bs_minus_triplet_kcal_mol": delta_elst,
            "target_somo_somo_elst_corrected_signed_kcal_mol": raw - delta_elst,
            "target_somo_somo_elst_corrected_stabilization_kcal_mol": -(raw - delta_elst),
        },
        None,
    )


def load_homodimers() -> pd.DataFrame:
    base = pd.read_csv(triplet_core.INPUT)
    asym_path = ROOT / "Unified_7Target_StructuralAsymmetry_plus_CPBP" / "ALL_SYSTEMS_7target_structural_asymmetry.csv"
    asym = pd.read_csv(asym_path)
    add_cols = [c for c in ASYM4 if c in asym.columns]
    base = base.drop(columns=[c for c in add_cols if c in base.columns], errors="ignore")
    merged = base.merge(asym[["id", *add_cols]], on="id", how="left", validate="one_to_one")
    merged["dimer_class"] = "homodimer"
    merged["dataset"] = merged["system"]
    return merged


def load_cross_dimer() -> pd.DataFrame:
    desc_path = (
        ROOT
        / "Unified_7Target_StructuralAsymmetry_plus_CPBP"
        / "CrossDimer_BlindTest_phenalenyl_olympicenyl"
        / "blind_xyz_descriptors_with_structural_asymmetry.csv"
    )
    target_path = (
        ROOT
        / "Unified_7Target_StructuralAsymmetry_plus_CPBP"
        / "CrossDimer_BlindTest_phenalenyl_olympicenyl"
        / "dft_targets_from_bs_orca.csv"
    )
    desc = pd.read_csv(desc_path)
    targets = pd.read_csv(target_path)
    df = desc.merge(targets, on=["id", "cross_system", "orca_index"], how="left", validate="one_to_one")
    df["system"] = df["cross_system"]
    df["dataset"] = df["cross_system"]
    df["dimer_class"] = "crossdimer"
    return df


def assemble_dataset() -> pd.DataFrame:
    homo = load_homodimers()
    cross = load_cross_dimer()
    common_cols = list(dict.fromkeys([*homo.columns, *cross.columns]))
    df = pd.concat([homo.reindex(columns=common_cols), cross.reindex(columns=common_cols)], ignore_index=True)
    if "target_mean_somo_stabilization_mev" in df.columns:
        if "target_somo_stabilization_mev" not in df.columns:
            df["target_somo_stabilization_mev"] = df["target_mean_somo_stabilization_mev"]
        else:
            df["target_somo_stabilization_mev"] = df["target_somo_stabilization_mev"].fillna(
                df["target_mean_somo_stabilization_mev"]
            )

    triplet_values = []
    triplet_failures = []
    for _, row in df.iterrows():
        values, failure = compute_optional_triplet_targets(row)
        triplet_values.append(values)
        if failure is not None:
            failure["system"] = row.get("system")
            failure["dimer_class"] = row.get("dimer_class")
            triplet_failures.append(failure)
    triplet_df = pd.DataFrame(triplet_values)
    for col in triplet_df.columns:
        df[col] = triplet_df[col]
    if "bs_spin_s2_audit" not in df.columns or df["bs_spin_s2_audit"].isna().any():
        missing = df.loc[df.get("bs_spin_s2_audit", pd.Series(index=df.index, dtype=float)).isna(), "id"].tolist()
        raise ValueError(f"Missing direct BS <S^2> audit values for {len(missing)} structures: {missing[:10]}")
    # Use one consistent source for every homo- and cross-dimer: the final value
    # printed in the original broken-symmetry ORCA output.
    df["target_spin_s2"] = df["bs_spin_s2_audit"].astype(float)

    low = df["target_spin_s2"] < BS_S2_MIN
    high = df["target_spin_s2"] > BS_S2_MAX
    # No-S2-filter workflow: keep every cleaned, modeling-eligible structure.
    # The historical thresholds remain as diagnostics only.
    df["bs_s2_filter_status"] = np.select(
        [low, high],
        ["LOW_S2_INCLUDED_NO_FILTER", "HIGH_S2_INCLUDED_NO_FILTER"],
        default="INCLUDED_NO_FILTER",
    )
    df["bs_s2_filter_reason"] = np.select(
        [low, high],
        [
            f"diagnostic only: final BS <S^2> below {BS_S2_MIN:.2f}; retained in no-S2-filter workflow",
            f"diagnostic only: final BS <S^2> above {BS_S2_MAX:.2f}; retained in no-S2-filter workflow",
        ],
        default="retained in no-S2-filter workflow",
    )
    df["bs_s2_diagnostic_min"] = BS_S2_MIN
    df["bs_s2_diagnostic_max"] = BS_S2_MAX
    df["bs_s2_filter_applied"] = False

    OUT.mkdir(parents=True, exist_ok=True)
    failure_columns = [
        "id",
        "system",
        "dimer_class",
        "reason",
        "triplet_out",
        "bs_out",
        "triplet_candidates",
    ]
    pd.DataFrame(triplet_failures, columns=failure_columns).to_csv(
        OUT / "triplet_target_missing_or_failed_rows.csv", index=False
    )
    audit_cols = [
        "id",
        "system",
        "dimer_class",
        "orca_index",
        "xyz_file",
        "target_spin_s2",
        "bs_s2_filter_status",
        "bs_s2_filter_reason",
        "contact_min_cc_distance",
        "mean_top10_closest_c_distances",
        "n_interfragment_c_contacts_3p4",
        "target_bs_interaction_energy_kcal_mol",
        "target_triplet_interaction_energy_kcal_mol",
        "target_singlet_triplet_gap_kcal_mol",
        "target_somo_somo_raw_signed_kcal_mol",
        "target_somo_somo_elst_corrected_signed_kcal_mol",
    ]
    df[audit_cols].to_csv(OUT / "BS_S2_ALL_STRUCTURE_AUDIT_NO_FILTER.csv", index=False)
    df.loc[low | high, audit_cols].to_csv(
        OUT / "BS_S2_LOW_OR_HIGH_DIAGNOSTIC_STRUCTURES_INCLUDED.csv", index=False
    )
    filtered = df.copy().reset_index(drop=True)
    filtered.to_csv(OUT / "ALL_DIMERS_19descriptors_7targets_NO_S2_FILTER.csv", index=False)
    # Compatibility copy for archived helper scripts; this no-filter workflow
    # uses every row in the file despite the historical filename.
    filtered.to_csv(OUT / "ALL_DIMERS_19descriptors_7targets_BS_filtered.csv", index=False)

    counts = df.groupby(["system", "bs_s2_filter_status"], dropna=False).size().unstack(fill_value=0)
    flagged = df.loc[low | high, audit_cols].sort_values(["system", "orca_index"])
    report = [
        "# Broken-Symmetry <S^2> Diagnostic Report: No-Filter Workflow",
        "",
        "## Rule used for this run",
        "",
        "- No structure is removed by an <S^2> criterion in this workflow.",
        f"- The historical diagnostic window `{BS_S2_MIN:.2f} <= final <S^2> <= {BS_S2_MAX:.2f}` is retained only to label low- or high-<S^2> structures.",
        "- The value is the last `Expectation value of <S**2>` printed in the original GuessMix broken-symmetry ORCA output.",
        "- Model fitting, cross-validation, feature analysis, and train/validation/test splitting use all cleaned modeling-eligible structures.",
        "",
        "## Dataset accounting",
        "",
        f"- Structures before diagnostic labeling: **{len(df)}**",
        f"- Structures used for modeling: **{len(filtered)}**",
        f"- Low-<S^2> structures included: **{int(low.sum())}**",
        f"- High-<S^2> structures included: **{int(high.sum())}**",
        "",
        "| System | Included | Low-<S^2> included | High-<S^2> included |",
        "|---|---:|---:|---:|",
    ]
    for system, row in counts.iterrows():
        included = int(row.get("INCLUDED_NO_FILTER", 0))
        low_included = int(row.get("LOW_S2_INCLUDED_NO_FILTER", 0))
        high_included = int(row.get("HIGH_S2_INCLUDED_NO_FILTER", 0))
        report.append(f"| {system} | {included} | {low_included} | {high_included} |")
    report.extend(
        [
            "",
            "## Low/high-<S^2> diagnostic structures included",
            "",
            "| Structure | System | <S^2> | Minimum C...C distance (A) | Raw E_SOMO-SOMO (kcal/mol) | Reason |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for _, row in flagged.iterrows():
        report.append(
            f"| {row['id']} | {row['system']} | {row['target_spin_s2']:.6f} | "
            f"{row['contact_min_cc_distance']:.4f} | {row['target_somo_somo_raw_signed_kcal_mol']:.4f} | "
            f"{row['bs_s2_filter_reason']} |"
        )
    report.extend(
        [
            "",
            "## Reproducibility files",
            "",
            "- `BS_S2_ALL_STRUCTURE_AUDIT_NO_FILTER.csv`: every structure and its diagnostic label.",
            "- `BS_S2_LOW_OR_HIGH_DIAGNOSTIC_STRUCTURES_INCLUDED.csv`: low/high-<S^2> structures retained in the model input.",
            "- `ALL_DIMERS_19descriptors_7targets_NO_S2_FILTER.csv`: exact modeling input.",
            "- `ALL_DIMERS_19descriptors_7targets_BS_filtered.csv`: compatibility copy of the same no-filter modeling input.",
            "",
        ]
    )
    (OUT / "BS_S2_NO_FILTER_DIAGNOSTIC_REPORT.md").write_text("\n".join(report))
    return filtered


def extra_trees_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )


def mlp_models() -> dict[str, TransformedTargetRegressor]:
    specs = [
        ("mlp_64_32_16_a1e-4", (64, 32, 16), 1e-4),
        ("mlp_64_32_16_a1e-3", (64, 32, 16), 1e-3),
        ("mlp_128_64_32_a1e-4", (128, 64, 32), 1e-4),
        ("mlp_128_64_32_a1e-3", (128, 64, 32), 1e-3),
        ("mlp_64_64_32_16_a1e-4", (64, 64, 32, 16), 1e-4),
        ("mlp_32_16_a1e-4", (32, 16), 1e-4),
    ]
    models: dict[str, TransformedTargetRegressor] = {}
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
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        models[name] = TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
    return models


def models_for_family(family: str) -> dict[str, Any]:
    if family == "extra_trees":
        return {"extra_trees": extra_trees_model()}
    if family == "mlp":
        return mlp_models()
    raise ValueError(family)


def plot_correlation(pred: pd.DataFrame, out_dir: Path, label_prefix: str, pred_col: str, split_label: str = "5-fold CV") -> None:
    plot_dir = out_dir / "correlation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, frame in pred.groupby("target"):
        y = frame["true"].to_numpy(dtype=float)
        yp = frame[pred_col].to_numpy(dtype=float)
        metrics = metric_dict(y, yp)
        rows.append({"target": target, "n": int(len(frame)), **metrics})
        lower = float(min(np.min(y), np.min(yp)))
        upper = float(max(np.max(y), np.max(yp)))
        pad = 0.06 * (upper - lower) if upper > lower else 1.0
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        grouped = {system: group for system, group in frame.groupby("system")}
        for system in SYSTEM_ORDER + sorted(set(grouped) - set(SYSTEM_ORDER)):
            group = grouped.get(system)
            if group is None:
                continue
            ax.scatter(
                group["true"],
                group[pred_col],
                s=20,
                alpha=0.82,
                color=COLORS.get(system, "#666666"),
                edgecolor="white",
                linewidth=0.25,
                label=system,
            )
        ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], color="#444444", lw=1.0)
        ax.set_xlim(lower - pad, upper + pad)
        ax.set_ylim(lower - pad, upper + pad)
        label = TARGET_LABELS.get(target, target)
        ax.set_xlabel(f"DFT {label}")
        ax.set_ylabel(f"{label_prefix} predicted {label}")
        ax.text(
            1.03,
            0.98,
            f"{split_label}\n$R^2$ = {metrics['r2']:.2f}\nMAE = {metrics['mae']:.2f}\nRMSE = {metrics['rmse']:.2f}",
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
        suffix = "test" if split_label.lower().startswith("test") else "cv"
        fig.savefig(plot_dir / f"{safe}_{suffix}_predicted_vs_dft.png", dpi=500)
        fig.savefig(plot_dir / f"{safe}_{suffix}_predicted_vs_dft.pdf")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(plot_dir / f"{split_label.lower().replace(' ', '_').replace('-', '')}_plot_metrics.csv", index=False)


def extract_feature_importance(
    family: str,
    target: str,
    fitted: Any,
    features: list[str],
    x: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    if family == "extra_trees":
        importances = fitted.named_steps["model"].feature_importances_
        return pd.DataFrame(
            {
                "target": target,
                "model_family": family,
                "importance_type": "extra_trees_impurity",
                "feature": features,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)
    result = permutation_importance(
        fitted,
        x,
        y,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "target": target,
            "model_family": family,
            "importance_type": "permutation_neg_mae_drop",
            "feature": features,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance", ascending=False)


def plot_feature_importance(fi: pd.DataFrame, out_dir: Path, family: str) -> None:
    plot_dir = out_dir / "feature_importance_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for target, frame in fi.groupby("target"):
        frame = frame.sort_values("importance", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.barh(frame["feature"], frame["importance"], color="#4C78A8")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        ax.set_title(TARGET_LABELS.get(target, target), fontsize=11)
        ax.grid(axis="x", color="#E8EAED", linewidth=0.7)
        fig.tight_layout()
        safe = target.replace("target_", "")
        fig.savefig(plot_dir / f"{safe}_{family}_feature_importance.png", dpi=500)
        fig.savefig(plot_dir / f"{safe}_{family}_feature_importance.pdf")
        plt.close(fig)


def train_cv_family(df: pd.DataFrame, family: str) -> None:
    out_dir = OUT / family
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    features = [f for f in FEATURES19 if f in df.columns]
    if len(features) != len(FEATURES19):
        raise ValueError(f"Missing 19-descriptor columns: {sorted(set(FEATURES19) - set(features))}")
    models = models_for_family(family)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    summary_rows = []
    pred_rows = []
    individual_rows = []
    fi_rows = []
    bundle: dict[str, Any] = {"models": {}, "feature_groups": FEATURE_GROUPS, "targets": TARGETS}

    for target in TARGETS:
        if target not in df.columns:
            continue
        keep = df[target].notna()
        if int(keep.sum()) < 20 or df.loc[keep, target].nunique() < 2:
            continue
        y = df.loc[keep, target].astype(float)
        x = df.loc[keep, features]
        best: dict[str, Any] | None = None
        best_mae = float("inf")
        for model_name, estimator in models.items():
            pred = cross_val_predict(clone(estimator), x, y, cv=cv)
            metrics = metric_dict(y.to_numpy(), pred)
            row = {
                "target": target,
                "model_family": family,
                "model": model_name,
                "feature_group": "core15_plus_asymmetry19",
                "n_samples": int(len(y)),
                "n_features": int(len(features)),
                **metrics,
            }
            summary_rows.append(row)
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best = {"row": row, "pred": pred, "estimator": estimator}
        if best is None:
            continue
        final = clone(best["estimator"])
        final.fit(x, y)
        bundle["models"][target] = {
            "feature_group": "core15_plus_asymmetry19",
            "model_name": best["row"]["model"],
            "features": features,
            "cv_metrics": best["row"],
            "model": final,
        }
        for idx, pred_value in zip(df.loc[keep].index, best["pred"]):
            pred_rows.append(
                {
                    "id": df.loc[idx, "id"],
                    "system": df.loc[idx, "system"],
                    "dimer_class": df.loc[idx, "dimer_class"],
                    "orca_index": int(df.loc[idx, "orca_index"]),
                    "target": target,
                    "model_family": family,
                    "model": best["row"]["model"],
                    "feature_group": "core15_plus_asymmetry19",
                    "true": float(df.loc[idx, target]),
                    "cv_prediction": float(pred_value),
                    "residual": float(df.loc[idx, target] - pred_value),
                }
            )
        fi = extract_feature_importance(family, target, final, features, x, y)
        fi["best_model"] = best["row"]["model"]
        fi["best_feature_group"] = "core15_plus_asymmetry19"
        fi_rows.extend(fi.to_dict("records"))

    summary = pd.DataFrame(summary_rows).sort_values(["target", "mae", "rmse"])
    predictions = pd.DataFrame(pred_rows)
    fi_table = pd.DataFrame(fi_rows)
    summary.to_csv(out_dir / f"{family}_cv_summary.csv", index=False)
    predictions.to_csv(out_dir / f"{family}_cv_predictions.csv", index=False)
    fi_table.to_csv(out_dir / f"{family}_feature_importance.csv", index=False)
    if not predictions.empty:
        for (system, target), group in predictions.groupby(["system", "target"]):
            if len(group) < 5 or group["true"].nunique() < 2:
                continue
            first = group.iloc[0]
            individual_rows.append(
                {
                    "system": system,
                    "target": target,
                    "model_family": family,
                    "model": first["model"],
                    "feature_group": first["feature_group"],
                    "n_samples": int(len(group)),
                    **metric_dict(group["true"].to_numpy(), group["cv_prediction"].to_numpy()),
                }
            )
    pd.DataFrame(individual_rows).sort_values(["target", "system"]).to_csv(
        out_dir / f"{family}_individual_system_metrics.csv", index=False
    )
    with (model_dir / f"{family}_seven_target_models.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)
    (out_dir / f"{family}_metadata.json").write_text(
        json.dumps(
            {
                "input": str(OUT / "ALL_DIMERS_19descriptors_7targets_NO_S2_FILTER.csv"),
                "targets": TARGETS,
                "features": features,
                "feature_group": "core15_plus_asymmetry19",
                "best_models": {
                    target: {
                        "model": item["model_name"],
                        "features": item["features"],
                        "cv_metrics": item["cv_metrics"],
                    }
                    for target, item in bundle["models"].items()
                },
            },
            indent=2,
        )
    )
    plot_correlation(predictions, out_dir, "Extra Trees" if family == "extra_trees" else "MLP", "cv_prediction")
    plot_feature_importance(fi_table, out_dir, family)
    print(f"\nBest {family} 5-fold CV models:")
    if not summary.empty:
        print(summary.groupby("target", as_index=False).first()[["target", "model", "n_samples", "mae", "rmse", "r2"]].round(4).to_string(index=False))


def bootstrap_ci(y_true: np.ndarray, pred: np.ndarray, seed: int, n_boot: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = {"mae": [], "rmse": [], "r2": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = metric_dict(y_true[idx], pred[idx])
        for key in samples:
            samples[key].append(m[key])
    out: dict[str, float] = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=float)
        out[f"{key}_ci95_low"] = float(np.nanpercentile(arr, 2.5)) if len(arr) else math.nan
        out[f"{key}_ci95_high"] = float(np.nanpercentile(arr, 97.5)) if len(arr) else math.nan
    return out


def make_split(df: pd.DataFrame, seed: int) -> pd.Series:
    idx = np.arange(len(df))
    stratify = df["system"] if df["system"].value_counts().min() >= 2 else None
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=seed, shuffle=True, stratify=stratify)
    temp = df.iloc[temp_idx]
    temp_stratify = temp["system"] if temp["system"].value_counts().min() >= 2 else None
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed + 1, shuffle=True, stratify=temp_stratify)
    split = pd.Series(index=df.index, dtype=object)
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "validation"
    split.iloc[test_idx] = "test"
    return split


def train_validate_test_family(df: pd.DataFrame, family: str) -> None:
    out_dir = OUT / "train_validate_test" / family
    out_dir.mkdir(parents=True, exist_ok=True)
    features = [f for f in FEATURES19 if f in df.columns]
    models = models_for_family(family)
    rows = []
    pred_rows = []
    bundle: dict[str, Any] = {"models": {}, "feature_group": "core15_plus_asymmetry19", "features": features}

    for target in TARGETS:
        if target not in df.columns:
            continue
        valid = df[target].notna()
        if int(valid.sum()) < 40 or df.loc[valid, target].nunique() < 2:
            continue
        sub = df.loc[valid].copy().reset_index(drop=False).rename(columns={"index": "source_index"})
        sub["split"] = make_split(sub, RANDOM_STATE)
        train = sub["split"] == "train"
        val = sub["split"] == "validation"
        test = sub["split"] == "test"
        if train.sum() < 20 or val.sum() < 10 or test.sum() < 10:
            continue

        best = None
        best_val_mae = float("inf")
        for model_name, estimator in models.items():
            fit = clone(estimator)
            fit.fit(sub.loc[train, features], sub.loc[train, target].to_numpy())
            val_pred = fit.predict(sub.loc[val, features])
            val_metrics = metric_dict(sub.loc[val, target].to_numpy(), val_pred)
            rows.append(
                {
                    "family": family,
                    "target": target,
                    "stage": "validation_selection",
                    "feature_group": "core15_plus_asymmetry19",
                    "model": model_name,
                    "n_train": int(train.sum()),
                    "n_validation": int(val.sum()),
                    "n_test": int(test.sum()),
                    "n_features": len(features),
                    **val_metrics,
                }
            )
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best = {"model_name": model_name, "estimator": estimator, "validation_metrics": val_metrics}
        assert best is not None
        selected = clone(best["estimator"])
        train_val = sub["split"].isin(["train", "validation"])
        selected.fit(sub.loc[train_val, features], sub.loc[train_val, target].to_numpy())

        for split_name, mask in [("train", train), ("validation", val), ("test", test)]:
            y = sub.loc[mask, target].to_numpy(dtype=float)
            pred = selected.predict(sub.loc[mask, features])
            metrics = metric_dict(y, pred)
            row = {
                "family": family,
                "target": target,
                "stage": f"selected_{split_name}",
                "feature_group": "core15_plus_asymmetry19",
                "model": best["model_name"],
                "n": int(mask.sum()),
                "n_features": len(features),
                **metrics,
            }
            if split_name == "test":
                row.update(bootstrap_ci(y, pred, BOOTSTRAP_SEED, N_BOOTSTRAP))
            rows.append(row)
            for _, r in sub.loc[mask].assign(prediction=pred).iterrows():
                pred_rows.append(
                    {
                        "family": family,
                        "target": target,
                        "split": split_name,
                        "id": r["id"],
                        "system": r["system"],
                        "dimer_class": r["dimer_class"],
                        "feature_group": "core15_plus_asymmetry19",
                        "model": best["model_name"],
                        "true": float(r[target]),
                        "prediction": float(r["prediction"]),
                        "residual": float(r[target] - r["prediction"]),
                    }
                )
        bundle["models"][target] = {
            "model_name": best["model_name"],
            "feature_group": "core15_plus_asymmetry19",
            "features": features,
            "validation_metrics": best["validation_metrics"],
            "model": selected,
        }
    metrics = pd.DataFrame(rows)
    predictions = pd.DataFrame(pred_rows)
    metrics.to_csv(out_dir / f"{family}_train_validate_test_metrics.csv", index=False)
    predictions.to_csv(out_dir / f"{family}_train_validate_test_predictions.csv", index=False)
    if not predictions.empty:
        plot_correlation(
            predictions[predictions["split"] == "test"].rename(columns={"prediction": "test_prediction"}),
            out_dir,
            "Extra Trees" if family == "extra_trees" else "MLP",
            "test_prediction",
            "Test set",
        )
    with (out_dir / f"{family}_selected_models.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    df = assemble_dataset()
    manifest = {
        "output_dir": str(OUT),
        "n_rows": int(len(df)),
        "bs_s2_diagnostic_no_filter": {
            "diagnostic_min": BS_S2_MIN,
            "diagnostic_max": BS_S2_MAX,
            "filter_applied": False,
            "source": "last <S^2> printed in each original broken-symmetry ORCA output",
        },
        "systems": {str(k): int(v) for k, v in df["system"].value_counts().sort_index().items()},
        "dimer_classes": {str(k): int(v) for k, v in df["dimer_class"].value_counts().sort_index().items()},
        "features19": FEATURES19,
        "target_non_null_counts": {target: int(df[target].notna().sum()) for target in TARGETS if target in df.columns},
        "electrostatic_fragment_mapping": {
            "homodimers": "equal contiguous halves",
            "cross_dimers": (
                "Monomer_A/Monomer_B atom counts and elemental compositions; "
                "both A--B and B--A dimer ordering are detected explicitly"
            ),
        },
        "note": "Cross dimer is included as a dataset. No structures are removed by an <S^2> filter in this run; <S^2> values are retained as diagnostics only.",
    }
    (OUT / "unified_19descriptor_no_s2_filter_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    for family in ["extra_trees", "mlp"]:
        train_cv_family(df, family)
        train_validate_test_family(df, family)
    print(f"\nUnified 19-descriptor NO-S2-filter homo+cross workflow complete: {OUT}")


if __name__ == "__main__":
    main()
