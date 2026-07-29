#!/usr/bin/env python3
"""Copy and patch archived validation/plot scripts for the no-S2-filter FO-DFT run."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/home/rb1820/BLA-ML")
OLD_SOURCE = ROOT / "Unified_19Descriptor_BSFiltered_ElstOrderFixed_plus_CPBP"
OLD_TARGET = "target_somo_stabilization_mev"
NEW_TARGET = "target_fodft_coupling_abs_ev"


def patch_file(path: Path, final_dir: Path) -> None:
    text = path.read_text()
    replacements = {
        str(OLD_SOURCE): str(final_dir),
        "Unified_19Descriptor_BSFiltered_ElstOrderFixed_plus_CPBP": final_dir.name,
        "YRandomization_500x_BSFiltered_ElstOrderFixed_4Targets": str(final_dir / "YRandomization_500x_FODFT_4Targets"),
        "GeometryClustered_ExternalTest_19Descriptor_4Targets": str(final_dir / "GeometryClustered_ExternalTest_FODFT_4Targets"),
        "BS_Filter_Sensitivity_19Descriptor_4Targets": str(final_dir / "BS_Filter_Sensitivity_FODFT_4Targets"),
        "ExtraTrees_ComplexitySensitivity_ElstOrderFixed_4Targets": str(final_dir / "ExtraTrees_ComplexitySensitivity_FODFT_4Targets"),
        "MLP_ComplexitySensitivity_ElstOrderFixed_4Targets": str(final_dir / "MLP_ComplexitySensitivity_FODFT_4Targets"),
        OLD_TARGET: NEW_TARGET,
        "SOMO stabilization (meV)": "|t_FO| (eV)",
        "SOMO stabilization": "|t_FO|",
        "somo_stabilization_mev": "fodft_coupling_abs_ev",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text)


def copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns("logs", "results_by_task", "plots", "plots_zoomed_shuffled_r2", "models", "__pycache__", "*.csv", "*.json", "*.md", "*.png", "*.pdf", "*.svg", "*.pkl", "*.out", "*.err")
    shutil.copytree(src, dest, ignore=ignore)


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--mode", choices=["prepare", "run-light"], default="prepare")
    args = parser.parse_args()

    final = args.final_dir
    support = final / "supporting_analyses"
    support.mkdir(parents=True, exist_ok=True)

    sources = {
        "YRandomization_500x_FODFT_4Targets": ROOT / "YRandomization_500x_BSFiltered_ElstOrderFixed_4Targets",
        "GeometryClustered_ExternalTest_FODFT_4Targets": ROOT / "GeometryClustered_ExternalTest_19Descriptor_4Targets",
        "ExtraTrees_ComplexitySensitivity_FODFT_4Targets": ROOT / "ExtraTrees_ComplexitySensitivity_ElstOrderFixed_4Targets",
        "MLP_ComplexitySensitivity_FODFT_4Targets": ROOT / "MLP_ComplexitySensitivity_ElstOrderFixed_4Targets",
        "Publication_Top7_FeatureImportance_FODFT": ROOT / "Publication_Top7_FeatureImportance_BSFiltered_ElstOrderFixed",
    }

    for name, src in sources.items():
        dest = support / name
        copy_tree(src, dest)
        for script in list(dest.rglob("*.py")) + list(dest.rglob("*.sh")) + list(dest.rglob("*.sbatch")):
            patch_file(script, final)
            if script.suffix == ".sbatch":
                text = script.read_text()
                if "#SBATCH --partition=" not in text:
                    text = text.replace("#SBATCH --nodes=1\n", "#SBATCH --partition=base1\n#SBATCH --nodes=1\n")
                script.write_text(text)

    top7 = support / "Publication_Top7_FeatureImportance_FODFT"
    (top7 / "inputs").mkdir(exist_ok=True)
    shutil.copy2(final / "extra_trees" / "extra_trees_feature_importance.csv", top7 / "inputs" / "extra_trees_feature_importance.csv")
    shutil.copy2(final / "mlp" / "mlp_feature_importance.csv", top7 / "inputs" / "mlp_feature_importance.csv")

    if args.mode == "run-light":
        run([str(args.python), "plot_top7_feature_importance_publication.py"], top7)
        run([str(args.python), "geometry_clustered_external_validation.py"], support / "GeometryClustered_ExternalTest_FODFT_4Targets")
        run([str(args.python), "extra_trees_complexity_sensitivity.py"], support / "ExtraTrees_ComplexitySensitivity_FODFT_4Targets")
        run([str(args.python), "mlp_complexity_sensitivity.py"], support / "MLP_ComplexitySensitivity_FODFT_4Targets")

    print(f"Prepared supporting analyses in {support}")


if __name__ == "__main__":
    main()
