#!/usr/bin/env python3
"""Run the 19-descriptor no-S2-filter workflow with |t_FO| as the first target."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("old_core_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["old_core_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-script", type=Path, required=True)
    parser.add_argument("--fodft-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    core = load_core(args.core_script)
    core.OUT = args.out_dir
    core.TARGETS = [
        "target_fodft_coupling_abs_ev",
        "target_interaction_energy_kcal_mol",
        "target_frontier_gap_ev",
        "target_somo_somo_elst_corrected_signed_kcal_mol",
    ]
    core.TARGET_LABELS.update(
        {
            "target_fodft_coupling_abs_ev": "|t_FO| (eV)",
            "target_interaction_energy_kcal_mol": "Interaction energy (kcal/mol)",
            "target_frontier_gap_ev": "Spin-conserving frontier gap (eV)",
            "target_somo_somo_elst_corrected_signed_kcal_mol": "Corrected SOMO-SOMO energy (kcal/mol)",
        }
    )

    original_assemble = core.assemble_dataset

    def assemble_with_fodft() -> pd.DataFrame:
        df = original_assemble()
        fodft = pd.read_csv(args.fodft_csv)
        # The established dataset labels structures as system_orca_N, whereas
        # the projection manifest stores system and the integer index
        # separately. Build the merge key explicitly to avoid relying on a
        # presentation-oriented ID convention in either table.
        fodft["id"] = fodft["system"].astype(str) + "_orca_" + fodft["orca_index"].astype(int).astype(str)
        if fodft["id"].duplicated().any():
            duplicates = fodft.loc[fodft["id"].duplicated(keep=False), "id"].unique()[:20].tolist()
            raise ValueError(f"Duplicate FO-DFT structure IDs after normalization: {duplicates}")
        needed = [
            "id",
            "target_fodft_coupling_abs_ev",
            "target_fodft_coupling_abs_mev",
            "lowdin_corrected_t_fo_hartree",
            "raw_fragment_somo_overlap_s_ab",
            "site_energy_mismatch_abs_ev",
            "generalized_two_state_splitting_ev",
            "half_generalized_splitting_ev",
            "fragment_somo_a_norm",
            "fragment_somo_b_norm",
        ]
        merged = df.merge(fodft[needed], on="id", how="left", validate="one_to_one")
        missing = merged["target_fodft_coupling_abs_ev"].isna()
        if missing.any():
            missing_ids = merged.loc[missing, "id"].head(20).tolist()
            raise ValueError(f"Missing |t_FO| values for {int(missing.sum())} structures, examples: {missing_ids}")
        canonical = args.out_dir / "ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv"
        merged.to_csv(canonical, index=False)
        # Compatibility copies for archived helper scripts. In this no-filter
        # workflow, these files contain all cleaned modeling-eligible rows.
        shutil.copy2(canonical, args.out_dir / "ALL_DIMERS_19descriptors_4targets_FODFT_BS_filtered.csv")
        shutil.copy2(canonical, args.out_dir / "ALL_DIMERS_19descriptors_7targets_BS_filtered.csv")
        return merged

    core.assemble_dataset = assemble_with_fodft
    if args.validate_only:
        validated = assemble_with_fodft()
        print(
            json.dumps(
                {
                    "n_rows": int(len(validated)),
                    "n_fodft_missing": int(validated["target_fodft_coupling_abs_ev"].isna().sum()),
                    "n_unique_ids": int(validated["id"].nunique()),
                },
                indent=2,
            )
        )
        return
    core.main()

    manifest_path = args.out_dir / "unified_19descriptor_fodft_4target_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "core_script": str(args.core_script),
                "fodft_csv": str(args.fodft_csv),
                "output_dir": str(args.out_dir),
                "production_targets": core.TARGETS,
                "note": "|t_FO| replaces the previous SOMO orbital-shift target. No S2 filter is applied; legacy BS-filtered filenames are compatibility copies only.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
