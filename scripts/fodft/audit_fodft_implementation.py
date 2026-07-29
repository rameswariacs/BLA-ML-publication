#!/usr/bin/env python3
"""Independent implementation audit for the production FO-DFT coupling table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HARTREE_TO_EV = 27.211386245988


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(errors="ignore").splitlines()
    n_atoms = int(lines[0].strip())
    rows = [line.split() for line in lines[2 : 2 + n_atoms]]
    return [row[0] for row in rows], np.array([[float(value) for value in row[1:4]] for row in rows])


def highest_occupied_alpha(path: Path) -> int:
    text = path.read_text(errors="ignore")
    start = text.rfind("ORBITAL ENERGIES")
    if start < 0:
        raise ValueError(f"No ORBITAL ENERGIES section in {path}")
    section = text[start:]
    up = section.find("SPIN UP ORBITALS")
    down = section.find("SPIN DOWN ORBITALS")
    if up < 0 or down < 0 or down <= up:
        raise ValueError(f"Could not isolate unrestricted alpha orbitals in {path}")
    occupied = []
    for line in section[up:down].splitlines():
        match = re.match(r"\s*(\d+)\s+([-+0-9.]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)", line)
        if match and float(match.group(2)) > 0.5:
            occupied.append((int(match.group(1)), float(match.group(3))))
    if not occupied:
        raise ValueError(f"No occupied alpha orbital in {path}")
    return max(occupied, key=lambda item: item[1])[0]


def xyz_multiplicity(path: Path) -> int | None:
    for line in path.read_text(errors="ignore").splitlines():
        match = re.match(r"\s*\*\s+xyzfile\s+[-+]?\d+\s+(\d+)\s+", line, re.I)
        if match:
            return int(match.group(1))
    return None


def route(path: Path) -> str:
    return next((line.strip() for line in path.read_text(errors="ignore").splitlines() if line.lstrip().startswith("!")), "")


def lowdin(row: pd.Series) -> tuple[float, float, float]:
    overlap = float(row["raw_fragment_somo_overlap_s_ab"])
    s_matrix = np.array([[1.0, overlap], [overlap, 1.0]])
    h_matrix = np.array(
        [
            [float(row["h_aa_hartree"]), float(row["h_ab_hartree"])],
            [float(row["h_ab_hartree"]), float(row["h_bb_hartree"])],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(s_matrix)
    inverse_sqrt = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    orthogonal_h = inverse_sqrt @ h_matrix @ inverse_sqrt
    levels = np.linalg.eigvalsh(orthogonal_h)
    return float(orthogonal_h[0, 1]), float((levels[1] - levels[0]) * HARTREE_TO_EV), float(eigenvalues.min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--couplings", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    couplings = pd.read_csv(args.couplings)
    coupling_by_key = couplings.set_index(["system", "orca_index"], verify_integrity=True)

    coordinate_failures = []
    somo_failures = []
    input_failures = []
    block_failures = []
    numeric_failures = []
    records = []
    maximum_coordinate_deviation = 0.0
    maximum_t_error = 0.0
    maximum_split_error = 0.0
    minimum_overlap_eigenvalue = np.inf
    fragment_routes: set[str] = set()
    triplet_routes: set[str] = set()

    for _, item in manifest.iterrows():
        system = str(item["system"])
        index = int(item["orca_index"])
        key = (system, index)
        row = coupling_by_key.loc[key]

        dimer_symbols, dimer_coordinates = read_xyz(Path(item["source_xyz"]))
        fragment_a_dir = Path(item["fragment_a_dir"])
        fragment_b_dir = Path(item["fragment_b_dir"])
        a_xyz = fragment_a_dir / "fragment_A.xyz"
        b_xyz = fragment_b_dir / "fragment_B.xyz"
        a_symbols, a_coordinates = read_xyz(a_xyz)
        b_symbols, b_coordinates = read_xyz(b_xyz)
        if item["split_method"] == "Monomer_B_then_Monomer_A":
            expected_symbols = b_symbols + a_symbols
            expected_coordinates = np.vstack([b_coordinates, a_coordinates])
            expected_order = "B_then_A"
        else:
            expected_symbols = a_symbols + b_symbols
            expected_coordinates = np.vstack([a_coordinates, b_coordinates])
            expected_order = "A_then_B"
        coordinate_deviation = (
            float(np.max(np.abs(dimer_coordinates - expected_coordinates)))
            if dimer_coordinates.shape == expected_coordinates.shape
            else np.inf
        )
        maximum_coordinate_deviation = max(maximum_coordinate_deviation, coordinate_deviation)
        coordinate_ok = dimer_symbols == expected_symbols and coordinate_deviation <= 1.0e-8
        if not coordinate_ok:
            coordinate_failures.append({"system": system, "orca_index": index, "deviation": coordinate_deviation})

        a_out = fragment_a_dir / "fragment_A.out"
        b_out = fragment_b_dir / "fragment_B.out"
        a_index = highest_occupied_alpha(a_out)
        b_index = highest_occupied_alpha(b_out)
        somo_ok = (
            a_index == int(row["fragment_a_alpha_somo_index"])
            and b_index == int(row["fragment_b_alpha_somo_index_local"])
        )
        if not somo_ok:
            somo_failures.append(
                {
                    "system": system,
                    "orca_index": index,
                    "parsed_a": a_index,
                    "stored_a": int(row["fragment_a_alpha_somo_index"]),
                    "parsed_b": b_index,
                    "stored_b": int(row["fragment_b_alpha_somo_index_local"]),
                }
            )

        a_inp = fragment_a_dir / "fragment_A.inp"
        b_inp = fragment_b_dir / "fragment_B.inp"
        triplet_gbw = Path(item["triplet_gbw"])
        triplet_inputs = sorted(triplet_gbw.parent.glob("*.inp"))
        triplet_inp = next((path for path in triplet_inputs if path.stem == triplet_gbw.stem), triplet_inputs[0] if triplet_inputs else None)
        fragment_routes.update([route(a_inp), route(b_inp)])
        if triplet_inp is not None:
            triplet_routes.add(route(triplet_inp))
        input_ok = (
            xyz_multiplicity(a_inp) == 2
            and xyz_multiplicity(b_inp) == 2
            and triplet_inp is not None
            and xyz_multiplicity(triplet_inp) == 3
            and "GUESSMIX" not in route(a_inp).upper()
            and "GUESSMIX" not in route(b_inp).upper()
            and "GUESSMIX" not in route(triplet_inp).upper()
        )
        if not input_ok:
            input_failures.append({"system": system, "orca_index": index, "triplet_input": str(triplet_inp)})

        block_log = args.case_root / system / f"orca_{index}" / "block.log"
        block_text = block_log.read_text(errors="ignore")
        orbital_dim = re.search(r"orbitals matrix of dimension\s+(\d+)", block_text)
        fock_dim = re.search(r"Fock matrix of dimension\s+(\d+)", block_text)
        block_ok = (
            orbital_dim is not None
            and fock_dim is not None
            and orbital_dim.group(1) == fock_dim.group(1)
            and "leaving gracefully" in block_text
        )
        if not block_ok:
            block_failures.append({"system": system, "orca_index": index, "block_log": str(block_log)})

        t_orthogonal, splitting_ev, min_overlap_eigenvalue = lowdin(row)
        t_formula = (
            float(row["h_ab_hartree"])
            - 0.5
            * float(row["raw_fragment_somo_overlap_s_ab"])
            * (float(row["h_aa_hartree"]) + float(row["h_bb_hartree"]))
        ) / (1.0 - float(row["raw_fragment_somo_overlap_s_ab"]) ** 2)
        t_error = max(
            abs(t_formula - float(row["lowdin_corrected_t_fo_hartree"])),
            abs(t_orthogonal - float(row["lowdin_corrected_t_fo_hartree"])),
        )
        splitting_error = abs(splitting_ev - float(row["generalized_two_state_splitting_ev"]))
        maximum_t_error = max(maximum_t_error, t_error)
        maximum_split_error = max(maximum_split_error, splitting_error)
        minimum_overlap_eigenvalue = min(minimum_overlap_eigenvalue, min_overlap_eigenvalue)
        numeric_ok = (
            t_error <= 1.0e-10
            and splitting_error <= 1.0e-8
            and abs(float(row["fragment_somo_a_norm"]) - 1.0) <= 1.0e-8
            and abs(float(row["fragment_somo_b_norm"]) - 1.0) <= 1.0e-8
            and min_overlap_eigenvalue > 0.0
            and str(row["projection_fragment_order"]) == expected_order
        )
        if not numeric_ok:
            numeric_failures.append({"system": system, "orca_index": index, "t_error": t_error, "splitting_error": splitting_error})

        records.append(
            {
                "system": system,
                "orca_index": index,
                "coordinate_order_ok": coordinate_ok,
                "fragment_somo_indices_ok": somo_ok,
                "input_multiplicity_and_guessmix_ok": input_ok,
                "block_dimensions_ok": block_ok,
                "numeric_reconstruction_ok": numeric_ok,
                "expected_projection_order": expected_order,
                "stored_projection_order": row["projection_fragment_order"],
                "coordinate_deviation_angstrom": coordinate_deviation,
                "t_reconstruction_error_hartree": t_error,
                "splitting_reconstruction_error_ev": splitting_error,
            }
        )

    details = pd.DataFrame(records)
    details.to_csv(args.out_dir / "fodft_implementation_audit_all_cases.csv", index=False)
    summary = {
        "n_manifest": int(len(manifest)),
        "n_couplings": int(len(couplings)),
        "coordinate_order_failures": len(coordinate_failures),
        "fragment_somo_index_failures": len(somo_failures),
        "input_multiplicity_or_guessmix_failures": len(input_failures),
        "orca_block_dimension_failures": len(block_failures),
        "numeric_reconstruction_failures": len(numeric_failures),
        "maximum_coordinate_deviation_angstrom": maximum_coordinate_deviation,
        "maximum_t_reconstruction_error_hartree": maximum_t_error,
        "maximum_splitting_reconstruction_error_ev": maximum_split_error,
        "minimum_overlap_matrix_eigenvalue": minimum_overlap_eigenvalue,
        "fragment_route_sections": sorted(fragment_routes),
        "triplet_route_sections": sorted(triplet_routes),
        "all_checks_passed": not any([coordinate_failures, somo_failures, input_failures, block_failures, numeric_failures]),
    }
    (args.out_dir / "fodft_implementation_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "FODFT_IMPLEMENTATION_AUDIT_REPORT.md").write_text(
        "# FO-DFT implementation audit\n\n"
        f"Cases audited: **{len(manifest)}**.\n\n"
        f"All checks passed: **{summary['all_checks_passed']}**.\n\n"
        "The audit independently checked dimer/fragment atom order, frozen coordinates, final occupied alpha-orbital indices, "
        "doublet/triplet input multiplicities, absence of GuessMix from fragment and triplet calculations, equality of the "
        "orbital and Fock dimensions reported by ORCA blockf, fragment-orbital normalization, overlap conditioning, and "
        "both the analytic and matrix Loewdin reconstructions of the coupling and generalized two-state splitting.\n\n"
        "```json\n" + json.dumps(summary, indent=2) + "\n```\n"
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_checks_passed"]:
        raise SystemExit("FO-DFT implementation audit found failures")


if __name__ == "__main__":
    main()
