#!/usr/bin/env python3
"""Calculate Loewdin-corrected fragment-orbital SOMO couplings for all cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

HARTREE_TO_EV = 27.211386245988
ORCA_DIR = Path("/home/rb1820/orca_6_1_1")


def run(cmd: list[str], cwd: Path, log_name: str) -> None:
    with (cwd / log_name).open("w") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed in {cwd}: {' '.join(cmd)}; see {log_name}")


def link_or_copy(source: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(source)
    except OSError:
        shutil.copy2(source, dest)


def write_json_config(path: Path, include_basis: bool = False) -> None:
    path.write_text(
        '{\n'
        '  "MOCoefficients": true,\n'
        f'  "Basisset": {"true" if include_basis else "false"},\n'
        '  "1elIntegrals": ["S"],\n'
        '  "MullikenCharge": false,\n'
        '  "LoewdinCharge": false,\n'
        '  "JSONFormats": ["json"]\n'
        '}\n'
    )


def load_mos(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["Molecule"]["MolecularOrbitals"]["MOs"]


def alpha_somo_index(fragment_json: Path) -> int:
    mos = load_mos(fragment_json)
    # ORCA writes unrestricted alpha MOs followed by beta MOs. The fragment SOMO
    # is the highest-energy occupied alpha orbital in the first spin block.
    alpha = mos[: len(mos) // 2]
    occupied = [
        (i, float(mo.get("OrbitalEnergy", -1.0e9)))
        for i, mo in enumerate(alpha)
        if float(mo.get("Occupancy", 0.0)) > 0.5
    ]
    if not occupied:
        raise ValueError(f"No occupied alpha orbital detected in {fragment_json}")
    return max(occupied, key=lambda item: item[1])[0]


def mo_overlap(json_path: Path, first: int, second: int) -> tuple[float, float, float]:
    molecule = json.loads(json_path.read_text())["Molecule"]
    overlap = molecule["S-Matrix"]
    mos = molecule["MolecularOrbitals"]["MOs"]

    def one(i: int, j: int) -> float:
        ci = mos[i]["MOCoefficients"]
        cj = mos[j]["MOCoefficients"]
        total = 0.0
        for row in range(len(ci)):
            total += ci[row] * sum(overlap[row][col] * cj[col] for col in range(len(cj)))
        return float(total)

    return one(first, first), one(second, second), one(first, second)


def move_second_fragment_somo(source_json: Path, dest_json: Path, second_somo: int, destination_index: int) -> None:
    data = json.loads(source_json.read_text())
    mos = data["Molecule"]["MolecularOrbitals"]["MOs"]
    mos[destination_index], mos[second_somo] = mos[second_somo], mos[destination_index]
    data["Molecule"]["BaseName"] = dest_json.stem
    dest_json.write_text(json.dumps(data))


def parse_block_log(path: Path) -> tuple[float, float, float, float, float]:
    text = path.read_text(errors="ignore")
    matrix = re.search(
        r"Local Fock Matrix.*?\n\s*0\s+1\s*\n\s*0\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
        r"\s*\n\s*1\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)",
        text,
        re.S,
    )
    eigen = re.search(
        r"Local Eigenvalues \(eV\).*?\n\s*0\s+([-+0-9.Ee]+)\s*\n\s*1\s+([-+0-9.Ee]+)",
        text,
        re.S,
    )
    if not matrix or not eigen:
        raise ValueError(f"Could not parse local Fock matrix/eigenvalues from {path}")
    haa, hab, hba, hbb = map(float, matrix.groups())
    e0, e1 = map(float, eigen.groups())
    return haa, hbb, 0.5 * (hab + hba), e0, e1


def process_case(row: dict[str, str], out_dir: Path, scratch_base: Path, force: bool) -> dict[str, object]:
    system = row["system"]
    idx = int(row["orca_index"])
    split_method = row.get("split_method", "")
    expected_order = "B_then_A" if split_method == "Monomer_B_then_Monomer_A" else "A_then_B"
    archive = out_dir / "case_summaries" / system / f"orca_{idx}"
    archive.mkdir(parents=True, exist_ok=True)
    done = archive / "fodft_done.json"
    if done.exists() and not force:
        cached = json.loads(done.read_text())
        # Results made before the cross-dimer AO-order correction do not carry
        # this marker. Recompute only those B-then-A cases; established
        # homodimer caches remain valid and are reused.
        if expected_order != "B_then_A" or cached.get("projection_fragment_order") == expected_order:
            cached.setdefault("split_method", split_method)
            cached.setdefault("projection_fragment_order", expected_order)
            return cached

    job_tag = os.environ.get("SLURM_JOB_ID") or f"pid_{os.getpid()}"
    work = scratch_base / f"fodft4_{job_tag}" / system / f"orca_{idx}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    sources = {
        "fragA.gbw": Path(row["fragment_a_dir"]) / "fragment_A.gbw",
        "fragB.gbw": Path(row["fragment_b_dir"]) / "fragment_B.gbw",
        "triplet.gbw": Path(row["triplet_gbw"]),
        "dimer.xyz": Path(row["source_xyz"]),
    }
    missing = [str(path) for path in sources.values() if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing FO-DFT inputs for {system} orca_{idx}: {missing}")
    for name, source in sources.items():
        link_or_copy(source, work / name)

    write_json_config(work / "fragA.json.conf")
    write_json_config(work / "fragB.json.conf")
    # The merged JSON must include the basis set because it is converted back to
    # a GBW after placing the two fragment SOMOs in an adjacent MO block.
    write_json_config(work / "merged.json.conf", include_basis=True)

    run([str(ORCA_DIR / "orca_2json"), "fragA.gbw", "-json"], work, "fragA_2json.log")
    run([str(ORCA_DIR / "orca_2json"), "fragB.gbw", "-json"], work, "fragB_2json.log")
    a_somo = alpha_somo_index(work / "fragA.json")
    b_somo_local = alpha_somo_index(work / "fragB.json")

    if split_method == "Monomer_B_then_Monomer_A":
        projection_order = "B_then_A"
        first_gbw, second_gbw = "fragB.gbw", "fragA.gbw"
        first_somo, second_somo_local = b_somo_local, a_somo
    else:
        projection_order = "A_then_B"
        first_gbw, second_gbw = "fragA.gbw", "fragB.gbw"
        first_somo, second_somo_local = a_somo, b_somo_local

    # orca_blockf requires the fragment-orbital AO rows to have exactly the
    # same atom/basis order as the dimer GBW. The manifest records that order;
    # cross dimers in this data set are B then A, unlike the homodimers.
    run([str(ORCA_DIR / "orca_mergefrag"), first_gbw, second_gbw, "merged.gbw"], work, "mergefrag.log")
    run([str(ORCA_DIR / "orca_2json"), "merged.gbw", "-json"], work, "merged_2json.log")

    # In an ORCA mergefrag GBW, the occupied alpha orbitals of the second
    # fragment follow the occupied alpha block of the first fragment.
    n_occ_alpha_first = first_somo + 1
    second_somo_merged = n_occ_alpha_first + second_somo_local
    destination = first_somo + 1
    if destination == second_somo_merged:
        destination = first_somo + 2
    move_second_fragment_somo(
        work / "merged.json", work / "raw_reordered.json", second_somo_merged, destination
    )
    norm_first, norm_second, s_ab = mo_overlap(work / "raw_reordered.json", first_somo, destination)
    run([str(ORCA_DIR / "orca_2json"), "raw_reordered.json", "-gbw"], work, "raw_import.log")
    raw_gbw = work / "raw_reordered_copy.gbw"
    if not raw_gbw.exists():
        raw_gbw = work / "raw_reordered.gbw"
    if not raw_gbw.exists():
        raise FileNotFoundError(f"orca_2json did not create a raw reordered GBW in {work}")

    run(
        [str(ORCA_DIR / "orca_blockf"), "triplet.gbw", raw_gbw.name, "block.gbw", str(first_somo), str(destination)],
        work,
        "block.log",
    )
    h_first, h_second, hab, eig0, eig1 = parse_block_log(work / "block.log")
    if projection_order == "B_then_A":
        haa, hbb = h_second, h_first
        norm_a, norm_b = norm_second, norm_first
        b_somo_merged = first_somo
    else:
        haa, hbb = h_first, h_second
        norm_a, norm_b = norm_first, norm_second
        b_somo_merged = second_somo_merged
    denom = 1.0 - s_ab**2
    if abs(denom) < 1.0e-8:
        raise ValueError(f"Near-singular SOMO overlap for {system} orca_{idx}: S_AB={s_ab}")
    t_h = (hab - 0.5 * s_ab * (haa + hbb)) / denom
    linear = 2.0 * s_ab * hab - haa - hbb
    constant = haa * hbb - hab**2
    disc = max(0.0, linear**2 - 4.0 * denom * constant)
    roots = sorted(((-linear - math.sqrt(disc)) / (2.0 * denom), (-linear + math.sqrt(disc)) / (2.0 * denom)))

    result = {
        "id": f"{system}_{idx}",
        "system": system,
        "orca_index": idx,
        "fodft_status": "OK",
        "fragment_a_alpha_somo_index": a_somo,
        "fragment_b_alpha_somo_index_local": b_somo_local,
        "fragment_b_alpha_somo_index_merged": b_somo_merged,
        "split_method": split_method,
        "projection_fragment_order": projection_order,
        "projected_block_first": first_somo,
        "projected_block_second": destination,
        "h_aa_hartree": haa,
        "h_bb_hartree": hbb,
        "h_ab_hartree": hab,
        "raw_fragment_somo_overlap_s_ab": s_ab,
        "fragment_somo_a_norm": norm_a,
        "fragment_somo_b_norm": norm_b,
        "lowdin_corrected_t_fo_hartree": t_h,
        "target_fodft_coupling_abs_ev": abs(t_h) * HARTREE_TO_EV,
        "target_fodft_coupling_abs_mev": abs(t_h) * HARTREE_TO_EV * 1000.0,
        "site_energy_mismatch_abs_ev": abs(haa - hbb) * HARTREE_TO_EV,
        "generalized_two_state_splitting_ev": (roots[1] - roots[0]) * HARTREE_TO_EV,
        "half_generalized_splitting_ev": 0.5 * (roots[1] - roots[0]) * HARTREE_TO_EV,
        "block_eigenvalue_low_ev": eig0,
        "block_eigenvalue_high_ev": eig1,
        "case_archive_dir": str(archive),
    }
    done.write_text(json.dumps(result, indent=2) + "\n")
    for log_name in [
        "fragA_2json.log",
        "fragB_2json.log",
        "mergefrag.log",
        "merged_2json.log",
        "raw_import.log",
        "block.log",
    ]:
        source = work / log_name
        if source.exists():
            shutil.copy2(source, archive / log_name)
    shutil.rmtree(work, ignore_errors=True)
    for parent in [work.parent, work.parent.parent]:
        try:
            parent.rmdir()
        except OSError:
            pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scratch-base", type=Path, default=Path("/home/scratch/rb1820/ML-scratch"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-system", action="append", default=[])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_base.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.manifest.open(newline="")))
    force_systems = set(args.force_system)
    results = []
    failures = []
    for count, row in enumerate(rows, start=1):
        label = f"{row['system']} orca_{row['orca_index']}"
        print(f"[{count}/{len(rows)}] {label}", flush=True)
        try:
            force_case = args.force or row["system"] in force_systems
            results.append(process_case(row, args.out_dir, args.scratch_base, force_case))
        except Exception as exc:  # noqa: BLE001 - write complete failure table and continue
            failures.append(
                {
                    "id": f"{row['system']}_{row['orca_index']}",
                    "system": row["system"],
                    "orca_index": row["orca_index"],
                    "fodft_status": "FAILED",
                    "reason": repr(exc),
                }
            )

    if results:
        fieldnames: list[str] = []
        for result in results:
            for field in result:
                if field not in fieldnames:
                    fieldnames.append(field)
        with (args.out_dir / "fodft_couplings_all.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    if failures:
        with (args.out_dir / "fodft_coupling_failures.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
    summary = {"n_manifest": len(rows), "n_ok": len(results), "n_failed": len(failures)}
    (args.out_dir / "fodft_coupling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        raise SystemExit(f"FO-DFT coupling failed for {len(failures)} cases. See {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
