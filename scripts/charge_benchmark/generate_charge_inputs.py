#!/usr/bin/env python3
"""
Generate ORCA inputs for the charge-scheme benchmark (Reviewer 1, comment 4).

WHY
---
The electrostatically corrected SOMO-SOMO energy uses an interfragment Coulomb
term built from Mulliken atomic charges (manuscript Eq. 7). Reviewer 1 notes
that Mulliken charges are basis-set dependent and asks for a benchmark against
alternative schemes.

The quantity actually entering Eq. 7 is a DIFFERENCE between two states,

    dE_elst = E_elst(BS) - E_elst(triplet)

evaluated at the same geometry with the same basis and the same charge scheme.
In the published data E_elst(BS) and E_elst(triplet) average 8.170 kcal/mol each
and correlate at r = 0.9995, so their difference is 0.113 kcal/mol on average --
1.4% of the absolute terms. Systematic, scheme-dependent bias therefore cancels
to first order. This benchmark tests that empirically.

WHAT IT GENERATES
-----------------
For each of the 22 selected structures, two ORCA inputs:

    <id>_bs.inp        broken-symmetry singlet
    <id>_triplet.inp   high-spin triplet

Each requests four population analyses in a single run:

    Mulliken   printed by default (this is what the manuscript used)
    Loewdin    printed by default
    Hirshfeld  keyword !HIRSHFELD
    CHELPG     keyword !CHELPG -- charges fitted to the electrostatic potential,
               arguably the most appropriate scheme for a Coulomb model, and a
               stronger comparison than Mulliken against Mulliken-like schemes

So 22 x 2 = 44 single points, all four schemes obtained per run. No separate
job per scheme is needed.

BEFORE YOU RUN
--------------
Check LEVEL_OF_THEORY and the BS block below against one of your original ORCA
inputs. The settings here follow the manuscript (M05-2X/6-311G(d), broken
symmetry, no GuessMix) but your production inputs may contain convergence or
grid settings that should be carried over for consistency.

Usage
    python generate_charge_inputs.py
    python generate_charge_inputs.py --xyz-root /home/rb1820/BLA-ML
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# --- production inputs, reproduced verbatim --------------------------------
# These are the author's own production inputs, unchanged. This is a revision:
# the benchmark must use the same protocol as the published calculations, so
# nothing here is "improved".
#
# Broken symmetry (from /home/rb1820/BLA-ML/CPBP/orca_456/*.inp):
#
#   ! UKS LibXC(M052X) 6-311G(d) TightSCF GUESSMIX RIJCOSX AutoAux EnGrad
#   %pal   nprocs 20   end
#   %scf   MaxIter 400   FinalMs 0.0   end
#   %output  JSONPropFile True  end
#   * xyzfile 0 1 str_456.xyz
#
# Triplet:
#
#   ! UKS LibXC(M052X) 6-311G(d) TightSCF RIJCOSX AutoAux EnGrad
#   %pal   nprocs 16   end
#   %scf   MaxIter 400   end
#   %output  JSONPropFile True  end
#   * xyzfile 0 3 str_2.xyz
#
# Note the BS solution comes from GUESSMIX with FinalMs 0.0 at multiplicity 1,
# not from the BrokenSym keyword at the high-spin multiplicity. The two are not
# interchangeable and only the former reproduces the published data.
#
# THE ONLY ADDITION is the "! HIRSHFELD CHELPG" line, which requests two extra
# population analyses of the converged density. It does not alter the SCF, the
# geometry, or the energies -- it only makes additional charge definitions
# available from the same wavefunction. Without it the benchmark cannot be done.
EXTRA_ANALYSES = "! HIRSHFELD CHELPG"

# state -> (keyword line, nprocs, multiplicity, extra %scf lines)
STATES = {
    "bs": ("! UKS LibXC(M052X) 6-311G(d) TightSCF GUESSMIX RIJCOSX AutoAux EnGrad",
           20, 1, ["FinalMs 0.0"]),
    "triplet": ("! UKS LibXC(M052X) 6-311G(d) TightSCF RIJCOSX AutoAux EnGrad",
                16, 3, []),
}
MAXITER = 400
CHARGE = 0


def make_input(xyz_path: str, state: str) -> str:
    head, nprocs, mult, scf_extra = STATES[state]
    scf = ["%scf", f"  MaxIter {MAXITER}"] + [f"  {s}" for s in scf_extra] + ["end"]
    lines = [head, EXTRA_ANALYSES, "",
             "%pal", f"  nprocs {nprocs}", "end", ""] + scf + ["",
             "%output", "  JSONPropFile True", "end", "",
             f"* xyzfile {CHARGE} {mult} {xyz_path}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path,
                    default=HERE / "charge_benchmark_structures.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "charge_benchmark")
    ap.add_argument("--xyz-root", type=str, default=None,
                    help="only needed if the geometries have MOVED. Replaces the "
                         "common leading directory of the stored paths, preserving "
                         "everything below it. The stored paths are absolute and "
                         "already correct on the machine where the data was "
                         "generated, so normally omit this.")
    a = ap.parse_args()

    sel = pd.read_csv(a.selection)
    inputs = a.out_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    # Determine the common root of the stored paths, so --xyz-root swaps only
    # that prefix. Subdirectory structure below it is preserved -- important
    # because the cross-dimers sit one level deeper, under CrossDimers/, than
    # the homodimers do.
    old_root = None
    if a.xyz_root:
        parts = [Path(p).parts for p in sel["xyz_file"]]
        common = []
        for tup in zip(*parts):
            if len(set(tup)) == 1:
                common.append(tup[0])
            else:
                break
        old_root = Path(*common) if common else None
        print(f"replacing common root {old_root} -> {a.xyz_root}")

    written, missing = [], []
    for _, r in sel.iterrows():
        xyz = r["xyz_file"]
        if old_root is not None:
            rel = Path(xyz).relative_to(old_root)
            xyz = str(Path(a.xyz_root) / rel)
        if not Path(xyz).exists():
            missing.append(xyz)
        for state in ("bs", "triplet"):
            name = f"{r['id']}_{state}"
            (inputs / f"{name}.inp").write_text(make_input(xyz, state))
            written.append({"job": name, "id": r["id"], "system": r["system"],
                            "state": state, "xyz_file": xyz,
                            "published_delta_elst_kcal_mol":
                                r["target_delta_elst_bs_minus_triplet_kcal_mol"],
                            "stratum": r["stratum"]})

    manifest = pd.DataFrame(written)
    manifest.to_csv(a.out_dir / "job_manifest.csv", index=False)
    (a.out_dir / "job_list.txt").write_text(
        "\n".join(manifest["job"]) + "\n")

    print(f"wrote {len(manifest)} inputs to {inputs}")
    print(f"  {manifest['id'].nunique()} structures x 2 states")
    print(f"  manifest: {a.out_dir/'job_manifest.csv'}")
    print(f"  job list: {a.out_dir/'job_list.txt'}  (for the array job)")
    if missing:
        print(f"\nWARNING: {len(set(missing))} geometry files not found, e.g.")
        for m in sorted(set(missing))[:3]:
            print(f"    {m}")
        print("  Use --xyz-root to point at the correct location.")
    else:
        print("\nAll geometry files located.")


if __name__ == "__main__":
    main()
