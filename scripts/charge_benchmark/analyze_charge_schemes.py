#!/usr/bin/env python3
"""
Recompute the electrostatic correction under four charge schemes (Reviewer 1.4).

WHAT IT DOES
------------
Parses the ORCA outputs from the benchmark, extracts Mulliken, Loewdin,
Hirshfeld and CHELPG charges for the broken-symmetry and triplet states of each
structure, and recomputes the interfragment Coulomb term of manuscript Eq. 7,

    E_elst(X) = sum_{i in A} sum_{j in B}  q_i q_j / r_ij           (state X)

and the quantity that actually enters the correction,

    dE_elst = E_elst(BS) - E_elst(triplet)

THE POINT
---------
The absolute E_elst values are expected to DIFFER between schemes, possibly a
lot -- Mulliken and Hirshfeld charges are not comparable in magnitude. The
claim being tested is that their DIFFERENCE between two states at the same
geometry is stable, because systematic scheme-dependent bias cancels. In the
published data the two absolute terms average 8.170 kcal/mol each and correlate
at r = 0.9995, leaving a difference of 0.113 kcal/mol.

FRAGMENT ASSIGNMENT
-------------------
The two fragments are identified by bond connectivity, using a covalent-radius
criterion, rather than by splitting the atom list in half. This is deliberate:
it works for the phenalenyl-olympicenyl heterodimer and does not depend on the
atom ordering in the xyz file. The script asserts that exactly two fragments are
found and reports their sizes, so a bad assignment fails loudly instead of
silently producing wrong Coulomb sums.

INPUT
-----
  charge_benchmark/outputs/<id>_bs.out
  charge_benchmark/outputs/<id>_triplet.out
  charge_benchmark/job_manifest.csv
  (the published Mulliken values come along in the manifest for comparison)

OUTPUT -> charge_benchmark/analysis/
  charges_by_scheme.csv        per structure, state and scheme: E_elst
  correction_comparison.csv    dE_elst per scheme vs the published value
  CHARGE_SCHEME_REPORT.md
  fig_charge_schemes.png/.pdf

Usage
    python analyze_charge_schemes.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# e^2 / Angstrom -> kcal/mol
COULOMB_K = 332.0637

# covalent radii in Angstrom, enough for C/H/N/O/S
COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}
BOND_TOL = 1.30          # bonded if r < BOND_TOL * (rA + rB)

SCHEMES = ["mulliken", "loewdin", "hirshfeld", "chelpg"]


# ---------------------------------------------------------------------------
# ORCA output parsing
# ---------------------------------------------------------------------------

def _block(text: str, header_pat: str, stop_blank: int = 1) -> list[str]:
    m = re.search(header_pat, text)
    if not m:
        return []
    lines, blanks = [], 0
    for ln in text[m.end():].split("\n"):
        if not ln.strip():
            blanks += 1
            if blanks >= stop_blank and lines:
                break
            continue
        lines.append(ln)
    return lines


def parse_geometry(text: str):
    """Last CARTESIAN COORDINATES (ANGSTROEM) block."""
    idx = text.rfind("CARTESIAN COORDINATES (ANGSTROEM)")
    if idx < 0:
        raise ValueError("no cartesian coordinates found")
    sym, xyz = [], []
    for ln in text[idx:].split("\n")[2:]:
        p = ln.split()
        if len(p) != 4:
            break
        sym.append(p[0])
        xyz.append([float(v) for v in p[1:]])
    return sym, np.asarray(xyz)


def parse_charges(text: str, scheme: str, natoms: int) -> np.ndarray | None:
    """
    Return the LAST occurrence of each charge block, since ORCA prints
    intermediate analyses during a broken-symmetry run.
    """
    pats = {
        "mulliken":  r"MULLIKEN ATOMIC CHARGES(?: AND SPIN POPULATIONS)?\s*\n-+",
        "loewdin":   r"LOEWDIN ATOMIC CHARGES(?: AND SPIN POPULATIONS)?\s*\n-+",
        "hirshfeld": r"HIRSHFELD ANALYSIS\s*\n-+",
        "chelpg":    r"CHELPG Charges\s*\n-+",
    }
    hits = list(re.finditer(pats[scheme], text))
    if not hits:
        return None
    tail = text[hits[-1].end():]
    q = []
    for ln in tail.split("\n"):
        s = ln.strip()
        if not s or s.startswith("-"):
            if q:
                break
            continue
        # Mulliken/Loewdin:  "  0 C :   -0.123456   0.98"
        m = re.match(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s*:\s*(-?\d+\.\d+)", ln)
        if m:
            q.append(float(m.group(3)))
            continue
        # Hirshfeld:  "   0   C      -0.012345   0.001234"
        m = re.match(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+(-?\d+\.\d+)", ln)
        if m:
            q.append(float(m.group(3)))
            continue
        # CHELPG:  "  0   C   :   -0.123456"
        m = re.match(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+:\s+(-?\d+\.\d+)", ln)
        if m:
            q.append(float(m.group(3)))
            continue
        if q:
            break
    if len(q) < natoms:
        return None
    return np.asarray(q[:natoms])


# ---------------------------------------------------------------------------
# fragments and the Coulomb sum
# ---------------------------------------------------------------------------

def fragments(sym, xyz) -> tuple[np.ndarray, int]:
    """
    Label atoms by connected component using a covalent-radius bond criterion.
    Returns (labels, n_components). For a pi-stacked dimer there should be
    exactly two components; anything else means the geometry is not a clean
    dimer and the Coulomb partition would be meaningless.
    """
    n = len(sym)
    r = np.array([COVALENT.get(s.capitalize(), 0.76) for s in sym])
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    cutoff = BOND_TOL * (r[:, None] + r[None, :])
    adj = (d < cutoff) & ~np.eye(n, dtype=bool)

    labels = -np.ones(n, dtype=int)
    comp = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = comp
        while stack:
            k = stack.pop()
            for j in np.flatnonzero(adj[k]):
                if labels[j] < 0:
                    labels[j] = comp
                    stack.append(j)
        comp += 1
    return labels, comp


def e_elst(q, xyz, labels) -> float:
    """Interfragment Coulomb energy in kcal/mol."""
    a = labels == 0
    b = labels == 1
    d = np.linalg.norm(xyz[a][:, None, :] - xyz[b][None, :, :], axis=-1)
    return float(COULOMB_K * np.sum(np.outer(q[a], q[b]) / d))


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=HERE / "charge_benchmark")
    a = ap.parse_args()

    man = pd.read_csv(a.bench / "job_manifest.csv")
    out = a.bench / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows, problems = [], []
    for jid, grp in man.groupby("id"):
        rec = {"id": jid, "system": grp["system"].iloc[0],
               "stratum": grp["stratum"].iloc[0],
               "published_delta_elst": grp["published_delta_elst_kcal_mol"].iloc[0]}
        ok = True
        for state in ("bs", "triplet"):
            f = a.bench / "outputs" / f"{jid}_{state}.out"
            if not f.exists():
                problems.append(f"missing output: {f.name}"); ok = False; break
            text = f.read_text(errors="ignore")
            if "ORCA TERMINATED NORMALLY" not in text:
                problems.append(f"did not terminate normally: {f.name}"); ok = False; break
            sym, xyz = parse_geometry(text)
            labels, ncomp = fragments(sym, xyz)
            if ncomp != 2:
                problems.append(f"{f.name}: found {ncomp} fragments, expected 2"); ok = False; break
            rec[f"n_atoms"] = len(sym)
            rec[f"frag_sizes"] = f"{int((labels==0).sum())}/{int((labels==1).sum())}"
            for s in SCHEMES:
                q = parse_charges(text, s, len(sym))
                rec[f"E_elst_{s}_{state}"] = e_elst(q, xyz, labels) if q is not None else np.nan
                if q is not None:
                    rec[f"qsum_{s}_{state}"] = float(q.sum())
        if ok:
            for s in SCHEMES:
                b, t = rec.get(f"E_elst_{s}_bs"), rec.get(f"E_elst_{s}_triplet")
                rec[f"delta_{s}"] = (b - t) if (b is not None and t is not None) else np.nan
            rows.append(rec)

    if not rows:
        print("No usable results.")
        for p in problems[:10]:
            print("  ", p)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out / "charges_by_scheme.csv", index=False)

    comp = df[["id", "system", "stratum", "published_delta_elst"]
              + [f"delta_{s}" for s in SCHEMES]].copy()
    for s in SCHEMES:
        comp[f"dev_{s}"] = comp[f"delta_{s}"] - comp["published_delta_elst"]
    comp.to_csv(out / "correction_comparison.csv", index=False)

    # ---- report ----------------------------------------------------------
    L = ["# Charge-scheme benchmark for the electrostatic correction", "",
         f"{len(df)} structures, four population analyses per state.", "",
         "## Absolute interfragment Coulomb energy, by scheme", "",
         "These are expected to differ; the schemes are not comparable in magnitude.", "",
         "| Scheme | mean E_elst(BS) | mean E_elst(T) | correlation BS vs T |",
         "|---|---|---|---|"]
    for s in SCHEMES:
        b, t = df[f"E_elst_{s}_bs"], df[f"E_elst_{s}_triplet"]
        if b.notna().sum() < 2:
            L.append(f"| {s} | not parsed | | |"); continue
        L.append(f"| {s} | {b.mean():.3f} | {t.mean():.3f} | {b.corr(t):.6f} |")
    L += ["", "## The quantity that enters Eq. 7: dE_elst = E_elst(BS) - E_elst(T)", "",
          "| Scheme | mean dE_elst | mean abs deviation from Mulliken | max abs deviation |",
          "|---|---|---|---|"]
    for s in SCHEMES:
        if comp[f"delta_{s}"].notna().sum() == 0:
            L.append(f"| {s} | not parsed | | |"); continue
        L.append(f"| {s} | {comp[f'delta_{s}'].mean():+.4f} "
                 f"| {comp[f'dev_{s}'].abs().mean():.4f} | {comp[f'dev_{s}'].abs().max():.4f} |")
    L += ["", "All values in kcal mol^-1. For reference the model's own test-set MAE",
          "on this target is 0.550 kcal mol^-1, so a deviation well below that is",
          "immaterial to the reported results.", ""]
    if problems:
        L += ["## Problems", ""] + [f"- {p}" for p in problems] + [""]
    (out / "CHARGE_SCHEME_REPORT.md").write_text("\n".join(L))

    # ---- figure ----------------------------------------------------------
    avail = [s for s in SCHEMES if comp[f"delta_{s}"].notna().sum() > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for s in avail:
        axes[0].scatter(comp["published_delta_elst"], comp[f"delta_{s}"],
                        s=34, alpha=0.8, label=s, edgecolor="black", lw=0.4)
    lim = [comp["published_delta_elst"].min() - 0.2, comp["published_delta_elst"].max() + 0.2]
    axes[0].plot(lim, lim, "k--", lw=1, label="identity")
    axes[0].set_xlabel("published Mulliken $\\Delta E_{elst}$ (kcal/mol)")
    axes[0].set_ylabel("recomputed $\\Delta E_{elst}$ (kcal/mol)")
    axes[0].set_title("Correction reproduced under four charge schemes")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].boxplot([comp[f"dev_{s}"].dropna() for s in avail], labels=avail)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].axhline(0.550, color="red", ls="--", lw=1, label="model MAE on this target")
    axes[1].axhline(-0.550, color="red", ls="--", lw=1)
    axes[1].set_ylabel("deviation from Mulliken (kcal/mol)")
    axes[1].set_title("Scheme dependence against the model's own error")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    for e in ("png", "pdf"):
        fig.savefig(out / f"fig_charge_schemes.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"parsed {len(df)} structures")
    print(f"fragment sizes seen: {sorted(df['frag_sizes'].unique())}")
    for s in SCHEMES:
        if comp[f"dev_{s}"].notna().sum():
            print(f"  {s:<10} mean |deviation from Mulliken| = "
                  f"{comp[f'dev_{s}'].abs().mean():.4f} kcal/mol "
                  f"(max {comp[f'dev_{s}'].abs().max():.4f})")
    if problems:
        print(f"\n{len(problems)} problems, see the report")
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
