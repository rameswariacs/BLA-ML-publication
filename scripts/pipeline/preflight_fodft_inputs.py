#!/usr/bin/env python3
"""Preflight checks for the all-system FO-DFT postprocessing campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def terminated_normally(path: Path) -> bool:
    try:
        return "ORCA TERMINATED NORMALLY" in path.read_text(errors="ignore")
    except OSError:
        return False


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.manifest.open(newline="")))
    details: list[dict[str, object]] = []

    for row in rows:
        case_dir = Path(row["case_dir"])
        frag_a = Path(row["fragment_a_dir"])
        frag_b = Path(row["fragment_b_dir"])
        checks = {
            "fragment_A_out_normal": terminated_normally(frag_a / "fragment_A.out"),
            "fragment_B_out_normal": terminated_normally(frag_b / "fragment_B.out"),
            "fragment_A_gbw_present": nonempty(frag_a / "fragment_A.gbw"),
            "fragment_B_gbw_present": nonempty(frag_b / "fragment_B.gbw"),
            "triplet_gbw_present": nonempty(Path(row["triplet_gbw"])),
            "source_xyz_present": nonempty(Path(row["source_xyz"])),
            "case_dir_present": case_dir.is_dir(),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        details.append(
            {
                "system": row["system"],
                "orca_index": int(row["orca_index"]),
                "case_dir": str(case_dir),
                "status": status,
                **checks,
            }
        )

    failures = [r for r in details if r["status"] != "PASS"]
    by_system: dict[str, dict[str, int]] = {}
    for item in details:
        system = str(item["system"])
        by_system.setdefault(system, {"total": 0, "pass": 0, "fail": 0})
        by_system[system]["total"] += 1
        by_system[system]["pass" if item["status"] == "PASS" else "fail"] += 1

    with (args.out_dir / "fodft_preflight_detail.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    with (args.out_dir / "fodft_preflight_summary.json").open("w") as handle:
        json.dump(
            {
                "manifest": str(args.manifest),
                "n_cases": len(details),
                "n_pass": len(details) - len(failures),
                "n_fail": len(failures),
                "by_system": by_system,
                "strict": args.strict,
            },
            handle,
            indent=2,
        )

    report = [
        "# FO-DFT input preflight",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Cases checked: **{len(details)}**",
        f"- Passed: **{len(details) - len(failures)}**",
        f"- Failed: **{len(failures)}**",
        "",
        "## System accounting",
        "",
        "| system | total | pass | fail |",
        "|---|---:|---:|---:|",
    ]
    for system, counts in sorted(by_system.items()):
        report.append(f"| {system} | {counts['total']} | {counts['pass']} | {counts['fail']} |")
    if failures:
        report += ["", "## First failed cases", ""]
        for item in failures[:25]:
            failed_checks = [k for k, v in item.items() if k not in {"system", "orca_index", "case_dir", "status"} and not v]
            report.append(f"- {item['system']} orca_{item['orca_index']}: {', '.join(failed_checks)}")
    (args.out_dir / "FODFT_PREFLIGHT_REPORT.md").write_text("\n".join(report) + "\n")

    if failures and args.strict:
        raise SystemExit(f"Preflight failed for {len(failures)} cases. See {args.out_dir}")

    print(json.dumps({"n_cases": len(details), "n_fail": len(failures), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
