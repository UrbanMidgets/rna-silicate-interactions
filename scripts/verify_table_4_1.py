#!/usr/bin/env python3
"""Verify the generated classifications against thesis Table 4.1.

Table 4.1 reports twenty-six optimised mononucleotide-silicate complexes. This
script checks that the pipeline in `parsers/` reproduces it from the deposited
calculations, and writes a row-by-row reconciliation.

Structure labels are mapped onto repository paths below. For the thirteen
linked structures the mapping is pinned by the published Mayer bond order,
which is a unique fingerprint. For the unlinked structures the conformer letter
within a system is not pinned, since distinguishing 6a from 6b would require
reproducing the relative-energy ordering; those are therefore verified as a set
("no linkage forms in either conformer of system 6") rather than individually.

Usage:
    python3 scripts/verify_table_4_1.py
    python3 scripts/verify_table_4_1.py --output data/table_4_1_reconciliation.csv

Exits non-zero if any structure fails to reconcile.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSIFICATIONS = REPO_ROOT / "data" / "classifications.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "table_4_1_reconciliation.csv"

# Tolerances. Table 4.1 quotes bond orders to three decimals and distances to
# two, so these are the tightest meaningful windows.
BOND_ORDER_TOL = 0.002
DISTANCE_TOL = 0.01

# (label, system, surface, frame, published bond order, published d(O-Si) / A)
LINKED = [
    ("1a", "amp", "canonical", "frame12", 0.913, 1.77),
    ("1b", "amp", "canonical", "frame15", 0.936, 1.76),
    ("2a", "amp", "aluminium", "frame15", 0.859, 1.81),
    ("3a", "amp_protonated", "canonical", "frame02", 0.885, 1.79),
    ("5a", "cmp", "canonical", "frame14", 0.661, 1.93),
    ("7a", "cmp_protonated", "canonical", "frame11", 0.901, 1.78),
    ("7b", "cmp_protonated", "canonical", "frame09", 0.852, 1.81),
    ("9a", "gmp", "canonical", "frame16", 0.666, 1.94),
    ("10b", "gmp", "aluminium", "frame59", 0.918, 1.79),
    ("11a", "ump", "canonical", "frame14", 0.952, 1.75),
    ("11b", "ump", "canonical", "frame15", 0.936, 1.76),
    ("11c", "ump", "canonical", "frame16", 0.940, 1.76),
    ("12b", "ump", "aluminium", "frame15", 0.896, 1.79),
]

# Marked with a dagger in Table 4.1: migrated to the slab periphery during
# dehydration and excluded from all comparison under section 3.5.3.
EXCLUDED = [
    ("4a", "amp_protonated", "aluminium", "frame12"),
    ("4b", "amp_protonated", "aluminium", "frame14"),
    ("10a", "gmp", "aluminium", "frame35"),
    ("12a", "ump", "aluminium", "frame14"),
]

# No linkage forms. Conformer letters within a system are not pinned; see the
# module docstring.
UNLINKED = [
    ("1c", "amp", "canonical", "frame11"),
    ("2b", "amp", "aluminium", "frame16"),
    ("3b", "amp_protonated", "canonical", "frame10"),
    ("5b", "cmp", "canonical", "frame15"),
    ("6a/6b", "cmp", "aluminium", "frame14"),
    ("6a/6b", "cmp", "aluminium", "frame15"),
    ("8a/8b", "cmp_protonated", "aluminium", "frame02"),
    ("8a/8b", "cmp_protonated", "aluminium", "frame03"),
    ("9b", "gmp", "canonical", "frame15"),
]

LINKED_STATE = "Covalently Condensed"
EXCLUDED_STATE = "Excluded (Off-Basal)"

# Structures where the deposited geometry and Table 4.1 disagree, with the
# reason the deposited value is taken to be correct. Reported prominently but
# not treated as a pipeline failure; --strict overrides.
#
# For both, the published Mayer bond order matches the deposited calculation
# exactly, the converged .xyz agrees with the final trajectory frame, and the
# next-nearest phosphate O-Si contact lies beyond 3.4 A, so there is no
# ambiguity about which atom pair carries the bond. Ten of the twelve other
# linkages match the table to the quoted precision.
KNOWN_DISCREPANCIES = {
    "2a": "Table 4.1 quotes 1.81 A; deposited geometry gives 1.79 A",
    "3a": "Table 4.1 quotes 1.79 A; deposited geometry gives 1.77 A",
}


def _lookup(df: pd.DataFrame, system: str, surface: str, frame: str) -> pd.Series | None:
    match = df[
        (df["system"] == system)
        & (df["surface"] == surface)
        & (df["frame"] == frame)
        & (df["calc_state"] == "dry")
    ]
    if len(match) == 0:
        return None
    if len(match) > 1:
        raise RuntimeError(
            f"{system}/{surface}/{frame}: {len(match)} dry rows; expected one. "
            "Deduplication in generate_classifications.py has regressed."
        )
    return match.iloc[0]


def _row(label, system, surface, frame, expected, row, notes):
    return {
        "structure": label,
        "system": system,
        "surface": surface,
        "frame": frame,
        "expected": expected,
        "observed": "ABSENT" if row is None else row["predicted_state"],
        "published_bond_order": None,
        "observed_bond_order": None if row is None else row.get("Final_Bond_Order"),
        "published_distance": None,
        "observed_distance": None if row is None else row.get("Final_Linkage_Dist"),
        "verdict": notes,
    }


def verify(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows: list[dict] = []
    failures = 0

    for label, system, surface, frame, bond_order, distance in LINKED:
        row = _lookup(df, system, surface, frame)
        entry = _row(label, system, surface, frame, LINKED_STATE, row, "")
        entry["published_bond_order"] = bond_order
        entry["published_distance"] = distance

        if row is None:
            entry["verdict"] = "ABSENT from repository"
        else:
            problems = []
            if row["predicted_state"] != LINKED_STATE:
                problems.append(f"state is {row['predicted_state']!r}")
            if abs(row["Final_Bond_Order"] - bond_order) > BOND_ORDER_TOL:
                problems.append(
                    f"bond order {row['Final_Bond_Order']:.3f} vs published {bond_order:.3f}"
                )
            observed_d = row.get("Final_Linkage_Dist")
            if pd.isna(observed_d):
                problems.append("no linkage distance recorded")
            elif abs(observed_d - distance) > DISTANCE_TOL:
                problems.append(f"distance {observed_d:.2f} A vs published {distance:.2f} A")
            entry["verdict"] = "OK" if not problems else "; ".join(problems)

        if entry["verdict"] != "OK":
            if label in KNOWN_DISCREPANCIES:
                entry["verdict"] = f"KNOWN: {KNOWN_DISCREPANCIES[label]}"
            else:
                failures += 1
        rows.append(entry)

    for label, system, surface, frame in EXCLUDED:
        row = _lookup(df, system, surface, frame)
        entry = _row(label + " (dagger)", system, surface, frame, EXCLUDED_STATE, row, "")
        if row is None:
            entry["verdict"] = "ABSENT from repository"
        elif row["predicted_state"] != EXCLUDED_STATE:
            entry["verdict"] = f"state is {row['predicted_state']!r}, expected exclusion"
        else:
            entry["verdict"] = "OK"
        if entry["verdict"] != "OK":
            failures += 1
        rows.append(entry)

    for label, system, surface, frame in UNLINKED:
        row = _lookup(df, system, surface, frame)
        entry = _row(label, system, surface, frame, "no linkage", row, "")
        if row is None:
            entry["verdict"] = "ABSENT from repository"
        elif row["predicted_state"] == LINKED_STATE:
            entry["verdict"] = "linkage reported where Table 4.1 reports none"
        else:
            entry["verdict"] = "OK"
        if entry["verdict"] != "OK":
            failures += 1
        rows.append(entry)

    return pd.DataFrame(rows), failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifications", type=Path, default=DEFAULT_CLASSIFICATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail on the documented Table 4.1 discrepancies",
    )
    args = parser.parse_args()

    if not args.classifications.exists():
        print(
            f"error: {args.classifications} not found. "
            "Run scripts/generate_classifications.py first.",
            file=sys.stderr,
        )
        return 2

    df = pd.read_csv(args.classifications)
    report, failures = verify(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    if not args.quiet:
        width = max(len(str(v)) for v in report["verdict"])
        print(f"{'struct':<13}{'system':<16}{'surface':<11}{'frame':<9}{'verdict'}")
        print("-" * (49 + min(width, 60)))
        for _, r in report.iterrows():
            print(
                f"{r['structure']:<13}{r['system']:<16}{r['surface']:<11}"
                f"{r['frame']:<9}{r['verdict']}"
            )
        print()

    known = report[report["verdict"].str.startswith("KNOWN:")]
    total = len(report)
    reconciled = total - failures - len(known)

    print(f"Table 4.1 reconciliation: {reconciled} / {total} structures verified exactly")
    print(f"  linked   : {len(LINKED)}")
    print(f"  excluded : {len(EXCLUDED)} (marked with a dagger in Table 4.1)")
    print(f"  unlinked : {len(UNLINKED)}")

    if not known.empty:
        print(f"\n{len(known)} documented discrepancy/ies between Table 4.1 and the deposited data:")
        for _, r in known.iterrows():
            print(f"  {r['structure']}: {r['verdict'][7:]}")

    print(f"\nWrote {args.output}")

    if failures:
        print(f"\n{failures} structure(s) did not reconcile.", file=sys.stderr)
        return 1
    if known.empty is False and args.strict:
        print("\n--strict: documented discrepancies treated as failures.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
