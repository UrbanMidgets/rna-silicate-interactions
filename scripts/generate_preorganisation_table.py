#!/usr/bin/env python3
"""Measure pre-organisation in the solvated state against the dehydrated outcome.

Section 4.4.1 of the thesis argues that whether a P-O-Si linkage forms on
dehydration is decided while the complex is still solvated: reacting structures
already place a *free* phosphate oxygen on a near-vertical axis above a surface
silicon. The thesis enumerates that distance for seven structures. This script
computes it for every structure, so the claim can be assessed across the whole
dataset rather than the subset quoted.

"Free" means a non-bridging phosphate oxygen that is neither the C5' ester
oxygen nor the protonated oxygen of the monoanion. Both are distinguished by
connectivity rather than by index order:

    ester     - a phosphate oxygen also bonded to carbon
    protonated- a phosphate oxygen also bonded to hydrogen
    free      - the remainder

The approach angle is measured between the oxygen-to-silicon vector and the
slab normal, so a near-vertical approach gives a small angle.

Usage:
    python3 scripts/generate_preorganisation_table.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import MDAnalysis as mda  # noqa: E402
from MDAnalysis.lib.distances import distance_array  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

MANIFEST = REPO_ROOT / "data" / "MANIFEST.tsv"
CLASSIFICATIONS = REPO_ROOT / "data" / "classifications.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "preorganisation.csv"

P_O_CUTOFF = 1.9
O_C_CUTOFF = 1.6
O_H_CUTOFF = 1.2
FRAMEWORK = ("Si", "Al")


def _select_xyz(candidates: pd.DataFrame) -> Path | None:
    """Pick the converged solvated geometry.

    Several solvated directories also hold a copy of the initial docked pose
    named `frameNN.xyz`, which sorts before the converged file and is not an
    optimised structure. Prefer the optimisation trajectory, whose final frame
    is the converged geometry by construction, then a plain output file, and
    only then anything else. This matches the preference used by
    generate_classifications.py.
    """
    trajectory = candidates[candidates["repo_path"].str.endswith("_trj.xyz")]
    plain = candidates[
        ~candidates["repo_path"].str.endswith("_trj.xyz")
        & ~candidates["repo_path"].str.contains(r"/frame\d+\.xyz$", regex=True)
    ]
    for group in (trajectory, plain, candidates):
        for _, row in group.iterrows():
            path = REPO_ROOT / row["repo_path"]
            if path.exists():
                return path
    return None


def analyse(xyz_path: Path) -> dict | None:
    u = mda.Universe(str(xyz_path))
    u.trajectory[-1]
    names = np.asarray(u.atoms.names, dtype=str)
    pos = u.atoms.positions

    p_hits = np.where(names == "P")[0]
    if len(p_hits) == 0:
        return None
    p_idx = int(p_hits[0])

    idx = np.arange(len(names))
    framework = idx[np.isin(names, FRAMEWORK) & (idx < p_idx)]
    silicon = idx[(names == "Si") & (idx < p_idx)]
    if len(framework) < 3 or len(silicon) == 0:
        return None

    # Phosphate oxygens are found by distance to phosphorus, not by index.
    # Atom ordering is not guaranteed to place the whole nucleotide after P:
    # in the UMP structures the slab occupies indices 0-125 while P sits at 142,
    # so the C5' ester oxygen lies *below* P and an index filter drops it.
    # Slab oxygens are >3 A from phosphorus in every structure, so a distance
    # criterion is unambiguous.
    oxygens = idx[names == "O"]
    p_distances = distance_array(pos[p_idx][None, :], pos[oxygens])[0]
    phosphate_o = oxygens[p_distances <= P_O_CUTOFF]
    if len(phosphate_o) == 0:
        return None

    carbon = idx[(names == "C")]
    hydrogen = idx[(names == "H")]

    def bonded_to(group: np.ndarray, cutoff: float) -> np.ndarray:
        if len(group) == 0:
            return np.zeros(len(phosphate_o), dtype=bool)
        d = distance_array(pos[phosphate_o], pos[group])
        return np.any(d <= cutoff, axis=1)

    is_ester = bonded_to(carbon, O_C_CUTOFF)
    is_protonated = bonded_to(hydrogen, O_H_CUTOFF)
    free = phosphate_o[~is_ester & ~is_protonated]
    ester = phosphate_o[is_ester]

    # Slab normal, oriented so the adsorbate lies on the positive side.
    centre = pos[framework].mean(axis=0)
    _, _, vt = np.linalg.svd(pos[framework] - centre)
    normal = vt[2]
    # Orient using phosphorus, which is unambiguously part of the adsorbate.
    if float(np.dot(pos[p_idx] - centre, normal)) < 0:
        normal = -normal

    result = {
        "n_phosphate_oxygens": len(phosphate_o),
        "n_free_oxygens": len(free),
        "free_min_O_Si": np.nan,
        "free_O_index": -1,
        "free_Si_index": -1,
        "free_approach_angle_deg": np.nan,
        "ester_min_O_Si": np.nan,
    }

    if len(free):
        d = distance_array(pos[free], pos[silicon])
        flat = int(np.argmin(d))
        o_pos, si_pos = np.unravel_index(flat, d.shape)
        o_atom, si_atom = int(free[o_pos]), int(silicon[si_pos])
        vector = pos[si_atom] - pos[o_atom]
        norm = np.linalg.norm(vector)
        # Angle between the O->Si vector and the inward slab normal.
        cosine = float(np.dot(vector / norm, -normal))
        result.update(
            {
                "free_min_O_Si": float(d[o_pos, si_pos]),
                "free_O_index": o_atom,
                "free_Si_index": si_atom,
                "free_approach_angle_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
            }
        )

    if len(ester):
        result["ester_min_O_Si"] = float(distance_array(pos[ester], pos[silicon]).min())

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"error: {MANIFEST} not found", file=sys.stderr)
        return 2

    manifest = pd.read_csv(MANIFEST, sep="\t").fillna("")
    xyz = manifest[manifest["repo_path"].str.endswith(".xyz")]

    outcomes = {}
    if CLASSIFICATIONS.exists():
        cls = pd.read_csv(CLASSIFICATIONS)
        dry = cls[cls["calc_state"] == "dry"]
        for _, r in dry.iterrows():
            outcomes[(r["system"], r["surface"], r["frame"])] = r

    solvated = xyz[xyz["state"] == "solvated"]
    keys = solvated[["system", "surface", "frame"]].drop_duplicates()

    rows = []
    skipped = []
    for _, key in keys.iterrows():
        system, surface, frame = key["system"], key["surface"], key["frame"]
        if not frame:
            continue
        candidates = solvated[
            (solvated["system"] == system)
            & (solvated["surface"] == surface)
            & (solvated["frame"] == frame)
        ]
        path = _select_xyz(candidates)
        if path is None:
            skipped.append((f"{system}/{surface}/{frame}", "no solvated .xyz"))
            continue

        try:
            measured = analyse(path)
        except Exception as exc:  # noqa: BLE001
            skipped.append((f"{system}/{surface}/{frame}", repr(exc)))
            continue
        if measured is None:
            skipped.append((f"{system}/{surface}/{frame}", "no phosphorus or no slab"))
            continue

        outcome = outcomes.get((system, surface, frame))
        rows.append(
            {
                "system": system,
                "surface": surface,
                "frame": frame,
                **measured,
                "dry_state": None if outcome is None else outcome["predicted_state"],
                "dry_bond_order": None if outcome is None else outcome.get("Final_Bond_Order"),
                "dry_linkage_dist": None if outcome is None else outcome.get("Final_Linkage_Dist"),
                "solvated_xyz": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            }
        )

    if not rows:
        print("No solvated structures analysed.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows).sort_values(["surface", "system", "frame"])
    # Negative means the free oxygen is nearer a silicon than the ester oxygen,
    # which is the configuration section 4.4.1 identifies as productive.
    df["free_minus_ester"] = df["free_min_O_Si"] - df["ester_min_O_Si"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    linked = df[df["dry_state"] == "Covalently Condensed"]
    other = df[(df["dry_state"].notna()) & (df["dry_state"] != "Covalently Condensed")]

    print(f"{'system':<16}{'surface':<11}{'frame':<9}{'freeO-Si':>10}{'angle':>8}"
          f"{'esterO-Si':>11}  dry outcome")
    print("-" * 86)
    for _, r in df.iterrows():
        fo = "-" if pd.isna(r["free_min_O_Si"]) else f"{r['free_min_O_Si']:.2f}"
        an = "-" if pd.isna(r["free_approach_angle_deg"]) else f"{r['free_approach_angle_deg']:.0f}"
        es = "-" if pd.isna(r["ester_min_O_Si"]) else f"{r['ester_min_O_Si']:.2f}"
        print(f"{r['system']:<16}{r['surface']:<11}{r['frame']:<9}{fo:>10}{an:>8}{es:>11}"
              f"  {r['dry_state'] or 'n/a'}")

    def summarise(subset: pd.DataFrame, title: str) -> None:
        if subset.empty:
            return
        v = subset["free_min_O_Si"].dropna()
        a = subset["free_approach_angle_deg"].dropna()
        print(f"{title} (n={len(subset)}):")
        if len(v):
            print(f"  solvated free O-Si  : {v.min():.2f} - {v.max():.2f} A (mean {v.mean():.2f})")
        if len(a):
            print(f"  approach angle      : {a.min():.0f} - {a.max():.0f} deg (mean {a.mean():.0f})")
        d = subset["free_minus_ester"].dropna()
        if len(d):
            closer = int((d < 0).sum())
            print(f"  free oxygen nearer than ester: {closer}/{len(d)}")

    print()
    summarise(linked, "Structures that go on to react")
    summarise(other, "Structures that do not")

    # The discriminator proposed in section 4.4.1 is not merely that a phosphate
    # oxygen is close, but that a *free* one occupies the position above a
    # silicon rather than the unreactive C5' ester oxygen. Report how well that
    # separates outcomes, since the distance ranges themselves overlap.
    scored = df[df["free_minus_ester"].notna() & df["dry_state"].notna()]
    scored = scored[scored["dry_state"] != "Excluded (Off-Basal)"]
    if not scored.empty:
        reacted = scored["dry_state"] == "Covalently Condensed"
        predicted = scored["free_minus_ester"] < 0
        agree = int((reacted == predicted).sum())
        print(
            f"\n'free oxygen nearer than ester' predicts the dehydrated outcome in "
            f"{agree}/{len(scored)} basal structures ({agree / len(scored):.0%})"
        )
        wrong = scored[reacted != predicted]
        if not wrong.empty:
            print("  does not hold for:")
            for _, r in wrong.iterrows():
                print(
                    f"    {r['system']}/{r['surface']}/{r['frame']}: "
                    f"free {r['free_min_O_Si']:.2f} A, ester {r['ester_min_O_Si']:.2f} A"
                    f" -> {r['dry_state']}"
                )

    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for name, why in skipped:
            print(f"  {name}: {why}")

    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
