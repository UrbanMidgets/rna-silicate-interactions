from __future__ import annotations

import warnings
import argparse
from pathlib import Path
import sys

import MDAnalysis as mda
import pandas as pd
from MDAnalysis.analysis.distances import distance_array

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parsers import (
    ClassicalConformationParser,
    ClassicalTopologyConfig,
    QuantumTopologyParser,
    StateClassifier,
)


warnings.filterwarnings("ignore", category=UserWarning)

MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.tsv"
OUTPUT_CSV = REPO_ROOT / "data" / "classifications.csv"


def _select_first_existing_xyz(candidates: pd.DataFrame) -> Path | None:
    if candidates.empty:
        return None

    trj = candidates[candidates["repo_path"].str.endswith("_trj.xyz")]
    ordered = trj if not trj.empty else candidates
    for _, row in ordered.iterrows():
        path = REPO_ROOT / row["repo_path"]
        if path.exists():
            return path
    return None


def _dedupe_outputs(out_files: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Keep one ORCA output per (system, surface, frame, state).

    Several keys carry both an optimisation output and its `_engrad` gradient
    companion. The companion holds no Mayer population block for the final
    geometry, so classifying it produces a spurious second row reporting bond
    order 0 for a structure that is in fact linked. Prefer the optimisation.
    """
    if out_files.empty:
        return out_files, []

    ranked = out_files.copy()
    ranked["_is_engrad"] = ranked["repo_path"].str.contains(r"_engrad\.out$", regex=True)
    ranked["_status_rank"] = (ranked.get("status", "") == "normal").astype(int)
    ranked = ranked.sort_values(
        ["_is_engrad", "_status_rank", "repo_path"], ascending=[True, False, True]
    )

    keys = ["system", "surface", "frame", "state"]
    kept = ranked.drop_duplicates(subset=keys, keep="first")
    dropped = ranked.loc[~ranked.index.isin(kept.index)]

    notes = [
        (row["repo_path"], "duplicate output for this key; superseded by the optimisation run")
        for _, row in dropped.iterrows()
    ]
    return kept.drop(columns=["_is_engrad", "_status_rank"]), notes


def _linkage_distance(xyz_path: Path, q_df: pd.DataFrame) -> float:
    """Distance of the specific O-Si pair that population analysis identified.

    `min_dist_phosphateO_surfaceSi` is a minimum over every phosphate oxygen and
    every surface silicon, which need not be the pair that actually carries the
    bond. Table 4.1 reports the bonded pair, so report that here too.
    """
    import numpy as np

    if q_df.empty:
        return float("nan")

    ordered = q_df.sort_values("Step") if "Step" in q_df.columns else q_df
    final = ordered.iloc[-1]
    target = final.get("Target_Idx")
    partner = final.get("Si_Idx")
    if pd.isna(target) or pd.isna(partner):
        return float("nan")

    u = mda.Universe(str(xyz_path))
    u.trajectory[-1]  # converged geometry, not the starting frame

    try:
        a = u.atoms[int(target)].position
        b = u.atoms[int(partner)].position
    except IndexError:
        return float("nan")
    return float(np.linalg.norm(a - b))


def _build_topology_config(u: mda.Universe) -> ClassicalTopologyConfig:
    names = pd.Series(u.atoms.names)
    p_indices = names[names == "P"].index.tolist()
    p_idx = p_indices[0] if p_indices else None
    surface_stop = p_idx

    return ClassicalTopologyConfig(
        p_index=p_idx,
        surface_index_stop=surface_stop,
    )


def _surface_si_indices(u: mda.Universe, p_idx: int) -> list[int]:
    return [
        int(atom.index)
        for atom in u.atoms
        if atom.name == "Si" and atom.index < p_idx
    ]


def _phosphate_oxygen_indices(u: mda.Universe, p_idx: int) -> list[int]:
    oxygen_indices = [int(atom.index) for atom in u.atoms if atom.name == "O" and atom.index > p_idx]
    if not oxygen_indices:
        return []
    distances = distance_array(u.atoms[p_idx].position[None, :], u.atoms[oxygen_indices].positions)[0]
    return [
        oxygen_indices[i]
        for i, distance in enumerate(distances)
        if distance <= 1.9
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QMD final-state classifications from ORCA .out and XYZ files.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to data/MANIFEST.tsv")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="CSV file to write")
    parser.add_argument("--system", help="Optional system filter, e.g. amp or amp_protonated")
    parser.add_argument("--surface", help="Optional surface filter, e.g. canonical or aluminium")
    parser.add_argument("--state", help="Optional state filter, e.g. solvated or dry")
    parser.add_argument("--frame", help="Optional frame filter, e.g. frame15")
    parser.add_argument("--limit", type=int, help="Process at most this many ORCA rows after filtering")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    print(f"Loading manifest from {manifest_path}...")
    manifest = pd.read_csv(manifest_path, sep="\t").fillna("")

    for column, value in {
        "system": args.system,
        "surface": args.surface,
        "state": args.state,
        "frame": args.frame,
    }.items():
        if value:
            manifest = manifest[manifest[column] == value]

    out_files = manifest[manifest["repo_path"].str.endswith(".out")]
    out_files, duplicate_notes = _dedupe_outputs(out_files)
    if args.limit:
        out_files = out_files.head(args.limit)

    xyz_files = manifest[manifest["repo_path"].str.endswith(".xyz")]
    rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = list(duplicate_notes)

    for _, out_row in out_files.iterrows():
        system = out_row["system"]
        surface = out_row["surface"]
        frame = out_row["frame"]
        state = out_row["state"]
        out_path = REPO_ROOT / out_row["repo_path"]
        if not out_path.exists():
            skipped.append((out_row["repo_path"], "output file missing on disk"))
            continue

        matching_xyz = xyz_files[
            (xyz_files["system"] == system)
            & (xyz_files["surface"] == surface)
            & (xyz_files["frame"] == frame)
            & (xyz_files["state"] == state)
        ]
        xyz_path = _select_first_existing_xyz(matching_xyz)
        if xyz_path is None:
            skipped.append((out_row["repo_path"], "no matching .xyz for this key"))
            continue

        print(f"Processing: {system} | {surface} | {frame} | {state}")

        try:
            u = mda.Universe(str(xyz_path))
            p_atoms = u.select_atoms("name P")
            if len(p_atoms) == 0:
                print(f"  Skipping: no phosphorus atom found in {xyz_path.name}")
                continue

            p_idx = int(p_atoms.indices[0])
            si_indices = _surface_si_indices(u, p_idx)
            phosphate_oxygens = _phosphate_oxygen_indices(u, p_idx)

            q_df = QuantumTopologyParser(
                p_idx=p_idx,
                target_indices=phosphate_oxygens,
                target_element="O",
                partner_indices=si_indices,
                partner_element="Si",
            ).parse_file(out_path)
            c_df = ClassicalConformationParser(_build_topology_config(u)).parse_trajectory(xyz_path)
            result = StateClassifier().classify_pair(q_df, c_df, label=out_path.stem)
            result["Final_Linkage_Dist"] = _linkage_distance(xyz_path, q_df)
            result.update(
                {
                    "system": system,
                    "surface": surface,
                    "frame": frame,
                    "calc_state": state,
                    "role": str(out_row.get("role", "")),
                    "orca_status": str(out_row.get("status", "")),
                    "out_file": out_path.name,
                    "xyz_file": xyz_path.name,
                }
            )
            rows.append(result)
        except Exception as exc:
            failures.append((out_path.name, repr(exc)))
            print(f"  Error processing {out_path.name}: {exc}")

    results = pd.DataFrame(rows)
    if results.empty:
        print("\nNo classifications generated.")
        return 1

    # 'calc_state' is the calculation stage (solvated/dry/...); 'predicted_state'
    # is the classifier label. These were previously 'state' and 'State', which
    # differ only by case and break every case-insensitive CSV consumer.
    results = results.rename(columns={"State": "predicted_state"})

    preferred_columns = [
        "system",
        "surface",
        "frame",
        "calc_state",
        "role",
        "orca_status",
        "predicted_state",
        "Selection_Pass",
        "Reason",
        "Final_Bond_Order",
        "Final_Si_Idx",
        "Final_Intramol_Dist",
        "Final_Intramol_HBond_Count",
        "Final_Anchoring_Dist",
        "Final_NH2_SurfaceO_Dist",
        "Final_RiboseOH_SurfaceSilanol_Dist",
        "Final_PhosphateO_SurfaceO_Dist",
        "Final_PhosphateO_SurfaceSi_Dist",
        "Final_Linkage_Dist",
        "Final_PO_Siloxane_Ring_Centroid_Dist",
        "Final_Nucleotide_EdgeOH_Dist",
        "Final_Basal_Height",
        "Final_Overhang_Fraction",
        "out_file",
        "xyz_file",
    ]
    results = results[[col for col in preferred_columns if col in results.columns]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\nSuccessfully processed {len(results)} trajectories.")
    print(f"Saved classifications to {output_path}")

    if skipped:
        print(f"\n{len(skipped)} input(s) skipped:")
        for name, why in skipped:
            print(f"  {name}: {why}")

    if failures:
        print(f"\n{len(failures)} input(s) FAILED:")
        for name, exc in failures:
            print(f"  {name}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
