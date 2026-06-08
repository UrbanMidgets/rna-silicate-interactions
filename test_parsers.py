from __future__ import annotations

import warnings
from pathlib import Path

import MDAnalysis as mda

from parsers import (
    ClassicalConformationParser,
    ClassicalTopologyConfig,
    QuantumTopologyParser,
    StateClassifier,
)


warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_DIR = Path("data/primary_calculations/aluminium_surface/ump/selected_frames/frame15/dry")
SAMPLE_OUT = SAMPLE_DIR / "ump_al_frame15_dry.out"
SAMPLE_XYZ = SAMPLE_DIR / "ump_al_frame15_dry_trj.xyz"


def test() -> None:
    if not SAMPLE_OUT.exists() or not SAMPLE_XYZ.exists():
        raise FileNotFoundError("Sample AMP frame15 solvated files are missing")

    u = mda.Universe(str(SAMPLE_XYZ))
    p_idx = int(u.select_atoms("name P").indices[0])
    si_indices = [int(atom.index) for atom in u.atoms if atom.name == "Si" and atom.index < p_idx]
    oxygen_indices = [int(atom.index) for atom in u.atoms if atom.name == "O" and atom.index > p_idx]
    p_distances = mda.lib.distances.distance_array(u.atoms[p_idx].position[None, :], u.atoms[oxygen_indices].positions)[0]
    phosphate_oxygens = [
        oxygen_indices[i]
        for i, distance in enumerate(p_distances)
        if distance <= 1.9
    ]

    print("Testing QuantumTopologyParser...")
    q_df = QuantumTopologyParser(
        p_idx=p_idx,
        target_indices=phosphate_oxygens,
        target_element="O",
        partner_indices=si_indices,
        partner_element="Si",
    ).parse_file(SAMPLE_OUT)
    print(q_df[["File", "Step", "Target_Idx", "Si_Idx", "Bond_Order", "Target_Pair_Found"]].tail())

    print("\nTesting ClassicalConformationParser...")
    topology_config = ClassicalTopologyConfig(p_index=p_idx, surface_index_stop=p_idx)
    c_df = ClassicalConformationParser(topology_config).parse_trajectory(SAMPLE_XYZ)
    print(
        c_df[
            [
                "File",
                "Frame",
                "min_dist_intramol_OH_PO",
                "min_dist_phosphateO_surfaceSi",
                "steric_trap_hbond_count",
                "dist_PO_siloxane_COM",
            ]
        ].tail()
    )

    print("\nTesting StateClassifier...")
    result = StateClassifier().classify_pair(q_df, c_df, label=SAMPLE_OUT.stem)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test()
