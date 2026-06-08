from __future__ import annotations

import glob
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis.distances import distance_array


SelectionValue = str | int | range | Sequence[int]


def _as_index_array(value: SelectionValue) -> Optional[np.ndarray]:
    if isinstance(value, str):
        return None
    if isinstance(value, int):
        return np.array([value], dtype=int)
    if isinstance(value, range):
        return np.array(list(value), dtype=int)
    return np.array(list(value), dtype=int)


@dataclass(frozen=True)
class ClassicalTopologyConfig:
    """
    Configurable atom groups for XYZ trajectories.

    Values can be MDAnalysis selection strings, integer indices, ranges, or index lists.
    Indices are zero-based by default; set index_base=1 for human-facing atom numbers.
    """

    groups: Mapping[str, SelectionValue] = field(default_factory=dict)
    index_base: int = 0
    surface_index_stop: Optional[int] = None
    p_index: Optional[int] = None
    siloxane_rings: Mapping[str, SelectionValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index_base not in (0, 1):
            raise ValueError("index_base must be 0 or 1")

    def _normalize_indices(self, value: SelectionValue) -> SelectionValue:
        indices = _as_index_array(value)
        if indices is None:
            return value
        return (indices - self.index_base).tolist()

    def group(self, name: str) -> Optional[SelectionValue]:
        value = self.groups.get(name)
        if value is None:
            return None
        return self._normalize_indices(value)

    def ring_groups(self) -> Mapping[str, SelectionValue]:
        return {name: self._normalize_indices(value) for name, value in self.siloxane_rings.items()}


class ClassicalConformationParser:
    """
    Parse XYZ trajectories and measure conformation metrics needed for state assignment.

    XYZ files lack bonds, residue names, and donor/acceptor topology. This parser therefore
    supports exact atom-index groups and uses distance cutoffs as the default H-bond proxy.
    """

    def __init__(
        self,
        topology_config: ClassicalTopologyConfig | Mapping[str, SelectionValue] | None = None,
        hbond_dist_cutoff: float = 3.0,
        hbond_angle_cutoff: float = 130.0,
    ) -> None:
        if topology_config is None:
            topology_config = ClassicalTopologyConfig()
        elif not isinstance(topology_config, ClassicalTopologyConfig):
            topology_config = ClassicalTopologyConfig(groups=topology_config)

        self.topology_config = topology_config
        self.hbond_dist_cutoff = hbond_dist_cutoff
        self.hbond_angle_cutoff = hbond_angle_cutoff

    def _empty_result(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "File",
                "Frame",
                "Time",
                "min_dist_nh2_surfO",
                "hbond_nh2_surfO_count",
                "min_dist_riboseOH_surfSilanol",
                "hbond_riboseOH_surfSilanol_count",
                "min_dist_phosphateO_surfaceO",
                "phosphateO_surfaceO_hbond_count",
                "min_dist_intramol_OH_PO",
                "steric_trap_hbond_count",
                "min_dist_phosphateO_surfaceSi",
                "dist_PO_siloxane_COM",
            ]
        )

    @staticmethod
    def _select(u: mda.Universe, value: Optional[SelectionValue]) -> mda.AtomGroup:
        if value is None:
            return u.atoms[:0]

        indices = _as_index_array(value)
        if indices is not None:
            valid = indices[(indices >= 0) & (indices < len(u.atoms))]
            return u.atoms[valid]

        return u.select_atoms(value)

    @staticmethod
    def _element_indices(u: mda.Universe, element: str) -> np.ndarray:
        names = np.asarray(u.atoms.names, dtype=str)
        return np.where(np.char.upper(names) == element.upper())[0]

    def _infer_config(self, u: mda.Universe) -> dict[str, SelectionValue]:
        groups = dict(self.topology_config.groups)

        p_indices = self._element_indices(u, "P")
        p_idx = self.topology_config.p_index
        if p_idx is None and len(p_indices):
            p_idx = int(p_indices[0])
        if p_idx is not None:
            p_idx -= self.topology_config.index_base

        surface_stop = self.topology_config.surface_index_stop
        if surface_stop is None:
            surface_stop = p_idx if p_idx is not None else len(u.atoms)
        else:
            surface_stop -= self.topology_config.index_base

        oxygen_indices = self._element_indices(u, "O")
        nitrogen_indices = self._element_indices(u, "N")
        silicon_indices = self._element_indices(u, "Si")
        hydrogen_indices = self._element_indices(u, "H")

        surface_o = oxygen_indices[oxygen_indices < surface_stop]
        surface_si = silicon_indices[silicon_indices < surface_stop]
        nucleotide_o = oxygen_indices[oxygen_indices > p_idx] if p_idx is not None else oxygen_indices

        phosphate_oxygens: np.ndarray
        if p_idx is not None and len(nucleotide_o):
            p_pos = u.atoms[p_idx].position[None, :]
            distances = distance_array(p_pos, u.atoms[nucleotide_o].positions)[0]
            phosphate_oxygens = nucleotide_o[distances <= 1.9]
        else:
            phosphate_oxygens = np.array([], dtype=int)

        nucleotide_h = hydrogen_indices[hydrogen_indices > surface_stop]
        non_phosphate_o = np.setdiff1d(nucleotide_o, phosphate_oxygens, assume_unique=False)
        if len(non_phosphate_o) and len(nucleotide_h):
            oh_distances = distance_array(u.atoms[non_phosphate_o].positions, u.atoms[nucleotide_h].positions)
            ribose_oh = non_phosphate_o[np.any(oh_distances <= 1.25, axis=1)]
        else:
            ribose_oh = np.array([], dtype=int)
        nucleobase_n = nitrogen_indices[nitrogen_indices > surface_stop]
        siloxane_atoms = np.sort(np.concatenate([silicon_indices[silicon_indices < surface_stop], surface_o]))

        groups.setdefault("adenine_nh2", nucleobase_n.tolist())
        groups.setdefault("surface_oxygens", surface_o.tolist())
        groups.setdefault("surface_silicons", surface_si.tolist())
        groups.setdefault("ribose_oh", ribose_oh.tolist())
        groups.setdefault("surface_silanols", surface_o.tolist())
        groups.setdefault("phosphate_oxygens", phosphate_oxygens.tolist())
        groups.setdefault("phosphate_surface_hbond_acceptors", phosphate_oxygens.tolist())
        groups.setdefault("phosphate_oxygen_target", phosphate_oxygens[:1].tolist())
        groups.setdefault("siloxane_rings", siloxane_atoms.tolist())

        return groups

    def _pair_metrics(
        self,
        u: mda.Universe,
        group_a: SelectionValue,
        group_b: SelectionValue,
        min_name: str,
        count_name: str,
    ) -> dict[str, float]:
        atoms_a = self._select(u, group_a)
        atoms_b = self._select(u, group_b)
        if len(atoms_a) == 0 or len(atoms_b) == 0:
            return {min_name: np.nan, count_name: 0}

        distances = distance_array(atoms_a.positions, atoms_b.positions)
        return {
            min_name: float(np.min(distances)),
            count_name: int(np.sum(distances <= self.hbond_dist_cutoff)),
        }

    def _siloxane_metric(self, u: mda.Universe, groups: Mapping[str, SelectionValue]) -> dict[str, float]:
        phosphate_o = self._select(u, groups.get("phosphate_oxygen_target"))
        if len(phosphate_o) == 0:
            return {"dist_PO_siloxane_COM": np.nan, "nearest_siloxane_ring": ""}

        ring_groups = self.topology_config.ring_groups()
        if not ring_groups and "siloxane_rings" in groups:
            ring_groups = {"siloxane_rings": groups["siloxane_rings"]}

        best_name = ""
        best_distance = np.inf
        po_pos = phosphate_o.positions[0]
        for ring_name, ring_value in ring_groups.items():
            ring_atoms = self._select(u, ring_value)
            if len(ring_atoms) == 0:
                continue
            ring_com = ring_atoms.center_of_mass()
            distance = float(np.linalg.norm(po_pos - ring_com))
            if distance < best_distance:
                best_distance = distance
                best_name = ring_name

        if not np.isfinite(best_distance):
            best_distance = np.nan

        return {"dist_PO_siloxane_COM": best_distance, "nearest_siloxane_ring": best_name}

    def parse_trajectory(self, xyz_path: str | os.PathLike[str]) -> pd.DataFrame:
        path = Path(xyz_path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        u = mda.Universe(str(path))
        inferred_groups = self._infer_config(u)
        rows: list[dict[str, object]] = []

        for ts in u.trajectory:
            row: dict[str, object] = {
                "File": path.name,
                "Frame": ts.frame,
                "Time": float(getattr(ts, "time", ts.frame)),
                "file": path.name,
                "frame": ts.frame,
                "time": float(getattr(ts, "time", ts.frame)),
            }
            row.update(
                self._pair_metrics(
                    u,
                    inferred_groups["adenine_nh2"],
                    inferred_groups["surface_oxygens"],
                    "min_dist_nh2_surfO",
                    "hbond_nh2_surfO_count",
                )
            )
            row.update(
                self._pair_metrics(
                    u,
                    inferred_groups["ribose_oh"],
                    inferred_groups["surface_silanols"],
                    "min_dist_riboseOH_surfSilanol",
                    "hbond_riboseOH_surfSilanol_count",
                )
            )
            row.update(
                self._pair_metrics(
                    u,
                    inferred_groups["ribose_oh"],
                    inferred_groups["phosphate_oxygens"],
                    "min_dist_intramol_OH_PO",
                    "steric_trap_hbond_count",
                )
            )
            row.update(
                self._pair_metrics(
                    u,
                    inferred_groups["phosphate_surface_hbond_acceptors"],
                    inferred_groups["surface_oxygens"],
                    "min_dist_phosphateO_surfaceO",
                    "phosphateO_surfaceO_hbond_count",
                )
            )
            row.update(
                self._pair_metrics(
                    u,
                    inferred_groups["phosphate_oxygens"],
                    inferred_groups["surface_silicons"],
                    "min_dist_phosphateO_surfaceSi",
                    "phosphateO_surfaceSi_contact_count",
                )
            )
            row.update(self._siloxane_metric(u, inferred_groups))
            rows.append(row)

        if not rows:
            return self._empty_result()
        return pd.DataFrame(rows)

    def process_directory(self, dir_path: str | os.PathLike[str], n_jobs: int = 4) -> pd.DataFrame:
        xyz_files = sorted(glob.glob(os.path.join(str(dir_path), "*.xyz")))
        return self.process_files(xyz_files, n_jobs=n_jobs)

    def process_files(self, xyz_files: Sequence[str | os.PathLike[str]], n_jobs: int = 4) -> pd.DataFrame:
        files = [str(path) for path in xyz_files]
        if not files:
            return self._empty_result()

        if n_jobs <= 1 or len(files) == 1:
            results = [self.parse_trajectory(path) for path in files]
        else:
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.map(self.parse_trajectory, files)

        results = [df for df in results if not df.empty]
        if not results:
            return self._empty_result()
        return pd.concat(results, ignore_index=True)
