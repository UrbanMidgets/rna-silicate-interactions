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

    # Chemistry cutoffs. Previously hard-coded in method bodies, which made the
    # class non-configurable despite existing for that purpose.
    p_o_bond_cutoff: float = 1.9
    o_h_bond_cutoff: float = 1.25
    si_o_bond_cutoff: float = 1.85
    ring_size: int = 6
    # Tetrahedral framework cations. Aluminium must be included: in the
    # Al-substituted slab the four oxygens around the substitution bridge Al-O-Si,
    # so counting Si neighbours alone makes them look terminal and wrongly marks
    # them as edge hydroxyls.
    framework_elements: tuple[str, ...] = ("Si", "Al")
    # Half-thickness (A) of the band around the slab mid-plane within which a
    # terminating oxygen counts as edge- rather than face-terminating.
    face_plane_tolerance: float = 1.0

    def __post_init__(self) -> None:
        if self.index_base not in (0, 1):
            raise ValueError("index_base must be 0 or 1")
        if self.ring_size < 3:
            raise ValueError("ring_size must be at least 3")

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
                "min_dist_nucleotide_edgeOH",
                "nucleotide_edgeOH_contact_count",
                "min_dist_nucleotide_apicalOH",
                "nucleotide_apicalOH_contact_count",
                "nucleotide_min_height_above_plane",
                "nucleotide_overhang_fraction",
                "dist_PO_siloxane_ring_centroid",
                "nearest_siloxane_ring",
                "siloxane_ring_phosphate_oxygen",
                "n_siloxane_rings",
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

    @staticmethod
    def _find_cycles(neighbours: Mapping[int, set[int]], length: int) -> list[tuple[int, ...]]:
        """Enumerate simple cycles of exactly `length` nodes in an undirected graph.

        Each cycle is returned once, as a tuple in traversal order starting from
        its lowest-numbered member.
        """
        seen: set[tuple[int, ...]] = set()
        cycles: list[tuple[int, ...]] = []
        for start in sorted(neighbours):
            stack: list[tuple[int, list[int]]] = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) == length:
                    if start in neighbours[node]:
                        key = tuple(sorted(path))
                        if key not in seen:
                            seen.add(key)
                            cycles.append(tuple(path))
                    continue
                for nb in sorted(neighbours[node]):
                    # Only extend through nodes above `start` so each cycle is
                    # discovered from its lowest member exactly once.
                    if nb <= start or nb in path:
                        continue
                    stack.append((nb, path + [nb]))
        return cycles

    def _framework_indices(self, u: mda.Universe, surface_stop: Optional[int]) -> np.ndarray:
        """Tetrahedral framework cations (Si and any substituted Al) in the slab."""
        parts = [self._element_indices(u, el) for el in self.topology_config.framework_elements]
        if not parts:
            return np.array([], dtype=int)
        framework = np.unique(np.concatenate(parts)) if len(parts) > 1 else parts[0]
        if surface_stop is not None:
            framework = framework[framework < surface_stop]
        return framework

    def _surface_bond_matrix(
        self,
        u: mda.Universe,
        framework: np.ndarray,
        surface_o: np.ndarray,
    ) -> np.ndarray:
        """Boolean (n_framework, n_o) cation-O bonding matrix for the slab."""
        if len(framework) == 0 or len(surface_o) == 0:
            return np.zeros((len(framework), len(surface_o)), dtype=bool)
        d = distance_array(u.atoms[framework].positions, u.atoms[surface_o].positions)
        return d <= self.topology_config.si_o_bond_cutoff

    def _classify_surface_hydroxyls(
        self,
        u: mda.Universe,
        framework: np.ndarray,
        surface_o: np.ndarray,
        bonded: np.ndarray,
        nucleotide_indices: np.ndarray,
    ) -> dict[str, list[int]]:
        """Split terminating oxygens into apical and edge sets.

        Faces are assigned from the perpendicular height of each terminating
        oxygen above the mean plane of the slab silicons, the plane obtained by
        singular value decomposition and oriented so the adsorbate lies on the
        positive side. Oxygens lying close to that plane but far from the slab
        axis terminate the edges; those displaced away from the adsorbate
        terminate the apical face.

        Both sets exist only to close a finite cluster and have no counterpart in
        an extended phyllosilicate, which is why contact with them disqualifies a
        pose.
        """
        empty = {"apical_hydroxyl_oxygens": [], "edge_hydroxyl_oxygens": []}
        if len(framework) < 3 or len(surface_o) == 0 or bonded.size == 0:
            return empty

        terminal = surface_o[bonded.sum(axis=0) == 1]
        if len(terminal) == 0:
            return empty

        centre = u.atoms[framework].positions.mean(axis=0)
        _, _, vt = np.linalg.svd(u.atoms[framework].positions - centre)
        normal = vt[2]

        # Orient so the adsorbate is on the positive side (thesis convention).
        if len(nucleotide_indices):
            probe = u.atoms[nucleotide_indices].positions.mean(axis=0)
            if float(np.dot(probe - centre, normal)) < 0:
                normal = -normal

        offsets = u.atoms[terminal].positions - centre
        height = offsets @ normal
        tolerance = self.topology_config.face_plane_tolerance

        edge = terminal[np.abs(height) <= tolerance]
        apical = terminal[height < -tolerance]
        return {
            "apical_hydroxyl_oxygens": apical.tolist(),
            "edge_hydroxyl_oxygens": edge.tolist(),
        }

    def _detect_siloxane_rings(
        self,
        u: mda.Universe,
        framework: np.ndarray,
        surface_o: np.ndarray,
        bonded: np.ndarray,
    ) -> dict[str, list[int]]:
        """Identify n-membered siloxane rings from Si-O connectivity.

        A ring is n silicons joined in a cycle by n bridging oxygens, where a
        bridging oxygen is one bonded to exactly two silicons. The returned atom
        list for each ring contains both the silicons and the bridging oxygens
        that join them, in cyclic order.

        Returns an empty mapping when the topology yields no rings. Callers must
        treat that as "not measurable" rather than substituting another group.
        """
        if len(framework) == 0 or len(surface_o) == 0 or bonded.size == 0:
            return {}

        # A bridging oxygen connects exactly two framework cations.
        bridging_local = np.where(bonded.sum(axis=0) == 2)[0]
        if len(bridging_local) == 0:
            return {}

        bridge_of: dict[tuple[int, int], int] = {}
        neighbours: dict[int, set[int]] = {int(s): set() for s in framework}
        for local_o in bridging_local:
            left, right = np.where(bonded[:, local_o])[0]
            a, b = int(framework[left]), int(framework[right])
            bridge_of[(a, b)] = int(surface_o[local_o])
            bridge_of[(b, a)] = int(surface_o[local_o])
            neighbours[a].add(b)
            neighbours[b].add(a)

        rings: dict[str, list[int]] = {}
        for ordinal, cycle in enumerate(self._find_cycles(neighbours, self.topology_config.ring_size)):
            size = len(cycle)
            oxygens = [bridge_of[(cycle[i], cycle[(i + 1) % size])] for i in range(size)]
            rings[f"ring_{ordinal:02d}"] = list(cycle) + oxygens
        return rings

    def _infer_config(self, u: mda.Universe) -> tuple[dict[str, SelectionValue], dict[str, list[int]]]:
        """Derive atom groups and siloxane-ring membership from frame 0.

        Returns (groups, rings). Ring membership is topological and is held fixed
        for the trajectory; ring centroids are recomputed per frame by the caller.
        """
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
            phosphate_oxygens = nucleotide_o[distances <= self.topology_config.p_o_bond_cutoff]
        else:
            phosphate_oxygens = np.array([], dtype=int)

        nucleotide_h = hydrogen_indices[hydrogen_indices > surface_stop]
        non_phosphate_o = np.setdiff1d(nucleotide_o, phosphate_oxygens, assume_unique=False)
        if len(non_phosphate_o) and len(nucleotide_h):
            oh_distances = distance_array(u.atoms[non_phosphate_o].positions, u.atoms[nucleotide_h].positions)
            ribose_oh = non_phosphate_o[np.any(oh_distances <= self.topology_config.o_h_bond_cutoff, axis=1)]
        else:
            ribose_oh = np.array([], dtype=int)
        nucleobase_n = nitrogen_indices[nitrogen_indices > surface_stop]

        groups.setdefault("adenine_nh2", nucleobase_n.tolist())
        groups.setdefault("surface_oxygens", surface_o.tolist())
        groups.setdefault("surface_silicons", surface_si.tolist())
        groups.setdefault("ribose_oh", ribose_oh.tolist())
        groups.setdefault("surface_silanols", surface_o.tolist())
        groups.setdefault("phosphate_oxygens", phosphate_oxygens.tolist())
        groups.setdefault("phosphate_surface_hbond_acceptors", phosphate_oxygens.tolist())
        groups.setdefault("phosphate_oxygen_target", phosphate_oxygens[:1].tolist())

        framework = self._framework_indices(u, surface_stop)
        bonded = self._surface_bond_matrix(u, framework, surface_o)

        # Everything above the surface cut-off is adsorbate, used both to orient
        # the slab normal and to test the selection criteria.
        all_indices = np.arange(len(u.atoms))
        nucleotide_indices = all_indices[all_indices > surface_stop] if surface_stop is not None else np.array([], dtype=int)
        groups.setdefault("nucleotide_atoms", nucleotide_indices.tolist())
        groups.setdefault("framework_cations", framework.tolist())

        hydroxyls = self._classify_surface_hydroxyls(u, framework, surface_o, bonded, nucleotide_indices)
        for name, value in hydroxyls.items():
            groups.setdefault(name, value)

        # Explicit rings from config win; otherwise detect them from connectivity.
        # There is deliberately no fallback to "all surface atoms" here: the
        # centroid of the whole slab is not a siloxane-ring centre, and silently
        # substituting it previously made dist_PO_siloxane_COM report slab-scale
        # distances (3.8-12.3 A) under a ring-centre label.
        configured_rings = self.topology_config.ring_groups()
        if configured_rings:
            rings = {
                name: list(_as_index_array(value) or [])
                for name, value in configured_rings.items()
            }
        else:
            rings = self._detect_siloxane_rings(u, framework, surface_o, bonded)

        return groups, rings

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

    def _basal_face_metrics(
        self,
        u: mda.Universe,
        groups: Mapping[str, SelectionValue],
    ) -> dict[str, float]:
        """How far the adsorbate sits above the slab mid-plane, and how far it overhangs.

        A pose that migrates around the cluster periphery drops toward, or below,
        the plane of the framework cations and extends beyond the slab footprint.
        A pose resting on the basal face does neither. These two numbers are the
        measurable form of "lies entirely on the basal face".
        """
        empty = {
            "nucleotide_min_height_above_plane": np.nan,
            "nucleotide_overhang_fraction": np.nan,
        }
        framework = self._select(u, groups.get("framework_cations"))
        nucleotide = self._select(u, groups.get("nucleotide_atoms"))
        if len(framework) < 3 or len(nucleotide) == 0:
            return empty

        centre = framework.positions.mean(axis=0)
        _, _, vt = np.linalg.svd(framework.positions - centre)
        normal = vt[2]
        if float(np.dot(nucleotide.positions.mean(axis=0) - centre, normal)) < 0:
            normal = -normal

        def resolve(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            offsets = positions - centre
            height = offsets @ normal
            radial = np.linalg.norm(offsets - np.outer(height, normal), axis=1)
            return height, radial

        fw_height, fw_radial = resolve(framework.positions)
        nuc_height, nuc_radial = resolve(nucleotide.positions)

        return {
            "nucleotide_min_height_above_plane": float(np.min(nuc_height)),
            "nucleotide_overhang_fraction": float(np.mean(nuc_radial > fw_radial.max())),
        }

    def _siloxane_metric(
        self,
        u: mda.Universe,
        groups: Mapping[str, SelectionValue],
        rings: Mapping[str, Sequence[int]],
    ) -> dict[str, object]:
        """Distance from the target phosphate oxygen to the nearest ring centroid.

        The centroid is geometric (unweighted). A mass-weighted centre would be
        pulled toward the silicons and is not the geometric ring centre that the
        ring-adsorption configuration refers to.

        Returns NaN when no ring could be identified. That is a real "not
        measurable" outcome and must not be confused with "measured far away".
        """
        empty = {
            "dist_PO_siloxane_ring_centroid": np.nan,
            "nearest_siloxane_ring": "",
            "siloxane_ring_phosphate_oxygen": -1,
            "n_siloxane_rings": int(len(rings)),
        }

        # Minimise over every phosphate oxygen, not an arbitrary file-order one.
        # The question this metric answers is whether *any* phosphate oxygen is
        # directed at a ring centre.
        phosphate_o = self._select(u, groups.get("phosphate_oxygens"))
        if len(phosphate_o) == 0 or not rings:
            return empty

        centroids = []
        names: list[str] = []
        for ring_name, ring_indices in rings.items():
            ring_atoms = self._select(u, list(ring_indices))
            if len(ring_atoms) == 0:
                continue
            centroids.append(ring_atoms.positions.mean(axis=0))
            names.append(ring_name)

        if not centroids:
            return empty

        distances = distance_array(phosphate_o.positions, np.asarray(centroids, dtype=np.float32))
        flat = int(np.argmin(distances))
        o_pos, ring_pos = np.unravel_index(flat, distances.shape)

        return {
            "dist_PO_siloxane_ring_centroid": float(distances[o_pos, ring_pos]),
            "nearest_siloxane_ring": names[int(ring_pos)],
            "siloxane_ring_phosphate_oxygen": int(phosphate_o.indices[int(o_pos)]),
            "n_siloxane_rings": int(len(rings)),
        }

    def parse_trajectory(self, xyz_path: str | os.PathLike[str]) -> pd.DataFrame:
        path = Path(xyz_path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        u = mda.Universe(str(path))
        inferred_groups, siloxane_rings = self._infer_config(u)
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
            row.update(
                self._pair_metrics(
                    u,
                    inferred_groups["nucleotide_atoms"],
                    inferred_groups["edge_hydroxyl_oxygens"],
                    "min_dist_nucleotide_edgeOH",
                    "nucleotide_edgeOH_contact_count",
                )
            )
            row.update(
                self._pair_metrics(
                    u,
                    inferred_groups["nucleotide_atoms"],
                    inferred_groups["apical_hydroxyl_oxygens"],
                    "min_dist_nucleotide_apicalOH",
                    "nucleotide_apicalOH_contact_count",
                )
            )
            row.update(self._basal_face_metrics(u, inferred_groups))
            row.update(self._siloxane_metric(u, inferred_groups, siloxane_rings))
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
