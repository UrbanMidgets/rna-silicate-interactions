from __future__ import annotations

import glob
import multiprocessing as mp
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


_MAYER_START_RE = re.compile(r"^\s*Mayer bond orders larger than\s+([0-9.Ee+-]+)")
_BOND_RE = re.compile(
    r"B\(\s*(?P<i>\d+)\s*-\s*(?P<ei>[A-Za-z]+)\s*,"
    r"\s*(?P<j>\d+)\s*-\s*(?P<ej>[A-Za-z]+)\s*\)\s*:\s*"
    r"(?P<bo>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)


@dataclass(frozen=True)
class MayerBondTarget:
    """Atom-pair definition for the condensation bond tracked in ORCA output."""

    phosphorus_index: int
    silicon_indices: Optional[tuple[int, ...]] = None
    target_indices: Optional[tuple[int, ...]] = None
    partner_indices: Optional[tuple[int, ...]] = None
    index_base: int = 0
    phosphorus_element: str = "P"
    silicon_element: str = "Si"
    target_element: str = "P"
    partner_element: str = "Si"

    def __post_init__(self) -> None:
        if self.index_base not in (0, 1):
            raise ValueError("index_base must be 0 for Python/ORCA indices or 1 for human-facing indices")

        p_idx = self.phosphorus_index - self.index_base
        object.__setattr__(self, "phosphorus_index", p_idx)

        target_indices = self.target_indices
        if target_indices is None:
            target_indices = (p_idx,)
        else:
            target_indices = tuple(idx - self.index_base for idx in target_indices)
        object.__setattr__(self, "target_indices", target_indices)

        partner_indices = self.partner_indices if self.partner_indices is not None else self.silicon_indices
        if partner_indices is not None:
            partner_indices = tuple(idx - self.index_base for idx in partner_indices)
        object.__setattr__(self, "partner_indices", partner_indices)

        if self.silicon_indices is not None:
            object.__setattr__(
                self,
                "silicon_indices",
                tuple(idx - self.index_base for idx in self.silicon_indices),
            )

    @property
    def p_idx(self) -> int:
        return self.phosphorus_index

    def is_target_bond(self, i: int, ei: str, j: int, ej: str) -> tuple[bool, Optional[int]]:
        target_indices = self.target_indices or ()
        left_is_target = i in target_indices and ei == self.target_element and ej == self.partner_element
        right_is_target = j in target_indices and ej == self.target_element and ei == self.partner_element

        if left_is_target:
            partner_idx = j
        elif right_is_target:
            partner_idx = i
        else:
            return False, None

        if self.partner_indices is not None and partner_idx not in self.partner_indices:
            return False, None

        return True, partner_idx


class QuantumTopologyParser:
    """
    Stream ORCA output files and track the target Si-P Mayer bond order.

    ORCA prints Mayer bond-order lists above a threshold, not a dense matrix. Missing
    target pairs are therefore reported as 0.0 for that Mayer section.
    """

    def __init__(
        self,
        p_idx: Optional[int] = None,
        si_indices: Optional[Sequence[int]] = None,
        *,
        si_idx: Optional[int] = None,
        target_indices: Optional[Sequence[int]] = None,
        target_element: str = "P",
        partner_indices: Optional[Sequence[int]] = None,
        partner_element: str = "Si",
        target: Optional[MayerBondTarget] = None,
        index_base: int = 0,
    ) -> None:
        if target is not None:
            self.target = target
            return

        if p_idx is None:
            raise ValueError("Provide p_idx or a MayerBondTarget")

        if si_idx is not None:
            if si_indices is not None:
                raise ValueError("Use either si_idx or si_indices, not both")
            si_indices = (si_idx,)

        self.target = MayerBondTarget(
            phosphorus_index=p_idx,
            silicon_indices=tuple(si_indices) if si_indices is not None else None,
            target_indices=tuple(target_indices) if target_indices is not None else None,
            partner_indices=tuple(partner_indices) if partner_indices is not None else None,
            target_element=target_element,
            partner_element=partner_element,
            index_base=index_base,
        )

    def _empty_result(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "File",
                "Step",
                "P_Idx",
                "Target_Idx",
                "Si_Idx",
                "Partner_Idx",
                "Bond_Order",
                "Mayer_Threshold",
                "Target_Pair_Found",
                "file",
                "step",
                "p_index",
                "target_index",
                "si_index",
                "partner_index",
                "si_p_bond_order",
                "mayer_threshold",
                "target_pair_found",
            ]
        )

    def parse_file(self, filepath: str | os.PathLike[str]) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Output file not found: {path}")

        rows: list[dict[str, object]] = []
        in_mayer_block = False
        step = 0
        threshold = 0.0
        best_order = 0.0
        best_target_idx: Optional[int] = None
        best_si_idx: Optional[int] = None
        target_pair_found = False

        def flush_block() -> None:
            nonlocal step, best_order, best_target_idx, best_si_idx, target_pair_found
            rows.append(
                {
                    "File": path.name,
                    "Step": step,
                    "P_Idx": self.target.p_idx,
                    "Target_Idx": best_target_idx,
                    "Si_Idx": best_si_idx,
                    "Partner_Idx": best_si_idx,
                    "Bond_Order": best_order,
                    "Mayer_Threshold": threshold,
                    "Target_Pair_Found": target_pair_found,
                    "file": path.name,
                    "step": step,
                    "p_index": self.target.p_idx,
                    "target_index": best_target_idx,
                    "si_index": best_si_idx,
                    "partner_index": best_si_idx,
                    "si_p_bond_order": best_order,
                    "mayer_threshold": threshold,
                    "target_pair_found": target_pair_found,
                }
            )
            step += 1
            best_order = 0.0
            best_target_idx = None
            best_si_idx = None
            target_pair_found = False

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                start_match = _MAYER_START_RE.match(line)
                if start_match:
                    if in_mayer_block:
                        flush_block()
                    in_mayer_block = True
                    threshold = float(start_match.group(1))
                    best_order = 0.0
                    best_target_idx = None
                    best_si_idx = None
                    target_pair_found = False
                    continue

                if not in_mayer_block:
                    continue

                if not line.strip():
                    flush_block()
                    in_mayer_block = False
                    continue

                if "B(" not in line:
                    continue

                for match in _BOND_RE.finditer(line):
                    i = int(match.group("i"))
                    j = int(match.group("j"))
                    is_target, si_idx = self.target.is_target_bond(
                        i,
                        match.group("ei"),
                        j,
                        match.group("ej"),
                    )
                    if not is_target:
                        continue

                    target_pair_found = True
                    bond_order = float(match.group("bo"))
                    if bond_order >= best_order:
                        best_order = bond_order
                        best_si_idx = si_idx
                        if i in (self.target.target_indices or ()):
                            best_target_idx = i
                        elif j in (self.target.target_indices or ()):
                            best_target_idx = j

        if in_mayer_block:
            flush_block()

        if not rows:
            return self._empty_result()

        return pd.DataFrame(rows)

    def process_directory(self, dir_path: str | os.PathLike[str], n_jobs: int = 4) -> pd.DataFrame:
        out_files = sorted(glob.glob(os.path.join(str(dir_path), "*.out")))
        return self.process_files(out_files, n_jobs=n_jobs)

    def process_files(self, out_files: Iterable[str | os.PathLike[str]], n_jobs: int = 4) -> pd.DataFrame:
        files = [str(path) for path in out_files]
        if not files:
            return self._empty_result()

        if n_jobs <= 1 or len(files) == 1:
            results = [self.parse_file(path) for path in files]
        else:
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.map(self.parse_file, files)

        results = [df for df in results if not df.empty]
        if not results:
            return self._empty_result()
        return pd.concat(results, ignore_index=True)
