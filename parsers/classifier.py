from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateClassificationThresholds:
    covalent_bond_order: float = 0.5
    no_bond_order: float = 0.2
    hbond_distance: float = 3.0
    anchored_distance: float = 3.0


class StateClassifier:
    """
    Combine quantum and classical metrics into an interpretable final-state label.
    """

    def __init__(
        self,
        quantum_df: Optional[pd.DataFrame] = None,
        classical_df: Optional[pd.DataFrame] = None,
        thresholds: StateClassificationThresholds | None = None,
    ) -> None:
        self.quantum_df = quantum_df if quantum_df is not None else pd.DataFrame()
        self.classical_df = classical_df if classical_df is not None else pd.DataFrame()
        self.thresholds = thresholds or StateClassificationThresholds()

    @staticmethod
    def _file_stem(filename: str) -> str:
        stem = Path(filename).stem
        return stem.removesuffix("_trj").removesuffix("_dry")

    @staticmethod
    def _last_row(df: pd.DataFrame, order_column: str) -> pd.Series:
        if df.empty:
            raise ValueError("Cannot classify an empty dataframe")
        if order_column in df.columns:
            return df.sort_values(order_column).iloc[-1]
        return df.iloc[-1]

    @staticmethod
    def _get(row: pd.Series, *names: str, default: float = np.nan) -> Any:
        for name in names:
            if name in row and pd.notna(row[name]):
                return row[name]
        return default

    def classify_pair(
        self,
        quantum_df: pd.DataFrame,
        classical_df: pd.DataFrame,
        *,
        label: str = "",
    ) -> dict[str, Any]:
        if quantum_df.empty and classical_df.empty:
            return {"File": label, "State": "Unknown (Missing Data)", "Reason": "No quantum or classical rows"}
        if quantum_df.empty:
            return {"File": label, "State": "Unknown (Missing Quantum Data)", "Reason": "No Mayer bond-order rows"}
        if classical_df.empty:
            return {"File": label, "State": "Unknown (Missing Classical Data)", "Reason": "No trajectory rows"}

        final_q = self._last_row(quantum_df, "Step")
        final_c = self._last_row(classical_df, "Frame")

        bond_order = float(self._get(final_q, "si_p_bond_order", "Bond_Order", default=0.0))
        si_idx = self._get(final_q, "si_index", "Si_Idx", default=np.nan)
        intramol_dist = float(self._get(final_c, "min_dist_intramol_OH_PO", default=np.inf))
        intramol_count = int(self._get(final_c, "steric_trap_hbond_count", default=0))
        nh2_anchor = float(self._get(final_c, "min_dist_nh2_surfO", default=np.inf))
        ribose_anchor = float(self._get(final_c, "min_dist_riboseOH_surfSilanol", default=np.inf))
        phosphate_surface_anchor = float(self._get(final_c, "min_dist_phosphateO_surfaceO", default=np.inf))
        phosphate_si_dist = float(self._get(final_c, "min_dist_phosphateO_surfaceSi", default=np.nan))
        siloxane_dist = float(self._get(final_c, "dist_PO_siloxane_COM", default=np.nan))
        anchoring_dist = min(nh2_anchor, ribose_anchor, phosphate_surface_anchor)

        state = "Free"
        reason = "No covalent Si-P bond, steric lock, or close surface anchor detected"

        if bond_order >= self.thresholds.covalent_bond_order:
            state = "Covalently Condensed"
            reason = f"Si-O(phosphate) Mayer bond order {bond_order:.3f} exceeds {self.thresholds.covalent_bond_order:.2f}"
        elif (
            bond_order < self.thresholds.no_bond_order
            and (intramol_dist <= self.thresholds.hbond_distance or intramol_count > 0)
        ):
            state = "Sterically Locked"
            reason = "Intramolecular ribose-phosphate contact persists without Si-O(phosphate) bond formation"
        elif anchoring_dist <= self.thresholds.anchored_distance:
            state = "Surface Anchored"
            reason = "Nucleobase, ribose, or phosphate contact is within the surface anchoring distance"

        return {
            "File": label,
            "State": state,
            "Reason": reason,
            "Final_Bond_Order": bond_order,
            "Final_Si_Idx": si_idx,
            "Final_Intramol_Dist": intramol_dist,
            "Final_Intramol_HBond_Count": intramol_count,
            "Final_Anchoring_Dist": anchoring_dist,
            "Final_NH2_SurfaceO_Dist": nh2_anchor,
            "Final_RiboseOH_SurfaceSilanol_Dist": ribose_anchor,
            "Final_PhosphateO_SurfaceO_Dist": phosphate_surface_anchor,
            "Final_PhosphateO_SurfaceSi_Dist": phosphate_si_dist,
            "Final_PO_Siloxane_COM_Dist": siloxane_dist,
        }

    def classify_trajectory(self, filename: str) -> dict[str, Any]:
        q_df = self.quantum_df.copy()
        c_df = self.classical_df.copy()
        target_stem = self._file_stem(filename)

        if "File" in q_df.columns:
            q_df = q_df[q_df["File"].map(self._file_stem) == target_stem]
        if "File" in c_df.columns:
            c_df = c_df[c_df["File"].map(self._file_stem) == target_stem]

        return self.classify_pair(q_df, c_df, label=filename)

    def summarize(self) -> pd.DataFrame:
        if self.quantum_df.empty or self.classical_df.empty:
            return pd.DataFrame()

        q_files = set(self.quantum_df["File"].map(self._file_stem)) if "File" in self.quantum_df else set()
        c_files = set(self.classical_df["File"].map(self._file_stem)) if "File" in self.classical_df else set()

        rows = [self.classify_trajectory(stem) for stem in sorted(q_files.intersection(c_files))]
        return pd.DataFrame(rows)
