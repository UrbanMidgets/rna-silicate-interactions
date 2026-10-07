"""Regression tests for the parsing and classification pipeline.

Run with:  pytest test_parsers.py

These lock in the behaviour fixed on the siloxane-metric branch:

  * siloxane ring detection finds real six-membered rings rather than falling
    back to the centroid of the whole slab;
  * the slab's terminating hydroxyls are split into apical and edge sets;
  * poses that migrate around the cluster periphery are excluded, reproducing
    the structures the thesis marks with a dagger in Table 4.1;
  * the retained linkages reproduce Table 4.1 bond orders and distances.

Tests that need calculation data are skipped when it is absent, so the suite
still runs on a sparse checkout.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MDAnalysis = pytest.importorskip("MDAnalysis")
import MDAnalysis as mda  # noqa: E402

from parsers import (  # noqa: E402
    ClassicalConformationParser,
    ClassicalTopologyConfig,
    QuantumTopologyParser,
    StateClassifier,
)
from parsers.classifier import StateClassificationThresholds  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent
PRIMARY = REPO_ROOT / "data" / "primary_calculations"

# AMP on the pure surface, dehydrated. Thesis structure 1a: P-O-Si at 1.77 A,
# Mayer bond order 0.913.
AMP_DRY_DIR = PRIMARY / "canonical_surface" / "amp" / "selected_frames" / "frame12" / "dry"
AMP_DRY_XYZ = AMP_DRY_DIR / "amp_frame12_dry.xyz"
AMP_DRY_OUT = AMP_DRY_DIR / "amp_frame12_dry.out"

# AMP on the pure surface, dehydrated, unlinked. Thesis structure 1c: the
# phosphate points at a siloxane ring centre instead of reacting.
AMP_RING_XYZ = (
    PRIMARY / "canonical_surface" / "amp" / "selected_frames" / "frame11" / "dry" / "amp_frame11_dry.xyz"
)

CLASSIFICATIONS = REPO_ROOT / "data" / "classifications.csv"


def _needs(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"calculation data not present: {path.relative_to(REPO_ROOT)}")


def _parse(xyz: Path) -> pd.Series:
    u = mda.Universe(str(xyz))
    names = np.asarray(u.atoms.names, dtype=str)
    p_idx = int(np.where(names == "P")[0][0])
    config = ClassicalTopologyConfig(p_index=p_idx, surface_index_stop=p_idx)
    return ClassicalConformationParser(config).parse_trajectory(xyz).iloc[-1]


# --------------------------------------------------------------------------
# Siloxane ring detection
# --------------------------------------------------------------------------

def test_slab_has_seven_six_membered_rings():
    """The 3x3 tetrahedral-sheet fragment presents a central ring plus six neighbours."""
    _needs(AMP_DRY_XYZ)
    assert _parse(AMP_DRY_XYZ)["n_siloxane_rings"] == 7


def test_ring_centroid_distance_is_ring_scale_not_slab_scale():
    """Regression: the metric used to report the centroid of the entire slab.

    That produced 3.8-12.3 A across the dataset. A real ring centroid for an
    adsorbed phosphate is a few angstrom.
    """
    _needs(AMP_DRY_XYZ)
    distance = _parse(AMP_DRY_XYZ)["dist_PO_siloxane_ring_centroid"]
    assert 1.0 < distance < 5.5, f"{distance:.2f} A is slab-scale, not ring-scale"


def test_ring_centred_conformer_sits_closer_to_a_ring_than_the_linked_one():
    """Thesis 4.3.1: the unlinked conformer directs an oxygen at a ring centre."""
    _needs(AMP_DRY_XYZ)
    _needs(AMP_RING_XYZ)
    linked = _parse(AMP_DRY_XYZ)
    ring_centred = _parse(AMP_RING_XYZ)

    # The linked structure forms the bond; the other does not.
    assert linked["min_dist_phosphateO_surfaceSi"] < 2.0
    assert ring_centred["min_dist_phosphateO_surfaceSi"] > 3.0
    # The unlinked structure is the one sitting over a ring centre.
    assert (
        ring_centred["dist_PO_siloxane_ring_centroid"]
        < linked["dist_PO_siloxane_ring_centroid"]
    )


def test_no_silent_fallback_when_rings_cannot_be_found():
    """With no framework cations there are no rings, and the metric must be NaN."""
    _needs(AMP_DRY_XYZ)
    config = ClassicalTopologyConfig(
        p_index=126, surface_index_stop=126, framework_elements=("Xx",)
    )
    row = ClassicalConformationParser(config).parse_trajectory(AMP_DRY_XYZ).iloc[-1]
    assert row["n_siloxane_rings"] == 0
    assert np.isnan(row["dist_PO_siloxane_ring_centroid"])


# --------------------------------------------------------------------------
# Surface hydroxyl classification
# --------------------------------------------------------------------------

def test_terminating_hydroxyls_split_into_apical_and_edge():
    """Thesis 4.9.1: 36 hydroxyls, 24 apical and 12 edge; the basal face carries none."""
    _needs(AMP_DRY_XYZ)
    u = mda.Universe(str(AMP_DRY_XYZ))
    names = np.asarray(u.atoms.names, dtype=str)
    p_idx = int(np.where(names == "P")[0][0])
    config = ClassicalTopologyConfig(p_index=p_idx, surface_index_stop=p_idx)
    parser = ClassicalConformationParser(config)
    groups, _ = parser._infer_config(u)

    assert len(groups["apical_hydroxyl_oxygens"]) == 24
    assert len(groups["edge_hydroxyl_oxygens"]) == 12


def test_aluminium_counts_as_framework():
    """Al-O-Si bridges must not be mistaken for terminating oxygens.

    Counting Si neighbours alone made the four oxygens around the substitution
    look terminal, which wrongly excluded two real Table 4.1 linkages.
    """
    assert "Al" in ClassicalTopologyConfig().framework_elements


# --------------------------------------------------------------------------
# Quantum parser
# --------------------------------------------------------------------------

def test_mayer_bond_order_matches_table_4_1():
    """Thesis structure 1a: Mayer bond order 0.913."""
    _needs(AMP_DRY_XYZ)
    _needs(AMP_DRY_OUT)
    u = mda.Universe(str(AMP_DRY_XYZ))
    names = np.asarray(u.atoms.names, dtype=str)
    pos = u.atoms.positions
    p_idx = int(np.where(names == "P")[0][0])

    phosphate_o = [
        i for i in range(p_idx + 1, len(names))
        if names[i] == "O" and np.linalg.norm(pos[i] - pos[p_idx]) <= 1.9
    ]
    si = [i for i in range(p_idx) if names[i] == "Si"]

    df = QuantumTopologyParser(
        p_idx=p_idx,
        target_indices=phosphate_o,
        target_element="O",
        partner_indices=si,
        partner_element="Si",
    ).parse_file(AMP_DRY_OUT)

    assert not df.empty
    assert df.sort_values("Step").iloc[-1]["Bond_Order"] == pytest.approx(0.913, abs=0.002)


# --------------------------------------------------------------------------
# Classifier branches
# --------------------------------------------------------------------------

def _classical_row(**overrides) -> pd.DataFrame:
    row = {
        "Frame": 0,
        "min_dist_intramol_OH_PO": 9.0,
        "steric_trap_hbond_count": 0,
        "min_dist_nh2_surfO": 9.0,
        "min_dist_riboseOH_surfSilanol": 9.0,
        "min_dist_phosphateO_surfaceO": 9.0,
        "min_dist_phosphateO_surfaceSi": 9.0,
        "dist_PO_siloxane_ring_centroid": 5.0,
        "min_dist_nucleotide_edgeOH": 6.0,
        "nucleotide_min_height_above_plane": 2.5,
        "nucleotide_overhang_fraction": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _quantum_row(bond_order: float) -> pd.DataFrame:
    return pd.DataFrame([{"Step": 0, "Bond_Order": bond_order, "Si_Idx": 1}])


def test_classifier_covalent_branch():
    result = StateClassifier().classify_pair(
        _quantum_row(0.91), _classical_row(min_dist_phosphateO_surfaceSi=1.77)
    )
    assert result["State"] == "Covalently Condensed"
    assert result["Selection_Pass"] is True


def test_classifier_requires_bonding_distance_not_just_bond_order():
    """A high bond order at an implausible separation must not be called a bond."""
    result = StateClassifier().classify_pair(
        _quantum_row(0.91), _classical_row(min_dist_phosphateO_surfaceSi=3.5)
    )
    assert result["State"] != "Covalently Condensed"


def test_classifier_excludes_off_basal_poses():
    """Poses that drop to the slab mid-plane have migrated around the edge."""
    result = StateClassifier().classify_pair(
        _quantum_row(0.0),
        _classical_row(nucleotide_min_height_above_plane=0.19, nucleotide_overhang_fraction=0.56),
    )
    assert result["State"] == "Excluded (Off-Basal)"
    assert result["Selection_Pass"] is False


def test_off_basal_exclusion_takes_precedence_over_bonding():
    """An edge-bound structure is outside the model regardless of bond order."""
    result = StateClassifier().classify_pair(
        _quantum_row(0.92),
        _classical_row(
            min_dist_phosphateO_surfaceSi=1.8, nucleotide_min_height_above_plane=0.2
        ),
    )
    assert result["State"] == "Excluded (Off-Basal)"


def test_classifier_steric_and_anchored_branches():
    locked = StateClassifier().classify_pair(
        _quantum_row(0.0), _classical_row(min_dist_intramol_OH_PO=2.4)
    )
    assert locked["State"] == "Sterically Locked"

    anchored = StateClassifier().classify_pair(
        _quantum_row(0.0), _classical_row(min_dist_nh2_surfO=2.2)
    )
    assert anchored["State"] == "Surface Anchored"

    free = StateClassifier().classify_pair(_quantum_row(0.0), _classical_row())
    assert free["State"] == "Free"


def test_basal_threshold_separates_the_thesis_dagger_structures():
    """Retained poses measured 1.60-2.35 A; excluded poses -0.08 to 0.89 A."""
    threshold = StateClassificationThresholds().min_basal_height
    assert 0.89 < threshold < 1.60


# --------------------------------------------------------------------------
# Generated dataset
# --------------------------------------------------------------------------

def test_classifications_have_no_case_colliding_columns():
    """'state'/'State' broke every case-insensitive CSV reader."""
    _needs(CLASSIFICATIONS)
    columns = pd.read_csv(CLASSIFICATIONS, nrows=0).columns
    lowered = [c.lower() for c in columns]
    assert len(lowered) == len(set(lowered)), "column names collide when lower-cased"


@pytest.mark.parametrize(
    "system,surface,frame,bond_order,distance",
    [
        ("amp", "canonical", "frame12", 0.913, 1.77),
        ("amp", "canonical", "frame15", 0.936, 1.76),
        ("cmp", "canonical", "frame14", 0.661, 1.93),
        ("gmp", "canonical", "frame16", 0.666, 1.94),
        ("ump", "canonical", "frame14", 0.952, 1.75),
        ("ump", "canonical", "frame15", 0.936, 1.76),
        ("ump", "canonical", "frame16", 0.940, 1.76),
        ("gmp", "aluminium", "frame59", 0.918, 1.79),
        ("ump", "aluminium", "frame15", 0.896, 1.79),
    ],
)
def test_reproduces_table_4_1_linkages(system, surface, frame, bond_order, distance):
    _needs(CLASSIFICATIONS)
    df = pd.read_csv(CLASSIFICATIONS)
    match = df[
        (df["system"] == system)
        & (df["surface"] == surface)
        & (df["frame"] == frame)
        & (df["calc_state"] == "dry")
    ]
    assert len(match) == 1, f"expected exactly one dry row, found {len(match)}"
    row = match.iloc[0]
    assert row["predicted_state"] == "Covalently Condensed"
    assert row["Final_Bond_Order"] == pytest.approx(bond_order, abs=0.002)
    assert row["Final_Linkage_Dist"] == pytest.approx(distance, abs=0.01)


@pytest.mark.parametrize(
    "system,surface,frame",
    [
        ("amp_protonated", "aluminium", "frame12"),
        ("amp_protonated", "aluminium", "frame14"),
        ("gmp", "aluminium", "frame35"),
        ("ump", "aluminium", "frame14"),
    ],
)
def test_reproduces_table_4_1_exclusions(system, surface, frame):
    """The four structures Table 4.1 marks with a dagger."""
    _needs(CLASSIFICATIONS)
    df = pd.read_csv(CLASSIFICATIONS)
    match = df[
        (df["system"] == system)
        & (df["surface"] == surface)
        & (df["frame"] == frame)
        & (df["calc_state"] == "dry")
    ]
    assert len(match) == 1
    assert match.iloc[0]["predicted_state"] == "Excluded (Off-Basal)"


def test_no_duplicate_dry_keys():
    """Each (system, surface, frame) must yield one dry classification."""
    _needs(CLASSIFICATIONS)
    df = pd.read_csv(CLASSIFICATIONS)
    dry = df[df["calc_state"] == "dry"]
    counts = dry.groupby(["system", "surface", "frame"]).size()
    assert counts.max() == 1, f"duplicated keys: {counts[counts > 1].to_dict()}"
