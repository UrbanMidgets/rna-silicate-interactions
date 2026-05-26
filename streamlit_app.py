import streamlit as st
import pandas as pd
import py3Dmol
import os
import base64
import inspect
import numpy as np
from pathlib import Path

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# Set page config
st.set_page_config(page_title="RNA Silicate Interactions", layout="wide")

st.title("RNA Silicate Interactions")
st.markdown("Supporting data for RNA nucleotide interactions with silicate surfaces.")

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.tsv"
MAX_TEXT_PREVIEW_BYTES = 200_000


def resolve_repo_path(path_value):
    try:
        candidate = Path(str(path_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        candidate.relative_to(REPO_ROOT)
        return candidate
    except Exception:
        return None

@st.cache_data
def load_manifest():
    if MANIFEST_PATH.exists():
        df = pd.read_csv(MANIFEST_PATH, sep="\t").fillna("")
        # Derive file_name and file_type
        df['file_name'] = df['repo_path'].apply(lambda x: os.path.basename(x))
        
        def classify(path):
            p = path.lower()
            if p.endswith("_trj.xyz"): return "trajectory"
            if p.endswith(".xyz"): return "structure"
            if p.endswith(".inp"): return "input"
            if p.endswith(".out"): return "output"
            return "other"
            
        df['file_type'] = df['repo_path'].apply(classify)
        return df
    return pd.DataFrame()

df = load_manifest()

if df.empty:
    st.error("Manifest not found. Please ensure data/MANIFEST.tsv exists.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

def get_options(dataframe, column):
    opts = sorted([str(x) for x in dataframe[column].unique() if x])
    return ["All"] + opts

def get_param_idx(opts, param_name, default="All", prefix=""):
    val = st.query_params.get(param_name, default)
    if val in opts:
        return opts.index(val)
    if prefix and f"{prefix}{val}" in opts:
        return opts.index(f"{prefix}{val}")
    return opts.index(default) if default in opts else 0

def update_param(param_name, value):
    if value == "All":
        if param_name in st.query_params:
            del st.query_params[param_name]
    else:
        # If it's structure, and value starts with 'frame', maybe strip it?
        # Let's just keep the value as is, or if they want exactly 14 instead of frame14:
        if param_name == "structure" and value.startswith("frame"):
            st.query_params[param_name] = value.replace("frame", "")
        else:
            st.query_params[param_name] = value

# Dynamic filtering logic for sidebar
filtered_df = df.copy()

surface_opts = get_options(df, 'surface')
surface = st.sidebar.selectbox("Surface", surface_opts, index=get_param_idx(surface_opts, "surface"))
update_param("surface", surface)
if surface != "All":
    filtered_df = filtered_df[filtered_df['surface'] == surface]

system_opts = get_options(filtered_df, 'system')
system = st.sidebar.selectbox("System", system_opts, index=get_param_idx(system_opts, "nuc"))
update_param("nuc", system)
if system != "All":
    filtered_df = filtered_df[filtered_df['system'] == system]

role_opts = get_options(filtered_df, 'role')
role = st.sidebar.selectbox("Role", role_opts, index=get_param_idx(role_opts, "role"))
update_param("role", role)
if role != "All":
    filtered_df = filtered_df[filtered_df['role'] == role]

frame_opts = get_options(filtered_df, 'frame')
frame = st.sidebar.selectbox("Frame", frame_opts, index=get_param_idx(frame_opts, "structure", prefix="frame"))
update_param("structure", frame)
if frame != "All":
    filtered_df = filtered_df[filtered_df['frame'] == frame]

st.sidebar.header("Settings")
show_surface = st.sidebar.checkbox("Show Surface (Si, Al, O)", value=True)
spin = st.sidebar.checkbox("Spin Molecule", value=False)
performance_mode = st.sidebar.checkbox("Performance Mode", value=True)

try:
    default_trj_frame = int(st.query_params.get("trj_frame", 0))
except ValueError:
    default_trj_frame = 0

trajectory_stride = st.sidebar.slider("Trajectory Frame Step", 1, 10, 1)

st.sidebar.markdown(f"**Matches:** {len(filtered_df)}")

# Grouping by system/surface/frame for comparison
# We want to identify pairs of (solvated, dry) for the same frame
comparison_options = filtered_df[filtered_df['state'].isin(['solvated', 'dry'])]
groups = comparison_options.groupby(['surface', 'system', 'frame', 'role'])

def update_tab():
    st.query_params["tab"] = st.session_state.active_tab

tab_options = ["Visualization", "Comparison", "Data Table"]
default_tab = st.query_params.get("tab", "Visualization")
if default_tab not in tab_options:
    default_tab = "Visualization"

# Main content area
tabs_kwargs = {}
tabs_params = inspect.signature(st.tabs).parameters
if "default" in tabs_params:
    tabs_kwargs["default"] = default_tab
if "key" in tabs_params:
    tabs_kwargs["key"] = "active_tab"
if "on_change" in tabs_params:
    tabs_kwargs["on_change"] = update_tab
tabs = st.tabs(tab_options, **tabs_kwargs)

@st.cache_data
def get_frame_count(xyz_path):
    resolved_path = resolve_repo_path(xyz_path)
    if resolved_path is None or not resolved_path.exists():
        return 0
    try:
        with resolved_path.open("r") as f:
            first_line = f.readline().strip()
            if not first_line.isdigit():
                return 1
            num_atoms = int(first_line)
            f.seek(0)
            content = f.read()
            # Count occurrences of the atom count at the start of a frame
            # This is a simple heuristic for XYZ trajectories
            return content.count(f"\n{num_atoms}\n") + (1 if content.startswith(str(num_atoms)) else 0)
    except (OSError, ValueError):
        return 1

@st.cache_data
def get_xyz_data(path):
    resolved_path = resolve_repo_path(path)
    if resolved_path is None or not resolved_path.exists():
        return None
    with resolved_path.open("r") as f:
        return f.read()


def load_structure_dataframe(xyz_path):
    resolved_path = resolve_repo_path(xyz_path)
    if resolved_path is None or not resolved_path.exists():
        return pd.DataFrame(columns=["atom", "x", "y", "z"])

    with resolved_path.open("r") as f:
        lines = [line.strip() for line in f.readlines()]

    if len(lines) < 3:
        return pd.DataFrame(columns=["atom", "x", "y", "z"])

    atom_count = None
    if lines[0].isdigit():
        atom_count = int(lines[0])

    records = []
    start_idx = 2 if atom_count is not None else 0
    end_idx = start_idx + atom_count if atom_count is not None else len(lines)

    for line in lines[start_idx:end_idx]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            records.append(
                {
                    "atom": parts[0],
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                }
            )
        except ValueError:
            continue

    return pd.DataFrame(records)


@st.cache_data
def load_xyz_frames(xyz_path):
    resolved_path = resolve_repo_path(xyz_path)
    if resolved_path is None or not resolved_path.exists():
        return []

    with resolved_path.open("r") as f:
        lines = [line.rstrip("\n") for line in f]

    frames = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if not line.isdigit():
            break

        atom_count = int(line)
        if i + 2 + atom_count > len(lines):
            break

        records = []
        start = i + 2
        end = start + atom_count
        for atom_line in lines[start:end]:
            parts = atom_line.split()
            if len(parts) < 4:
                continue
            try:
                records.append(
                    {
                        "atom": parts[0],
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "z": float(parts[3]),
                    }
                )
            except ValueError:
                continue

        if records:
            frames.append(pd.DataFrame(records))

        i = end

    return frames


def measure_surface_interactions(threshold: float = 3.5) -> str:
    """Measure close adsorbate-to-silicate contacts from the active structure.

    This function uses ``st.session_state.df_structure`` as the active atomic context,
    isolates silicate atoms (Al, Si), isolates adsorbate atoms (all non-surface atoms),
    computes full 3D Euclidean distances between those sets, and reports a concise
    text matrix of all boundaries within ``threshold`` angstroms.

    Args:
        threshold: Distance cutoff in angstroms for defining close interactions.

    Returns:
        A compact multiline text report of close contacts sorted by distance, or a
        clear reason if structure context is unavailable.
    """
    df_structure = st.session_state.get("df_structure")
    if df_structure is None or df_structure.empty:
        return "No active structure is loaded. Select an XYZ file first."

    silicate_mask = df_structure["atom"].isin(["Al", "Si"])
    adsorbate_mask = ~df_structure["atom"].isin(["Al", "Si", "O"])

    silicate = df_structure[silicate_mask].reset_index(drop=True)
    adsorbate = df_structure[adsorbate_mask].reset_index(drop=True)

    if silicate.empty:
        return "No silicate atoms (Al/Si) found in the active structure."
    if adsorbate.empty:
        return "No adsorbate atoms found in the active structure."

    sil_coords = silicate[["x", "y", "z"]].to_numpy(dtype=float)
    ads_coords = adsorbate[["x", "y", "z"]].to_numpy(dtype=float)

    delta = ads_coords[:, None, :] - sil_coords[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    contact_pairs = np.argwhere(distances <= threshold)

    if contact_pairs.size == 0:
        return f"No adsorbate-silicate contacts found within {threshold:.2f} A."

    lines = [
        f"Found {len(contact_pairs)} contacts within {threshold:.2f} A",
        "ads_atom | sil_atom | distance_A",
    ]

    contacts = []
    for ads_idx, sil_idx in contact_pairs:
        d = float(distances[ads_idx, sil_idx])
        contacts.append((d, ads_idx, sil_idx))
    contacts.sort(key=lambda x: x[0])

    for d, ads_idx, sil_idx in contacts[:60]:
        ads_row = adsorbate.iloc[ads_idx]
        sil_row = silicate.iloc[sil_idx]
        ads_label = f"{ads_row['atom']}[{ads_idx}]"
        sil_label = f"{sil_row['atom']}[{sil_idx}]"
        lines.append(f"{ads_label:>8} | {sil_label:>8} | {d:8.3f}")

    if len(contacts) > 60:
        lines.append(f"... truncated {len(contacts) - 60} additional contacts")

    return "\n".join(lines)


def _compute_interaction_tables_from_df(
    df_structure,
    hbond_distance: float = 2.6,
    contact_distance: float = 3.5,
    dispersion_min: float = 3.3,
    dispersion_max: float = 4.5,
):
    if df_structure is None or df_structure.empty:
        return {
            "error": "No active structure is loaded. Select an XYZ file first.",
            "hbond_df": pd.DataFrame(),
            "polar_df": pd.DataFrame(),
            "disp_df": pd.DataFrame(),
        }

    coords = df_structure[["x", "y", "z"]].to_numpy(dtype=float)
    atoms = df_structure["atom"].to_numpy()

    is_surface = np.isin(atoms, ["Al", "Si", "O"])
    is_surface_oxygen = atoms == "O"
    is_adsorbate = ~is_surface
    if not np.any(is_adsorbate):
        return {
            "error": "No adsorbate atoms identified (all atoms classified as surface Al/Si/O).",
            "hbond_df": pd.DataFrame(),
            "polar_df": pd.DataFrame(),
            "disp_df": pd.DataFrame(),
        }

    ads_idx = np.where(is_adsorbate)[0]
    surf_idx = np.where(is_surface)[0]
    surf_o_idx = np.where(is_surface_oxygen)[0]

    ads_coords = coords[ads_idx]
    surf_coords = coords[surf_idx]
    dmat = np.linalg.norm(ads_coords[:, None, :] - surf_coords[None, :, :], axis=2)

    # Polar contacts: adsorbate N/O/P/H to surface Al/Si/O
    polar_ads = np.isin(atoms[ads_idx], ["N", "O", "P", "H"])
    close_pairs = np.argwhere(dmat <= contact_distance)
    polar_rows = []
    for i, j in close_pairs:
        if not polar_ads[i]:
            continue
        ai = int(ads_idx[i])
        sj = int(surf_idx[j])
        polar_rows.append(
            {
                "ads_atom": f"{atoms[ai]}[{ai}]",
                "surface_atom": f"{atoms[sj]}[{sj}]",
                "distance_A": float(dmat[i, j]),
            }
        )
    polar_df = pd.DataFrame(polar_rows).sort_values("distance_A", ascending=True) if polar_rows else pd.DataFrame(columns=["ads_atom", "surface_atom", "distance_A"])

    # H-bond plausibility: adsorbate H close to surface O AND H bound to adsorbate N/O donor
    hbond_rows = []
    h_ads_idx = ads_idx[atoms[ads_idx] == "H"]
    donor_heavy_idx = ads_idx[np.isin(atoms[ads_idx], ["N", "O"])]
    if h_ads_idx.size > 0 and surf_o_idx.size > 0 and donor_heavy_idx.size > 0:
        h_coords = coords[h_ads_idx]
        so_coords = coords[surf_o_idx]
        donor_coords = coords[donor_heavy_idx]

        # Find nearest donor heavy atom to each adsorbate H
        h_donor_dmat = np.linalg.norm(h_coords[:, None, :] - donor_coords[None, :, :], axis=2)
        nearest_donor_pos = np.argmin(h_donor_dmat, axis=1)
        nearest_donor_dist = h_donor_dmat[np.arange(len(h_ads_idx)), nearest_donor_pos]

        # Heuristic donor-H covalent threshold
        plausible_donor_h = nearest_donor_dist <= 1.30

        h_o_dmat = np.linalg.norm(h_coords[:, None, :] - so_coords[None, :, :], axis=2)
        h_o_pairs = np.argwhere(h_o_dmat <= hbond_distance)
        for hi, oi in h_o_pairs:
            if not plausible_donor_h[hi]:
                continue
            h_global = int(h_ads_idx[hi])
            o_global = int(surf_o_idx[oi])
            donor_global = int(donor_heavy_idx[nearest_donor_pos[hi]])
            donor_o_dist = float(np.linalg.norm(coords[donor_global] - coords[o_global]))
            hbond_rows.append(
                {
                    "donor_atom": f"{atoms[donor_global]}[{donor_global}]",
                    "h_atom": f"H[{h_global}]",
                    "acceptor_surface_O": f"O[{o_global}]",
                    "H_O_A": float(h_o_dmat[hi, oi]),
                    "D_O_A": donor_o_dist,
                }
            )
    hbond_df = pd.DataFrame(hbond_rows).sort_values(["H_O_A", "D_O_A"], ascending=True) if hbond_rows else pd.DataFrame(columns=["donor_atom", "h_atom", "acceptor_surface_O", "H_O_A", "D_O_A"])

    # Likely dispersive shell: adsorbate C/H vs any surface atom in shell
    disp_ads_mask = np.isin(atoms[ads_idx], ["C", "H"])
    shell_pairs = np.argwhere((dmat >= dispersion_min) & (dmat <= dispersion_max))
    disp_rows = []
    for i, j in shell_pairs:
        if not disp_ads_mask[i]:
            continue
        ai = int(ads_idx[i])
        sj = int(surf_idx[j])
        disp_rows.append(
            {
                "ads_atom": f"{atoms[ai]}[{ai}]",
                "surface_atom": f"{atoms[sj]}[{sj}]",
                "distance_A": float(dmat[i, j]),
            }
        )
    disp_df = pd.DataFrame(disp_rows).sort_values("distance_A", ascending=True) if disp_rows else pd.DataFrame(columns=["ads_atom", "surface_atom", "distance_A"])

    return {
        "error": None,
        "hbond_df": hbond_df,
        "polar_df": polar_df,
        "disp_df": disp_df,
    }


def _compute_interaction_tables(
    hbond_distance: float = 2.6,
    contact_distance: float = 3.5,
    dispersion_min: float = 3.3,
    dispersion_max: float = 4.5,
):
    return _compute_interaction_tables_from_df(
        st.session_state.get("df_structure"),
        hbond_distance=hbond_distance,
        contact_distance=contact_distance,
        dispersion_min=dispersion_min,
        dispersion_max=dispersion_max,
    )


def characterize_surface_interactions(
    hbond_distance: float = 2.6,
    contact_distance: float = 3.5,
    dispersion_min: float = 3.3,
    dispersion_max: float = 4.5,
) -> str:
    """Characterize likely H-bond and dispersive contacts at the interface.

    Uses ``st.session_state.df_structure`` and reports three layers of interaction:
    1) likely hydrogen-bond-like contacts (H from adsorbate near surface O),
    2) close polar contacts to Al/Si/O within ``contact_distance``,
    3) likely dispersive contacts (C/H vs surface atoms in 3.3-4.5 A shell).

    This is a geometric heuristic over XYZ coordinates only (no bond-order graph),
    so outputs are intentionally labeled as "likely" rather than definitive.
    """
    tables = _compute_interaction_tables(
        hbond_distance=hbond_distance,
        contact_distance=contact_distance,
        dispersion_min=dispersion_min,
        dispersion_max=dispersion_max,
    )
    if tables["error"]:
        return tables["error"]

    hbond_df = tables["hbond_df"]
    polar_df = tables["polar_df"]
    disp_df = tables["disp_df"]

    lines = []
    lines.append("Interaction characterization (geometry heuristic from XYZ)")
    lines.append(
        f"likely_Hbond={len(hbond_df)}, polar_contacts_le_{contact_distance:.2f}A={len(polar_df)}, "
        f"likely_dispersion_{dispersion_min:.2f}-{dispersion_max:.2f}A={len(disp_df)}"
    )

    lines.append("\nTop likely H-bond-like contacts (donor-H ... surface O):")
    if not hbond_df.empty:
        for row in hbond_df.head(12).itertuples(index=False):
            lines.append(
                f"{row.donor_atom}-{row.h_atom}...{row.acceptor_surface_O} "
                f"H...O={row.H_O_A:6.3f} A, D...O={row.D_O_A:6.3f} A"
            )
    else:
        lines.append("None under cutoff.")

    lines.append(f"\nTop polar contacts <= {contact_distance:.2f} A:")
    if not polar_df.empty:
        for row in polar_df.head(12).itertuples(index=False):
            lines.append(f"{row.ads_atom}...{row.surface_atom}  {row.distance_A:6.3f} A")
    else:
        lines.append("None under cutoff.")

    lines.append(f"\nTop likely dispersive shell contacts ({dispersion_min:.2f}-{dispersion_max:.2f} A):")
    if not disp_df.empty:
        for row in disp_df.head(12).itertuples(index=False):
            lines.append(f"{row.ads_atom}...{row.surface_atom}  {row.distance_A:6.3f} A")
    else:
        lines.append("None in shell.")

    lines.append("\nNote: classifications are heuristic because XYZ lacks explicit bond topology and charges.")
    return "\n".join(lines)


def analyze_trajectory_interactions(frame_step: int = 1, max_frames: int = 200) -> str:
    """Analyze interaction persistence across trajectory frames.

    Iterates over trajectory frames cached in ``st.session_state.df_structure_frames`` and
    reports frame-wise and aggregate statistics for likely H-bond-like, polar, and likely
    dispersive contacts. This helps distinguish persistent behavior from single-frame events.

    Args:
        frame_step: Analyze every Nth frame (>=1).
        max_frames: Maximum number of sampled frames to process.

    Returns:
        A concise multiline trajectory summary suitable for chat responses.
    """
    frames = st.session_state.get("df_structure_frames", [])
    if not frames:
        return "No trajectory frames are loaded. Select a *_trj.xyz file first."

    frame_step = max(1, int(frame_step))
    max_frames = max(1, int(max_frames))

    sampled = list(range(0, len(frames), frame_step))[:max_frames]
    if not sampled:
        return "No frames selected for analysis."

    rows = []
    for frame_idx in sampled:
        tables = _compute_interaction_tables_from_df(frames[frame_idx])
        if tables["error"]:
            continue
        hbond_n = len(tables["hbond_df"])
        polar_n = len(tables["polar_df"])
        disp_n = len(tables["disp_df"])

        min_polar = float(tables["polar_df"]["distance_A"].min()) if not tables["polar_df"].empty else np.nan
        min_hbond = float(tables["hbond_df"]["H_O_A"].min()) if not tables["hbond_df"].empty else np.nan
        rows.append(
            {
                "frame": frame_idx,
                "hbond_like_count": hbond_n,
                "polar_count": polar_n,
                "dispersion_count": disp_n,
                "min_H_O_A": min_hbond,
                "min_polar_A": min_polar,
            }
        )

    if not rows:
        return "Trajectory analysis did not produce valid frame results."

    summary_df = pd.DataFrame(rows)
    n = len(summary_df)

    hbond_occ = float((summary_df["hbond_like_count"] > 0).mean() * 100.0)
    polar_occ = float((summary_df["polar_count"] > 0).mean() * 100.0)
    disp_occ = float((summary_df["dispersion_count"] > 0).mean() * 100.0)

    best_hbond_frame = int(summary_df.sort_values("hbond_like_count", ascending=False).iloc[0]["frame"])
    best_polar_frame = int(summary_df.sort_values("polar_count", ascending=False).iloc[0]["frame"])

    lines = [
        f"Trajectory summary across {n} sampled frames (step={frame_step}, total_frames={len(frames)}):",
        (
            f"H-bond-like occupancy={hbond_occ:.1f}%, polar occupancy={polar_occ:.1f}%, "
            f"dispersion occupancy={disp_occ:.1f}%"
        ),
        (
            f"Mean counts: H-bond-like={summary_df['hbond_like_count'].mean():.2f}, "
            f"polar={summary_df['polar_count'].mean():.2f}, dispersion={summary_df['dispersion_count'].mean():.2f}"
        ),
        (
            f"Strongest H-bond-like frame={best_hbond_frame} (count={int(summary_df['hbond_like_count'].max())}), "
            f"strongest polar-contact frame={best_polar_frame} (count={int(summary_df['polar_count'].max())})"
        ),
    ]

    if summary_df["min_H_O_A"].notna().any():
        lines.append(f"Shortest H...O observed: {summary_df['min_H_O_A'].min():.3f} A")
    if summary_df["min_polar_A"].notna().any():
        lines.append(f"Shortest polar contact observed: {summary_df['min_polar_A'].min():.3f} A")

    return "\n".join(lines)


def render_gemini_advisor(active_xyz_path):
    st.subheader("Gemini Chemistry Advisor")
    st.caption("Ask chemical questions about the active structure and run local interaction analysis.")

    if not active_xyz_path:
        st.info("Select an XYZ file in the viewer to enable structural analysis context.")
        return

    is_traj_file = str(active_xyz_path).lower().endswith("_trj.xyz")
    frame_mode = "Current frame"
    frame_idx = 0
    frame_step = 1

    if is_traj_file:
        frames = load_xyz_frames(active_xyz_path)
        st.session_state.df_structure_frames = frames
        st.session_state.df_structure_source = active_xyz_path

        if frames:
            frame_mode = st.radio(
                "Advisor analysis scope",
                ["Current frame", "Trajectory summary"],
                horizontal=True,
            )
            frame_idx = st.slider("Advisor frame", 0, len(frames) - 1, min(default_trj_frame, len(frames) - 1), 1)
            frame_step = st.slider("Trajectory sampling step", 1, 20, 1, 1)
            st.session_state.df_structure = frames[frame_idx]
            st.session_state.df_structure_frame_idx = frame_idx
            st.session_state.df_structure_frame_step = frame_step
        else:
            st.session_state.df_structure = pd.DataFrame(columns=["atom", "x", "y", "z"])
            st.session_state.df_structure_frames = []
    else:
        df_structure = load_structure_dataframe(active_xyz_path)
        st.session_state.df_structure = df_structure
        st.session_state.df_structure_frames = []
        st.session_state.df_structure_source = active_xyz_path

    with st.expander("Interaction summary tables", expanded=False):
        tables = _compute_interaction_tables()
        if tables["error"]:
            st.info(tables["error"])
        else:
            st.caption(
                "Geometry-based heuristic tables from active XYZ. "
                "H-bond rows include donor-H plausibility (nearest adsorbate N/O within 1.30 A)."
            )
            st.markdown("**Likely H-bond-like contacts (top 12)**")
            st.dataframe(tables["hbond_df"].head(12), use_container_width=True)
            st.markdown("**Polar contacts <= 3.5 A (top 12)**")
            st.dataframe(tables["polar_df"].head(12), use_container_width=True)
            st.markdown("**Likely dispersive shell 3.3-4.5 A (top 12)**")
            st.dataframe(tables["disp_df"].head(12), use_container_width=True)

    if is_traj_file and st.session_state.get("df_structure_frames"):
        with st.expander("Trajectory interaction overview", expanded=False):
            sampled_frames = list(range(0, len(st.session_state.df_structure_frames), frame_step))
            traj_rows = []
            for fi in sampled_frames:
                tables = _compute_interaction_tables_from_df(st.session_state.df_structure_frames[fi])
                if tables["error"]:
                    continue
                traj_rows.append(
                    {
                        "frame": fi,
                        "hbond_like_count": len(tables["hbond_df"]),
                        "polar_count": len(tables["polar_df"]),
                        "dispersion_count": len(tables["disp_df"]),
                        "min_H_O_A": float(tables["hbond_df"]["H_O_A"].min()) if not tables["hbond_df"].empty else np.nan,
                        "min_polar_A": float(tables["polar_df"]["distance_A"].min()) if not tables["polar_df"].empty else np.nan,
                    }
                )

            traj_df = pd.DataFrame(traj_rows)
            if traj_df.empty:
                st.info("No valid trajectory frame metrics available.")
            else:
                st.caption("Per-frame interaction metrics using the selected trajectory sampling step.")
                st.dataframe(traj_df, use_container_width=True)
                if frame_mode == "Trajectory summary":
                    st.info(
                        "Tip: ask Gemini about persistence/occupancy trends; it can call "
                        "`analyze_trajectory_interactions(frame_step=...)` automatically."
                    )

    if genai is None:
        st.error("google-genai is not installed. Add it to requirements and redeploy.")
        return

    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = []

    if "gemini_chat" not in st.session_state:
        try:
            client = genai.Client()
            chat_config = None
            if genai_types is not None:
                chat_config = genai_types.GenerateContentConfig(
                    system_instruction=(
                        "You are a chemical advisor for RNA-silicate interfaces. "
                        "Always run characterize_surface_interactions for questions about interfacial behavior, "
                        "and run analyze_trajectory_interactions for trajectory- or persistence-related questions. "
                        "then explain results with explicit caveats (heuristic geometry from XYZ). "
                        "Distinguish likely H-bond-like, polar, and likely dispersive interactions."
                    ),
                    tools=[measure_surface_interactions, characterize_surface_interactions, analyze_trajectory_interactions],
                )
            st.session_state.gemini_chat = client.chats.create(
                model="gemini-2.5-flash",
                config=chat_config,
            )
        except Exception as exc:
            st.error(f"Failed to initialize Gemini chat: {exc}")
            return

    messages_box = st.container()
    with messages_box:
        for message in st.session_state.gemini_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about interactions, geometry, or adsorption behavior")
    if not prompt:
        return

    st.session_state.gemini_messages.append({"role": "user", "content": prompt})
    try:
        response = st.session_state.gemini_chat.send_message(prompt)
        answer = getattr(response, "text", None)
        if not answer:
            answer = "I could not generate a text response for that query."
        st.session_state.gemini_messages.append({"role": "assistant", "content": answer})
    except Exception as exc:
        err = f"Gemini request failed: {exc}"
        st.session_state.gemini_messages.append({"role": "assistant", "content": err})

    st.rerun()


@st.cache_data
def get_text_preview(path, max_bytes=MAX_TEXT_PREVIEW_BYTES):
    resolved_path = resolve_repo_path(path)
    if resolved_path is None or not resolved_path.exists():
        return None, 0, False

    file_size = resolved_path.stat().st_size
    with resolved_path.open("r", errors="replace") as f:
        preview = f.read(max_bytes)
    was_truncated = file_size > max_bytes
    return preview, file_size, was_truncated

def inject_visibility_fix(html):
    """
    Injects an IntersectionObserver into the py3Dmol HTML to ensure that the
    viewer resizes and renders properly when a hidden Streamlit tab becomes visible.
    """
    # Expose the viewers to the window object so the observer can find them
    html = re.sub(r'var (viewer_\d+)', r'window.\1', html)
    html = re.sub(r'var (viewergrid_\d+)', r'window.\1', html)
    
    script = """
    <script>
    $3Dmolpromise.then(function() {
        var observer = new IntersectionObserver(function(entries) {
            if(entries[0].isIntersecting) {
                for(var key in window) {
                    if(key.startsWith('viewergrid_') && window[key] !== null) {
                        try { window[key][0][0].resize(); window[key][0][0].zoomTo(); window[key][0][0].render(); } catch(e) {}
                        try { window[key][0][1].resize(); window[key][0][1].zoomTo(); window[key][0][1].render(); } catch(e) {}
                    } else if(key.startsWith('viewer_') && window[key] !== null && typeof window[key].resize === 'function') {
                        try { window[key].resize(); window[key].zoomTo(); window[key].render(); } catch(e) {}
                    }
                }
            }
        });
        observer.observe(document.body);
    });
    </script>
    """
    return html + script

import re

def inject_viewer_ui(html, n_frames, stride, is_grid=False, n_sol=1, n_dry=1, start_frame=0):
    match = re.search(r'viewer_(\d+)', html)
    if not match: return html
    vid = match.group(1)
    
    if is_grid:
        html = html.replace(f'var viewergrid_{vid}', f'window.viewergrid_{vid}')

        js_update = f'''
            var frame_sol = Math.min(frame, {n_sol} - 1);
            var frame_dry = Math.min(frame, {n_dry} - 1);
            if (typeof window.viewergrid_{vid} !== 'undefined' && window.viewergrid_{vid} !== null) {{
                window.viewergrid_{vid}[0][0].setFrame(frame_sol);
                window.viewergrid_{vid}[0][0].render();
                window.viewergrid_{vid}[0][1].setFrame(frame_dry);
                window.viewergrid_{vid}[0][1].render();
            }}
        '''
    else:
        js_update = f'''
            viewer_{vid}.setFrame(frame);
            viewer_{vid}.render();
        '''

    ui_html = f'''
    <style>
        /* Hide scrollbars completely in the iframe */
        body {{ margin: 0; overflow: hidden; }}
    </style>
    <div style="width: 100%; padding: 5px 15px; display: flex; align-items: center; justify-content: center; font-family: sans-serif; box-sizing: border-box; background: white; overflow: hidden;">
        <button id="btn_play_{vid}" style="padding: 4px 10px; cursor: pointer; border: 1px solid #ccc; background: #eee; border-radius: 4px; margin-right: 15px;">Play</button>
        <button id="btn_prev_{vid}" style="padding: 4px 10px; cursor: pointer; border: 1px solid #ccc; background: #eee; border-radius: 4px;">&lt; Prev</button>
        <input type="range" id="slider_{vid}" min="0" max="{n_frames - 1}" value="{start_frame}" step="{stride}" style="flex-grow: 1; margin: 0 15px;">
        <button id="btn_next_{vid}" style="padding: 4px 10px; cursor: pointer; border: 1px solid #ccc; background: #eee; border-radius: 4px;">Next &gt;</button>
        <span id="frame_label_{vid}" style="margin-left: 15px; min-width: 100px; font-size: 14px; text-align: right; white-space: nowrap;">Frame: 1 / {n_frames}</span>
    </div>
    <script>
    $3Dmolpromise.then(function() {{
        var slider = document.getElementById("slider_{vid}");
        var lbl = document.getElementById("frame_label_{vid}");
        var btnPrev = document.getElementById("btn_prev_{vid}");
        var btnNext = document.getElementById("btn_next_{vid}");
        var btnPlay = document.getElementById("btn_play_{vid}");
        var isPlaying = false;
        var syncInterval = null;
        
        function stopAnimation() {{
            isPlaying = false;
            btnPlay.innerText = "Play";
            if (syncInterval) {{
                clearInterval(syncInterval);
                syncInterval = null;
            }}
        }}
        
        function updateFrame(val) {{
            var frame = parseInt(val);
            {js_update}
            lbl.innerText = "Frame: " + (frame + 1) + " / {n_frames}";
            slider.value = frame;
        }}
        
        slider.oninput = function() {{
            if (isPlaying) stopAnimation();
            updateFrame(this.value);
        }};
        
        btnPrev.onclick = function() {{
            if (isPlaying) stopAnimation();
            var v = parseInt(slider.value) - {stride};
            if (v >= 0) updateFrame(v);
            else updateFrame(0);
        }};
        
        btnNext.onclick = function() {{
            if (isPlaying) stopAnimation();
            var v = parseInt(slider.value) + {stride};
            if (v < {n_frames}) updateFrame(v);
            else updateFrame({n_frames - 1});
        }};
        
        btnPlay.onclick = function() {{
            if (isPlaying) {{
                stopAnimation();
            }} else {{
                isPlaying = true;
                btnPlay.innerText = "Pause";
                syncInterval = setInterval(function() {{
                    var current = parseInt(slider.value);
                    var nextFrame = current + {stride};
                    if (nextFrame >= {n_frames}) {{
                        nextFrame = 0;
                    }}
                    updateFrame(nextFrame);
                }}, 100);
            }}
        }};
    }});
    </script>
    '''
    return html + ui_html

def render_xyz(xyz_path, title=None, height=600, width=1000, fast_mode=False, start_frame=0):

    xyz_data = get_xyz_data(xyz_path)
    if xyz_data is None:
        st.error(f"File not found: {xyz_path}")
        return
    
    view = py3Dmol.view(width=width, height=height)
    # Reverting to white background
    view.setBackgroundColor('white')
    
    is_trajectory = xyz_path.lower().endswith("_trj.xyz")
    if is_trajectory:
        view.addModelsAsFrames(xyz_data, "xyz")
        view.setFrame(start_frame)
    else:
        view.addModel(xyz_data, "xyz")

    # Styling
    main_stick_radius = 0.14 if fast_mode else 0.18
    main_sphere_scale = 0.20 if fast_mode else 0.25
    surface_stick_radius = 0.08 if fast_mode else 0.12
    view.setStyle({'elem': ["C", "N", "P"]}, {'stick': {'radius': main_stick_radius}, 'sphere': {'scale': main_sphere_scale}})
    if show_surface:
        view.setStyle({'elem': ["Si", "Al", "O"]}, {'stick': {'radius': surface_stick_radius, 'opacity': 0.9}})
    else:
        view.setStyle({'elem': ["Si", "Al", "O"]}, {'sphere': {'radius': 0.01, 'opacity': 0}})
    h_stick_radius = 0.07 if fast_mode else 0.09
    h_sphere_scale = 0.14 if fast_mode else 0.18
    view.setStyle({'elem': "H"}, {'stick': {'radius': h_stick_radius, 'color': '#9aa0a6'}, 'sphere': {'scale': h_sphere_scale, 'color': '#9aa0a6'}})
    
    if spin:
        view.spin(True)
    
    view.zoomTo() # Always zoom to ensure visibility in new iframe
    if title:
        st.subheader(title)
    html = view._make_html()
    
    if is_trajectory:
        n_frames = get_frame_count(xyz_path)
        if n_frames > 1:
            html = inject_viewer_ui(html, n_frames, trajectory_stride, start_frame=start_frame)
            height += 60 # Accommodate UI
            
    html = inject_visibility_fix(html)
    encoded_html = base64.b64encode(html.encode("utf-8")).decode("ascii")
    st.components.v1.iframe(f"data:text/html;base64,{encoded_html}", height=height, width=width)

with tabs[0]:
    st.header("File Viewer")

    viewer_col, advisor_col = st.columns([2, 1])
    active_xyz_path = None

    with viewer_col:
        selectable_files = filtered_df.copy()

        show_all_files = st.checkbox("Include text/data files", value=False, help="By default, only 3D visualization files (.xyz) are shown.")

        def rank_viewer_file(path):
            p = str(path).lower()
            if p.endswith('_trj.xyz'):
                return 2
            if p.endswith('.xyz'):
                return 1
            return 0

        selectable_files['rank'] = selectable_files['repo_path'].apply(rank_viewer_file)

        if not show_all_files:
            selectable_files = selectable_files[selectable_files['rank'] > 0]

        selectable_files = selectable_files.sort_values(['rank', 'repo_path'], ascending=[False, True])

        if not selectable_files.empty:
            def get_viewer_file_idx(opts, param_name="file"):
                val = st.query_params.get(param_name, "")
                if val:
                    for i, idx in enumerate(opts):
                        if selectable_files.loc[idx, 'file_name'] == val:
                            return i
                return 0

            selected_file_row = st.selectbox(
                "Select file to view",
                selectable_files.index,
                index=get_viewer_file_idx(selectable_files.index),
                format_func=lambda x: f"{selectable_files.loc[x, 'repo_path']} ({selectable_files.loc[x, 'state']})"
            )
            selected_file = selectable_files.loc[selected_file_row]
            update_param("file", selected_file['file_name'])
            path = selected_file['repo_path']
            resolved_path = resolve_repo_path(path)

            if resolved_path is None:
                st.error("Invalid file path in manifest entry.")
                st.stop()

            if path.endswith('.xyz'):
                active_xyz_path = path
                render_xyz(
                    path,
                    f"Visualizing: {selected_file['file_name']}",
                    fast_mode=performance_mode,
                    start_frame=default_trj_frame,
                )
            else:
                st.subheader(f"Viewing: {selected_file['file_name']}")
                content, file_size, was_truncated = get_text_preview(path)
                if content is None:
                    st.error(f"File not found: {path}")
                else:
                    if was_truncated:
                        st.warning(
                            f"Showing first {MAX_TEXT_PREVIEW_BYTES:,} bytes of {file_size:,} byte file. "
                            "Use download to view full content."
                        )
                    st.code(content, language="text")
                    st.download_button(
                        "Download full file",
                        data=resolved_path.read_bytes(),
                        file_name=selected_file['file_name'],
                        mime="text/plain",
                    )
        else:
            st.info("No files found for current filters.")

    with advisor_col:
        render_gemini_advisor(active_xyz_path)

with tabs[1]:
    st.header("Wet vs Dry Comparison")

    sync_cameras = st.checkbox("Sync Cameras", value=True)
    compare_mode_labels = {
        "paired": "Paired (Wet vs Dry)",
        "custom": "Custom (Cross-Structure)",
    }
    compare_mode_keys = list(compare_mode_labels.keys())
    compare_mode_default = st.query_params.get("compare_mode", "paired")
    if compare_mode_default not in compare_mode_keys:
        compare_mode_default = "paired"
    compare_mode = st.selectbox(
        "Comparison Mode",
        compare_mode_keys,
        index=compare_mode_keys.index(compare_mode_default),
        format_func=lambda k: compare_mode_labels[k],
    )
    update_param("compare_mode", compare_mode)

    def rank_files(df_subset, state_label=""):
        if df_subset.empty:
            return df_subset

        def get_rank(row):
            score = 0
            path_lower = row['repo_path'].lower()
            file_lower = row['file_name'].lower()
            if path_lower.endswith('_trj.xyz'):
                score += 10
            if state_label and state_label.lower() in file_lower:
                score += 5
            if state_label and f"/{state_label.lower()}/" in path_lower:
                score += 2
            return score

        df_subset = df_subset.copy()
        df_subset['rank'] = df_subset.apply(get_rank, axis=1)
        return df_subset.sort_values(['rank', 'repo_path'], ascending=[False, True])

    def get_file_idx(df_subset, param_name):
        val = st.query_params.get(param_name, "")
        if val:
            for i, row in enumerate(df_subset.itertuples()):
                if row.file_name == val:
                    return i
        return 0

    def render_comparison_pair(left_file, right_file, left_label, right_label):
        left_data = get_xyz_data(left_file['repo_path'])
        right_data = get_xyz_data(right_file['repo_path'])

        if left_data is None or right_data is None:
            st.error("Could not load comparison files.")
            st.stop()

        is_left_trj = left_file['repo_path'].lower().endswith("_trj.xyz")
        is_right_trj = right_file['repo_path'].lower().endswith("_trj.xyz")

        n_left = get_frame_count(left_file['repo_path']) if is_left_trj else 1
        n_right = get_frame_count(right_file['repo_path']) if is_right_trj else 1

        viewer_width = 1200
        viewer_height = 600
        view = py3Dmol.view(width=viewer_width, height=viewer_height, viewergrid=(1, 2), linked=True)
        view.setBackgroundColor('white')

        def apply_comparison_style(v, model_data, viewer_idx, is_trj=False, start_frame=0):
            if is_trj:
                v.addModelsAsFrames(model_data, "xyz", viewer=viewer_idx)
                v.setFrame(start_frame, viewer=viewer_idx)
            else:
                v.addModel(model_data, "xyz", viewer=viewer_idx)

            main_stick_radius = 0.14 if performance_mode else 0.18
            main_sphere_scale = 0.20 if performance_mode else 0.25
            surface_stick_radius = 0.08 if performance_mode else 0.12
            v.setStyle({'elem': ["C", "N", "P"]}, {'stick': {'radius': main_stick_radius}, 'sphere': {'scale': main_sphere_scale}}, viewer=viewer_idx)
            if show_surface:
                v.setStyle({'elem': ["Si", "Al", "O"]}, {'stick': {'radius': surface_stick_radius, 'opacity': 0.9}}, viewer=viewer_idx)
            else:
                v.setStyle({'elem': ["Si", "Al", "O"]}, {'sphere': {'radius': 0.01, 'opacity': 0}}, viewer=viewer_idx)
            h_stick_radius = 0.07 if performance_mode else 0.09
            h_sphere_scale = 0.14 if performance_mode else 0.18
            v.setStyle(
                {'elem': "H"},
                {'stick': {'radius': h_stick_radius, 'color': '#9aa0a6'}, 'sphere': {'scale': h_sphere_scale, 'color': '#9aa0a6'}},
                viewer=viewer_idx,
            )
            v.zoomTo(viewer=viewer_idx)

        apply_comparison_style(view, left_data, (0, 0), is_trj=is_left_trj, start_frame=default_trj_frame)
        apply_comparison_style(view, right_data, (0, 1), is_trj=is_right_trj, start_frame=default_trj_frame)

        st.subheader(f"Left: {left_label} | Right: {right_label}")
        html = view._make_html()

        if is_left_trj or is_right_trj:
            max_frames = max(n_left, n_right)
            if max_frames > 1:
                html = inject_viewer_ui(
                    html,
                    max_frames,
                    trajectory_stride,
                    is_grid=True,
                    n_sol=n_left,
                    n_dry=n_right,
                    start_frame=default_trj_frame,
                )
                viewer_height += 60

        html = inject_visibility_fix(html)
        encoded_html = base64.b64encode(html.encode("utf-8")).decode("ascii")
        st.components.v1.iframe(
            f"data:text/html;base64,{encoded_html}",
            height=viewer_height,
            width=viewer_width,
        )

    if compare_mode == "paired":
        comp_groups = []
        for name, group in groups:
            states = group['state'].unique()
            if 'solvated' in states and 'dry' in states:
                solvated_xyz = group[(group['state'] == 'solvated') & (group['repo_path'].str.endswith('.xyz'))]
                dry_xyz = group[(group['state'] == 'dry') & (group['repo_path'].str.endswith('.xyz'))]
                if not solvated_xyz.empty and not dry_xyz.empty:
                    comp_groups.append(name)

        def get_comp_idx(opts, param_name="compare"):
            val = st.query_params.get(param_name, "")
            if val:
                for i, opt in enumerate(opts):
                    if f"{opt[0]}_{opt[1]}_{opt[2]}_{opt[3]}" == val:
                        return i
            return 0

        if comp_groups:
            selected_comp = st.selectbox(
                "Select Frame to Compare",
                comp_groups,
                index=get_comp_idx(comp_groups),
                format_func=lambda x: f"{x[0]} | {x[1]} | {x[2]} ({x[3]})",
            )

            update_param("compare", f"{selected_comp[0]}_{selected_comp[1]}_{selected_comp[2]}_{selected_comp[3]}")

            group_df = groups.get_group(selected_comp)
            solvated_files = rank_files(group_df[(group_df['state'] == 'solvated') & (group_df['repo_path'].str.endswith('.xyz'))], 'solvated')
            dry_files = rank_files(group_df[(group_df['state'] == 'dry') & (group_df['repo_path'].str.endswith('.xyz'))], 'dry')

            if len(solvated_files) > 1:
                sol_idx_pos = st.selectbox(
                    "Select Solvated File",
                    range(len(solvated_files)),
                    index=get_file_idx(solvated_files, "sol_file"),
                    format_func=lambda i: solvated_files.iloc[i]['file_name'],
                    key="sol_select",
                )
                solvated_file = solvated_files.iloc[sol_idx_pos]
                update_param("sol_file", solvated_file['file_name'])
            else:
                solvated_file = solvated_files.iloc[0]
                update_param("sol_file", "All")

            if len(dry_files) > 1:
                dry_idx_pos = st.selectbox(
                    "Select Dry File",
                    range(len(dry_files)),
                    index=get_file_idx(dry_files, "dry_file"),
                    format_func=lambda i: dry_files.iloc[i]['file_name'],
                    key="dry_select",
                )
                dry_file = dry_files.iloc[dry_idx_pos]
                update_param("dry_file", dry_file['file_name'])
            else:
                dry_file = dry_files.iloc[0]
                update_param("dry_file", "All")

            if sync_cameras:
                render_comparison_pair(solvated_file, dry_file, "Solvated", "Dry")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Solvated (Wet)")
                    render_xyz(solvated_file['repo_path'], height=400, fast_mode=performance_mode, start_frame=default_trj_frame)
                with col2:
                    st.subheader("Dry")
                    render_xyz(dry_file['repo_path'], height=400, fast_mode=performance_mode, start_frame=default_trj_frame)
        else:
            st.info("No frames found with both solvated and dry .xyz files for current filters.")

    else:
        custom_candidates = filtered_df[filtered_df['repo_path'].str.endswith('.xyz')].copy()
        custom_candidates = rank_files(custom_candidates)

        if custom_candidates.empty:
            st.info("No .xyz files available for custom comparison with current filters.")
        else:
            left_idx = st.selectbox(
                "Select Left File",
                range(len(custom_candidates)),
                index=get_file_idx(custom_candidates, "left_file"),
                format_func=lambda i: f"{custom_candidates.iloc[i]['repo_path']} ({custom_candidates.iloc[i]['state']})",
                key="custom_left_select",
            )
            right_idx = st.selectbox(
                "Select Right File",
                range(len(custom_candidates)),
                index=get_file_idx(custom_candidates, "right_file"),
                format_func=lambda i: f"{custom_candidates.iloc[i]['repo_path']} ({custom_candidates.iloc[i]['state']})",
                key="custom_right_select",
            )

            left_file = custom_candidates.iloc[left_idx]
            right_file = custom_candidates.iloc[right_idx]
            update_param("left_file", left_file['file_name'])
            update_param("right_file", right_file['file_name'])

            if sync_cameras:
                render_comparison_pair(left_file, right_file, left_file['file_name'], right_file['file_name'])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Left: {left_file['file_name']}")
                    render_xyz(left_file['repo_path'], height=400, fast_mode=performance_mode, start_frame=default_trj_frame)
                with col2:
                    st.subheader(f"Right: {right_file['file_name']}")
                    render_xyz(right_file['repo_path'], height=400, fast_mode=performance_mode, start_frame=default_trj_frame)

with tabs[2]:
    st.header("Filtered Data")
    safe_df = filtered_df.drop(columns=["source_path"], errors="ignore")
    st.dataframe(safe_df)

st.sidebar.markdown("---")
st.sidebar.info("RNA Silicate Interactions Streamlit App")

# Reorder and cleanup URL query parameters
current_params = st.query_params.to_dict()
st.query_params.clear()

# Enforce tab as the very first parameter
if "tab" in current_params:
    st.query_params["tab"] = current_params.pop("tab")

active_tab = st.query_params.get("tab", "Visualization")

# Clean up tab-specific parameters so they don't persist incorrectly
if active_tab == "Visualization":
    for p in ["compare_mode", "compare", "sol_file", "dry_file", "left_file", "right_file"]:
        current_params.pop(p, None)
elif active_tab == "Comparison":
    current_params.pop("file", None)
    compare_mode_value = current_params.get("compare_mode", "paired")
    if compare_mode_value == "paired":
        for p in ["left_file", "right_file"]:
            current_params.pop(p, None)
    else:
        for p in ["compare", "sol_file", "dry_file"]:
            current_params.pop(p, None)
elif active_tab == "Data Table":
    for p in ["compare_mode", "compare", "sol_file", "dry_file", "left_file", "right_file", "file"]:
        current_params.pop(p, None)

# Add remaining parameters back in
for k, v in current_params.items():
    st.query_params[k] = v
