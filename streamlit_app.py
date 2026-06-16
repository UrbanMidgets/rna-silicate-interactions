import streamlit as st
import pandas as pd
import py3Dmol
import os
import base64
from pathlib import Path
import warnings
import math

import altair as alt
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

from parsers import ClassicalConformationParser, ClassicalTopologyConfig, QuantumTopologyParser, StateClassifier

# Set page config
st.set_page_config(page_title="RNA Silicate Interactions", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning)

st.title("RNA Silicate Interactions")
st.markdown("Supporting data for RNA nucleotide interactions with silicate surfaces.")

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.tsv"
MAX_TEXT_PREVIEW_BYTES = 200_000
PARSER_CACHE_VERSION = "si-op-phosphate-v3"
HARTREE_TO_KJ_MOL = 2625.499638


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

def _path_mtime(path):
    return path.stat().st_mtime if path.exists() else 0


@st.cache_data
def load_manifest(manifest_mtime, classifications_mtime, parser_cache_version):
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
        
        # Load classifications if available
        classifications_path = REPO_ROOT / "data" / "classifications.csv"
        if classifications_path.exists():
            class_df = pd.read_csv(classifications_path)
            class_df = class_df.drop_duplicates(subset=['system', 'surface', 'frame', 'state'], keep='last')
            # Merge classifications based on system, surface, frame, state
            df = df.merge(
                class_df[['system', 'surface', 'frame', 'state', 'State', 'Final_Bond_Order', 'Final_Intramol_Dist', 'Final_Anchoring_Dist']], 
                on=['system', 'surface', 'frame', 'state'], 
                how='left'
            )
            # Rename 'State' to 'classification' for clarity
            df = df.rename(columns={'State': 'classification'})
            # Fill NaNs for classifications
            df['classification'] = df['classification'].fillna('Unclassified')
            
        return df
    return pd.DataFrame()

CLASSIFICATIONS_PATH = REPO_ROOT / "data" / "classifications.csv"
df = load_manifest(_path_mtime(MANIFEST_PATH), _path_mtime(CLASSIFICATIONS_PATH), PARSER_CACHE_VERSION)

if df.empty:
    st.error("Manifest not found. Please ensure data/MANIFEST.tsv exists.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

def get_options(dataframe, column):
    opts = sorted([str(x) for x in dataframe[column].unique() if x])
    return ["All"] + opts

def get_state_options(dataframe):
    state_order = ["solvated", "dry", "initial", "docking"]
    states = [str(x) for x in dataframe["state"].unique() if x]
    ordered_states = [state for state in state_order if state in states]
    extra_states = sorted([state for state in states if state not in state_order])
    return ["All"] + ordered_states + extra_states

def format_state_option(value):
    labels = {
        "All": "All",
        "solvated": "Solvated",
        "dry": "Dry",
        "initial": "Initial",
        "docking": "Docking",
    }
    return labels.get(value, str(value).title())

def format_file_option(row):
    context = [row.get("system"), row.get("surface"), row.get("frame"), row.get("state")]
    context = [str(value) for value in context if value]
    return f"{row['file_name']} | {' / '.join(context)}" if context else row["file_name"]

def get_param_idx(opts, param_name, default="All", prefix=""):
    val = st.query_params.get(param_name, default)
    if val in opts:
        return opts.index(val)
    if prefix and f"{prefix}{val}" in opts:
        return opts.index(f"{prefix}{val}")
    return opts.index(default) if default in opts else 0


def get_bool_param(param_name, default=False):
    value = st.query_params.get(param_name)
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def update_bool_param(param_name, value, default=False):
    if bool(value) == bool(default):
        if param_name in st.query_params:
            del st.query_params[param_name]
    else:
        st.query_params[param_name] = "1" if value else "0"


def get_int_param(param_name, default, minimum=None, maximum=None):
    try:
        value = int(st.query_params.get(param_name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def update_int_param(param_name, value, default):
    if int(value) == int(default):
        if param_name in st.query_params:
            del st.query_params[param_name]
    else:
        st.query_params[param_name] = str(int(value))

def update_param(param_name, value):
    if value == "All":
        if param_name in st.query_params:
            del st.query_params[param_name]
    else:
        # If it's structure, and value starts with 'frame', maybe strip it?
        # Let's just keep the value as is, or if they want exactly 14 instead of frame14:
        if param_name == "structure" and value.startswith("frame"):
            next_value = value.replace("frame", "")
        else:
            next_value = value
        if st.query_params.get(param_name) != next_value:
            st.query_params[param_name] = next_value

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

state_opts = get_state_options(filtered_df)
solvent_state = st.sidebar.selectbox(
    "Solvent State",
    state_opts,
    index=get_param_idx(state_opts, "state"),
    format_func=format_state_option,
)
update_param("state", solvent_state)
if solvent_state != "All":
    filtered_df = filtered_df[filtered_df['state'] == solvent_state]

if 'classification' in filtered_df.columns:
    class_opts = get_options(filtered_df, 'classification')
    classification = st.sidebar.selectbox("Classification", class_opts, index=get_param_idx(class_opts, "class"))
    update_param("class", classification)
    if classification != "All":
        filtered_df = filtered_df[filtered_df['classification'] == classification]

role_opts = get_options(filtered_df, 'role')
role = st.sidebar.selectbox("Role", role_opts, index=get_param_idx(role_opts, "role"))
update_param("role", role)
if role != "All":
    filtered_df = filtered_df[filtered_df['role'] == role]

frame_unfiltered_df = filtered_df.copy()

frame_opts = get_options(filtered_df, 'frame')
frame = st.sidebar.selectbox("Frame", frame_opts, index=get_param_idx(frame_opts, "structure", prefix="frame"))
update_param("structure", frame)
if frame != "All":
    filtered_df = filtered_df[filtered_df['frame'] == frame]

st.sidebar.header("Settings")
show_surface = st.sidebar.checkbox("Show Surface (Si, Al, O)", value=get_bool_param("show_surface", True))
update_bool_param("show_surface", show_surface, True)
spin = st.sidebar.checkbox("Spin Molecule", value=get_bool_param("spin", False))
update_bool_param("spin", spin, False)
performance_mode = st.sidebar.checkbox("Performance Mode", value=get_bool_param("performance", True))
update_bool_param("performance", performance_mode, True)

try:
    default_trj_frame = int(st.query_params.get("trj_frame", 0))
except ValueError:
    default_trj_frame = 0

trajectory_stride = st.sidebar.slider(
    "Trajectory Frame Step",
    1,
    10,
    get_int_param("trj_stride", 1, minimum=1, maximum=10),
)
update_int_param("trj_stride", trajectory_stride, 1)

st.sidebar.markdown(f"**Matches:** {len(filtered_df)}")

# Grouping by system/surface/frame for comparison
# We want to identify pairs of (solvated, dry) for the same frame
comparison_options = filtered_df[filtered_df['state'].isin(['solvated', 'dry'])]
groups = comparison_options.groupby(['surface', 'system', 'frame', 'role'])

tab_options = ["Visualization", "Comparison", "Relative Energies", "Data Table"]
default_tab = st.query_params.get("tab", "Visualization")
if default_tab not in tab_options:
    default_tab = "Visualization"

active_tab = st.segmented_control(
    "Section",
    tab_options,
    default=default_tab,
    key="active_tab",
    label_visibility="collapsed",
    width="stretch",
)
if active_tab is None:
    active_tab = default_tab
if st.query_params.get("tab") != active_tab:
    st.query_params["tab"] = active_tab

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
def get_xyz_atom_count(xyz_path):
    resolved_path = resolve_repo_path(xyz_path)
    if resolved_path is None or not resolved_path.exists():
        return None
    try:
        with resolved_path.open("r") as f:
            first_line = f.readline().strip()
        return int(first_line)
    except (OSError, ValueError):
        return None


@st.cache_data
def get_xyz_atom_signature(xyz_path):
    resolved_path = resolve_repo_path(xyz_path)
    if resolved_path is None or not resolved_path.exists():
        return None
    try:
        with resolved_path.open("r") as f:
            first_line = f.readline().strip()
            atom_count = int(first_line)
            f.readline()
            element_counts = {}
            for _ in range(atom_count):
                parts = f.readline().split()
                if not parts:
                    continue
                element = parts[0]
                element_counts[element] = element_counts.get(element, 0) + 1
        if not element_counts:
            return None
        return "; ".join(f"{element}{element_counts[element]}" for element in sorted(element_counts))
    except (OSError, ValueError):
        return None


def rank_xyz_files(df_subset, state_label=""):
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


def select_representative_files_by_frame(df_subset):
    if df_subset.empty:
        return df_subset
    ranked = rank_xyz_files(df_subset[df_subset['repo_path'].str.lower().str.endswith('.xyz')])
    if ranked.empty:
        return ranked
    return ranked.groupby("frame", group_keys=False).head(1).sort_values("frame")

@st.cache_data
def get_xyz_data(path):
    resolved_path = resolve_repo_path(path)
    if resolved_path is None or not resolved_path.exists():
        return None
    with resolved_path.open("r") as f:
        return f.read()


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


@st.cache_data(show_spinner=False)
def parse_orca_single_point_energies(out_path, out_mtime=0, parser_cache_version=PARSER_CACHE_VERSION):
    resolved_path = resolve_repo_path(out_path)
    if resolved_path is None or not resolved_path.exists():
        return pd.DataFrame()

    rows = []
    with resolved_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "FINAL SINGLE POINT ENERGY" not in line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                energy_hartree = float(parts[-1])
            except ValueError:
                continue
            rows.append(
                {
                    "energy_record": len(rows),
                    "energy_hartree": energy_hartree,
                    "energy_kj_mol": energy_hartree * HARTREE_TO_KJ_MOL,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["energy_record", "energy_hartree", "energy_kj_mol", "relative_energy_kj_mol"])

    energy_df = pd.DataFrame(rows)
    minimum_energy = energy_df["energy_hartree"].min()
    energy_df["relative_energy_kj_mol"] = (energy_df["energy_hartree"] - minimum_energy) * HARTREE_TO_KJ_MOL
    return energy_df


def get_output_energy_data(out_path):
    if not out_path:
        return pd.DataFrame()
    resolved_path = resolve_repo_path(out_path)
    if resolved_path is None or not resolved_path.exists():
        return pd.DataFrame()
    return parse_orca_single_point_energies(str(out_path), resolved_path.stat().st_mtime, PARSER_CACHE_VERSION)


def summarize_energy(energy_df):
    if energy_df.empty:
        return {}
    final = energy_df.iloc[-1]
    return {
        "final_energy_hartree": float(final["energy_hartree"]),
        "final_energy_kj_mol": float(final["energy_kj_mol"]),
        "final_energy_record": int(final["energy_record"]),
        "energy_record_count": int(len(energy_df)),
    }


def find_matching_output(row):
    group = df[
        (df['system'] == row['system'])
        & (df['surface'] == row['surface'])
        & (df['frame'] == row['frame'])
        & (df['state'] == row['state'])
        & (df['repo_path'].str.endswith('.out'))
    ].copy()
    if group.empty:
        return None

    selected_stem = Path(row['file_name']).stem.replace("_trj", "")
    group['stem_score'] = group['file_name'].apply(
        lambda name: 2 if Path(name).stem in selected_stem or selected_stem in Path(name).stem else 0
    )
    if 'status' in group.columns:
        group['status_score'] = group['status'].apply(lambda status: 1 if status == 'normal' else 0)
    else:
        group['status_score'] = 0
    group = group.sort_values(['stem_score', 'status_score', 'repo_path'], ascending=[False, False, True])
    return group.iloc[0]


@st.cache_data(show_spinner=False)
def parse_current_interactions(xyz_path, out_path=None, parser_cache_version=PARSER_CACHE_VERSION):
    xyz_resolved = resolve_repo_path(xyz_path)
    if xyz_resolved is None or not xyz_resolved.exists():
        return pd.DataFrame(), pd.DataFrame(), {"State": "Unknown (Missing XYZ)", "Reason": "Selected XYZ file was not found"}

    u = mda.Universe(str(xyz_resolved))
    p_atoms = u.select_atoms("name P")
    if len(p_atoms) == 0:
        return pd.DataFrame(), pd.DataFrame(), {"State": "Unknown (No Phosphorus)", "Reason": "No phosphorus atom was found in the selected XYZ"}

    p_idx = int(p_atoms.indices[0])
    oxygen_indices = [int(atom.index) for atom in u.atoms if atom.name == "O" and atom.index > p_idx]
    p_distances = distance_array(u.atoms[p_idx].position[None, :], u.atoms[oxygen_indices].positions)[0]
    phosphate_oxygen_indices = [
        oxygen_indices[i]
        for i, distance in enumerate(p_distances)
        if distance <= 1.9
    ]
    topology_config = ClassicalTopologyConfig(p_index=p_idx, surface_index_stop=p_idx)
    c_df = ClassicalConformationParser(topology_config).parse_trajectory(xyz_resolved)

    q_df = pd.DataFrame()
    if out_path:
        out_resolved = resolve_repo_path(out_path)
        if out_resolved is not None and out_resolved.exists():
            si_indices = [int(atom.index) for atom in u.atoms if atom.name == "Si" and atom.index < p_idx]
            q_df = QuantumTopologyParser(
                p_idx=p_idx,
                target_indices=phosphate_oxygen_indices,
                target_element="O",
                partner_indices=si_indices,
                partner_element="Si",
            ).parse_file(out_resolved)

    result = StateClassifier().classify_pair(q_df, c_df, label=Path(xyz_path).stem)
    return q_df, c_df, result


def render_interaction_overview(selected_file):
    if not selected_file['repo_path'].lower().endswith('.xyz'):
        return

    matching_output = find_matching_output(selected_file)
    out_path = matching_output['repo_path'] if matching_output is not None else None
    q_df, c_df, result = parse_current_interactions(selected_file['repo_path'], out_path, PARSER_CACHE_VERSION)
    energy_df = get_output_energy_data(out_path)
    energy_summary = summarize_energy(energy_df)

    st.subheader("Trajectory Interaction Overview")
    st.caption(
        f"{selected_file['system']} | {selected_file['surface']} | {selected_file['frame']} | "
        f"{selected_file['state']} | ORCA: {Path(out_path).name if out_path else 'not found'}"
    )

    render_metric_grid(
        [
            ("Final State", result.get("State", "Unknown")),
            ("Si-O(P) Mayer BO", f"{result.get('Final_Bond_Order', 0.0):.3f}"),
            ("Ribose-PO min dist", _format_metric(result.get("Final_Intramol_Dist"), "A")),
            ("P-O to surface O", _format_metric(result.get("Final_PhosphateO_SurfaceO_Dist"), "A")),
            ("P-O to surface Si", _format_metric(result.get("Final_PhosphateO_SurfaceSi_Dist"), "A")),
            ("PO-siloxane COM", _format_metric(result.get("Final_PO_Siloxane_COM_Dist"), "A")),
            ("FINAL SINGLE POINT ENERGY", _format_energy_hartree(energy_summary.get("final_energy_hartree"))),
            ("Final energy converted", _format_energy_kj_mol(energy_summary.get("final_energy_kj_mol"))),
        ]
    )

    reason = result.get("Reason")
    if reason:
        st.caption(reason)

    if not c_df.empty:
        classical_cols = [
            "min_dist_intramol_OH_PO",
            "min_dist_nh2_surfO",
            "min_dist_riboseOH_surfSilanol",
            "min_dist_phosphateO_surfaceO",
            "min_dist_phosphateO_surfaceSi",
            "dist_PO_siloxane_COM",
        ]
        available_cols = [col for col in classical_cols if col in c_df.columns]
        if available_cols:
            plot_df = c_df.set_index("Frame")[available_cols].rename(
                columns={
                    "min_dist_intramol_OH_PO": "Ribose hydroxyl oxygen to phosphate oxygen distance",
                    "min_dist_nh2_surfO": "Nucleobase nitrogen to surface oxygen distance",
                    "min_dist_riboseOH_surfSilanol": "Ribose hydroxyl oxygen to surface oxygen distance",
                    "min_dist_phosphateO_surfaceO": "Phosphate oxygen to surface oxygen distance",
                    "min_dist_phosphateO_surfaceSi": "Phosphate oxygen to surface silicon distance",
                    "dist_PO_siloxane_COM": "Phosphate oxygen to siloxane center-of-mass distance",
                }
            )
            render_interaction_line_chart(
                plot_df,
                x_title="Trajectory frame",
                y_title="Distance (A)",
                legend_title="Geometric interaction",
                height=320,
                y_zero=False,
                y_tick_count=10,
            )

    if energy_summary:
        st.caption(
            "`FINAL SINGLE POINT ENERGY` is shown for the matched ORCA output of the displayed structure. "
            f"The value is the final record in that file ({energy_summary.get('energy_record_count')} parsed records)."
        )
        energy_plot = energy_df.set_index("energy_record")[["relative_energy_kj_mol"]].rename(
            columns={
                "relative_energy_kj_mol": (
                    "Relative FINAL SINGLE POINT ENERGY during optimization "
                    "(0 = lowest parsed record)"
                )
            }
        )
        render_interaction_line_chart(
            energy_plot,
            x_title="ORCA energy record",
            y_title="Relative energy (kJ/mol)",
            legend_title="Energy development",
            height=260,
            points=True,
            y_zero=True,
            y_tick_count=10,
        )

    if not q_df.empty:
        bond_cols = [col for col in ["Step", "Target_Idx", "Si_Idx", "Bond_Order", "Target_Pair_Found"] if col in q_df.columns]
        q_plot = q_df.set_index("Step")[["Bond_Order"]].rename(columns={"Bond_Order": "Silicon to phosphate oxygen Mayer bond order"})
        st.caption(
            "The Mayer bond-order plot comes from ORCA bond-order sections, not XYZ trajectory frames. "
            "Most files contain only two sections: the first electronic-structure evaluation and the final optimized structure."
        )
        render_interaction_line_chart(
            q_plot,
            x_title="ORCA Mayer bond-order section",
            y_title="Mayer bond order",
            legend_title="Quantum topology metric",
            height=240,
            points=True,
            y_zero=True,
            y_tick_count=8,
        )
        with st.expander("Mayer bond-order sections"):
            st.dataframe(q_df[bond_cols], use_container_width=True)
    elif out_path is None:
        st.info("No matching ORCA .out file was found for this selected trajectory.")


def render_comparison_energy_summary(left_file, right_file, left_label, right_label):
    left_output = find_matching_output(left_file)
    right_output = find_matching_output(right_file)
    left_out_path = left_output['repo_path'] if left_output is not None else None
    right_out_path = right_output['repo_path'] if right_output is not None else None

    left_energy_df = get_output_energy_data(left_out_path)
    right_energy_df = get_output_energy_data(right_out_path)
    left_energy = summarize_energy(left_energy_df)
    right_energy = summarize_energy(right_energy_df)

    metrics = [
        (f"{left_label} final energy (Hartree)", _format_energy_hartree(left_energy.get("final_energy_hartree"))),
        (f"{left_label} final energy (kJ/mol)", _format_energy_kj_mol(left_energy.get("final_energy_kj_mol"))),
        (f"{right_label} final energy (Hartree)", _format_energy_hartree(right_energy.get("final_energy_hartree"))),
        (f"{right_label} final energy (kJ/mol)", _format_energy_kj_mol(right_energy.get("final_energy_kj_mol"))),
    ]

    if left_energy and right_energy:
        delta_energy = right_energy["final_energy_kj_mol"] - left_energy["final_energy_kj_mol"]
        metrics.append((f"Delta E ({right_label} - {left_label})", _format_signed_energy(delta_energy)))

    render_metric_grid(metrics)
    if left_energy or right_energy:
        st.caption(
            "Energies are the final `FINAL SINGLE POINT ENERGY` records from the matched ORCA outputs. "
            "Compare absolute or delta energies only between calculations with the same composition, charge, and method."
        )
        render_comparison_energy_development(left_energy_df, right_energy_df, left_label, right_label)


def render_comparison_energy_development(left_energy_df, right_energy_df, left_label, right_label):
    traces = []
    if not left_energy_df.empty and "relative_energy_kj_mol" in left_energy_df.columns:
        left_trace = left_energy_df[["energy_record", "relative_energy_kj_mol"]].copy()
        left_trace["structure"] = left_label
        traces.append(left_trace)
    if not right_energy_df.empty and "relative_energy_kj_mol" in right_energy_df.columns:
        right_trace = right_energy_df[["energy_record", "relative_energy_kj_mol"]].copy()
        right_trace["structure"] = right_label
        traces.append(right_trace)

    if not traces:
        return

    plot_df = pd.concat(traces, ignore_index=True)
    st.caption(
        "Energy development overlay: each curve is relative to its own lowest parsed "
        "`FINAL SINGLE POINT ENERGY` record, so the optimization profiles can be compared on the same scale."
    )
    chart = alt.Chart(plot_df).mark_line(point=True).encode(
        x=alt.X("energy_record:Q", title="ORCA energy record"),
        y=alt.Y(
            "relative_energy_kj_mol:Q",
            title="Relative energy (kJ/mol)",
            scale=alt.Scale(zero=True, nice=True),
            axis=alt.Axis(tickCount=10, grid=True),
        ),
        color=alt.Color(
            "structure:N",
            title="Compared structure",
            legend=alt.Legend(orient="bottom", labelLimit=0, titleLimit=0),
        ),
        tooltip=[
            alt.Tooltip("structure:N", title="Structure"),
            alt.Tooltip("energy_record:Q", title="ORCA energy record"),
            alt.Tooltip("relative_energy_kj_mol:Q", title="Relative energy (kJ/mol)", format=".4f"),
        ],
    )
    st.altair_chart(chart.properties(height=260), use_container_width=True)


def render_same_setup_energy_comparison(frame_files):
    rows = []
    for _, file_row in frame_files.iterrows():
        output_row = find_matching_output(file_row)
        out_path = output_row["repo_path"] if output_row is not None else None
        energy_df = get_output_energy_data(out_path)
        energy = summarize_energy(energy_df)
        atom_count = get_xyz_atom_count(file_row["repo_path"])
        rows.append(
            {
                "frame": file_row["frame"],
                "file": file_row["file_name"],
                "atom_count": atom_count,
                "orca_output": Path(out_path).name if out_path else "not found",
                "final_energy_hartree": energy.get("final_energy_hartree"),
                "final_energy_kj_mol": energy.get("final_energy_kj_mol"),
            }
        )

    energy_table = pd.DataFrame(rows)
    if energy_table.empty:
        return

    atom_counts = sorted([int(value) for value in energy_table["atom_count"].dropna().unique()])
    if len(atom_counts) == 1:
        st.caption(
            f"Same-setup energy comparison: all selected frame representatives contain {atom_counts[0]} atoms, "
            "so final energies are directly comparable when the ORCA method and charge are unchanged."
        )
    elif len(atom_counts) > 1:
        st.warning(
            "Selected frame representatives do not all contain the same atom count. "
            "Use the energy values cautiously or choose a narrower setup."
        )

    valid_energy = energy_table.dropna(subset=["final_energy_kj_mol"]).copy()
    if valid_energy.empty:
        st.info("No matched ORCA final energies were found for this setup.")
        st.dataframe(energy_table, use_container_width=True)
        return

    minimum_energy = valid_energy["final_energy_kj_mol"].min()
    valid_energy["relative_final_energy_kj_mol"] = valid_energy["final_energy_kj_mol"] - minimum_energy
    metrics = [
        (f"{row.frame} relative final energy", _format_signed_energy(row.relative_final_energy_kj_mol))
        for row in valid_energy.itertuples()
    ]
    render_metric_grid(metrics)

    chart = alt.Chart(valid_energy).mark_bar(size=42).encode(
        x=alt.X("frame:N", title="Frame", sort=list(valid_energy["frame"])),
        y=alt.Y(
            "relative_final_energy_kj_mol:Q",
            title="Relative final energy (kJ/mol)",
            scale=alt.Scale(zero=True, nice=True),
            axis=alt.Axis(tickCount=10, grid=True),
        ),
        tooltip=[
            alt.Tooltip("frame:N", title="Frame"),
            alt.Tooltip("file:N", title="File"),
            alt.Tooltip("atom_count:Q", title="Atom count"),
            alt.Tooltip("final_energy_hartree:Q", title="Final energy (Hartree)", format=".10f"),
            alt.Tooltip("final_energy_kj_mol:Q", title="Final energy (kJ/mol)", format=".2f"),
            alt.Tooltip("relative_final_energy_kj_mol:Q", title="Relative final energy (kJ/mol)", format=".4f"),
            alt.Tooltip("orca_output:N", title="ORCA output"),
        ],
    )
    st.altair_chart(chart.properties(height=260), use_container_width=True)

    display_table = energy_table.copy()
    display_table["final_energy_hartree"] = display_table["final_energy_hartree"].map(_format_energy_hartree)
    display_table["final_energy_kj_mol"] = display_table["final_energy_kj_mol"].map(_format_energy_kj_mol)
    st.dataframe(display_table, use_container_width=True)


def build_relative_energy_comparison_rows(source_df):
    xyz_source = source_df[
        source_df['state'].isin(['solvated', 'dry'])
        & source_df['repo_path'].str.lower().str.endswith('.xyz')
    ].copy()
    if xyz_source.empty:
        return pd.DataFrame()

    representatives = []
    for _, setup_df in xyz_source.groupby(['surface', 'system', 'state', 'role']):
        representatives.append(select_representative_files_by_frame(setup_df))
    if not representatives:
        return pd.DataFrame()

    representative_df = pd.concat(representatives, ignore_index=True)
    rows = []
    for _, file_row in representative_df.iterrows():
        atom_count = get_xyz_atom_count(file_row['repo_path'])
        atom_signature = get_xyz_atom_signature(file_row['repo_path'])
        output_row = find_matching_output(file_row)
        out_path = output_row['repo_path'] if output_row is not None else None
        energy = summarize_energy(get_output_energy_data(out_path))
        classification = file_row.get('classification', 'Unclassified')
        bond_order = file_row.get('Final_Bond_Order')
        try:
            bond_formed = pd.notna(bond_order) and float(bond_order) >= 0.5
        except (TypeError, ValueError):
            bond_formed = False

        rows.append(
            {
                "nucleotide": file_row['system'],
                "surface": file_row['surface'],
                "solvent_state": file_row['state'],
                "role": file_row['role'],
                "atom_count": atom_count,
                "atom_signature": atom_signature,
                "frame": file_row['frame'],
                "file": file_row['file_name'],
                "orca_output": Path(out_path).name if out_path else "not found",
                "classification": classification,
                "bond_formed": "Yes" if bond_formed else "No",
                "sterically_locked": "Yes" if classification == "Sterically Locked" else "No",
                "final_bond_order": bond_order,
                "intramolecular_lock_distance_A": file_row.get('Final_Intramol_Dist'),
                "anchoring_distance_A": file_row.get('Final_Anchoring_Dist'),
                "final_energy_hartree": energy.get("final_energy_hartree"),
                "final_energy_kj_mol": energy.get("final_energy_kj_mol"),
                "energy_record_count": energy.get("energy_record_count"),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    comparable_cols = ['surface', 'nucleotide', 'solvent_state', 'role', 'atom_signature']
    table['comparison_group_size'] = table.groupby(comparable_cols)['file'].transform('count')
    table['valid_energy_count'] = table.groupby(comparable_cols)['final_energy_kj_mol'].transform(
        lambda values: values.notna().sum()
    )
    table['relative_energy_kj_mol'] = pd.NA

    for _, idx in table.dropna(subset=['final_energy_kj_mol']).groupby(comparable_cols).groups.items():
        group_energies = table.loc[idx, 'final_energy_kj_mol']
        table.loc[idx, 'relative_energy_kj_mol'] = group_energies - group_energies.min()

    state_order = {"solvated": 0, "dry": 1}
    table["solvent_state_sort"] = table["solvent_state"].map(state_order).fillna(99)
    return table.sort_values(
        ['nucleotide', 'solvent_state_sort', 'surface', 'role', 'atom_count', 'relative_energy_kj_mol', 'frame', 'file'],
        na_position='last',
    ).drop(columns=["solvent_state_sort"])


def format_relative_energy_export_table(source_df, show_extended_data=False, include_nucleotide=False):
    columns = [
        'solvent_state',
        'surface',
        'frame',
        'file',
        'relative_energy_kj_mol',
        'final_energy_hartree',
        'classification',
    ]
    if include_nucleotide:
        columns.insert(0, 'nucleotide')
    if show_extended_data:
        columns.extend(
            [
                'role',
                'bond_formed',
                'sterically_locked',
                'final_bond_order',
                'intramolecular_lock_distance_A',
                'anchoring_distance_A',
                'atom_count',
                'orca_output',
                'energy_record_count',
                'atom_signature',
            ]
        )

    formatted = source_df[columns].rename(
        columns={
            'solvent_state': 'solvent state',
            'relative_energy_kj_mol': 'relative energy (kJ/mol)',
            'final_energy_hartree': 'final energy (Eh)',
            'bond_formed': 'bond formed',
            'sterically_locked': 'sterically locked',
            'final_bond_order': 'final bond order',
            'intramolecular_lock_distance_A': 'intramol lock dist (A)',
            'anchoring_distance_A': 'anchoring dist (A)',
            'atom_count': 'atom count',
            'orca_output': 'ORCA output',
            'energy_record_count': 'energy records',
            'atom_signature': 'atom composition',
        }
    )
    formatted['relative energy (kJ/mol)'] = formatted['relative energy (kJ/mol)'].map(_format_signed_energy)
    formatted['final energy (Eh)'] = formatted['final energy (Eh)'].map(_format_energy_hartree)
    for col in ['final bond order', 'intramol lock dist (A)', 'anchoring dist (A)']:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda value: _format_metric(value))
    return formatted.reset_index(drop=True)


def build_relative_energy_export_html(source_df, show_extended_data=False):
    export_df = source_df.copy().reset_index(drop=True)
    group_cols = ['nucleotide', 'solvent_state', 'surface', 'role', 'atom_signature']
    group_keys = export_df[group_cols].apply(tuple, axis=1)
    group_palette = [
        "#dbeafe",
        "#fef3c7",
        "#d1fae5",
        "#ede9fe",
        "#ffe4e6",
        "#cffafe",
    ]
    group_colours = {
        key: group_palette[i % len(group_palette)]
        for i, key in enumerate(group_keys.drop_duplicates())
    }
    row_colours = group_keys.map(group_colours).tolist()
    formatted = format_relative_energy_export_table(
        export_df,
        show_extended_data=show_extended_data,
        include_nucleotide=True,
    )

    def mark_comparable_group(row):
        colour = row_colours[row.name]
        return [f"background-color: {colour}" for _ in row]

    styled = formatted.style.apply(mark_comparable_group, axis=1).hide(axis="index")
    table_html = styled.to_html()
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Relative Energy Table</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1 {{ font-size: 20px; margin-bottom: 8px; }}
p {{ max-width: 960px; font-size: 13px; line-height: 1.4; }}
table {{ border-collapse: collapse; font-size: 11px; width: 100%; }}
th, td {{ border: 1px solid #d1d5db; padding: 5px 7px; text-align: left; vertical-align: top; }}
th {{ background: #f3f4f6; font-weight: 700; }}
</style>
</head>
<body>
<h1>Relative Energy Table</h1>
<p>Final ORCA energies are converted to relative energies only within matching setup groups: same nucleotide, surface, solvent state, role, and atom composition. Dry and solvated calculations are never mixed. Colour-coordinated rows indicate directly comparable structures.</p>
{table_html}
</body>
</html>"""


def render_relative_energy_comparison_table(source_df):
    st.header("Relative Energy Table")
    st.caption(
        "Final ORCA energies are converted to relative energies only within matching setup groups: "
        "same nucleotide, surface, solvent state, role, and atom composition. Dry and solvated calculations are never mixed. Directly comparable rows are colourcoordinated in groups, but use the relative energy values and atom signatures to check comparability between groups."
    )

    energy_table = build_relative_energy_comparison_rows(source_df)
    if energy_table.empty:
        st.info("No solvated or dry .xyz representatives were found for the current filters.")
        return

    show_singletons = st.checkbox(
        "Show groups with only one comparable structure",
        value=get_bool_param("show_singletons", False),
    )
    update_bool_param("show_singletons", show_singletons, False)
    show_extended_data = st.checkbox(
        "Extended data/information",
        value=get_bool_param("show_extended_data", False),
    )
    update_bool_param("show_extended_data", show_extended_data, False)
    display_source = energy_table if show_singletons else energy_table[energy_table['valid_energy_count'] >= 2]
    if display_source.empty:
        st.info("No comparable groups with two or more matched final energies were found for the current filters.")
        st.dataframe(energy_table, use_container_width=True)
        return

    group_count = display_source[
        ['surface', 'nucleotide', 'solvent_state', 'role', 'atom_signature']
    ].drop_duplicates().shape[0]
    st.caption(f"Showing {len(display_source)} structures across {group_count} comparable setup groups.")

    for nucleotide in sorted(display_source['nucleotide'].dropna().unique()):
        nucleotide_df = display_source[display_source['nucleotide'] == nucleotide].copy()
        st.subheader(str(nucleotide).upper())
        group_cols = ['solvent_state', 'surface', 'role', 'atom_signature']
        group_keys = nucleotide_df[group_cols].apply(tuple, axis=1)
        group_palette = [
            "rgba(59, 130, 246, 0.12)",
            "rgba(245, 158, 11, 0.12)",
            "rgba(16, 185, 129, 0.12)",
            "rgba(168, 85, 247, 0.12)",
            "rgba(244, 63, 94, 0.12)",
            "rgba(6, 182, 212, 0.12)",
        ]
        group_colours = {
            key: group_palette[i % len(group_palette)]
            for i, key in enumerate(group_keys.drop_duplicates())
        }
        row_colours = group_keys.map(group_colours).tolist()
        formatted = format_relative_energy_export_table(
            nucleotide_df,
            show_extended_data=show_extended_data,
        )

        def mark_comparable_group(row):
            colour = row_colours[row.name]
            return [f"background-color: {colour}" for _ in row]

        formatted = formatted.reset_index(drop=True)
        styled = formatted.style.apply(
            mark_comparable_group,
            axis=1,
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    export_html = build_relative_energy_export_html(
        display_source,
        show_extended_data=show_extended_data,
    )
    st.download_button(
        "Download coloured relative energy table (HTML)",
        data=export_html,
        file_name="relative_energy_table.html",
        mime="text/html",
        use_container_width=True,
    )


def _format_metric(value, unit=""):
    try:
        if pd.isna(value):
            return "n/a"
        suffix = f" {unit}" if unit else ""
        return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def _format_energy_kj_mol(value):
    try:
        if value is None or pd.isna(value) or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):,.0f} kJ/mol"
    except (TypeError, ValueError):
        return "n/a"


def _format_energy_hartree(value):
    try:
        if value is None or pd.isna(value) or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.12f} Eh"
    except (TypeError, ValueError):
        return "n/a"


def _format_signed_energy(value):
    try:
        if value is None or pd.isna(value) or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):+,.2f} kJ/mol"
    except (TypeError, ValueError):
        return "n/a"


def render_interaction_line_chart(
    plot_df,
    x_title,
    y_title,
    legend_title,
    height=280,
    points=False,
    y_zero=False,
    y_tick_count=10,
):
    if plot_df.empty:
        return

    index_name = plot_df.index.name or "index"
    chart_df = (
        plot_df.reset_index()
        .rename(columns={index_name: "x"})
        .melt(id_vars="x", var_name="interaction", value_name="value")
        .dropna(subset=["value"])
    )
    if chart_df.empty:
        return

    line = alt.Chart(chart_df).mark_line(point=points).encode(
        x=alt.X("x:Q", title=x_title),
        y=alt.Y(
            "value:Q",
            title=y_title,
            scale=alt.Scale(zero=y_zero, nice=True),
            axis=alt.Axis(tickCount=y_tick_count, grid=True),
        ),
        color=alt.Color(
            "interaction:N",
            title=legend_title,
            legend=alt.Legend(
                orient="bottom",
                columns=2,
                labelLimit=0,
                symbolLimit=0,
                titleLimit=0,
            ),
        ),
        tooltip=[
            alt.Tooltip("x:Q", title=x_title),
            alt.Tooltip("interaction:N", title=legend_title),
            alt.Tooltip("value:Q", title=y_title, format=".4f"),
        ],
    )
    st.altair_chart(line.properties(height=height), use_container_width=True)


def render_metric_grid(metrics):
    items = "\n".join(
        f"""
        <div class="metric-item">
            <div class="metric-label">{_html_escape(str(label))}</div>
            <div class="metric-value">{_html_escape(str(value))}</div>
        </div>
        """
        for label, value in metrics
    )
    st.markdown(
        f"""
        <style>
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
                gap: 1rem 1.5rem;
                margin: 1rem 0 0.75rem 0;
            }}
            .metric-item {{
                min-width: 0;
            }}
            .metric-label {{
                color: rgba(250, 250, 250, 0.72);
                font-size: 0.92rem;
                font-weight: 650;
                line-height: 1.25;
                overflow-wrap: anywhere;
            }}
            .metric-value {{
                color: rgb(250, 250, 250);
                font-size: clamp(1.55rem, 2.2vw, 2.35rem);
                font-weight: 500;
                line-height: 1.18;
                margin-top: 0.3rem;
                overflow-wrap: anywhere;
                white-space: normal;
            }}
        </style>
        <div class="metric-grid">
            {items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _html_escape(value):
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )

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

if active_tab == "Visualization":
    st.header("File Viewer")

    show_all_files = st.checkbox(
        "Include text/data files",
        value=get_bool_param("show_all", False),
        help="By default, the native file selector only shows .xyz structure and trajectory files.",
    )
    update_bool_param("show_all", show_all_files, False)
    
    def rank_viewer_file(path):
        p = str(path).lower()
        if p.endswith('_trj.xyz'): return 2
        if p.endswith('.xyz'): return 1
        return 0

    xyz_files = filtered_df[filtered_df['repo_path'].str.lower().str.endswith('.xyz')].copy()
    selectable_files = filtered_df.copy() if show_all_files else xyz_files
    selectable_files['rank'] = selectable_files['repo_path'].apply(rank_viewer_file)
    selectable_files = selectable_files.sort_values(['rank', 'repo_path'], ascending=[False, True])

    if not selectable_files.empty:
        def get_viewer_file_idx(opts, param_name="file"):
            val = st.query_params.get(param_name, "")
            if val:
                for i, idx in enumerate(opts):
                    row = selectable_files.loc[idx]
                    if row['repo_path'] == val or row['file_name'] == val:
                        return i
            return 0

        selected_file_row = st.selectbox(
            "Select file to view",
            selectable_files.index,
            index=get_viewer_file_idx(selectable_files.index),
            format_func=lambda x: format_file_option(selectable_files.loc[x]),
            key="viewer_file_select",
        )
        selected_file = selectable_files.loc[selected_file_row]
        update_param("file", selected_file['repo_path'])
        path = selected_file['repo_path']
        resolved_path = resolve_repo_path(path)

        if resolved_path is None:
            st.error("Invalid file path in manifest entry.")
            st.stop()
        
        if path.lower().endswith('.xyz'):
            render_interaction_overview(selected_file)
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
        if show_all_files:
            st.info("No files found for current filters.")
        else:
            st.info("No .xyz files found for current filters. Enable text/data files to inspect other manifest entries.")

if active_tab == "Comparison":
    st.header("Structure Comparison")

    sync_cameras = st.checkbox("Sync Cameras", value=get_bool_param("sync_cameras", True))
    update_bool_param("sync_cameras", sync_cameras, True)
    compare_mode_labels = {
        "same_setup": "Same Setup (Across Frames)",
        "paired": "Paired (Wet vs Dry)",
        "custom": "Custom (Cross-Structure)",
    }
    compare_mode_keys = list(compare_mode_labels.keys())
    compare_mode_default = st.query_params.get("compare_mode", "same_setup")
    if compare_mode_default not in compare_mode_keys:
        compare_mode_default = "same_setup"
    compare_mode = st.selectbox(
        "Comparison Mode",
        compare_mode_keys,
        index=compare_mode_keys.index(compare_mode_default),
        format_func=lambda k: compare_mode_labels[k],
    )
    update_param("compare_mode", compare_mode)

    rank_files = rank_xyz_files

    def get_file_idx(df_subset, param_name):
        val = st.query_params.get(param_name, "")
        if val:
            for i, row in enumerate(df_subset.itertuples()):
                if row.repo_path == val or row.file_name == val:
                    return i
        return 0

    def get_option_idx(df_subset, options, param_name, default_option=None):
        val = st.query_params.get(param_name, "")
        if val:
            for option_idx, row_idx in enumerate(options):
                row = df_subset.iloc[row_idx]
                if row['repo_path'] == val or row['file_name'] == val:
                    return option_idx
        if default_option in options:
            return options.index(default_option)
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
        render_comparison_energy_summary(left_file, right_file, left_label, right_label)
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

    if compare_mode == "same_setup":
        setup_source = frame_unfiltered_df[frame_unfiltered_df['repo_path'].str.lower().str.endswith('.xyz')].copy()
        setup_groups = []
        setup_grouped = setup_source.groupby(['surface', 'system', 'state', 'role'])
        for name, group in setup_grouped:
            frame_files = select_representative_files_by_frame(group)
            if frame_files['frame'].nunique() >= 2:
                setup_groups.append(name)

        def get_setup_idx(opts, param_name="setup"):
            val = st.query_params.get(param_name, "")
            if val:
                for i, opt in enumerate(opts):
                    if f"{opt[0]}_{opt[1]}_{opt[2]}_{opt[3]}" == val:
                        return i
            return 0

        if setup_groups:
            selected_setup = st.selectbox(
                "Select Setup to Compare",
                setup_groups,
                index=get_setup_idx(setup_groups),
                format_func=lambda x: f"{x[1]} | {x[0]} | {format_state_option(x[2])} ({x[3]})",
                key="same_setup_select",
            )
            update_param("setup", f"{selected_setup[0]}_{selected_setup[1]}_{selected_setup[2]}_{selected_setup[3]}")

            setup_df = setup_grouped.get_group(selected_setup)
            frame_files = select_representative_files_by_frame(setup_df).reset_index(drop=True)
            st.subheader("Comparable Frame Energies")
            render_same_setup_energy_comparison(frame_files)

            left_idx = st.selectbox(
                "Select Left Frame",
                range(len(frame_files)),
                index=get_file_idx(frame_files, "setup_left_file"),
                format_func=lambda i: format_file_option(frame_files.iloc[i]),
                key="setup_left_select",
            )
            right_options = [i for i in range(len(frame_files)) if i != left_idx]
            if not right_options:
                st.info("At least two distinct frames are needed for comparison.")
                st.stop()
            default_right_idx = right_options[0]
            right_idx = st.selectbox(
                "Select Right Frame",
                right_options,
                index=get_option_idx(
                    frame_files,
                    right_options,
                    "setup_right_file",
                    default_option=default_right_idx,
                ),
                format_func=lambda i: format_file_option(frame_files.iloc[i]),
                key="setup_right_select",
            )

            left_file = frame_files.iloc[left_idx]
            right_file = frame_files.iloc[right_idx]
            update_param("setup_left_file", left_file['repo_path'])
            update_param("setup_right_file", right_file['repo_path'])

            left_label = str(left_file['frame']) or left_file['file_name']
            right_label = str(right_file['frame']) or right_file['file_name']
            if sync_cameras:
                render_comparison_pair(left_file, right_file, left_label, right_label)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Left: {left_label}")
                    render_xyz(left_file['repo_path'], height=400, fast_mode=performance_mode, start_frame=default_trj_frame)
                with col2:
                    st.subheader(f"Right: {right_label}")
                    render_xyz(right_file['repo_path'], height=400, fast_mode=performance_mode, start_frame=default_trj_frame)
        else:
            st.info("No same-setup groups with two or more .xyz frames were found for current filters.")

    elif compare_mode == "paired":
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
                update_param("sol_file", solvated_file['repo_path'])
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
                update_param("dry_file", dry_file['repo_path'])
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
        elif len(custom_candidates) < 2:
            st.info("At least two distinct .xyz files are needed for custom comparison with current filters.")
        else:
            left_idx = st.selectbox(
                "Select Left File",
                range(len(custom_candidates)),
                index=get_file_idx(custom_candidates, "left_file"),
                format_func=lambda i: format_file_option(custom_candidates.iloc[i]),
                key="custom_left_select",
            )
            right_options = [i for i in range(len(custom_candidates)) if i != left_idx]
            right_idx = st.selectbox(
                "Select Right File",
                right_options,
                index=get_option_idx(custom_candidates, right_options, "right_file"),
                format_func=lambda i: format_file_option(custom_candidates.iloc[i]),
                key="custom_right_select",
            )

            left_file = custom_candidates.iloc[left_idx]
            right_file = custom_candidates.iloc[right_idx]
            update_param("left_file", left_file['repo_path'])
            update_param("right_file", right_file['repo_path'])

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

if active_tab == "Relative Energies":
    render_relative_energy_comparison_table(frame_unfiltered_df)

if active_tab == "Data Table":
    st.header("Filtered Data")
    safe_df = filtered_df.drop(columns=["source_path"], errors="ignore")
    st.dataframe(safe_df)

st.sidebar.markdown("---")
st.sidebar.info("RNA Silicate Interactions Streamlit App")

# Reorder and cleanup URL query parameters
current_params = st.query_params.to_dict()
next_params = current_params.copy()
active_tab = next_params.get("tab", "Visualization")

# Clean up tab-specific parameters so they don't persist incorrectly
if active_tab == "Visualization":
    for p in [
        "compare_mode", "compare", "sol_file", "dry_file", "left_file", "right_file",
        "setup", "setup_left_file", "setup_right_file", "sync_cameras", "show_singletons",
    ]:
        next_params.pop(p, None)
elif active_tab == "Comparison":
    next_params.pop("file", None)
    next_params.pop("show_all", None)
    next_params.pop("show_singletons", None)
    compare_mode_value = next_params.get("compare_mode", "same_setup")
    if compare_mode_value == "paired":
        for p in ["left_file", "right_file", "setup", "setup_left_file", "setup_right_file"]:
            next_params.pop(p, None)
    elif compare_mode_value == "same_setup":
        for p in ["compare", "sol_file", "dry_file", "left_file", "right_file"]:
            next_params.pop(p, None)
    else:
        for p in ["compare", "sol_file", "dry_file", "setup", "setup_left_file", "setup_right_file"]:
            next_params.pop(p, None)
elif active_tab == "Relative Energies":
    for p in [
        "compare_mode", "compare", "sol_file", "dry_file", "left_file", "right_file",
        "setup", "setup_left_file", "setup_right_file", "file", "show_all", "sync_cameras",
    ]:
        next_params.pop(p, None)
elif active_tab == "Data Table":
    for p in [
        "compare_mode", "compare", "sol_file", "dry_file", "left_file", "right_file",
        "setup", "setup_left_file", "setup_right_file", "file", "show_all", "sync_cameras", "show_singletons",
    ]:
        next_params.pop(p, None)

if next_params != current_params:
    st.query_params.clear()
    for k, v in next_params.items():
        st.query_params[k] = v
