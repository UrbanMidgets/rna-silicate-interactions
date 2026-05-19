import streamlit as st
import pandas as pd
import py3Dmol
import os
import base64
from pathlib import Path

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

# Dynamic filtering logic for sidebar
filtered_df = df.copy()

surface = st.sidebar.selectbox("Surface", get_options(df, 'surface'))
if surface != "All":
    filtered_df = filtered_df[filtered_df['surface'] == surface]

system = st.sidebar.selectbox("System", get_options(filtered_df, 'system'))
if system != "All":
    filtered_df = filtered_df[filtered_df['system'] == system]

role = st.sidebar.selectbox("Role", get_options(filtered_df, 'role'))
if role != "All":
    filtered_df = filtered_df[filtered_df['role'] == role]

frame = st.sidebar.selectbox("Frame", get_options(filtered_df, 'frame'))
if frame != "All":
    filtered_df = filtered_df[filtered_df['frame'] == frame]

st.sidebar.header("Settings")
show_surface = st.sidebar.checkbox("Show Surface (Si, Al, O)", value=True)
spin = st.sidebar.checkbox("Spin Molecule", value=False)
performance_mode = st.sidebar.checkbox("Performance Mode", value=True)
trajectory_stride = st.sidebar.slider("Trajectory Frame Step", 1, 10, 2)

st.sidebar.markdown(f"**Matches:** {len(filtered_df)}")

# Grouping by system/surface/frame for comparison
# We want to identify pairs of (solvated, dry) for the same frame
comparison_options = filtered_df[filtered_df['state'].isin(['solvated', 'dry'])]
groups = comparison_options.groupby(['surface', 'system', 'frame', 'role'])

# Main content area
tabs = st.tabs(["Visualization", "Comparison", "Data Table"])

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

def render_xyz(xyz_path, title=None, height=600, width=1000, frame_idx=None, fast_mode=False):
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
        if frame_idx is not None:
            view.setFrame(frame_idx)
        else:
            view.animate({'loop': 'forward', 'rebuild': True})
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
    h_line_width = 0.35 if fast_mode else 0.5
    view.setStyle({'elem': "H"}, {'line': {'linewidth': h_line_width}})
    
    if spin:
        view.spin(True)
    
    view.zoomTo() # Always zoom to ensure visibility in new iframe
    if title:
        st.subheader(title)
    html = view._make_html()
    encoded_html = base64.b64encode(html.encode("utf-8")).decode("ascii")
    st.iframe(f"data:text/html;base64,{encoded_html}", height=height, width=width)

with tabs[0]:
    st.header("File Viewer")
    
    # Select specific file to view
    selectable_files = filtered_df.copy()
    if not selectable_files.empty:
        selected_file_row = st.selectbox(
            "Select file to view",
            selectable_files.index,
            format_func=lambda x: f"{selectable_files.loc[x, 'repo_path']} ({selectable_files.loc[x, 'state']})"
        )
        selected_file = selectable_files.loc[selected_file_row]
        path = selected_file['repo_path']
        resolved_path = resolve_repo_path(path)

        if resolved_path is None:
            st.error("Invalid file path in manifest entry.")
            st.stop()
        
        if path.endswith('.xyz'):
            is_trj = path.lower().endswith("_trj.xyz")
            frame_to_show = None
            if is_trj:
                n_frames = get_frame_count(path)
                if n_frames > 1:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        frame_to_show = st.slider(
                            "Frame Selector",
                            0,
                            n_frames - 1,
                            0,
                            step=trajectory_stride,
                            key="viewer_frame_slider",
                        )
                    with col2:
                        st.metric("Current Frame", f"{frame_to_show + 1} / {n_frames}")
                    if st.checkbox("Auto Animate", value=False, key="viewer_animate"):
                        frame_to_show = None # This triggers animation in render_xyz
            
            render_xyz(
                path,
                f"Visualizing: {selected_file['file_name']}",
                frame_idx=frame_to_show,
                fast_mode=performance_mode,
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

with tabs[1]:
    st.header("Wet vs Dry Comparison")
    
    # Find frames that have both solvated and dry states
    comp_groups = []
    for name, group in groups:
        states = group['state'].unique()
        # Only include if both solvated and dry states exist AND have xyz files
        if 'solvated' in states and 'dry' in states:
            solvated_xyz = group[(group['state'] == 'solvated') & (group['repo_path'].str.endswith('.xyz'))]
            dry_xyz = group[(group['state'] == 'dry') & (group['repo_path'].str.endswith('.xyz'))]
            if not solvated_xyz.empty and not dry_xyz.empty:
                comp_groups.append(name)
    
    if comp_groups:
        sync_cameras = st.checkbox("Sync Cameras", value=True)
        
        selected_comp = st.selectbox(
            "Select Frame to Compare",
            comp_groups,
            format_func=lambda x: f"{x[0]} | {x[1]} | {x[2]} ({x[3]})"
        )
        
        group_df = groups.get_group(selected_comp)
        
        solvated_files = group_df[(group_df['state'] == 'solvated') & (group_df['repo_path'].str.endswith('.xyz'))].copy()
        dry_files = group_df[(group_df['state'] == 'dry') & (group_df['repo_path'].str.endswith('.xyz'))].copy()

        def rank_files(df_subset, state_label):
            if df_subset.empty:
                return df_subset
            def get_rank(row):
                score = 0
                if row['repo_path'].lower().endswith('_trj.xyz'):
                    score += 10
                if state_label.lower() in row['file_name'].lower():
                    score += 5
                # Prefer files that are in a 'dry' or 'solvated' subfolder if applicable
                if f"/{state_label.lower()}/" in row['repo_path'].lower():
                    score += 2
                return score
            df_subset['rank'] = df_subset.apply(get_rank, axis=1)
            return df_subset.sort_values('rank', ascending=False)

        solvated_files = rank_files(solvated_files, 'solvated')
        dry_files = rank_files(dry_files, 'dry')

        if len(solvated_files) > 1:
            sol_idx = st.selectbox("Select Solvated File", solvated_files.index, format_func=lambda x: solvated_files.loc[x, 'file_name'], key="sol_select")
            solvated_file = solvated_files.loc[sol_idx]
        else:
            solvated_file = solvated_files.iloc[0]

        if len(dry_files) > 1:
            dry_idx = st.selectbox("Select Dry File", dry_files.index, format_func=lambda x: dry_files.loc[x, 'file_name'], key="dry_select")
            dry_file = dry_files.loc[dry_idx]
        else:
            dry_file = dry_files.iloc[0]

        if sync_cameras:
            # Combined grid view
            sol_data = get_xyz_data(solvated_file['repo_path'])
            dry_data = get_xyz_data(dry_file['repo_path'])

            if sol_data is None or dry_data is None:
                st.error("Could not load comparison files.")
                st.stop()

            is_sol_trj = solvated_file['repo_path'].lower().endswith("_trj.xyz")
            is_dry_trj = dry_file['repo_path'].lower().endswith("_trj.xyz")
            
            sol_frame = None
            dry_frame = None
            
            if is_sol_trj or is_dry_trj:
                st.subheader("Trajectory Controls")
                n_sol = get_frame_count(solvated_file['repo_path']) if is_sol_trj else 1
                n_dry = get_frame_count(dry_file['repo_path']) if is_dry_trj else 1
                max_frames = max(n_sol, n_dry)
                
                # Single global slider for PyMOL-like behavior
                global_frame = st.slider(
                    "Global Frame Selector",
                    0,
                    max_frames - 1,
                    0,
                    step=trajectory_stride,
                )
                
                # Logic: stop at last frame if shorter than current global_frame
                sol_frame = min(global_frame, n_sol - 1) if is_sol_trj else 0
                dry_frame = min(global_frame, n_dry - 1) if is_dry_trj else 0
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Solvated Frame", f"{sol_frame + 1} / {n_sol}")
                with col_m2:
                    st.metric("Dry Frame", f"{dry_frame + 1} / {n_dry}")
                
                if st.checkbox("Auto Animate", value=False):
                    sol_frame = None
                    dry_frame = None

            # Reverting to the version the user liked
            viewer_width = 1200
            viewer_height = 600
            view = py3Dmol.view(width=viewer_width, height=viewer_height, viewergrid=(1,2), linked=True)
            view.setBackgroundColor('white')
            
            # Helper to apply styles to a specific viewer in the grid
            def apply_comparison_style(v, model_data, viewer_idx, is_trj=False, frame_idx=None):
                if is_trj:
                    v.addModelsAsFrames(model_data, "xyz", viewer=viewer_idx)
                    if frame_idx is not None:
                        v.setFrame(frame_idx, viewer=viewer_idx)
                    else:
                        v.animate({'loop': 'forward', 'rebuild': True}, viewer=viewer_idx)
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
                h_line_width = 0.35 if performance_mode else 0.5
                v.setStyle({'elem': "H"}, {'line': {'linewidth': h_line_width}}, viewer=viewer_idx)
                
                # We MUST call zoomTo every time because a new iframe is generated by Streamlit
                # If we don't, the molecule might be off-screen.
                v.zoomTo(viewer=viewer_idx)

            apply_comparison_style(view, sol_data, (0,0), is_trj=is_sol_trj, frame_idx=sol_frame)
            apply_comparison_style(view, dry_data, (0,1), is_trj=is_dry_trj, frame_idx=dry_frame)
            
            st.subheader(f"Left: Solvated | Right: Dry")
            html = view._make_html()
            encoded_html = base64.b64encode(html.encode("utf-8")).decode("ascii")
            st.iframe(
                f"data:text/html;base64,{encoded_html}",
                height=viewer_height,
                width=viewer_width,
            )
        else:
            # Separate columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Solvated (Wet)")
                is_sol_trj = solvated_file['repo_path'].lower().endswith("_trj.xyz")
                sol_frame = None
                if is_sol_trj:
                    n_sol = get_frame_count(solvated_file['repo_path'])
                    sol_frame = st.slider("Frame", 0, n_sol - 1, 0, key="sol_sep_slider")
                    st.write(f"Frame: {sol_frame + 1} / {n_sol}")
                    if st.checkbox("Animate", value=False, key="sol_sep_anim"):
                        sol_frame = None
                render_xyz(solvated_file['repo_path'], height=400, frame_idx=sol_frame, fast_mode=performance_mode)
            with col2:
                st.subheader("Dry")
                is_dry_trj = dry_file['repo_path'].lower().endswith("_trj.xyz")
                dry_frame = None
                if is_dry_trj:
                    n_dry = get_frame_count(dry_file['repo_path'])
                    dry_frame = st.slider("Frame", 0, n_dry - 1, 0, key="dry_sep_slider")
                    st.write(f"Frame: {dry_frame + 1} / {n_dry}")
                    if st.checkbox("Animate", value=False, key="dry_sep_anim"):
                        dry_frame = None
                render_xyz(dry_file['repo_path'], height=400, frame_idx=dry_frame, fast_mode=performance_mode)
    else:
        st.info("No frames found with both solvated and dry .xyz files for current filters.")

with tabs[2]:
    st.header("Filtered Data")
    safe_df = filtered_df.drop(columns=["source_path"], errors="ignore")
    st.dataframe(safe_df)

st.sidebar.markdown("---")
st.sidebar.info("RNA Silicate Interactions Streamlit App")
