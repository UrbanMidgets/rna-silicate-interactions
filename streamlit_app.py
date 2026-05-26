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
tabs = st.tabs(tab_options, default=default_tab, key="active_tab", on_change=update_tab)

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
    
    selectable_files = filtered_df.copy()
    
    show_all_files = st.checkbox("Include text/data files", value=False, help="By default, only 3D visualization files (.xyz) are shown.")
    
    def rank_viewer_file(path):
        p = str(path).lower()
        if p.endswith('_trj.xyz'): return 2
        if p.endswith('.xyz'): return 1
        return 0
        
    selectable_files['rank'] = selectable_files['repo_path'].apply(rank_viewer_file)
    
    if not show_all_files:
        selectable_files = selectable_files[selectable_files['rank'] > 0]
        
    selectable_files = selectable_files.sort_values(['rank', 'repo_path'], ascending=[False, True])

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
        
        def get_comp_idx(opts, param_name="compare"):
            val = st.query_params.get(param_name, "")
            if val:
                for i, opt in enumerate(opts):
                    if f"{opt[0]}_{opt[1]}_{opt[2]}_{opt[3]}" == val:
                        return i
            return 0

        selected_comp = st.selectbox(
            "Select Frame to Compare",
            comp_groups,
            index=get_comp_idx(comp_groups),
            format_func=lambda x: f"{x[0]} | {x[1]} | {x[2]} ({x[3]})"
        )
        
        update_param("compare", f"{selected_comp[0]}_{selected_comp[1]}_{selected_comp[2]}_{selected_comp[3]}")
        
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

        def get_file_idx(df_subset, param_name):
            val = st.query_params.get(param_name, "")
            if val:
                for i, row in enumerate(df_subset.itertuples()):
                    if row.file_name == val:
                        return i
            return 0

        if len(solvated_files) > 1:
            sol_idx_pos = st.selectbox("Select Solvated File", range(len(solvated_files)), index=get_file_idx(solvated_files, "sol_file"), format_func=lambda i: solvated_files.iloc[i]['file_name'], key="sol_select")
            solvated_file = solvated_files.iloc[sol_idx_pos]
            update_param("sol_file", solvated_file['file_name'])
        else:
            solvated_file = solvated_files.iloc[0]
            update_param("sol_file", "All")

        if len(dry_files) > 1:
            dry_idx_pos = st.selectbox("Select Dry File", range(len(dry_files)), index=get_file_idx(dry_files, "dry_file"), format_func=lambda i: dry_files.iloc[i]['file_name'], key="dry_select")
            dry_file = dry_files.iloc[dry_idx_pos]
            update_param("dry_file", dry_file['file_name'])
        else:
            dry_file = dry_files.iloc[0]
            update_param("dry_file", "All")

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
            
            n_sol = get_frame_count(solvated_file['repo_path']) if is_sol_trj else 1
            n_dry = get_frame_count(dry_file['repo_path']) if is_dry_trj else 1

            # Reverting to the version the user liked
            viewer_width = 1200
            viewer_height = 600
            view = py3Dmol.view(width=viewer_width, height=viewer_height, viewergrid=(1,2), linked=True)
            view.setBackgroundColor('white')
            
            # Helper to apply styles to a specific viewer in the grid
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
                
                # We MUST call zoomTo every time because a new iframe is generated by Streamlit
                # If we don't, the molecule might be off-screen.
                v.zoomTo(viewer=viewer_idx)

            apply_comparison_style(view, sol_data, (0,0), is_trj=is_sol_trj, start_frame=default_trj_frame)
            apply_comparison_style(view, dry_data, (0,1), is_trj=is_dry_trj, start_frame=default_trj_frame)
            
            st.subheader(f"Left: Solvated | Right: Dry")
            html = view._make_html()
            
            if is_sol_trj or is_dry_trj:
                max_frames = max(n_sol, n_dry)
                if max_frames > 1:
                    html = inject_viewer_ui(html, max_frames, trajectory_stride, is_grid=True, n_sol=n_sol, n_dry=n_dry, start_frame=default_trj_frame)
                    viewer_height += 60
            
            html = inject_visibility_fix(html)
            encoded_html = base64.b64encode(html.encode("utf-8")).decode("ascii")
            st.components.v1.iframe(
                f"data:text/html;base64,{encoded_html}",
                height=viewer_height,
                width=viewer_width,
            )
        else:
            # Separate columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Solvated (Wet)")
                render_xyz(solvated_file['repo_path'], height=400, fast_mode=performance_mode, start_frame=default_trj_frame)
            with col2:
                st.subheader("Dry")
                render_xyz(dry_file['repo_path'], height=400, fast_mode=performance_mode, start_frame=default_trj_frame)
    else:
        st.info("No frames found with both solvated and dry .xyz files for current filters.")

with tabs[2]:
    st.header("Filtered Data")
    safe_df = filtered_df.drop(columns=["source_path"], errors="ignore")
    st.dataframe(safe_df)

st.sidebar.markdown("---")
st.sidebar.info("RNA Silicate Interactions Streamlit App")
