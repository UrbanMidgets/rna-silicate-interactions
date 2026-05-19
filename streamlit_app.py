import streamlit as st
import pandas as pd
import py3Dmol
from stmol import showmol
import os
import streamlit.components.v1 as components
import json

# Set page config
st.set_page_config(page_title="RNA Silicate Interactions", layout="wide")

st.title("RNA Silicate Interactions")
st.markdown("Supporting data for RNA nucleotide interactions with silicate surfaces.")

@st.cache_data
def load_manifest():
    if os.path.exists("data/MANIFEST.tsv"):
        df = pd.read_csv("data/MANIFEST.tsv", sep="\t").fillna("")
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

st.sidebar.markdown(f"**Matches:** {len(filtered_df)}")

# Grouping by system/surface/frame for comparison
# We want to identify pairs of (solvated, dry) for the same frame
comparison_options = filtered_df[filtered_df['state'].isin(['solvated', 'dry'])]
groups = comparison_options.groupby(['surface', 'system', 'frame', 'role'])

# Main content area
tabs = st.tabs(["Visualization", "Comparison", "Data Table"])

def get_frame_count(xyz_path):
    if not os.path.exists(xyz_path):
        return 0
    try:
        with open(xyz_path, "r") as f:
            first_line = f.readline().strip()
            if not first_line.isdigit():
                return 1
            num_atoms = int(first_line)
            f.seek(0)
            content = f.read()
            return content.count(f"\n{num_atoms}\n") + (1 if content.startswith(str(num_atoms)) else 0)
    except:
        return 1

@st.cache_data
def get_xyz_data(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read()

def render_xyz_interactive(xyz_path, title=None, height=600, width=1000):
    xyz_data = get_xyz_data(xyz_path)
    if xyz_data is None:
        st.error(f"File not found: {xyz_path}")
        return
    
    is_trajectory = xyz_path.lower().endswith("_trj.xyz")
    xyz_json = json.dumps(xyz_data)
    
    html_code = f"""
    <div id="container" style="width: {width}px; height: {height}px; position: relative; background-color: white; border: 1px solid #ddd;"></div>
    <div id="controls" style="width: {width}px; padding: 10px; font-family: sans-serif; display: {'block' if is_trajectory else 'none'};">
        <input type="range" id="frame-slider" style="width: 70%;" min="0" value="0">
        <span id="frame-label" style="margin-left: 10px; font-weight: bold;">Frame: 1 / 1</span>
        <button id="play-btn" style="margin-left: 15px; padding: 5px 15px; cursor: pointer;">Play</button>
        <div id="error-log" style="color: red; font-size: 12px; margin-top: 5px;"></div>
    </div>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
        (function() {{
            function init() {{
                try {{
                    var element = document.getElementById('container');
                    var viewer = $3Dmol.createViewer(element, {{ backgroundColor: 'white' }});
                    var xyzData = {xyz_json};
                    
                    if ({'true' if is_trajectory else 'false'}) {{
                        viewer.addModelsAsFrames(xyzData, "xyz");
                    }} else {{
                        viewer.addModel(xyzData, "xyz");
                    }}
                    
                    viewer.setStyle({{elem: ["C", "N", "P"]}}, {{stick: {{radius: 0.18}}, sphere: {{scale: 0.25}}}});
                    if ({'true' if show_surface else 'false'}) {{
                        viewer.setStyle({{elem: ["Si", "Al", "O"]}}, {{stick: {{radius: 0.12, opacity: 0.9}}}});
                    }} else {{
                        viewer.setStyle({{elem: ["Si", "Al", "O"]}}, {{sphere: {{radius: 0.01, opacity: 0}}}});
                    }}
                    viewer.setStyle({{elem: "H"}}, {{line: {{linewidth: 0.5}}}});
                    
                    viewer.zoomTo();
                    viewer.render();
                    
                    if ({'true' if spin else 'false'}) {{
                        viewer.spin(true);
                    }}
                    
                    if ({'true' if is_trajectory else 'false'}) {{
                        var slider = document.getElementById("frame-slider");
                        var label = document.getElementById("frame-label");
                        var playBtn = document.getElementById("play-btn");
                        var nFrames = viewer.getFrameCount();
                        
                        slider.max = nFrames - 1;
                        label.innerText = "Frame: 1 / " + nFrames;
                        
                        slider.oninput = function() {{
                            viewer.setFrame(parseInt(this.value));
                            label.innerText = "Frame: " + (parseInt(this.value) + 1) + " / " + nFrames;
                            viewer.render();
                        }};
                        
                        var animating = false;
                        var interval;
                        playBtn.onclick = function() {{
                            if (animating) {{
                                clearInterval(interval);
                                playBtn.innerText = "Play";
                            }} else {{
                                interval = setInterval(function() {{
                                    var next = (viewer.getFrame() + 1) % nFrames;
                                    viewer.setFrame(next);
                                    slider.value = next;
                                    label.innerText = "Frame: " + (next + 1) + " / " + nFrames;
                                    viewer.render();
                                }}, 100);
                                playBtn.innerText = "Pause";
                            }}
                            animating = !animating;
                        }};
                    }}
                }} catch (e) {{
                    document.getElementById('error-log').innerText = "Error: " + e.message;
                }}
            }}
            
            if (window.$3Dmol) init();
            else {{
                var check = setInterval(function() {{
                    if (window.$3Dmol) {{ clearInterval(check); init(); }}
                }}, 100);
            }}
        }})();
    </script>
    """
    if title:
        st.subheader(title)
    components.html(html_code, width=width + 20, height=height + 150)

def render_comparison_interactive(sol_path, dry_path, width=1200, height=600):
    sol_data = get_xyz_data(sol_path)
    dry_data = get_xyz_data(dry_path)
    
    if sol_data is None or dry_data is None:
        st.error("Could not load comparison files.")
        return

    is_sol_trj = sol_path.lower().endswith("_trj.xyz")
    is_dry_trj = dry_path.lower().endswith("_trj.xyz")
    
    sol_json = json.dumps(sol_data)
    dry_json = json.dumps(dry_data)
    
    is_any_trj = is_sol_trj or is_dry_trj
    
    html_code = f"""
    <div id="container" style="width: {width}px; height: {height}px; position: relative; background-color: white; border: 1px solid #ddd;"></div>
    <div id="controls" style="width: {width}px; padding: 10px; font-family: sans-serif; display: {'block' if is_any_trj else 'none'};">
        <input type="range" id="frame-slider" style="width: 70%;" min="0" value="0">
        <span id="frame-label" style="margin-left: 10px; font-weight: bold;">Frame: 1 / 1</span>
        <button id="play-btn" style="margin-left: 15px; padding: 5px 15px; cursor: pointer;">Play</button>
        <div id="error-log" style="color: red; font-size: 12px; margin-top: 5px;"></div>
    </div>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
        (function() {{
            function init() {{
                try {
                    var element = document.getElementById('container');
                    var viewer = $3Dmol.createViewer(element, {
                        rows: 1,
                        cols: 2,
                        control_all: true,
                        linked: true,
                        backgroundColor: 'white'
                    });
                    
                    var solData = {sol_json};
                    var dryData = {dry_json};
                    
                    if ({'true' if is_sol_trj else 'false'}) {
                        viewer.addModelsAsFrames(solData, "xyz", {viewer: [0, 0]});
                    } else {
                        viewer.addModel(solData, "xyz", {viewer: [0, 0]});
                    }
                    
                    if ({'true' if is_dry_trj else 'false'}) {
                        viewer.addModelsAsFrames(dryData, "xyz", {viewer: [0, 1]});
                    } else {
                        viewer.addModel(dryData, "xyz", {viewer: [0, 1]});
                    }
                    
                    function applyStyles(v, idx) {
                        v.setStyle({elem: ["C", "N", "P"]}, {stick: {radius: 0.18}, sphere: {scale: 0.25}}, {viewer: idx});
                        if ({'true' if show_surface else 'false'}) {
                            v.setStyle({elem: ["Si", "Al", "O"]}, {stick: {radius: 0.12, opacity: 0.9}}, {viewer: idx});
                        } else {
                            v.setStyle({elem: ["Si", "Al", "O"]}, {sphere: {radius: 0.01, opacity: 0}}, {viewer: idx});
                        }
                        v.setStyle({elem: "H"}, {line: {linewidth: 0.5}}, {viewer: idx});
                        v.zoomTo({viewer: idx});
                    }
                    
                    applyStyles(viewer, [0, 0]);
                    applyStyles(viewer, [0, 1]);
                    
                    viewer.render();
                    
                    if ({'true' if spin else 'false'}) {{
                        viewer.spin(true);
                    }}
                    
                    if ({'true' if is_any_trj else 'false'}) {{
                        var slider = document.getElementById("frame-slider");
                        var label = document.getElementById("frame-label");
                        var playBtn = document.getElementById("play-btn");
                        
                        var nSol = {'viewer.getFrameCount({viewer: [0, 0]})' if is_sol_trj else '1'};
                        var nDry = {'viewer.getFrameCount({viewer: [0, 1]})' if is_dry_trj else '1'};
                        var nMax = Math.max(nSol, nDry);
                        
                        slider.max = nMax - 1;
                        label.innerText = "Frame: 1 / " + nMax;
                        
                        function setGlobalFrame(frame) {{
                            if ({'true' if is_sol_trj else 'false'}) {{
                                viewer.setFrame(Math.min(frame, nSol - 1), {{viewer: [0, 0]}});
                            }}
                            if ({'true' if is_dry_trj else 'false'}) {{
                                viewer.setFrame(Math.min(frame, nDry - 1), {{viewer: [0, 1]}});
                            }}
                            label.innerText = "Frame: " + (frame + 1) + " / " + nMax;
                            viewer.render();
                        }}
                        
                        slider.oninput = function() {{
                            setGlobalFrame(parseInt(this.value));
                        }};
                        
                        var animating = false;
                        var interval;
                        var currentFrame = 0;
                        playBtn.onclick = function() {{
                            if (animating) {{
                                clearInterval(interval);
                                playBtn.innerText = "Play";
                            }} else {{
                                interval = setInterval(function() {{
                                    currentFrame = (currentFrame + 1) % nMax;
                                    setGlobalFrame(currentFrame);
                                    slider.value = currentFrame;
                                }}, 100);
                                playBtn.innerText = "Pause";
                            }}
                            animating = !animating;
                        }};
                    }}
                }} catch (e) {{
                    document.getElementById('error-log').innerText = "Error: " + e.message;
                }}
            }}
            
            if (window.$3Dmol) init();
            else {{
                var check = setInterval(function() {{
                    if (window.$3Dmol) {{ clearInterval(check); init(); }}
                }}, 100);
            }}
        }})();
    </script>
    """
    st.subheader("Left: Solvated | Right: Dry")
    components.html(html_code, width=width + 20, height=height + 150)

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
        
        if path.endswith('.xyz'):
            render_xyz_interactive(path, f"Visualizing: {selected_file['file_name']}")
        else:
            st.subheader(f"Viewing: {selected_file['file_name']}")
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read()
                st.code(content, language="text")
            else:
                st.error(f"File not found: {path}")
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
        
        solvated_files = group_df[(group_df['state'] == 'solvated') & (group_df['repo_path'].str.endswith('.xyz'))]
        dry_files = group_df[(group_df['state'] == 'dry') & (group_df['repo_path'].str.endswith('.xyz'))]

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
            render_comparison_interactive(solvated_file['repo_path'], dry_file['repo_path'])
        else:
            # Separate columns
            col1, col2 = st.columns(2)
            with col1:
                render_xyz_interactive(solvated_file['repo_path'], title="Solvated (Wet)", height=400, width=500)
            with col2:
                render_xyz_interactive(dry_file['repo_path'], title="Dry", height=400, width=500)
    else:
        st.info("No frames found with both solvated and dry .xyz files for current filters.")

with tabs[2]:
    st.header("Filtered Data")
    st.dataframe(filtered_df)

st.sidebar.markdown("---")
st.sidebar.info("RNA Silicate Interactions Streamlit App")
