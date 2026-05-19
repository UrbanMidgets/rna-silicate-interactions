import streamlit as st
import pandas as pd
import py3Dmol
from stmol import showmol
import os

# Set page config
st.set_page_config(page_title="RNA Silicate Interactions", layout="wide")

st.title("RNA Silicate Interactions")
st.markdown("Supporting data for RNA nucleotide interactions with silicate surfaces.")

@st.cache_data
def load_manifest():
    if os.path.exists("data/MANIFEST.tsv"):
        df = pd.read_csv("data/MANIFEST.tsv", sep="\t").fillna("")
        # Derive file_name from repo_path
        df['file_name'] = df['repo_path'].apply(lambda x: os.path.basename(x))
        return df
    return pd.DataFrame()

df = load_manifest()

if df.empty:
    st.error("Manifest not found. Please ensure data/MANIFEST.tsv exists.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

def get_options(column):
    opts = sorted([str(x) for x in df[column].unique() if x])
    return ["All"] + opts

surface = st.sidebar.selectbox("Surface", get_options('surface'))
system = st.sidebar.selectbox("System", get_options('system'))
role = st.sidebar.selectbox("Role", get_options('role'))
frame = st.sidebar.selectbox("Frame", get_options('frame'))

st.sidebar.header("Settings")
show_surface = st.sidebar.checkbox("Show Surface (Si, Al, O)", value=True)
spin = st.sidebar.checkbox("Spin Molecule", value=False)

# Filter the dataframe
filtered_df = df.copy()
if surface != "All":
    filtered_df = filtered_df[filtered_df['surface'] == surface]
if system != "All":
    filtered_df = filtered_df[filtered_df['system'] == system]
if role != "All":
    filtered_df = filtered_df[filtered_df['role'] == role]
if frame != "All":
    filtered_df = filtered_df[filtered_df['frame'] == frame]

st.sidebar.markdown(f"**Matches:** {len(filtered_df)}")

# Grouping by system/surface/frame for comparison
# We want to identify pairs of (solvated, dry) for the same frame
comparison_options = filtered_df[filtered_df['state'].isin(['solvated', 'dry'])]
groups = comparison_options.groupby(['surface', 'system', 'frame', 'role'])

# Main content area
tabs = st.tabs(["Visualization", "Comparison", "Data Table"])

def render_xyz(xyz_path, title=None, height=600, width=1000):
    if not os.path.exists(xyz_path):
        st.error(f"File not found: {xyz_path}")
        return

    with open(xyz_path, "r") as f:
        xyz_data = f.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_data, "xyz")

    # Styling
    # Nucleotide: Stick + Sphere
    view.setStyle({'elem': ["C", "N", "P"]}, {'stick': {'radius': 0.18}, 'sphere': {'scale': 0.25}})

    # Surface: Smaller sticks
    if show_surface:
        view.setStyle({'elem': ["Si", "Al", "O"]}, {'stick': {'radius': 0.12, 'opacity': 0.9}})
    else:
        view.setStyle({'elem': ["Si", "Al", "O"]}, {'sphere': {'radius': 0.01, 'opacity': 0}}) # Hide

    # Hydrogens: Line
    view.setStyle({'elem': "H"}, {'line': {'linewidth': 0.5}})

    if spin:
        view.spin(True)

    view.zoomTo()
    if title:
        st.subheader(title)
    showmol(view, height=height, width=width)


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
            render_xyz(path, f"Visualizing: {selected_file['file_name']}")
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
            # Combined grid view
            with open(solvated_file['repo_path'], "r") as f:
                sol_data = f.read()
            with open(dry_file['repo_path'], "r") as f:
                dry_data = f.read()

            # More balanced dimensions
            viewer_width = 1000
            viewer_height = 500
            
            # 1x3 grid with models in 0 and 2. 
            view = py3Dmol.view(width=viewer_width, height=viewer_height, viewergrid=(1,3), linked=True)
            
            # Match Streamlit's dark background
            view.setBackgroundColor('#0e1117')
            
            # Helper to apply styles to a specific viewer in the grid
            def apply_comparison_style(v, model_data, viewer_idx):
                v.addModel(model_data, "xyz", viewer=viewer_idx)
                v.setStyle({'elem': ["C", "N", "P"]}, {'stick': {'radius': 0.18}, 'sphere': {'scale': 0.25}}, viewer=viewer_idx)
                if show_surface:
                    v.setStyle({'elem': ["Si", "Al", "O"]}, {'stick': {'radius': 0.12, 'opacity': 0.9}}, viewer=viewer_idx)
                else:
                    v.setStyle({'elem': ["Si", "Al", "O"]}, {'sphere': {'radius': 0.01, 'opacity': 0}}, viewer=viewer_idx)
                v.setStyle({'elem': "H"}, {'line': {'linewidth': 0.5}}, viewer=viewer_idx)
                v.zoomTo(viewer=viewer_idx)

            apply_comparison_style(view, sol_data, (0,0))
            apply_comparison_style(view, dry_data, (0,2))
            
            st.subheader(f"Left: Solvated | Right: Dry")
            showmol(view, height=viewer_height, width=viewer_width)
        else:
            # Separate columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Solvated (Wet)")
                render_xyz(solvated_file['repo_path'], height=400)
            with col2:
                st.subheader("Dry")
                render_xyz(dry_file['repo_path'], height=400)
    else:
        st.info("No frames found with both solvated and dry .xyz files for current filters.")

with tabs[2]:
    st.header("Filtered Data")
    st.dataframe(filtered_df)

st.sidebar.markdown("---")
st.sidebar.info("RNA Silicate Interactions Streamlit App")
