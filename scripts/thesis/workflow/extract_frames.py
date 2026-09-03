import os

# === USER SETTINGS ===
input_file = "amp/docking/docking.docker.struc1.all.optimized.xyz"

# Define which frames to extract:
#    - Use e.g. [0, 5, 10] for 0-based indexing (Python default)
#    - Leave empty [] to save all frames
frame_list = [10,11,12,14,15]  # <-- Add your list of frame numbers here

# Define the output directory
output_dir = "amp/frames"
os.makedirs(output_dir, exist_ok=True)
# ======================

# --- Core splitting logic ---
with open(input_file, "r") as f:
    lines = f.readlines()

i = 0
frame0 = 0
saved = 0

while i < len(lines):
    try:
        n_atoms = int(lines[i].strip())
    except ValueError:
        break  # probably end of file

    frame_lines = lines[i : i + n_atoms + 2]  # atoms + 2 header lines
    frame1 = frame0 + 1

    # Save all frames if frame_list is empty, or only selected ones
    if not frame_list or frame1 in frame_list:
        # create per-frame folder labelled frameXX (two-digit zero-padded)
        frame_name = f"frame{frame1:02d}"
        frame_dir = os.path.join(output_dir, frame_name)
        os.makedirs(frame_dir, exist_ok=True)

        output_file = os.path.join(frame_dir, f"{frame_name}.xyz")
        with open(output_file, "w") as out:
            out.writelines(frame_lines)
        saved += 1

    frame0 += 1
    i += n_atoms + 2

print(f"Done! Extracted {saved} frame(s) to '{output_dir}/' (each in its frameXX folder)")
print(f"(Selection: {'all frames' if not frame_list else frame_list}, 1-based indexing)")
