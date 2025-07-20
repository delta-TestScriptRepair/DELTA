import os
import shutil

# Set target folder paths
folder_path = "./input_files"
gpt_input_dir = "./gpt_input"
input_files_dir = "./input_files"

# 1. Rename .pkl files starting with "prediction_"
for filename in os.listdir(folder_path):
    if filename.endswith(".pkl") and "prediction_" in filename:
        # Construct new file name
        new_filename = filename.replace("prediction_", "")

        # Full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # Skip if the target file already exists
        if os.path.exists(new_path):
            print(f"Skipping rename (target already exists): {new_filename}")
            continue

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")

# 2. Delete .pkl files without corresponding .h5 files
pkl_files = [f for f in os.listdir(input_files_dir) if f.endswith(".pkl")]
h5_files = [f for f in os.listdir(input_files_dir) if f.endswith(".h5")]
h5_basenames = {os.path.splitext(f)[0] for f in h5_files}

for pkl_file in pkl_files:
    base_name = os.path.splitext(pkl_file)[0]
    if base_name not in h5_basenames:
        pkl_path = os.path.join(input_files_dir, pkl_file)
        os.remove(pkl_path)
        print(f"Deleted .pkl file with no corresponding .h5: {pkl_file}")
