import os
import numpy as np
import pickle

# Source and target directories
source_root = "./muffin_files"
output_root = "./input_files"

# Ensure target directory exists
os.makedirs(output_root, exist_ok=True)

# Traverse each subfolder in muffin_files
for subfolder in os.listdir(source_root):
    subfolder_path = os.path.join(source_root, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    npz_path = os.path.join(subfolder_path, "inputs.npz")
    target_pkl_path = os.path.join(output_root, f"{subfolder}.pkl")

    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            first_key = list(data.keys())[0]
            array = data[first_key]

            with open(target_pkl_path, "wb") as f:
                pickle.dump(array, f)

            print(f"✔ Success: {subfolder} ➝ {target_pkl_path} (shape: {array.shape})")

        except Exception as e:
            print(f"❌ Conversion failed: {subfolder}, error: {e}")
    else:
        print(f"⚠ Missing inputs.npz: {subfolder}")
