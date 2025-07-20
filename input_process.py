import os
import shutil
import json
import re
from collections import defaultdict
from keras.models import load_model
from test import test_model

#  Error classification function (updated version)
def classify_error(error_msg):
    msg_lower = error_msg.lower()
    if "ndim" in msg_lower or "layer" in msg_lower:
        return "Structure Error"
    elif "function" in msg_lower:
        return "Function Error"
    elif "attribute" in msg_lower:
        return "Attribute Error"
    elif "no module named" in msg_lower:
        return "Import Error"
    elif "shape" in msg_lower or "reshape" in msg_lower:
        return "Shape Error"
    elif "type" in msg_lower:
        return "Type Error"
    else:
        return "Other Error"

#  Normalize error messages (clean layer names, model names, etc.)
def normalize_error_message(error_msg: str) -> str:
    cleaned = re.sub(r'layer\s+"[^"]+"', 'layer', error_msg)  # remove layer names
    return cleaned.strip()

#  Initialize error dictionary
error_dict = defaultdict(lambda: defaultdict(set))

def process_files(input_dir, output_dir, gpt_input_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gpt_input_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".h5"):
            h5_path = os.path.join(input_dir, file)
            pkl_path = os.path.join(input_dir, file.replace(".h5", ".pkl"))

            # Try to load the model
            try:
                model = load_model(h5_path)
            except Exception as e:
                raw_error = f"{type(e).__name__}: {str(e)}"
                normalized_error = normalize_error_message(raw_error)
                error_type = classify_error(normalized_error)
                print(f"Model load failed, moving to gpt_input: {file}\nError type: {error_type}\nError message: {normalized_error}")
                error_dict[error_type][normalized_error].add(file)

                shutil.move(h5_path, os.path.join(gpt_input_dir, file))
                if os.path.exists(pkl_path):
                    shutil.move(pkl_path, os.path.join(gpt_input_dir, os.path.basename(pkl_path)))
                continue

            # Check if .pkl file is missing
            if not os.path.exists(pkl_path):
                print(f"{file} is missing .pkl → classified as No Input Error and moved to gpt_input")
                error_dict["No Input Error"]["Missing .pkl input file"].add(file)
                shutil.move(h5_path, os.path.join(gpt_input_dir, file))
                continue

            # Test the model normally
            result = test_model(h5_path, pkl_path)
            print(f"Processing {file} test result:\n{result}\n")

            if result == "Success":
                shutil.move(h5_path, os.path.join(output_dir, file))
                shutil.move(pkl_path, os.path.join(output_dir, file.replace(".h5", ".pkl")))
            else:
                error_type = classify_error(result)
                normalized_error = normalize_error_message(result)
                error_dict[error_type][normalized_error].add(file)
                shutil.move(h5_path, os.path.join(gpt_input_dir, file))
                shutil.move(pkl_path, os.path.join(gpt_input_dir, file.replace(".h5", ".pkl")))

    # Save error information as JSON
    formatted_error_dict = {}

    for etype, details in error_dict.items():
        type_counter = 1
        formatted_error_dict[etype] = {}
        for msg, models in details.items():
            key = f"{etype.lower().replace(' ', '')}{type_counter}"  # e.g., type1, shape2
            formatted_error_dict[etype][key] = {
                "message": msg,
                "models": list(models)
            }
            type_counter += 1

    with open("error_info.json", "w", encoding="utf-8") as f:
        json.dump(formatted_error_dict, f, indent=2, ensure_ascii=False)

    print("Processing completed. Error information saved to error_info.json")
