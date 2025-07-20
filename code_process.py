import os
import shutil
import json
import re
import importlib.util
from collections import defaultdict
from keras.models import load_model
from test import test_model
from api import run_error_repair
from input_generation import process_no_input_errors
from input_process import process_files

# -------- Error Classification and Cleaning Functions -------- #
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

def normalize_error_message(error_msg: str) -> str:
    cleaned = re.sub(r'layer\s+\"[^\"]+\"', 'layer', error_msg)
    return cleaned.strip()

def load_repair_function(py_file_path):
    spec = importlib.util.spec_from_file_location("repair_module", py_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "build_fixed_model", None)

# -------- Main Repair Pipeline -------- #
def process_repair(error_info_path, repair_dir,
                   gpt_input_dir="gpt_input",
                   output_dir="output_files",
                   failure_dir=None,
                   failure_info_path=None):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gpt_input_dir, exist_ok=True)
    if failure_dir:
        os.makedirs(failure_dir, exist_ok=True)

    if not os.path.exists(error_info_path):
        print(f"❌ {error_info_path} does not exist")
        return

    with open(error_info_path, "r", encoding="utf-8") as f:
        error_info = json.load(f)

    error_dict = defaultdict(lambda: defaultdict(set))

    for error_type, group in error_info.items():
        if error_type == "No Input Error":
            continue

        for error_code, data in group.items():
            message = data["message"]
            models = data["models"]
            repair_path = os.path.join(repair_dir, f"{error_code}.py")

            if not os.path.exists(repair_path):
                print(f"⚠️ Missing repair script: {repair_path}")
                for file in models:
                    error_dict["Other Error"]["Missing repair script"].add(file)
                continue

            build_model_fn = load_repair_function(repair_path)
            if not build_model_fn:
                print(f"❌ Failed to load repair function: {repair_path}")
                for file in models:
                    err_type = classify_error("Failed to load repair function")
                    norm = normalize_error_message("Failed to load repair function")
                    error_dict[err_type][norm].add(file)
                continue

            for file in models:
                h5_path = os.path.join(gpt_input_dir, file)
                pkl_path = h5_path.replace(".h5", ".pkl")

                try:
                    model = build_model_fn()
                    model.save(h5_path)
                except Exception as e:
                    err_msg = f"Failed to execute repair function: {str(e)}"
                    err_type = classify_error(err_msg)
                    norm = normalize_error_message(err_msg)
                    error_dict[err_type][norm].add(file)
                    continue

                try:
                    load_model(h5_path)
                except Exception as e:
                    err_msg = f"Model loading failed: {str(e)}"
                    err_type = classify_error(err_msg)
                    norm = normalize_error_message(err_msg)
                    error_dict[err_type][norm].add(file)
                    continue

                if not os.path.exists(pkl_path):
                    error_dict["No Input Error"]["Missing .pkl input file"].add(file)
                    continue

                result = test_model(h5_path, pkl_path)
                if result == "Success":
                    shutil.move(h5_path, os.path.join(output_dir, file))
                    shutil.move(pkl_path, os.path.join(output_dir, os.path.basename(pkl_path)))
                else:
                    if failure_dir and failure_info_path:
                        shutil.move(h5_path, os.path.join(failure_dir, file))
                        if os.path.exists(pkl_path):
                            shutil.move(pkl_path, os.path.join(failure_dir, os.path.basename(pkl_path)))
                    err_type = classify_error(result)
                    norm = normalize_error_message(result)
                    error_dict[err_type][norm].add(file)

    formatted_error_dict = {}
    for etype, details in error_dict.items():
        type_counter = 1
        formatted_error_dict[etype] = {}
        for msg, models in details.items():
            key = f"{etype.lower().replace(' ', '')}{type_counter}"
            formatted_error_dict[etype][key] = {
                "message": msg,
                "models": list(models)
            }
            type_counter += 1

    out_path = failure_info_path if failure_info_path else "fail_error_info.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted_error_dict, f, indent=2, ensure_ascii=False)

    print(f"✅ Repair attempts completed. Error records written to {out_path}")

# -------- Main Entry Point -------- #
def run_full_pipeline(api_key):
    process_files(input_dir="input_files", output_dir="output_files", gpt_input_dir="gpt_input")
    process_no_input_errors(api_key, error_info_path="error_info.json", gpt_input_dir="gpt_input", output_dir="output_files")
    run_error_repair(api_key, error_info_path="error_info.json", repair_dir="repairs")
    process_repair("error_info.json", "repairs")
    process_no_input_errors(api_key, error_info_path="fail_error_info.json", gpt_input_dir="gpt_input", output_dir="output_files")
    run_error_repair(api_key, error_info_path="fail_error_info.json", repair_dir="repairs2")
    process_repair("fail_error_info.json", "repairs2", failure_dir="failure_files", failure_info_path="failure_info.json")
