import os
import json
import openai
import pickle
import numpy as np
import importlib.util
from keras.models import load_model
from test import test_model

# Extract simplified model summary for prompt, avoid overly long input
def extract_model_summary(model_path):
    try:
        model = load_model(model_path)
        input_shape = model.input_shape
        output_shape = model.output_shape
        num_layers = len(model.layers)
        return f"Model input shape: {input_shape}, output shape: {output_shape}, number of layers: {num_layers}"
    except Exception as e:
        print(f"❌ Failed to extract model structure: {e}")
        return None

# Request input generation code from GPT
def generate_input_with_gpt(api_key, model_summary):
    prompt = f"""
You are an expert in generating input for Keras models.

Below is a summary of a Keras model. Please generate a test input using NumPy that allows successful execution of `model.predict(input_data)`.

Please return a function called `build_test_input()` which generates a NumPy array with a shape matching the model's `input_shape`. Only return code, no text or explanation:

import numpy as np

def build_test_input():
    # Generate input based on model structure
    input_data = np.random.rand(1, 32, 32, 3).astype('float32')
    return input_data

Model summary:
{model_summary}
""".strip()

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Keras expert. The returned input generation function must contain code only."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    raw_code = response["choices"][0]["message"]["content"]
    code = raw_code.replace("```python", "").replace("```", "").strip()
    return code

# Execute code to generate numpy input
def save_and_run_input_code(code, test_input_path="generated_input.py"):
    with open(test_input_path, "w", encoding="utf-8") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location("input_module", test_input_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_test_input()

# Generate input for a single model
def generate_input(h5_path, api_key):
    try:
        summary = extract_model_summary(h5_path)
        if not summary:
            return False

        code = generate_input_with_gpt(api_key, summary)
        input_data = save_and_run_input_code(code)

        if isinstance(input_data, dict):
            input_data = list(input_data.values())[0]

        pkl_path = h5_path.replace(".h5", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(input_data, f)

        print(f"✅ GPT successfully generated input: {pkl_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to generate input via GPT: {e}")
        return False

# Batch process models with No Input Error
def process_no_input_errors(
    api_key,
    error_info_path="error_info.json",
    gpt_input_dir="./gpt_input",
    output_dir="./output_files"
):
    if not os.path.exists(error_info_path):
        print(f"❌ Cannot find {error_info_path}")
        return

    with open(error_info_path, "r", encoding="utf-8") as f:
        error_data = json.load(f)

    if "No Input Error" not in error_data:
        print("✅ No 'No Input Error' entries found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for code, entry in error_data["No Input Error"].items():
        for model_file in entry["models"]:
            h5_path = os.path.join(gpt_input_dir, model_file)
            pkl_path = h5_path.replace(".h5", ".pkl")

            print(f"\n🚧 Processing: {model_file}")
            if not generate_input(h5_path, api_key):
                continue

            result = test_model(h5_path, pkl_path)
            if result == "Success":
                print("✅ Test passed → moving to output_files")
                os.rename(h5_path, os.path.join(output_dir, model_file))
                os.rename(pkl_path, os.path.join(output_dir, os.path.basename(pkl_path)))
            else:
                print("⚠️ Test failed → staying in gpt_input")

    print("📌 No Input Error processing complete. Original JSON was not modified or deleted.")
