import openai
import json
import os
import re

PROMPT_MAP = {
    "Structure Error":
"""This error indicates that the input dimension to the model does not match the requirement of certain layers (e.g., LSTM, RNN, Dense), typically due to a 4D input when a 3D input is expected. Please generate a model-building function that explicitly sets the input shape to 3D format `(timesteps, features)`, avoids dimensionality-reducing expressions like `input_shape[:-1]`, and uses `InputLayer` to specify input clearly.

You must return code in the following format, and do NOT use markdown backticks:


from keras.models import Sequential
from keras.layers import LSTM, Dense, InputLayer

def build_fixed_model(input_shape=(49, 1), units=64):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape[0], input_shape[1])))
    model.add(LSTM(units))
    model.add(Dense(1))
    return model

The error message is:
""",

    "Shape Error":
"""This error suggests that the convolutional kernel dimensions are incompatible with the input shape (e.g., using unsupported Conv3D or having spatial dimensions that are too small).

Please construct a safe and standard convolutional model using Conv2D (Conv3D and other high-dim convolutions are prohibited), with the following constraints:
- Input shape = (32, 32, 3)
- All Conv2D layers must use padding='same'
- kernel_size must not exceed (3, 3)
- MaxPooling2D is recommended after each Conv2D layer
- Use Flatten + Dense layers at the end to complete classification

Return code in the following format. Do NOT use markdown backticks:

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_fixed_model(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

The error message is:
""",

    "Value Error":
"""This is a ValueError encountered during model construction or execution, such as illegal parameter values, shape mismatches, or missing input.

Please provide a concise and reasonable model-building function to fix such configuration issues. Use the following format:

from keras.models import Sequential
from keras.layers import Dense, Input

def build_fixed_model(input_shape=(10,), units=64):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    return model

The error message is:
""",

    "Function Error":
"""This error suggests that the model uses unrecognized function parameters (e.g., invalid activation name or initializer).

Please fix these invalid settings by replacing them with valid activation functions (e.g., 'relu' or LeakyReLU()). Return code in the format below:

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU

def build_fixed_model(input_shape=(100,)):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Dense(10))
    return model

The error message is:
""",

    "Type Error":
"""This error indicates an incorrect data type or illegal parameter was passed during model definition (e.g., `name` has illegal characters, or a dict was passed instead of a tensor).

Please rewrite the model-building function, ensuring all parameter types are correct. Return format:

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense

def build_fixed_model(input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, name='conv1_conv'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    return model

The error message is:
""",

    "Attribute Error":
"""This error suggests your code incorrectly applied `.astype()` to a dictionary. Please fix the input preprocessing logic to ensure `.astype()` is applied to a NumPy array rather than a dict. Return the fixed model-building function `build_fixed_model`, and include a comment example of the input fix:

from keras.models import Sequential
from keras.layers import Flatten, Dense

def build_fixed_model(input_shape=(28, 28)):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

# Example input fix (for reference, not part of the model function):
# if isinstance(inputs, dict):
#     inputs = list(inputs.values())[0]
# inputs = inputs.astype(np.float32)

The error message is:
""",

    "Other":
"""The following are other types of exceptions during model execution, possibly due to data inconsistencies, abnormal model structure, or unknown configuration errors.

Please provide a cleanly structured and reasonably parameterized model definition function as a repair attempt. Do not return markdown backticks. Use this format:

from keras.models import Sequential
from keras.layers import Dense

def build_fixed_model(input_shape=(20,)):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    return model

The error message is:
"""
}


def clean_gpt_code(raw_text):
    """
    Clean up code blocks returned by GPT wrapped in ```python ... ```
    """
    # Extract contents inside triple backticks
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return raw_text.strip()

def run_error_repair(api_key, error_info_path, repair_dir):
    openai.api_key = api_key
    os.makedirs(repair_dir, exist_ok=True)

    if not os.path.exists(error_info_path):
        raise FileNotFoundError(f"{error_info_path} does not exist. Please run input_process.py first.")

    with open(error_info_path, "r", encoding="utf-8") as f:
        error_data = json.load(f)

    for error_type, error_list in error_data.items():
        if error_type == "No Input Error":
            print(f"⏭ Skipping No Input Error")
            continue

        for error_code, error_entry in error_list.items():
            error_msg = error_entry["message"]

            print(f"\n🟡 Processing {error_code} ({error_type})")
            prompt_template = PROMPT_MAP.get(error_type, PROMPT_MAP["Other"])
            user_prompt = f"{prompt_template}\n\n{error_msg}"

            # Call GPT
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior Keras model repair expert. You only return the fixed Python function code. No natural language or explanation is allowed."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300
            )

            # Extract clean code
            raw_reply = response["choices"][0]["message"]["content"]
            clean_code = clean_gpt_code(raw_reply)

            # Save repaired file, use error_code as filename
            py_path = os.path.join(repair_dir, f"{error_code}.py")
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(clean_code)

            print(f"✅ Repaired code saved: {py_path}")

    print("\n🎉 All repair code has been generated.")
