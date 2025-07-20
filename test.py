import pickle
import numpy as np
from keras.models import load_model
import os

def test_model(model_path, pkl_path):
    try:
        model = load_model(model_path)

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                input_data = pickle.load(f)

            # ✅ Fix for dict-type input
            if isinstance(input_data, dict):
                print("⚠️ Input is a dict, extracting the first value")
                input_data = list(input_data.values())[0]
        else:
            # If no .pkl file, generate random input from model input_shape
            input_shape = model.input_shape[1:]
            input_data = np.random.random(input_shape)

        # ✅ Ensure input is a NumPy array (avoid list or other types)
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        # ✅ Convert dtype and normalize
        if hasattr(input_data, "astype"):
            input_data = input_data.astype('float32') / 255.0

        # ✅ Optionally add batch dimension (disabled here)
        # input_data = np.expand_dims(input_data, axis=0)

        try:
            predictions = model.predict(input_data)
            print(f"Predictions: {predictions}")
            return "Success"
        except Exception as e:
            error_msg = f"Error during prediction: {e}"
            print(error_msg)
            return error_msg

    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        return error_msg
