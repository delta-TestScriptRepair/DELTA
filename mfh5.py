import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input
from layer_map import LAYER_NAME_MAP

# Root directories
source_root = "./muffin_files"
output_root = "./input_files"

os.makedirs(output_root, exist_ok=True)

def build_model(model_json_path):
    with open(model_json_path, "r") as f:
        j = json.load(f)
    graph = j["model_structure"]
    input_ids = j["input_id_list"]
    output_ids = j["output_id_list"]

    tensor_map = {}
    skip_count = 0

    def get_tensor(pid: str):
        while pid not in tensor_map and int(pid) > 0:
            pid = str(int(pid) - 1)
        return tensor_map.get(pid)

    for idx in sorted(graph, key=lambda x: int(x)):
        info = graph[idx]
        ltype = info["type"]
        args = info.get("args", {})
        pres = info.get("pre_layers", [])

        kname = LAYER_NAME_MAP.get(ltype)
        if not kname:
            print(f"⚠ Unsupported layer {ltype}, skipping idx={idx}")
            skip_count += 1
            continue

        lname = args.get("name", f"{idx.zfill(2)}_{ltype}")

        if kname == "Input":
            shape = tuple(args["shape"])
            x = Input(shape, name=lname)
            tensor_map[idx] = x
            continue

        if ltype == "softmax":
            layer = layers.Activation("softmax", name=lname)
        elif ltype == "time_distributed":
            inner = args["layer"]
            inner_kname = LAYER_NAME_MAP.get(inner["type"])
            inner_args = inner["args"]
            if not inner_kname:
                print(f"⚠ Unsupported inner layer {inner['type']}, skipping idx={idx}")
                skip_count += 1
                continue
            inner_layer = getattr(layers, inner_kname)(**inner_args)
            layer = layers.TimeDistributed(inner_layer, name=lname)
        elif ltype in ("average",):
            layer = getattr(layers, kname)(name=lname)
        else:
            layer = getattr(layers, kname)(**args)
            layer._name = lname

        inputs = [get_tensor(str(p)) for p in pres]
        if any(i is None for i in inputs):
            print(f"⚠ Missing predecessor, skipping idx={idx}")
            skip_count += 1
            continue

        x = layer(inputs[0] if len(inputs) == 1 else inputs)
        tensor_map[idx] = x

    input_tensors = [tensor_map[str(i)] for i in input_ids if str(i) in tensor_map]
    output_tensors = [tensor_map[str(i)] for i in output_ids if str(i) in tensor_map]
    if not input_tensors or not output_tensors:
        raise ValueError("Missing input or output nodes, model assembly failed.")

    return Model(inputs=input_tensors, outputs=output_tensors), graph

# Traverse each model directory
for model_name in os.listdir(source_root):
    model_dir = os.path.join(source_root, model_name)
    json_path = os.path.join(model_dir, "model.json")
    weight_dir = os.path.join(model_dir, "initial_weights")

    if not os.path.exists(json_path) or not os.path.exists(weight_dir):
        print(f"❌ Skipping: {model_name}, missing required files")
        continue

    try:
        model, graph = build_model(json_path)
        success, failed = 0, 0

        for idx in graph:
            args = graph[idx].get("args", {})
            lname = args.get("name", f"{idx.zfill(2)}_{graph[idx]['type']}")
            weight_file = os.path.join(weight_dir, f"{lname}.npz")
            if not os.path.exists(weight_file):
                continue
            try:
                data = np.load(weight_file, allow_pickle=True)
                if not data.files:
                    success += 1  # Empty file counts as success
                    continue
                weights = [data[k] for k in sorted(data.files)]
                layer = model.get_layer(name=lname)
                layer.set_weights(weights)
                success += 1
            except Exception as e:
                print(f"⚠ Failed to load weights: {lname}, error: {e}")
                failed += 1

        save_path = os.path.join(output_root, f"{model_name}.h5")
        model.save(save_path)
        print(f"✔ Model converted successfully: {model_name}, weights loaded: {success}, failed: {failed}")

    except Exception as e:
        print(f"❌ Conversion failed: {model_name}, error: {e}")
