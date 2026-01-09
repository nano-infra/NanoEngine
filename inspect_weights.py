import json
import os
import sys

from safetensors.torch import load_file


def inspect_weights(model_dir):
    print(f"Inspecting model in: {model_dir}")

    # Check for index
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        print("Found index file.")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        # Get list of unique files
        files = list(set(weight_map.values()))
        files.sort()

        for file in files:
            full_path = os.path.join(model_dir, file)
            print(f"\n--- File: {file} ---")
            try:
                state_dict = load_file(full_path)
                keys = list(state_dict.keys())
                keys.sort()
                for k in keys:
                    t = state_dict[k]
                    print(f"{k}: {t.shape} {t.dtype}")
            except Exception as e:
                print(f"Error loading {file}: {e}")

    else:
        # Check for single file
        single_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single_path):
            print("Found single model.safetensors.")
            try:
                state_dict = load_file(single_path)
                keys = list(state_dict.keys())
                keys.sort()
                for k in keys:
                    t = state_dict[k]
                    # Filter for attention weights only to keep log short
                    if "layers.0" in k or "embed" in k:
                        print(f"{k}: {t.shape} {t.dtype}")
            except Exception as e:
                print(f"Error loading {single_path}: {e}")
        else:
            print("No safetensors found.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the path we know
        model_dir = "/mnt/c/Users/majinming/Downloads/qwen3-0.6b-local"
    else:
        model_dir = sys.argv[1]

    inspect_weights(model_dir)
