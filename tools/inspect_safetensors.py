import sys

from safetensors.torch import load_file

if len(sys.argv) < 2:
    print("Usage: python inspect_safetensors.py <path_to_model_dir>")
    sys.exit(1)

model_path = sys.argv[1] + "/model.safetensors"
print(f"Loading {model_path}...")

try:
    weights = load_file(model_path)
    print("Keys found:")
    for key in sorted(weights.keys()):
        print(f"  {key} : {weights[key].shape}")
except Exception as e:
    print(f"Error: {e}")
