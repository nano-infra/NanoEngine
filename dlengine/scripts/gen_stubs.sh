#!/bin/bash
set -e
pybind11-stubgen dlengine._dlengine_cpp --output-dir .
echo "✅ Stubs generated and patched! (Fixed json and CapsuleType)"
