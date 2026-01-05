#!/bin/bash
set -e
pybind11-stubgen nanodeploy._nanodeploy_cpp --output-dir .
echo "✅ Stubs generated and patched! (Fixed json and CapsuleType)"
