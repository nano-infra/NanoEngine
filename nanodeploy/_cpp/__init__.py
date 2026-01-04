import importlib.util
import os
import sys

# Try to find the compiled module
# It should be named _nanodeploy_cpp.cp3x-win_amd64.pyd on Windows or .so on Linux
# We can just import it if it's in the path or in this directory

try:
    from nanodeploy._nanodeploy_cpp import *
except ImportError as e:
    # Propagate the error so that the caller can see why the import failed
    # This is crucial for debugging (e.g. missing dependencies, symbol errors)
    raise e
