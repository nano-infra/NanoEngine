import importlib.util
import sys
import os

# Try to find the compiled module
# It should be named _nanodeploy_cpp.cp3x-win_amd64.pyd on Windows or .so on Linux
# We can just import it if it's in the path or in this directory

try:
    from ._nanodeploy_cpp import *
except ImportError:
    # If not found, maybe it's not compiled or not in the right place
    # We can try to look for it in the build directory if we are in dev mode
    pass
