import os
import subprocess
import sys
from pathlib import Path


def _get_binary_path(name: str) -> Path:
    # Look for binary in the package directory
    # structure:
    #   site-packages/
    #     nanodeploy/
    #       bin/
    #         nanodeploy_agent
    #       cli.py
    package_dir = Path(__file__).parent
    binary_path = package_dir / "bin" / name

    if not binary_path.exists():
        # Fallback for dev/editable mode if build artifacts are elsewhere?
        # But our CMake install rule puts it in nanodeploy/bin inside the build/install tree.
        # Scikit-build-core in editable mode allows importing the package,
        # so accessing resources relative to __file__ should work if the layout is preserved.
        pass

    return binary_path


def agent_entry():
    bin_path = _get_binary_path("nanodeploy_agent")
    if not bin_path.exists():
        print(f"Error: nanodeploy_agent binary not found at {bin_path}")
        sys.exit(1)

    # Exec the binary, replacing the current process
    try:
        os.execv(str(bin_path), [str(bin_path)] + sys.argv[1:])
    except OSError as e:
        print(f"Error executing {bin_path}: {e}")
        sys.exit(1)


def hub_entry():
    bin_path = _get_binary_path("nanodeploy_hub")
    if not bin_path.exists():
        print(f"Error: nanodeploy_hub binary not found at {bin_path}")
        sys.exit(1)

    try:
        os.execv(str(bin_path), [str(bin_path)] + sys.argv[1:])
    except OSError as e:
        print(f"Error executing {bin_path}: {e}")
        sys.exit(1)
