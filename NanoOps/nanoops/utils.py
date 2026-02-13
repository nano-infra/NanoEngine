"""Utility functions for NanoOps."""

import re
import socket
from typing import Optional


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format (alphanumeric + hyphens/underscores)."""
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", session_id))


def check_port_listening(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if TCP port is listening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def allocate_port(session_id: str, component_type: str) -> int:
    """Allocate unique port for component based on session ID and type.

    Uses deterministic hash to ensure same session+component always gets same port.
    """
    base_ports = {"route": 3001, "prefill": 5000, "decode": 6000}

    if component_type not in base_ports:
        raise ValueError(f"Unknown component type: {component_type}")

    base = base_ports[component_type]
    offset = hash(f"{session_id}_{component_type}") % 1000
    return base + offset


def find_binary(name: str, search_paths: Optional[list[str]] = None) -> Optional[str]:
    """Find binary in PATH or search paths."""
    import shutil

    # Check PATH first
    path_binary = shutil.which(name)
    if path_binary:
        return path_binary

    # Check search paths
    if search_paths:
        import os

        for path in search_paths:
            full_path = os.path.join(path, name)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                return full_path

    return None
