"""Network utility helpers."""

import socket

WILDCARD_HOSTS = {"", "0.0.0.0", "::"}


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def get_bind_host(host: str | None) -> str:
    """Return the local listen host for an optional user-supplied host."""
    return "0.0.0.0" if host is None or host == "" else host


def get_advertise_host(host: str | None) -> str:
    """Return an address peers can connect to (never a wildcard address)."""
    return get_local_ip() if host is None or host in WILDCARD_HOSTS else host


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]
