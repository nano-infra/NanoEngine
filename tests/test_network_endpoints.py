import socket

from dlengine.utils.network import get_advertise_host, get_bind_host


def test_bind_host_defaults_to_wildcard():
    assert get_bind_host(None) == "0.0.0.0"
    assert get_bind_host("") == "0.0.0.0"
    assert get_bind_host("127.0.0.1") == "127.0.0.1"


def test_advertise_host_never_returns_wildcard(monkeypatch):
    monkeypatch.setattr("dlengine.utils.network.get_local_ip", lambda: "10.0.0.8")

    assert get_advertise_host(None) == "10.0.0.8"
    assert get_advertise_host("") == "10.0.0.8"
    assert get_advertise_host("0.0.0.0") == "10.0.0.8"
    assert get_advertise_host("::") == "10.0.0.8"
    assert get_advertise_host("node.example") == "node.example"


def test_os_allocates_port_when_binding_zero():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        assert sock.getsockname()[1] > 0
