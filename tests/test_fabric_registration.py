from unittest.mock import Mock

import dlslime.ctrl
from dlengine.server import openai_server


def test_http_registration_copies_backend_placement_resource(monkeypatch):
    resource = {
        "schema_version": 1,
        "placements": [{"rank": 0, "fabric_domain_id": "fabric-a"}],
    }
    client = Mock()
    client.get_entity_info.return_value = {"resource": resource}
    client.register.return_value = True
    monkeypatch.setattr(dlslime.ctrl, "NanoCtrlClient", Mock(return_value=client))
    monkeypatch.setattr(openai_server, "_advertise_host", lambda _host: "node-a")

    result = openai_server.register_with_ctrl(
        ctrl_address="http://ctrl",
        ctrl_scope=None,
        host="0.0.0.0",
        port=8000,
        served_model_name="model",
        model_path="/models/model",
        role="decode",
        engine_id="engine-a",
    )

    assert result is client
    assert client.register.call_args.kwargs["resource"] is resource
    client.get_entity_info.assert_called_once_with("engine-a")


def test_http_registration_without_backend_keeps_empty_resource(monkeypatch):
    client = Mock()
    client.register.return_value = True
    monkeypatch.setattr(dlslime.ctrl, "NanoCtrlClient", Mock(return_value=client))
    monkeypatch.setattr(openai_server, "_advertise_host", lambda _host: "node-a")

    openai_server.register_with_ctrl(
        ctrl_address="http://ctrl",
        ctrl_scope=None,
        host="0.0.0.0",
        port=8000,
        served_model_name="model",
        model_path="/models/model",
    )

    assert client.register.call_args.kwargs["resource"] is None
