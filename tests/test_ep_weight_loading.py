from types import SimpleNamespace

import dlengine.runtime.runner.loader as loader
from dlengine.runtime.models.deepseek_v2 import deepseek_v2_mtp_loader
import pytest
import torch


def test_discover_weight_files_rejects_worker_invisible_path(monkeypatch):
    monkeypatch.setattr(loader, "glob", lambda _: [])

    with pytest.raises(FileNotFoundError, match="visible from every Ray worker"):
        loader._discover_weight_files("/missing/model")


def test_iterate_weights_fails_before_tensor_iteration(monkeypatch):
    monkeypatch.setattr(loader, "glob", lambda _: [])

    with pytest.raises(FileNotFoundError, match="/missing/model"):
        next(loader.iterate_weights("/missing/model"))


def test_iterate_mtp_weights_fails_before_tensor_iteration(monkeypatch):
    monkeypatch.setattr(loader, "glob", lambda _: [])

    with pytest.raises(FileNotFoundError, match="/missing/model"):
        next(loader.iterate_mtp_weights("/missing/model"))


def test_ep_filter_keeps_only_local_experts(monkeypatch):
    ctx = SimpleNamespace(ffn_ep_world_size=4, ffn_ep_rank=2)
    monkeypatch.setattr(loader, "get_dist_context", lambda: ctx)
    pred = loader._make_ep_weight_filter(SimpleNamespace(num_experts=256))

    assert pred("model.layers.3.mlp.experts.128.gate_proj.weight")
    assert pred("model.layers.3.mlp.experts.191.down_proj.weight_scale")
    assert not pred("model.layers.3.mlp.experts.127.gate_proj.weight")
    assert not pred("model.layers.3.mlp.experts.192.gate_proj.weight")
    assert pred("model.layers.3.self_attn.q_proj.weight")


def test_iterate_weights_filters_before_get_tensor(monkeypatch):
    calls = []

    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return [
                "model.layers.3.mlp.experts.0.gate_proj.weight",
                "model.layers.3.mlp.experts.1.gate_proj.weight",
                "model.layers.3.self_attn.q_proj.weight",
            ]

        def get_tensor(self, name):
            calls.append(name)
            return torch.zeros(1)

    monkeypatch.setattr(loader, "glob", lambda _: ["fake.safetensors"])
    monkeypatch.setattr(loader, "safe_open", lambda *args: FakeSafeOpen())
    pred = lambda name: ".experts.1." not in name

    names = [
        name for name, _, _ in loader.iterate_weights("unused", weight_name_filter=pred)
    ]
    assert names == [
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
    ]
    assert calls == names


def test_mtp_loader_delegates_expert_layout_to_backend(monkeypatch):
    calls = []

    class Experts:
        def load_expert_weight(self, expert_idx, projection, kind, tensor, *, ep_rank):
            calls.append((expert_idx, projection, kind, tensor.shape, ep_rank))
            return True

    class Model:
        config = SimpleNamespace(
            num_hidden_layers=78,
            num_nextn_predict_layers=1,
        )
        quantization_config = None
        mtp_start_layer_idx = 78

        def get_submodule(self, name):
            assert name == "layers.78.mtp_block.mlp.routed_experts"
            return Experts()

        def named_parameters(self):
            return []

        def modules(self):
            return []

    monkeypatch.setattr(
        deepseek_v2_mtp_loader,
        "get_dist_context",
        lambda: SimpleNamespace(ffn_ep_rank=2),
    )
    tensor = torch.zeros(2048, 1024, dtype=torch.uint8)
    weights = iter([("model.layers.78.mlp.experts.3.down_proj.weight", "raw", tensor)])
    deepseek_v2_mtp_loader.load_weights(Model(), weights)

    assert calls == [(3, "down_proj", "weight", tensor.shape, 2)]
