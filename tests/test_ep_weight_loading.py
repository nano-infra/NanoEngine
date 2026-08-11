from types import SimpleNamespace

import dlengine.worker.loader as loader
import torch


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
