from types import SimpleNamespace

import dlengine.runtime.models.deepseek_v2.deepseek_v2 as deepseek_module
import torch
from dlengine.runtime.models import pp_utils
from torch import nn


class _FakeDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        quantization_config,
        layer_idx,
        cache_layer_idx=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_layer_idx = cache_layer_idx


def test_deepseek_pp_remaps_global_layers_to_local_cache_slots(monkeypatch):
    context = SimpleNamespace(
        pp_world_size=16,
        pp_rank=3,
        is_first_pp_stage=False,
        is_last_pp_stage=False,
    )
    monkeypatch.setattr(deepseek_module, "get_dist_context", lambda: context)
    monkeypatch.setattr(pp_utils, "get_dist_context", lambda: context)
    monkeypatch.setattr(deepseek_module, "DeepseekV2DecoderLayer", _FakeDecoderLayer)

    config = SimpleNamespace(
        vocab_size=1024,
        hidden_size=64,
        dtype=None,
        num_hidden_layers=30,
        rms_norm_eps=1e-6,
    )
    model = deepseek_module.DeepseekV2Model(config, quantization_config=None)

    assert (model.start_layer, model.end_layer) == (6, 8)
    assert model.layers[6].layer_idx == 6
    assert model.layers[6].cache_layer_idx == 0
    assert model.layers[7].layer_idx == 7
    assert model.layers[7].cache_layer_idx == 1


def test_deepseek_pp_send_boundary_orders_hidden_header_and_indexer(monkeypatch):
    context = SimpleNamespace(pp_next_global_rank=11)
    monkeypatch.setattr(deepseek_module, "get_dist_context", lambda: context)
    captured = {}
    pending = object()

    def isend_tensors(tensors, *, dst, profiler_names=None):
        captured["tensors"] = tuple(tensors)
        captured["dst"] = dst
        captured["profiler_names"] = tuple(profiler_names or ())
        return pending

    monkeypatch.setattr(deepseek_module, "pp_isend_tensors", isend_tensors)
    hidden = torch.ones(2, 4)
    logical = torch.full((2, 3), 4, dtype=torch.int32)
    physical = torch.full((2, 3), 9, dtype=torch.int32)
    state = deepseek_module._IndexerTopKState()
    state.publish(17, logical, physical)

    assert deepseek_module._pp_send_boundary(hidden, state) is pending
    tensors = captured["tensors"]
    assert captured["dst"] == 11
    assert captured["profiler_names"] == (
        "nano_pp/send_hidden",
        "nano_pp/send_indexer_header",
        "nano_pp/send_indexer_logical",
        "nano_pp/send_indexer_physical",
    )
    assert tensors[0] is hidden
    assert tensors[1].tolist() == [17, 1]
    assert tensors[2] is logical
    assert tensors[3] is physical


def test_deepseek_pp_send_boundary_skips_indexer_for_full_next_layer(monkeypatch):
    context = SimpleNamespace(pp_next_global_rank=11)
    monkeypatch.setattr(deepseek_module, "get_dist_context", lambda: context)
    captured = {}
    pending = object()

    def isend_tensors(tensors, *, dst, profiler_names=None):
        captured["tensors"] = tuple(tensors)
        captured["dst"] = dst
        captured["profiler_names"] = tuple(profiler_names or ())
        return pending

    monkeypatch.setattr(deepseek_module, "pp_isend_tensors", isend_tensors)
    hidden = torch.ones(2, 4)
    state = deepseek_module._IndexerTopKState()
    state.publish(
        17,
        torch.full((2, 3), 4, dtype=torch.int32),
        torch.full((2, 3), 9, dtype=torch.int32),
    )

    assert (
        deepseek_module._pp_send_boundary(hidden, state, include_indexer=False)
        is pending
    )
    assert captured["dst"] == 11
    assert captured["profiler_names"] == (
        "nano_pp/send_hidden",
        "nano_pp/send_indexer_header",
    )
    assert captured["tensors"][0] is hidden
    assert captured["tensors"][1].tolist() == [-1, 0]


def test_deepseek_pp_recv_boundary_groups_known_buffers(monkeypatch):
    context = SimpleNamespace(pp_prev_global_rank=7)
    monkeypatch.setattr(deepseek_module, "get_dist_context", lambda: context)
    calls = []

    def irecv_tensors(tensors, *, src, profiler_names=None):
        tensors = tuple(tensors)
        calls.append((tensors, src, tuple(profiler_names or ())))
        if len(calls) == 1:
            tensors[0].fill_(2)
            tensors[1].copy_(torch.tensor([13, 1]))
        else:
            tensors[0].fill_(5)
            tensors[1].fill_(8)

    monkeypatch.setattr(deepseek_module, "pp_irecv_tensors", irecv_tensors)
    hidden, state = deepseek_module._pp_recv_boundary(
        2, 4, torch.float32, 3, device="cpu"
    )

    assert [len(tensors) for tensors, _, _ in calls] == [2, 2]
    assert [src for _, src, _ in calls] == [7, 7]
    assert calls[0][2] == (
        "nano_pp/recv_hidden",
        "nano_pp/recv_indexer_header",
    )
    assert calls[1][2] == (
        "nano_pp/recv_indexer_logical",
        "nano_pp/recv_indexer_physical",
    )
    assert hidden.tolist() == [[2.0] * 4] * 2
    logical, physical = state.require(13, 2, 3, require_physical=True)
    assert torch.equal(logical, torch.full((2, 3), 5, dtype=torch.int32))
    assert torch.equal(physical, torch.full((2, 3), 8, dtype=torch.int32))
