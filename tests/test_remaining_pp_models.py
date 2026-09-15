from types import SimpleNamespace

import dlengine.runtime.models.deepseek_v4.deepseek_v4 as deepseek_v4_module
import dlengine.runtime.models.gemma4.gemma4 as gemma4_module
from dlengine.runtime.models import pp_utils
from torch import nn


class _FakeDecoderLayer(nn.Module):
    def __init__(self, _config, *args):
        super().__init__()
        self.layer_idx = args[-1]


def test_deepseek_v4_builds_only_local_pp_layers(monkeypatch):
    context = SimpleNamespace(
        pp_world_size=16,
        pp_rank=3,
        is_first_pp_stage=False,
        is_last_pp_stage=False,
    )
    monkeypatch.setattr(deepseek_v4_module, "get_dist_context", lambda: context)
    monkeypatch.setattr(pp_utils, "get_dist_context", lambda: context)
    monkeypatch.setattr(deepseek_v4_module, "DeepseekV4DecoderLayer", _FakeDecoderLayer)

    config = SimpleNamespace(
        vocab_size=1024,
        hidden_size=64,
        dtype=None,
        num_hidden_layers=30,
        rms_norm_eps=1e-6,
        hc_mult=4,
        hc_eps=1e-6,
    )
    model = deepseek_v4_module.DeepseekV4Model(config, quantization_config=None)

    assert (model.start_layer, model.end_layer) == (6, 8)
    assert model.embed_tokens is None
    assert model.hc_head is None
    assert model.norm is None
    assert model.layers[6].layer_idx == 6
    assert model.layers[7].layer_idx == 7


def test_gemma4_reserves_shared_kv_suffix_for_last_stage(monkeypatch):
    context = SimpleNamespace(
        pp_world_size=4,
        pp_rank=1,
        is_first_pp_stage=False,
        is_last_pp_stage=False,
    )
    monkeypatch.setattr(gemma4_module, "get_dist_context", lambda: context)
    monkeypatch.setattr(pp_utils, "get_dist_context", lambda: context)
    monkeypatch.setattr(gemma4_module, "Gemma4DecoderLayer", _FakeDecoderLayer)

    config = SimpleNamespace(
        vocab_size=1024,
        hidden_size=64,
        dtype=None,
        num_hidden_layers=12,
        num_kv_shared_layers=4,
        layer_types=["sliding_attention", "full_attention"] * 6,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        hidden_size_per_layer_input=0,
    )
    model = gemma4_module.Gemma4Model(config)

    assert (model.start_layer, model.end_layer) == (2, 4)
    assert model.embed_tokens is None
    assert model.norm is None
    assert model.layers[2].layer_idx == 2
    assert model.layers[3].layer_idx == 3
