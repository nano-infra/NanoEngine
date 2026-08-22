from types import SimpleNamespace

import dlengine.runtime.models.deepseek_v2.deepseek_v2 as deepseek_module
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
