from types import SimpleNamespace

import dlengine.models.deepseek_v2.deepseek_v2_loader as loader_module
import torch
from dlengine.models import pp_utils
from torch import nn


class _FakeDeepseekModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.quantization_config = SimpleNamespace(block_size=[128, 128])
        self.model = nn.Module()
        self.model.start_layer = 2
        self.model.end_layer = 4


def test_deepseek_pp_loader_skips_nonlocal_kv_projection(monkeypatch):
    context = SimpleNamespace(
        pp_world_size=2,
        is_last_pp_stage=False,
    )
    monkeypatch.setattr(loader_module, "get_dist_context", lambda: context)
    monkeypatch.setattr(pp_utils, "get_dist_context", lambda: context)

    def fail_if_processed(*args, **kwargs):
        raise AssertionError("nonlocal kv_b_proj must not be buffered or processed")

    monkeypatch.setattr(loader_module, "_handle_kv_b_proj", fail_if_processed)
    weight_name = "model.layers.0.self_attn.kv_b_proj.weight"

    loader_module.load_weights(
        _FakeDeepseekModel(),
        iter([(weight_name, weight_name, torch.zeros(1))]),
    )
