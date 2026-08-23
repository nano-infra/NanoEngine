from types import MethodType, SimpleNamespace

import torch
from torch import nn

import dlengine.runtime.models.deepseek_v2.deepseek_v2_mtp as mtp_module
import dlengine.runtime.models.deepseek_v2.deepseek_v2_mtp_loader as loader_module
from dlengine.engine.llm_component import LLMComponent
from dlengine.runtime.runner.model_runner import _wire_mla_hisparse_modules


class _FakeMTPLayer(nn.Module):
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


class _FakeMLACacheModule:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None


def test_hisparse_wires_predictor_after_target_layers():
    target = [_FakeMLACacheModule(), _FakeMLACacheModule()]
    predictor = [_FakeMLACacheModule()]
    hot = torch.empty(1, 3, 2, 4, 1, 1)

    next_layer = _wire_mla_hisparse_modules(target, hot, 0, "cpu")
    next_layer = _wire_mla_hisparse_modules(predictor, hot, next_layer, "cpu")

    assert next_layer == 3
    assert target[0].k_cache.data_ptr() == hot[0, 0].data_ptr()
    assert target[1].k_cache.data_ptr() == hot[0, 1].data_ptr()
    assert predictor[0].k_cache.data_ptr() == hot[0, 2].data_ptr()
    assert predictor[0].v_cache.numel() == 0


def test_glm_mtp_uses_final_pp_stage_local_cache_slot(monkeypatch):
    monkeypatch.setattr(mtp_module, "VocabParallelEmbedding", nn.Embedding)
    monkeypatch.setattr(mtp_module, "DeepSeekMTPLayer", _FakeMTPLayer)
    monkeypatch.setattr(mtp_module, "get_pp_layer_range", lambda _layers: (69, 78))

    config = SimpleNamespace(
        quantization_config={},
        num_nextn_predict_layers=1,
        num_hidden_layers=78,
        vocab_size=16,
        hidden_size=8,
    )

    model = mtp_module.DeepSeekMTP(config)

    assert model.mtp_cache_start_idx == 9
    assert model.layers["78"].cache_layer_idx == 9


class _EmbeddingOnlyMTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=78,
            num_nextn_predict_layers=1,
        )
        self.quantization_config = SimpleNamespace(block_size=[128, 128])
        self.mtp_start_layer_idx = 78
        self.embed_tokens = nn.Embedding(4, 3)


def test_mtp_loader_can_replicate_base_embedding_on_final_pp_stage():
    model = _EmbeddingOnlyMTP()
    expected = torch.arange(12, dtype=model.embed_tokens.weight.dtype).view(4, 3)

    loader_module.load_weights(
        model,
        iter([("model.embed_tokens.weight", "model.embed_tokens.weight", expected)]),
    )

    assert torch.equal(model.embed_tokens.weight, expected)


def test_pp_cache_metadata_appends_predictor_to_final_stage():
    hf_config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        num_hidden_layers=78,
    )
    config = SimpleNamespace(
        hf_config=hf_config,
        pp=8,
        num_speculative_tokens=5,
        enable_hisparse=False,
    )
    component = SimpleNamespace(config=config)
    component._pp_layer_ranges = MethodType(LLMComponent._pp_layer_ranges, component)
    component._mtp_num_kv_layers = MethodType(
        LLMComponent._mtp_num_kv_layers, component
    )

    layers = LLMComponent._pp_cache_layer_indices(component)

    assert layers[-1] == list(range(69, 79))
    assert [len(stage) for stage in layers] == [10, 10, 10, 10, 10, 10, 9, 10]
    assert [layer for stage in layers for layer in stage] == list(range(79))


def test_non_mla_mtp_does_not_publish_predictor_as_primary_kv():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        num_speculative_tokens=1,
    )
    component = SimpleNamespace(config=config)

    assert LLMComponent._mtp_num_kv_layers(component) == 0
