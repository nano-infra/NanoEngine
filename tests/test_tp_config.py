from types import SimpleNamespace

import pytest
from jsonargparse import ArgumentParser

import dlengine.config as config_module
from dlengine.config import Config
from dlengine.runtime.context.distributed import DistContext


@pytest.fixture
def hf_config(monkeypatch):
    hf = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        dtype="bfloat16",
        max_position_embeddings=131072,
        num_hidden_layers=2,
        num_attention_heads=64,
        num_key_value_heads=16,
    )
    monkeypatch.setattr(config_module.AutoConfig, "from_pretrained", lambda *a, **kw: hf)
    monkeypatch.setattr(
        config_module.PretrainedConfig, "get_config_dict", lambda *a, **kw: ({}, {})
    )
    monkeypatch.setattr(config_module.torch.cuda, "is_available", lambda: False)
    return hf


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
    ],
)
def test_tp16_accepts_compatible_gqa_head_partitions(hf_config, architecture):
    hf_config.architectures = [architecture]
    config = Config(model="unused", attention_tp=16, ffn_tp=16)
    assert config.attn_world_size == config.ffn_world_size == config.world_size == 16


@pytest.mark.parametrize("tp", [16, 32])
def test_kimi_mla_tp_shares_compressed_kv(hf_config, tp):
    hf_config.architectures = ["KimiK3ForConditionalGeneration"]
    hf_config.num_key_value_heads = 1
    config = Config(model="unused", attention_tp=tp, ffn_ep=tp)
    context = DistContext(
        world_size=config.world_size,
        attention_tp=config.attention_tp,
        ffn_ep=config.ffn_ep,
        initialize_meshes=False,
    )
    assert context.world_size == tp
    assert config.attn_world_size == config.ffn_world_size == tp
    assert config.hf_config.num_key_value_heads == 1


@pytest.mark.parametrize("parameter", ["attention_tp", "ffn_tp"])
@pytest.mark.parametrize("value", [0, -1])
def test_tp_sizes_must_be_positive(hf_config, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        Config(model="unused", **{parameter: value})


@pytest.mark.parametrize("field,count", [("num_attention_heads", 24), ("num_key_value_heads", 8)])
def test_tp16_rejects_incompatible_head_partitions(hf_config, field, count):
    setattr(hf_config, field, count)
    with pytest.raises(ValueError, match=rf"{field}={count}.*attention_tp=16"):
        Config(model="unused", attention_tp=16, ffn_tp=16)


def test_kimi_still_requires_query_head_divisibility(hf_config):
    hf_config.architectures = ["KimiK3ForConditionalGeneration"]
    hf_config.num_attention_heads = 24
    hf_config.num_key_value_heads = 1
    with pytest.raises(ValueError, match="num_attention_heads=24.*attention_tp=16"):
        Config(model="unused", attention_tp=16, ffn_ep=16)


@pytest.mark.parametrize(
    "architecture",
    [
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Glm5NextForConditionalGeneration",
    ],
)
def test_model_specific_tp1_constraints_remain(hf_config, architecture):
    hf_config.architectures = [architecture]
    with pytest.raises(ValueError, match="attention_tp"):
        Config(model="unused", attention_tp=16, ffn_ep=16)


def test_hisparse_tp1_constraint_remains(hf_config):
    with pytest.raises(ValueError, match="enable_hisparse requires attention_tp == 1"):
        Config(model="unused", attention_tp=16, ffn_tp=16, enable_hisparse=True)


def test_cli_accepts_tp16(hf_config):
    parser = ArgumentParser()
    parser.add_class_arguments(Config, fail_untyped=False)
    parsed = parser.parse_args(
        ["--model", "unused", "--attention_tp", "16", "--ffn_tp", "16"]
    )
    config = Config(**vars(parsed))
    assert config.attention_tp == config.ffn_tp == 16
