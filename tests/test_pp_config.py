from types import SimpleNamespace

import dlengine.config as config_module
import pytest
from dlengine.config import Config


def _qwen35_config():
    return SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        dtype="bfloat16",
        max_position_embeddings=32768,
        num_hidden_layers=2,
        num_key_value_heads=2,
    )


def _deepseek_config(architecture):
    return SimpleNamespace(
        architectures=[architecture],
        dtype="bfloat16",
        max_position_embeddings=32768,
        num_hidden_layers=30,
        num_key_value_heads=1,
    )


def test_pp_allows_hybrid_dlslime_ctrl_address(monkeypatch):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _qwen35_config(),
    )

    config = Config(
        model="unused",
        pp=2,
        mode="hybrid",
        executor_backend="dlslime",
        ctrl_address="127.0.0.1:4479",
        num_speculative_tokens=0,
    )

    assert config.ctrl_address == "http://127.0.0.1:4479"
    assert config.world_size == 2
    assert config.enforce_eager is True


def test_pp_still_rejects_pd_role_with_ctrl_address(monkeypatch):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _qwen35_config(),
    )

    with pytest.raises(ValueError, match="mode='hybrid' only"):
        Config(
            model="unused",
            pp=2,
            mode="prefill",
            executor_backend="dlslime",
            ctrl_address="127.0.0.1:4479",
            num_speculative_tokens=0,
        )


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"],
)
def test_pp16_allows_supported_deepseek_architectures(monkeypatch, architecture):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _deepseek_config(architecture),
    )

    config = Config(
        model="unused",
        pp=16,
        mode="hybrid",
        num_speculative_tokens=0,
    )

    assert config.world_size == 16
    assert config.kvcache_block_size == 64
    assert config.enforce_eager is True


def test_pp_rejects_deepseek_v32_until_indexer_cache_is_stage_local(monkeypatch):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _deepseek_config("DeepseekV32ForCausalLM"),
    )

    with pytest.raises(ValueError, match="currently only supported"):
        Config(
            model="unused",
            pp=16,
            mode="hybrid",
            num_speculative_tokens=0,
        )
