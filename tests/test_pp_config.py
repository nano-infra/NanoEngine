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
