from types import SimpleNamespace

import dlengine.config as config_module
import pytest
from dlengine.config import Config
from dlengine.engine.scheduler import scheduler_token_budget


@pytest.fixture(autouse=True)
def _mock_raw_config(monkeypatch):
    monkeypatch.setattr(
        config_module.PretrainedConfig,
        "get_config_dict",
        lambda *args, **kwargs: ({}, {}),
    )


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


def _gemma4_config():
    return SimpleNamespace(
        architectures=["Gemma4ForCausalLM"],
        dtype="bfloat16",
        max_position_embeddings=32768,
        num_hidden_layers=12,
        num_key_value_heads=1,
        head_dim=128,
        global_head_dim=128,
        layer_types=["sliding_attention", "full_attention"] * 6,
        num_kv_shared_layers=4,
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


def test_pp_allows_prefill_pd_role_with_ctrl_address(monkeypatch):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _qwen35_config(),
    )

    config = Config(
        model="unused",
        pp=2,
        mode="prefill",
        executor_backend="dlslime",
        ctrl_address="127.0.0.1:4479",
        num_speculative_tokens=0,
    )

    assert config.mode == "prefill"
    assert config.world_size == 2


def test_pp_uses_max_batched_tokens_as_prefill_microbatch_size(monkeypatch):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _qwen35_config(),
    )

    config = Config(
        model="unused",
        pp=2,
        mode="prefill",
        max_num_batched_tokens=1024,
        pp_prefill_pipeline_depth=2,
        num_speculative_tokens=0,
    )

    assert config.max_num_batched_tokens == 1024
    assert scheduler_token_budget(config) == 2048
    assert config.pp_prefill_pipeline_depth == 2


def test_pp_still_rejects_decode_pd_role(monkeypatch):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _qwen35_config(),
    )

    with pytest.raises(ValueError, match="PP prefill.*pp=1 decode"):
        Config(
            model="unused",
            pp=2,
            mode="decode",
            num_speculative_tokens=0,
        )


@pytest.mark.parametrize(
    "architecture",
    [
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
    ],
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
    assert scheduler_token_budget(config) == 16384 * 16
    assert config.kvcache_block_size == 64
    assert config.enforce_eager is True


def test_pp16_allows_deepseek_v4(monkeypatch):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _deepseek_config("DeepseekV4ForCausalLM"),
    )

    config = Config(
        model="unused",
        pp=16,
        mode="hybrid",
        num_speculative_tokens=0,
    )

    assert config.world_size == 16
    assert config.enforce_eager is True


@pytest.mark.parametrize(
    "architecture", ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"]
)
def test_pp4_allows_gemma4_with_shared_kv(monkeypatch, architecture):
    hf_config = _gemma4_config()
    hf_config.architectures = [architecture]
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: hf_config,
    )

    config = Config(
        model="unused",
        pp=4,
        mode="hybrid",
        num_speculative_tokens=0,
    )

    assert config.world_size == 4
    assert config.enforce_eager is True


def test_pp4_allows_gemma4_hisparse_cache_layout(monkeypatch):
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _gemma4_config(),
    )

    config = Config(
        model="unused",
        pp=4,
        mode="prefill",
        enable_hisparse=True,
        num_speculative_tokens=0,
    )

    assert config.enable_hisparse is True
    assert config.world_size == 4


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_glm_multistep_mtp_allows_pd_roles(monkeypatch, mode):
    hf_config = _deepseek_config("GlmMoeDsaForCausalLM")
    hf_config.num_nextn_predict_layers = 1
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: hf_config,
    )

    config = Config(
        model="unused",
        mode=mode,
        num_speculative_tokens=5,
        max_num_seqs=8,
    )

    assert config.mode == mode
    assert config.num_speculative_tokens == 5


def test_glm_multistep_mtp_allows_pp_prefill(monkeypatch):
    hf_config = _deepseek_config("GlmMoeDsaForCausalLM")
    hf_config.num_nextn_predict_layers = 1
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: hf_config,
    )

    config = Config(
        model="unused",
        pp=8,
        mode="prefill",
        num_speculative_tokens=5,
        max_num_seqs=8,
    )

    assert config.pp == 8
    assert config.mode == "prefill"
    assert config.num_speculative_tokens == 5
    assert config.enforce_eager is True


@pytest.mark.parametrize("mode", ["hybrid", "decode"])
def test_glm_multistep_mtp_rejects_pp_outside_prefill(monkeypatch, mode):
    hf_config = _deepseek_config("GlmMoeDsaForCausalLM")
    hf_config.num_nextn_predict_layers = 1
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: hf_config,
    )

    with pytest.raises(
        ValueError,
        match="supported only for GLM prefill",
    ):
        Config(
            model="unused",
            pp=8,
            mode=mode,
            num_speculative_tokens=5,
            max_num_seqs=8,
        )
