from types import SimpleNamespace

import pytest

from nanodeploy.config import Config


def _hf_config(architecture: str = "DeepseekV3ForCausalLM") -> SimpleNamespace:
    return SimpleNamespace(
        architectures=[architecture],
        max_position_embeddings=16_384,
        num_key_value_heads=1,
        num_attention_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        torch_dtype="bfloat16",
    )


@pytest.fixture(autouse=True)
def mock_auto_config(monkeypatch):
    monkeypatch.setattr(
        "nanodeploy.config.AutoConfig.from_pretrained",
        lambda *args, **kwargs: _hf_config(),
    )


def _ls_kwargs() -> dict:
    return {
        "enable_ls_decode_core_scheduler": True,
        "mode": "decode",
        "dummy_prefill": True,
        "scheduler_mode": "centralized",
        "loop_count": 1,
        "attention_dp": 4,
        "attention_sp": 8,
        "attention_tp": 1,
        "ffn_ep": 32,
        "ffn_dp": 1,
        "ffn_tp": 1,
        "use_dlslime_rpc": True,
        "sp_backend": "hao_basic",
        "fixed_sp_size": 0,
        "enable_dynamic_sp_size": False,
        "dynamic_sp_size_strategy": "legacy",
        "use_new_decode_dynamic_sp_scheduler": False,
        "enable_non_uniform_split": False,
        "sp_debug": False,
        "kvcache_block_size": 64,
    }


def test_ls_decode_core_supported_configuration(tmp_path):
    config = Config(
        model=str(tmp_path),
        **_ls_kwargs(),
        ls_decode_initial_kv_dop=8,
        ls_decode_batch_per_master=128,
        ls_decode_enable_memory_scale_up=False,
    )

    assert config.enable_ls_decode_core_scheduler is True
    assert config.ls_decode_initial_kv_dop == 8
    assert config.ls_decode_batch_per_master == 128
    assert config.ls_decode_enable_memory_scale_up is False


def test_kv_consolidation_p2p_scratch_is_opt_in(tmp_path):
    config = Config(model=str(tmp_path), **_ls_kwargs())
    assert config.ls_kv_consolidation_mode == "off"
    assert config.ls_kv_consolidation_candidate_util == 0.50
    assert config.ls_kv_consolidation_target_high_watermark == 0.80
    assert config.ls_kv_consolidation_stable_steps == 32
    assert config.ls_kv_consolidation_cooldown_steps == 64
    assert config.ls_kv_consolidation_check_interval_steps == 8
    assert config.ls_kv_consolidation_max_source_blocks_per_event == 0
    assert config.ls_kv_consolidation_migration_chunk_tokens == 0

    enabled = Config(
        model=str(tmp_path),
        **_ls_kwargs(),
        ls_kv_consolidation_migration_chunk_tokens=128,
    )
    assert enabled.ls_kv_consolidation_migration_chunk_tokens == 128

    with pytest.raises(ValueError, match="must be >= 0"):
        Config(
            model=str(tmp_path),
            **_ls_kwargs(),
            ls_kv_consolidation_migration_chunk_tokens=-1,
        )


def test_kv_consolidation_shadow_and_execute_validation(tmp_path):
    shadow = Config(
        model=str(tmp_path),
        **_ls_kwargs(),
        ls_kv_consolidation_mode="shadow",
    )
    assert shadow.ls_kv_consolidation_mode == "shadow"

    with pytest.raises(ValueError, match="migration_chunk_tokens > 0"):
        Config(
            model=str(tmp_path),
            **_ls_kwargs(),
            ls_kv_consolidation_mode="execute",
        )

    with pytest.raises(ValueError, match="max_source_blocks_per_event > 0"):
        Config(
            model=str(tmp_path),
            **_ls_kwargs(),
            ls_kv_consolidation_mode="execute",
            ls_kv_consolidation_migration_chunk_tokens=128,
        )

    execute = Config(
        model=str(tmp_path),
        **_ls_kwargs(),
        ls_kv_consolidation_mode="execute",
        ls_kv_consolidation_migration_chunk_tokens=128,
        ls_kv_consolidation_max_source_blocks_per_event=16,
    )
    assert execute.ls_kv_consolidation_mode == "execute"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"ls_kv_consolidation_candidate_util": 0.0}, r"must be in \(0, 1\]"),
        ({"ls_kv_consolidation_candidate_util": 1.1}, r"must be in \(0, 1\]"),
        ({"ls_kv_consolidation_target_high_watermark": 0.0}, r"must be in \(0, 1\]"),
        ({"ls_kv_consolidation_target_high_watermark": 1.1}, r"must be in \(0, 1\]"),
        ({"ls_kv_consolidation_stable_steps": 0}, "must be > 0"),
        ({"ls_kv_consolidation_cooldown_steps": -1}, "must be >= 0"),
        ({"ls_kv_consolidation_check_interval_steps": 0}, "must be > 0"),
        ({"ls_kv_consolidation_max_source_blocks_per_event": -1}, "must be >= 0"),
    ],
)
def test_kv_consolidation_threshold_validation(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        Config(model=str(tmp_path), kvcache_block_size=64, **override)


def test_kv_consolidation_non_off_mode_requires_ls_scheduler(tmp_path):
    with pytest.raises(ValueError, match="requires enable_ls_decode_core_scheduler"):
        Config(
            model=str(tmp_path),
            kvcache_block_size=64,
            ls_kv_consolidation_mode="shadow",
        )


def test_ls_decode_core_supports_single_node_8gpu_preflight(tmp_path):
    kwargs = _ls_kwargs()
    kwargs.update(attention_dp=1, ffn_ep=8)

    config = Config(model=str(tmp_path), **kwargs)

    assert config.attn_world_size == 8
    assert config.ffn_world_size == 8


def test_ls_decode_core_supports_two_node_16gpu_validation(tmp_path):
    kwargs = _ls_kwargs()
    kwargs.update(attention_dp=2, ffn_ep=16)

    config = Config(model=str(tmp_path), **kwargs)

    assert config.attn_world_size == 16
    assert config.ffn_world_size == 16


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mode": "hybrid"}, "mode must be 'decode'"),
        ({"dummy_prefill": False}, "dummy_prefill must be True"),
        ({"scheduler_mode": "decentralized"}, "scheduler_mode must be 'centralized'"),
        ({"loop_count": 2}, "loop_count must be 1"),
        ({"attention_dp": 1}, "parallel topology must be"),
        ({"use_dlslime_rpc": False}, "use_dlslime_rpc must be True"),
        ({"sp_backend": "legacy_ll"}, "sp_backend must be 'hao_basic'"),
        ({"fixed_sp_size": 1}, "fixed_sp_size must be 0"),
        ({"enable_dynamic_sp_size": True}, "enable_dynamic_sp_size must be False"),
        (
            {"dynamic_sp_size_strategy": "long_short_sp8"},
            "dynamic_sp_size_strategy must be 'legacy'",
        ),
        (
            {"use_new_decode_dynamic_sp_scheduler": True},
            "use_new_decode_dynamic_sp_scheduler must be False",
        ),
        ({"enable_non_uniform_split": True}, "enable_non_uniform_split must be False"),
        ({"sp_debug": True}, "sp_debug must be False"),
    ],
)
def test_ls_decode_core_rejects_unsupported_combinations(tmp_path, override, message):
    kwargs = _ls_kwargs()
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        Config(model=str(tmp_path), **kwargs)


def test_ls_decode_core_requires_deepseek_v3(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nanodeploy.config.AutoConfig.from_pretrained",
        lambda *args, **kwargs: _hf_config("Qwen3ForCausalLM"),
    )

    with pytest.raises(ValueError, match="only supports the DeepseekV3ForCausalLM"):
        Config(model=str(tmp_path), **_ls_kwargs())


@pytest.mark.parametrize("initial_kv_dop", [-1, 9])
def test_ls_decode_initial_kv_dop_range_is_always_validated(
    tmp_path, initial_kv_dop
):
    with pytest.raises(ValueError, match=r"must be in \[0, attention_sp\]"):
        Config(
            model=str(tmp_path),
            attention_sp=8,
            kvcache_block_size=64,
            ls_decode_initial_kv_dop=initial_kv_dop,
        )


def test_ls_decode_batch_per_master_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="ls_decode_batch_per_master must be > 0"):
        Config(
            model=str(tmp_path),
            kvcache_block_size=64,
            ls_decode_batch_per_master=0,
        )


def test_reserved_blocks_per_request_must_be_non_negative(tmp_path):
    with pytest.raises(ValueError, match="reserved_blocks_per_req must be >= 0"):
        Config(
            model=str(tmp_path),
            kvcache_block_size=64,
            reserved_blocks_per_req=-0.1,
        )


def test_ls_decode_feature_flag_defaults_off(tmp_path):
    config = Config(model=str(tmp_path), kvcache_block_size=64)

    assert config.enable_ls_decode_core_scheduler is False
    assert config.ls_decode_initial_kv_dop == 0
    assert config.ls_decode_batch_per_master == 64
    assert config.ls_decode_enable_memory_scale_up is True
