from types import SimpleNamespace

import dlengine.config as config_module
import pytest
from dlengine.config import Config
from jsonargparse import ArgumentParser


@pytest.fixture
def hf(monkeypatch):
    hf = SimpleNamespace(
        architectures=["KimiK3ForConditionalGeneration"],
        dtype="bfloat16",
        max_position_embeddings=1048576,
        num_hidden_layers=2,
        num_attention_heads=96,
        num_key_value_heads=1,
    )
    monkeypatch.setattr(
        config_module.AutoConfig, "from_pretrained", lambda *a, **kw: hf
    )
    monkeypatch.setattr(
        config_module.PretrainedConfig, "get_config_dict", lambda *a, **kw: ({}, {})
    )
    monkeypatch.setattr(config_module.torch.cuda, "is_available", lambda: False)
    return hf


@pytest.mark.parametrize(
    "chunk,expected",
    [(1, 256), (8192, 1024), (16384, 2048), (32768, 4096), (8193, 1025)],
)
@pytest.mark.parametrize("dp", [1, 2, 4])
def test_auto_capacity_covers_one_busy_dp_group(hf, chunk, expected, dp):
    config = Config(
        model="unused",
        attention_tp=8,
        attention_dp=dp,
        ffn_ep=dp * 8,
        max_num_batched_tokens=chunk,
    )
    assert config.mega_moe_max_tokens_per_rank == expected


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_auto_capacity_follows_actual_attention_row_sharding(hf, tp):
    config = Config(
        model="unused",
        attention_tp=tp,
        ffn_ep=max(2, tp),
        attention_dp=2 if tp == 1 else 1,
        max_num_batched_tokens=16384,
    )
    assert config.mega_moe_max_tokens_per_rank == 16384 // tp


def test_total_context_does_not_inflate_forward_capacity(hf):
    config = Config(
        model="unused",
        attention_tp=8,
        ffn_ep=8,
        pp_prefill_scheduler_depth=32,
        max_num_batched_tokens=8192,
        max_model_len=1048576,
    )
    assert config.mega_moe_max_tokens_per_rank == 1024


def test_capacity_also_covers_decode_and_graph_batch(hf):
    config = Config(
        model="unused",
        attention_tp=8,
        ffn_ep=8,
        max_num_batched_tokens=128,
        max_num_seqs=4096,
    )
    assert config.mega_moe_max_tokens_per_rank == 512


@pytest.mark.parametrize("capacity", [2048, 4096])
def test_sufficient_explicit_capacity_is_preserved(hf, capacity):
    config = Config(
        model="unused",
        attention_tp=8,
        ffn_ep=8,
        max_num_batched_tokens=16384,
        mega_moe_max_tokens_per_rank=capacity,
    )
    assert config.mega_moe_max_tokens_per_rank == capacity


def test_undersized_explicit_capacity_fails_during_config_validation(hf):
    with pytest.raises(ValueError, match="at least 2048"):
        Config(
            model="unused",
            attention_tp=8,
            ffn_ep=8,
            max_num_batched_tokens=16384,
            mega_moe_max_tokens_per_rank=1024,
        )


def test_negative_capacity_is_rejected(hf):
    with pytest.raises(ValueError, match="mega_moe_max_tokens_per_rank"):
        Config(model="unused", mega_moe_max_tokens_per_rank=-1)


def test_non_k3_default_is_preserved(hf):
    hf.architectures = ["Qwen3ForCausalLM"]
    assert Config(model="unused").mega_moe_max_tokens_per_rank == 256


def test_cli_auto_capacity_survives_worker_serialization(hf):
    parser = ArgumentParser()
    parser.add_class_arguments(Config, fail_untyped=False)
    args = parser.parse_args(
        [
            "--model",
            "unused",
            "--attention_tp",
            "8",
            "--ffn_ep",
            "8",
            "--max_num_batched_tokens",
            "16384",
        ]
    )
    config = Config(**vars(args))
    assert config.model_dump()["mega_moe_max_tokens_per_rank"] == 2048
