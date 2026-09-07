from types import SimpleNamespace

import pytest
import torch
from dlengine.config import Config
from dlengine.runtime.context.cache.mla import (
    get_mla_block_bytes,
    resolve_mla_cache_format,
)
from dlengine.runtime.context.cache.plan import (
    deepseek_mla_cache_plan,
    gqa_cache_plan,
    kimi_k3_cache_plan,
)
from jsonargparse import ArgumentParser


def _config(dtype="auto", **overrides):
    values = dict(
        kv_cache_dtype=dtype,
        disable_nsa=False,
        enable_mla_reference_fallback=False,
        hf_config=SimpleNamespace(
            kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128
        ),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("auto", (False, False)),
        ("bfloat16", (False, False)),
        ("fp8_e4m3", (True, True)),
    ],
)
def test_k3_dense_cache_selection(dtype, expected):
    assert (
        resolve_mla_cache_format(_config(dtype), kimi_k3_cache_plan(), "blackwell")
        == expected
    )


@pytest.mark.parametrize("hardware,raw", [("blackwell", True), ("hopper", False)])
@pytest.mark.parametrize("dtype", ["auto", "fp8_e4m3"])
def test_sparse_cache_layout_is_preserved(hardware, raw, dtype):
    plan = deepseek_mla_cache_plan(use_indexer=True)
    assert resolve_mla_cache_format(_config(dtype), plan, hardware) == (True, raw)


def test_disable_nsa_does_not_disable_explicit_dense_blackwell_fp8():
    plan = deepseek_mla_cache_plan(use_indexer=True)
    assert resolve_mla_cache_format(_config(disable_nsa=True), plan, "blackwell") == (
        False,
        False,
    )
    assert resolve_mla_cache_format(
        _config("fp8_e4m3", disable_nsa=True), plan, "blackwell"
    ) == (True, True)


@pytest.mark.parametrize("hardware", ["hopper", "gpu_generic"])
def test_dense_fp8_rejects_unsupported_hardware(hardware):
    with pytest.raises(ValueError, match="requires Blackwell"):
        resolve_mla_cache_format(_config("fp8_e4m3"), kimi_k3_cache_plan(), hardware)


@pytest.mark.parametrize("rank,rope", [(256, 256), (512, 0)])
def test_explicit_fp8_rejects_unsupported_cache_shape(rank, rope):
    cfg = _config(
        "fp8_e4m3", hf_config=SimpleNamespace(kv_lora_rank=rank, qk_rope_head_dim=rope)
    )
    with pytest.raises(ValueError, match="kv_lora_rank=512"):
        resolve_mla_cache_format(cfg, kimi_k3_cache_plan(), "blackwell")


def test_reference_decode_preserves_auto_fallback_and_rejects_explicit_fp8():
    plan = deepseek_mla_cache_plan(use_indexer=True)
    cfg = _config(enable_mla_reference_fallback=True)
    cfg.hf_config.kv_lora_rank = 256
    cfg.hf_config.qk_rope_head_dim = 64
    assert resolve_mla_cache_format(cfg, plan, "hopper") == (False, False)
    cfg.kv_cache_dtype = "fp8_e4m3"
    with pytest.raises(ValueError, match="reference decode"):
        resolve_mla_cache_format(cfg, plan, "blackwell")


def test_forced_reference_rejects_explicit_fp8_even_for_native_dimensions():
    with pytest.raises(ValueError, match="reference decode"):
        resolve_mla_cache_format(
            _config("fp8_e4m3"), kimi_k3_cache_plan(), "blackwell", force_reference=True
        )


def test_bf16_does_not_silently_disable_sparse_decode():
    plan = deepseek_mla_cache_plan(use_indexer=True)
    with pytest.raises(ValueError, match="disable_nsa=True"):
        resolve_mla_cache_format(_config("bfloat16"), plan, "blackwell")


def test_hisparse_still_requires_fp8():
    plan = deepseek_mla_cache_plan(use_hisparse=True)
    with pytest.raises(ValueError, match="HiSparse requires FP8"):
        resolve_mla_cache_format(_config("bfloat16"), plan, "blackwell")


def test_non_mla_explicit_dtype_is_rejected():
    assert resolve_mla_cache_format(_config(), gqa_cache_plan(), "blackwell") == (
        False,
        False,
    )
    with pytest.raises(ValueError, match="only for MLA"):
        resolve_mla_cache_format(_config("fp8_e4m3"), gqa_cache_plan(), "blackwell")


def test_k3_fp8_page_accounting_halves_only_mla_bytes():
    context = SimpleNamespace(
        num_hidden_layers=24,
        block_size=64,
        num_local_kv_heads=1,
        head_dim=576,
        dtype=torch.bfloat16,
        is_fp8_kvcache=False,
        raw_fp8_mla_layout=False,
    )
    bf16_bytes = get_mla_block_bytes(context)
    context.is_fp8_kvcache, context.raw_fp8_mla_layout = resolve_mla_cache_format(
        _config("fp8_e4m3"), kimi_k3_cache_plan(), "blackwell"
    )
    assert get_mla_block_bytes(context) * 2 == bf16_bytes
    assert context._fp8_head_dim == 576


@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8_e4m3"])
def test_cli_preserves_cache_dtype_for_worker_serialization(dtype):
    parser = ArgumentParser()
    parser.add_class_arguments(Config, fail_untyped=False)
    parsed = parser.parse_args(["--model", "unused", "--kv_cache_dtype", dtype])
    config = Config.model_construct(**vars(parsed))
    assert config.model_dump()["kv_cache_dtype"] == dtype


def test_runner_passes_k3_fp8_format_to_cache_allocation(monkeypatch):
    from dlengine.runtime.context.cache import CacheContext
    from dlengine.runtime.models import pp_utils
    from dlengine.runtime.runner import model_runner

    hf = SimpleNamespace(
        architectures=["KimiK3ForConditionalGeneration"],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        hidden_size=7168,
        num_attention_heads=96,
        num_key_value_heads=96,
        num_hidden_layers=93,
        layer_types=["full_attention"] * 24 + ["linear_attention"] * 69,
    )
    config = Config.model_construct(
        model="unused", hf_config=hf, kv_cache_dtype="fp8_e4m3"
    )
    runner_cls = model_runner.ModelRunner.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner.config = config
    runner.cache_plan = kimi_k3_cache_plan()
    runner.mtp_runner = None
    runner._mla_cache_format = resolve_mla_cache_format(
        config, runner.cache_plan, "blackwell"
    )
    monkeypatch.setattr(pp_utils, "get_pp_layer_range", lambda _count: (0, 93))
    monkeypatch.setattr(
        CacheContext, "estimate_gdn_state_bytes", lambda *args, **kw: 123456
    )
    captured = {}

    class AllocationReached(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise AllocationReached

    monkeypatch.setattr(model_runner, "set_cache_context", capture)
    with pytest.raises(AllocationReached):
        runner.preallocate_kvcache()
    assert captured["is_fp8_kvcache"] is True
    assert captured["raw_fp8_mla_layout"] is True
    assert captured["num_hidden_layers"] == 24
    assert captured["reserved_state_bytes"] >= 123456
