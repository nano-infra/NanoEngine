from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nanodeploy.config import Config
from nanodeploy.worker.decode_backend_compat import (
    EXPECTED_BACKEND_VERSIONS,
    build_decode_backend_worker_env,
    resolve_decode_deepep_config,
    validate_decode_backend_compat,
    validate_deepseek_decode_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_MODEL = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3"
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _backend_modules() -> dict[str, ModuleType]:
    deep_gemm = _module(
        "deep_gemm",
        fp8_gemm_nt=object(),
        m_grouped_fp8_gemm_nt_masked=object(),
    )

    class DeepEPBuffer:
        set_num_sms = object()
        destroy = object()
        low_latency_dispatch = object()
        low_latency_combine = object()

    class DlBLASDeepEPBuffer:
        set_explicitly_destroy = object()
        destroy = object()

    return {
        "deep_gemm": deep_gemm,
        "deep_ep": _module("deep_ep", Buffer=DeepEPBuffer),
        "dlblas.layers.moe.token_dispatcher": _module(
            "dlblas.layers.moe.token_dispatcher",
            DeepEPBuffer=DlBLASDeepEPBuffer,
        ),
    }


def _validate_with(
    versions: dict[str, str], modules: dict[str, ModuleType]
) -> dict[str, str]:
    return validate_decode_backend_compat(
        7,
        version_getter=versions.__getitem__,
        module_importer=modules.__getitem__,
    )


def test_expected_decode_backend_versions_and_symbols_pass():
    actual = _validate_with(
        dict(EXPECTED_BACKEND_VERSIONS), _backend_modules()
    )

    assert actual == EXPECTED_BACKEND_VERSIONS


def test_dlblas_version_mismatch_reports_rank_and_versions():
    versions = dict(EXPECTED_BACKEND_VERSIONS)
    versions["dlblas"] = "0.0.5"

    with pytest.raises(
        RuntimeError,
        match=r"rank 7.*dlblas: expected 0\.0\.7, got 0\.0\.5",
    ):
        _validate_with(versions, _backend_modules())


def test_missing_deep_gemm_v2_symbol_fails_before_model_execution():
    modules = _backend_modules()
    delattr(modules["deep_gemm"], "fp8_gemm_nt")

    with pytest.raises(RuntimeError, match="deep_gemm missing symbols: fp8_gemm_nt"):
        _validate_with(dict(EXPECTED_BACKEND_VERSIONS), modules)


def test_missing_deepep_explicit_destroy_symbol_fails():
    modules = _backend_modules()
    buffer_type = modules[
        "dlblas.layers.moe.token_dispatcher"
    ].DeepEPBuffer
    delattr(buffer_type, "destroy")

    with pytest.raises(
        RuntimeError, match="dlBLAS DeepEPBuffer missing symbols: destroy"
    ):
        _validate_with(dict(EXPECTED_BACKEND_VERSIONS), modules)


@pytest.mark.parametrize("value", ["0", "3", "-2", "not-an-int"])
def test_deepep_sms_requires_a_positive_even_integer(value):
    with pytest.raises(ValueError, match="DEEPEP_SMS"):
        resolve_decode_deepep_config(256, {"DEEPEP_SMS": value})


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_deepep_max_tokens_per_rank_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="DEEPEP_MAX_TOKENS_PER_RANK"):
        resolve_decode_deepep_config(
            256, {"DEEPEP_MAX_TOKENS_PER_RANK": value}
        )


def test_deepep_max_tokens_per_rank_must_cover_decode_batch():
    with pytest.raises(ValueError, match=r"128 < 256"):
        resolve_decode_deepep_config(
            256, {"DEEPEP_MAX_TOKENS_PER_RANK": "128"}
        )


def test_decode_backend_rejects_non_auto_deepep_mode():
    with pytest.raises(ValueError, match="requires DEEPEP_MODE=auto"):
        resolve_decode_deepep_config(256, {"DEEPEP_MODE": "low_latency"})


@pytest.mark.parametrize("value", ["-1", "2", "not-an-int"])
def test_deepep_mnnvl_flag_requires_zero_or_one(value):
    with pytest.raises(ValueError, match="DEEPEP_ENABLE_MNNVL"):
        resolve_decode_deepep_config(256, {"DEEPEP_ENABLE_MNNVL": value})


def test_worker_env_contains_effective_deepep_defaults():
    assert build_decode_backend_worker_env(256, {}) == {
        "DEEPEP_SMS": "16",
        "DEEPEP_MAX_TOKENS_PER_RANK": "256",
        "DEEPEP_ENABLE_MNNVL": "0",
    }


@pytest.mark.skipif(
    not (DEEPSEEK_MODEL / "config.json").is_file(),
    reason=f"DeepSeek-V3 config not found at {DEEPSEEK_MODEL}",
)
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DEEPEP_SMS", "18"),
        ("DEEPEP_MAX_TOKENS_PER_RANK", "512"),
        ("DEEPEP_ENABLE_MNNVL", "1"),
    ],
)
def test_collective_fingerprint_changes_with_deepep_config(
    monkeypatch, name, value
):
    for env_name in (
        "DEEPEP_SMS",
        "DEEPEP_MAX_TOKENS_PER_RANK",
        "DEEPEP_ENABLE_MNNVL",
        "DEEPEP_MODE",
    ):
        monkeypatch.delenv(env_name, raising=False)
    config = Config(
        model=str(DEEPSEEK_MODEL),
        mode="decode",
        kvcache_block_size=64,
        max_num_seqs=256,
        ffn_ep=32,
    )
    baseline = config.collective_fingerprint()

    monkeypatch.setenv(name, value)

    assert config.collective_fingerprint() != baseline


@pytest.mark.parametrize(
    ("num_experts", "ep_size"), [(256, 32), (384, 16)]
)
def test_deepseek_and_kimi_expert_topologies_pass(num_experts, ep_size):
    hf_config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        n_routed_experts=num_experts,
        num_experts_per_tok=8,
        quantization_config={"weight_block_size": [128, 128]},
    )

    validate_deepseek_decode_contract(hf_config, ep_size)


def test_deepseek_uniform_distribution_contract_is_preserved():
    source = (
        REPOSITORY_ROOT / "nanodeploy/models/deepseek_v2.py"
    ).read_text()

    assert 'self.distribution = "uniform"' in source
    assert 'if self.distribution == "uniform":' in source
    assert "selected_experts = torch.randint(" in source


def test_dense_fp8_wrapper_uses_only_deep_gemm_v2_api():
    source = (
        REPOSITORY_ROOT / "nanodeploy/kernels/block_gemm_fp8.py"
    ).read_text()

    assert "fp8_gemm_nt" in source
    assert "gemm_fp8_fp8_bf16_nt" not in source
