import re
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nanodeploy.config import Config
from nanodeploy.worker.decode_backend_compat import (
    DEEP_EP_BUFFER_REQUIRED_SYMBOLS,
    DEEP_EP_REQUIRED_SYMBOLS,
    DEEP_GEMM_REQUIRED_SYMBOLS,
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
        **{name: object() for name in DEEP_GEMM_REQUIRED_SYMBOLS},
    )

    class DeepEPBuffer:
        pass

    for name in DEEP_EP_BUFFER_REQUIRED_SYMBOLS:
        setattr(DeepEPBuffer, name, object())

    deep_ep_symbols = {
        name: object() for name in DEEP_EP_REQUIRED_SYMBOLS
    }
    deep_ep_symbols["Buffer"] = DeepEPBuffer

    return {
        "deep_gemm": deep_gemm,
        "deep_ep": _module("deep_ep", **deep_ep_symbols),
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


def test_backend_contract_has_only_target_native_versions():
    assert EXPECTED_BACKEND_VERSIONS == {
        "deep_gemm": "2.3.0+477618c",
        "deep_ep": "1.2.1+73b6ea4",
    }


@pytest.mark.parametrize(
    ("distribution", "actual", "expected"),
    [
        ("deep_gemm", "2.1.1+c9f8b34", "2.3.0+477618c"),
        ("deep_ep", "1.2.1+9af0e0d", "1.2.1+73b6ea4"),
    ],
)
def test_backend_version_mismatch_reports_rank_and_versions(
    distribution, actual, expected
):
    versions = dict(EXPECTED_BACKEND_VERSIONS)
    versions[distribution] = actual

    with pytest.raises(
        RuntimeError,
        match=(
            rf"rank 7.*{distribution}: expected {re.escape(expected)}, "
            rf"got {re.escape(actual)}"
        ),
    ):
        _validate_with(versions, _backend_modules())


def test_backend_validation_imports_only_native_modules():
    modules = _backend_modules()
    imported = []

    def module_importer(name):
        imported.append(name)
        return modules[name]

    validate_decode_backend_compat(
        7,
        version_getter=EXPECTED_BACKEND_VERSIONS.__getitem__,
        module_importer=module_importer,
    )

    assert imported == ["deep_gemm", "deep_ep"]


@pytest.mark.parametrize(
    "symbol",
    [
        "fp8_gemm_nt",
        "m_grouped_fp8_gemm_nt_contiguous",
        "m_grouped_fp8_gemm_nt_masked",
        "get_mk_alignment_for_contiguous_layout",
        "transform_sf_into_required_layout",
    ],
)
def test_missing_deep_gemm_native_symbol_fails_before_model_execution(symbol):
    modules = _backend_modules()
    delattr(modules["deep_gemm"], symbol)

    with pytest.raises(
        RuntimeError, match=rf"deep_gemm missing symbols:.*{symbol}"
    ):
        _validate_with(dict(EXPECTED_BACKEND_VERSIONS), modules)


@pytest.mark.parametrize("symbol", ["topk_idx_t", "Config", "EventOverlap"])
def test_missing_deep_ep_native_module_symbol_fails(symbol):
    modules = _backend_modules()
    delattr(modules["deep_ep"], symbol)

    with pytest.raises(
        RuntimeError, match=rf"deep_ep missing symbols:.*{symbol}"
    ):
        _validate_with(dict(EXPECTED_BACKEND_VERSIONS), modules)


@pytest.mark.parametrize(
    "symbol",
    [
        "get_dispatch_layout",
        "dispatch",
        "combine",
        "low_latency_dispatch",
        "low_latency_combine",
        "clean_low_latency_buffer",
        "get_low_latency_rdma_size_hint",
        "destroy",
    ],
)
def test_missing_deep_ep_buffer_ll_or_ht_symbol_fails(symbol):
    modules = _backend_modules()
    delattr(modules["deep_ep"].Buffer, symbol)

    with pytest.raises(
        RuntimeError, match=rf"deep_ep.Buffer missing symbols:.*{symbol}"
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


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_nvshmem_qp_depth_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="NVSHMEM_QP_DEPTH"):
        resolve_decode_deepep_config(
            256, {"NVSHMEM_QP_DEPTH": value}
        )


def test_nvshmem_qp_depth_must_cover_max_tokens_per_rank():
    with pytest.raises(
        ValueError,
        match=r"got 1024 < 1026 for max_tokens_per_rank=512",
    ):
        resolve_decode_deepep_config(
            256,
            {
                "DEEPEP_MAX_TOKENS_PER_RANK": "512",
                "NVSHMEM_QP_DEPTH": "1024",
            },
        )


@pytest.mark.parametrize(
    ("max_tokens_per_rank", "expected_depth"),
    [(256, 1024), (511, 1024), (512, 2048), (1024, 4096)],
)
def test_nvshmem_qp_depth_default_is_normalized(
    max_tokens_per_rank, expected_depth
):
    config = resolve_decode_deepep_config(
        256,
        {"DEEPEP_MAX_TOKENS_PER_RANK": str(max_tokens_per_rank)},
    )

    assert config.nvshmem_qp_depth == expected_depth
    assert config.worker_env()["NVSHMEM_QP_DEPTH"] == str(expected_depth)


def test_explicit_sufficient_nvshmem_qp_depth_is_preserved():
    config = resolve_decode_deepep_config(
        256, {"NVSHMEM_QP_DEPTH": "1536"}
    )

    assert config.nvshmem_qp_depth == 1536


def test_decode_backend_rejects_non_auto_deepep_mode():
    with pytest.raises(
        ValueError, match="native DeepEP decode backend requires DEEPEP_MODE=auto"
    ):
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
        "NVSHMEM_QP_DEPTH": "1024",
    }


def test_worker_env_passes_through_gemm_debug_settings():
    environ = {
        "DG_PRINT_CONFIGS": "1",
        "DG_JIT_DEBUG": "0",
        "NANODEPLOY_MOE_GEMM_DEBUG": "1",
        "NANODEPLOY_MOE_GEMM_DEBUG_RANKS": "0,8",
        "NANODEPLOY_MOE_GEMM_DEBUG_LAYERS": "1",
        "NANODEPLOY_MOE_GEMM_DEBUG_GEMMS": "gate_up",
        "NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS": "2",
        "NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS": "8",
    }

    worker_env = build_decode_backend_worker_env(32, environ)

    assert worker_env == {
        "DEEPEP_SMS": "16",
        "DEEPEP_MAX_TOKENS_PER_RANK": "32",
        "DEEPEP_ENABLE_MNNVL": "0",
        "NVSHMEM_QP_DEPTH": "1024",
        **environ,
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
        ("NVSHMEM_QP_DEPTH", "2048"),
    ],
)
def test_collective_fingerprint_changes_with_deepep_config(
    monkeypatch, name, value
):
    for env_name in (
        "DEEPEP_SMS",
        "DEEPEP_MAX_TOKENS_PER_RANK",
        "DEEPEP_ENABLE_MNNVL",
        "NVSHMEM_QP_DEPTH",
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


def test_dense_fp8_wrapper_uses_only_deep_gemm_native_api():
    source = (
        REPOSITORY_ROOT / "nanodeploy/kernels/block_gemm_fp8.py"
    ).read_text()

    assert "fp8_gemm_nt" in source
    assert "gemm_fp8_fp8_bf16_nt" not in source
