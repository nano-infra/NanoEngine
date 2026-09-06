"""Policy-provider factory tests.

Assert that hardware tiers are pure policy providers: a single
``PolicyBackendFactory`` is parameterised by a tier key into ``TIER_POLICIES``,
which drives the linear/experts implementation families (matching the
pre-refactor per-tier hardcoding). There are no per-tier factory classes.
"""

import pytest
from dlengine.runtime.layers.backend_policy import TIER_POLICIES
from dlengine.runtime.layers.policy_backend import PolicyBackendFactory
from dlengine.runtime.models.quant_config import QuantizationConfig


@pytest.mark.parametrize(
    ("hardware", "linear", "experts", "quant_override", "fp8"),
    [
        ("gpu_generic", "generic", "generic", False, False),
        ("hopper", "deepseek", "deepseek", False, True),
        ("blackwell", "deepseek", "deepseek", True, True),
    ],
)
def test_factory_pins_tier_policy(hardware, linear, experts, quant_override, fp8):
    factory = PolicyBackendFactory(QuantizationConfig(), tier=hardware)

    assert factory.hardware_backend == hardware
    assert factory.policy is TIER_POLICIES[hardware]
    assert factory.policy.linear == linear
    assert factory.policy.experts == experts
    assert factory.policy.experts_quant_override is quant_override
    assert factory.policy.supports_fp8 is fp8
    # Ref fallback defaults to disabled; set by create_backend from config.
    assert factory.ref_fallback_allowed is False


def test_factory_exposes_kimi_delta_attention_contract():
    # KDA is reached through the factory contract, not a direct model import.
    from dlengine.runtime.layers.base_backend import BackendFactory

    assert hasattr(BackendFactory, "get_kimi_delta_attention")
    factory = PolicyBackendFactory(QuantizationConfig(), tier="gpu_generic")
    assert callable(factory.get_kimi_delta_attention)


def test_experts_ref_fallback_when_deepseek_unavailable(monkeypatch):
    # With ref_fallback_allowed, a failing deepseek experts import degrades to
    # the generic experts instead of raising.
    from dlengine.runtime.layers.backends import selector

    sentinel = object()

    def _boom(**_):
        raise RuntimeError("deepseek experts unavailable in this build")

    def _generic(**_):
        return sentinel

    # Force the deepseek path to fail and capture the generic fallback.
    import dlengine.runtime.layers.backends.experts.deep_gemm as ds

    monkeypatch.setattr(ds, "DeepGemmExperts", _boom)
    monkeypatch.setattr(selector, "_create_generic_experts", _generic)

    with pytest.raises(RuntimeError):
        selector.create_experts(
            family="deepseek", ref_fallback_allowed=False, hidden_size=8
        )

    got = selector.create_experts(
        family="deepseek", ref_fallback_allowed=True, hidden_size=8
    )
    assert got is sentinel


def test_models_do_not_import_kda_implementation_directly():
    # Topologies must depend on the abstract contract, not the KDA impl.
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "dlengine" / "runtime"
    offenders = []
    for path in (root / "models").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "delta_net.kda" in node.module:
                    offenders.append(str(path))
            if isinstance(node, ast.alias) and node.name in (
                "FlashInferKDA",
                "FlashInferKda",
            ):
                offenders.append(str(path))
    assert offenders == [], f"model files import KDA impl directly: {offenders}"
