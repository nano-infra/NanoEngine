"""Policy-provider factory tests.

Assert that hardware tiers are pure policy providers: the tier maps to a static
TierPolicy that drives which implementation family linear/experts resolve to,
matching the pre-refactor per-tier hardcoding.
"""

import pytest
from dlengine.runtime.layers.backend_policy import TIER_POLICIES
from dlengine.runtime.layers.blackwell import BlackwellBackendFactory
from dlengine.runtime.layers.generic import GenericBackendFactory
from dlengine.runtime.layers.hopper import HopperBackendFactory
from dlengine.runtime.models.quant_config import QuantizationConfig


@pytest.mark.parametrize(
    ("factory_cls", "hardware", "linear", "experts", "quant_override", "fp8"),
    [
        (GenericBackendFactory, "gpu_generic", "generic", "generic", False, False),
        (HopperBackendFactory, "hopper", "deepseek", "deepseek", False, True),
        (BlackwellBackendFactory, "blackwell", "deepseek", "deepseek", True, True),
    ],
)
def test_factory_pins_tier_policy(
    factory_cls, hardware, linear, experts, quant_override, fp8
):
    factory = factory_cls(QuantizationConfig())

    assert factory.hardware_backend == hardware
    assert factory.policy is TIER_POLICIES[hardware]
    assert factory.policy.linear == linear
    assert factory.policy.experts == experts
    assert factory.policy.experts_quant_override is quant_override
    assert factory.policy.supports_fp8 is fp8
    # Ref fallback defaults to disabled; set by create_backend from config.
    assert factory.ref_fallback_allowed is False


def test_blackwell_is_not_a_hopper_subclass():
    # The BlackwellBackendFactory(HopperBackendFactory) inheritance-for-config
    # pattern is gone; both are thin siblings over PolicyBackendFactory.
    assert not issubclass(BlackwellBackendFactory, HopperBackendFactory)
