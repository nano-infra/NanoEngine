"""Hopper (NVIDIA H100/H200) backend factory.

Selects the ``hopper`` tier policy (FP8 + DeepGEMM + DeepEP implementations).
All construction logic lives in ``PolicyBackendFactory``; this class only pins
the tier.
"""

from dlengine.runtime.layers.policy_backend import PolicyBackendFactory


class HopperBackendFactory(PolicyBackendFactory):
    """Factory that returns Hopper-specific (FP8-capable) layer instances."""

    def __init__(self, quant_config):
        super().__init__(quant_config, tier="hopper")
