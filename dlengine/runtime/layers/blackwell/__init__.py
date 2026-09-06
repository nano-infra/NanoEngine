"""Blackwell (NVIDIA SM100+) backend factory.

Selects the ``blackwell`` tier policy. The policy's ``experts_quant_override``
flag drives NVFP4/MXFP4 expert selection from the checkpoint format, so no
per-method override is needed here. All construction logic lives in
``PolicyBackendFactory``; this class only pins the tier.
"""

from dlengine.runtime.layers.policy_backend import PolicyBackendFactory


class BlackwellBackendFactory(PolicyBackendFactory):
    """Factory that returns Blackwell-specific layer instances."""

    def __init__(self, quant_config):
        super().__init__(quant_config, tier="blackwell")
