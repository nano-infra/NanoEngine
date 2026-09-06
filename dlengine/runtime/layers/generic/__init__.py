"""Generic GPU (BF16) backend factory.

Selects the ``gpu_generic`` tier policy. All construction logic lives in
``PolicyBackendFactory``; this class only pins the tier. Expert parallelism
(ep_size > 1) is not supported by the generic experts implementation.
"""

from dlengine.runtime.layers.policy_backend import PolicyBackendFactory


class GenericBackendFactory(PolicyBackendFactory):
    """Factory that returns generic BF16 layer instances."""

    def __init__(self, quant_config):
        super().__init__(quant_config, tier="gpu_generic")
