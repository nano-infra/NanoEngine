"""Blackwell backend using FA4 attention and Hopper-compatible linear layers."""

from dlengine.layers.base_backend import AttentionBase
from dlengine.layers.hopper import HopperBackendFactory


class BlackwellBackendFactory(HopperBackendFactory):
    """Blackwell factory; only attention differs from the Hopper backend today."""

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.hardware_backend = "blackwell"

    def get_attention(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        v_head_dim: int,
        attention_type: str = "MLA",
        nsa_index_topk: int = 0,
        **kwargs,
    ) -> AttentionBase:
        from dlengine.layers.backends import create_attention

        return create_attention(
            requested=self.attention_backend,
            hardware_backend=self.hardware_backend,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_dim=v_head_dim,
            attention_type=attention_type,
            nsa_index_topk=nsa_index_topk,
            **kwargs,
        )
