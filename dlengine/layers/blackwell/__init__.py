"""Blackwell hardware mapping for native layer backends."""

from dlengine.layers.base_backend import AttentionBase, DistributedRoutedExpertsBase
from dlengine.layers.hopper import HopperBackendFactory


class BlackwellBackendFactory(HopperBackendFactory):
    """Map Blackwell checkpoint formats to their preferred kernel backends."""

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.hardware_backend = "blackwell"

    def get_distributed_routed_experts(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        tp_size: int,
        **kwargs,
    ) -> DistributedRoutedExpertsBase:
        if bool(getattr(self.quant_config, "is_mxfp4", False)):
            from dlengine.layers.backends.megamoe import MegaMoEExperts

            kwargs.pop("quantization_config", None)
            return MegaMoEExperts(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                ep_size=ep_size,
                tp_size=tp_size,
                quantization_config=self.quant_config,
                **kwargs,
            )
        return super().get_distributed_routed_experts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            ep_size=ep_size,
            tp_size=tp_size,
            **kwargs,
        )

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
