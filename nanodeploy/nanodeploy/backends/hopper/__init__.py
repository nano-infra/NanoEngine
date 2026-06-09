"""Hopper (NVIDIA H100/H200) backend factory.

Provides FP8 + DeepGEMM + DeepEP implementations for all layer types.
"""

from nanodeploy.backends.base_backend import (
    AttentionBase,
    BackendFactory,
    ColumnParallelLinearBase,
    DistributedRoutedExpertsBase,
    GatedDeltaNetBase,
    MergedColumnParallelLinearBase,
    QKVParallelLinearBase,
    ReplicatedLinearBase,
    RowParallelLinearBase,
)


class HopperBackendFactory(BackendFactory):
    """Factory that returns Hopper-specific (FP8-capable) layer instances."""

    def __init__(self, quant_config):
        self.quant_config = quant_config

    # ------------------------------------------------------------------
    # Linear layers
    # ------------------------------------------------------------------

    def get_row_parallel_linear(
        self,
        input_size,
        output_size,
        bias=False,
        meta=False,
        weight_tensor=None,
        bias_tensor=None,
        scale_tensor=None,
        tp_group=None,
        **kwargs,
    ) -> RowParallelLinearBase:
        from .layers.linear import HopperRowParallelLinear

        return HopperRowParallelLinear(
            input_size,
            output_size,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=self.quant_config,
            tp_group=tp_group,
        )

    def get_column_parallel_linear(
        self,
        input_size,
        output_size,
        bias=False,
        meta=False,
        weight_tensor=None,
        bias_tensor=None,
        scale_tensor=None,
        tp_group=None,
        **kwargs,
    ) -> ColumnParallelLinearBase:
        from .layers.linear import HopperColumnParallelLinear

        return HopperColumnParallelLinear(
            input_size,
            output_size,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=self.quant_config,
            tp_group=tp_group,
        )

    def get_merged_column_parallel_linear(
        self,
        input_size,
        output_sizes,
        bias=False,
        meta=False,
        weight_tensor=None,
        bias_tensor=None,
        scale_tensor=None,
        tp_group=None,
        **kwargs,
    ) -> MergedColumnParallelLinearBase:
        from .layers.linear import HopperMergedColumnParallelLinear

        return HopperMergedColumnParallelLinear(
            input_size,
            output_sizes,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=self.quant_config,
            tp_group=tp_group,
        )

    def get_qkv_parallel_linear(
        self,
        hidden_size,
        head_size,
        total_num_heads,
        total_num_kv_heads=None,
        bias=False,
        meta=False,
        weight_tensor=None,
        bias_tensor=None,
        scale_tensor=None,
        tp_group=None,
        **kwargs,
    ) -> QKVParallelLinearBase:
        from .layers.linear import HopperQKVParallelLinear

        return HopperQKVParallelLinear(
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=self.quant_config,
            tp_group=tp_group,
        )

    def get_replicated_linear(
        self,
        input_size,
        output_size,
        bias=False,
        meta=False,
        weight_tensor=None,
        bias_tensor=None,
        scale_tensor=None,
        **kwargs,
    ) -> ReplicatedLinearBase:
        from .layers.linear import HopperReplicatedLinear

        return HopperReplicatedLinear(
            input_size,
            output_size,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=self.quant_config,
        )

    def get_distributed_routed_experts(
        self,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        ep_size,
        tp_size,
        **kwargs,
    ) -> DistributedRoutedExpertsBase:
        from .layers.experts import HopperDistributedRoutedExperts

        # Pass quant config; ignore any stale quantization_config in kwargs
        kwargs.pop("quantization_config", None)
        return HopperDistributedRoutedExperts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            ep_size=ep_size,
            tp_size=tp_size,
            quantization_config=self.quant_config,
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
    ) -> AttentionBase:
        from .layers.attention import HopperAttention

        return HopperAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_dim=v_head_dim,
            attention_type=attention_type,
            nsa_index_topk=nsa_index_topk,
        )

    # ------------------------------------------------------------------
    # GatedDeltaNet (Linear Attention) layer
    # ------------------------------------------------------------------

    def get_gated_delta_net(
        self,
        layer_idx: int,
        config,
        quantization_config=None,
        **kwargs,
    ) -> GatedDeltaNetBase:
        from nanodeploy.backends.gpu_generic.layers.gated_delta_net import (
            GenericGatedDeltaNet,
        )

        quant_config = quantization_config or self.quant_config
        return GenericGatedDeltaNet(
            layer_idx=layer_idx,
            config=config,
            quantization_config=quant_config,
            **kwargs,
        )
