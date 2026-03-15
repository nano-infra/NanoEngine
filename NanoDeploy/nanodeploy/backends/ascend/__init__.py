"""Ascend NPU backend factory (BF16, HCCL, torch_npu ops)."""

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


class AscendBackendFactory(BackendFactory):
    """Factory that returns Ascend NPU BF16 layer instances."""

    def __init__(self, quant_config):
        self.quant_config = quant_config

    # ------------------------------------------------------------------
    # Linear layers — thin wrappers reusing the generic BF16 pattern
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
        from .layers.linear import AscendRowParallelLinear

        return AscendRowParallelLinear(
            input_size,
            output_size,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
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
        from .layers.linear import AscendColumnParallelLinear

        return AscendColumnParallelLinear(
            input_size,
            output_size,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
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
        from .layers.linear import AscendMergedColumnParallelLinear

        return AscendMergedColumnParallelLinear(
            input_size,
            output_sizes,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
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
        from .layers.linear import AscendQKVParallelLinear

        return AscendQKVParallelLinear(
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
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
        from .layers.linear import AscendReplicatedLinear

        return AscendReplicatedLinear(
            input_size,
            output_size,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
        )

    def get_gated_delta_net(
        self,
        layer_idx: int,
        config,
        quantization_config=None,
        **kwargs,
    ) -> GatedDeltaNetBase:
        raise NotImplementedError(
            "GatedDeltaNet is not supported by the Ascend backend. "
            "Qwen3-MoE uses GQA attention, not GatedDeltaNet."
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
        from .layers.experts import AscendDistributedRoutedExperts

        kwargs.pop("quantization_config", None)
        return AscendDistributedRoutedExperts(
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
        attention_type: str = "GQA",
        **kwargs,
    ) -> AttentionBase:
        from .layers.attention import AscendAttention

        if attention_type == "MLA":
            raise NotImplementedError(
                "MLA attention is not supported by the Ascend backend in this version. "
                "Use GQA (Qwen3-MoE) instead."
            )
        return AscendAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_dim=v_head_dim,
        )
