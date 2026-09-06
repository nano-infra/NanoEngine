"""Policy-driven backend factory.

A single factory implementation whose behavior is fully determined by a
``TierPolicy`` (data). This replaces the previous pattern where each hardware
tier had its own factory with hardcoded per-family construction logic, and
Blackwell subclassed Hopper only to override two methods.

The per-tier named factories (``GenericBackendFactory``, ``HopperBackendFactory``,
``BlackwellBackendFactory``) are thin subclasses that only select which
``TierPolicy`` to use; they carry no implementation logic.
"""

from __future__ import annotations

from dlengine.runtime.layers.backend_policy import TIER_POLICIES, TierPolicy
from dlengine.runtime.layers.base_backend import (
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


class PolicyBackendFactory(BackendFactory):
    """Backend factory driven entirely by a ``TierPolicy``."""

    def __init__(self, quant_config, tier: str):
        self.quant_config = quant_config
        self.policy: TierPolicy = TIER_POLICIES[tier]
        self.hardware_backend = self.policy.hardware
        self.attention_backend = self.policy.attention
        self.gdn_backend = self.policy.gdn
        self.ref_fallback_allowed = False

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
        from dlengine.runtime.layers.backends.selector import create_linear

        return create_linear(
            "row",
            family=self.policy.linear,
            quantization_config=self.quant_config,
            scale_tensor=scale_tensor,
            input_size=input_size,
            output_size=output_size,
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
        from dlengine.runtime.layers.backends.selector import create_linear

        return create_linear(
            "column",
            family=self.policy.linear,
            quantization_config=self.quant_config,
            scale_tensor=scale_tensor,
            input_size=input_size,
            output_size=output_size,
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
        from dlengine.runtime.layers.backends.selector import create_linear

        return create_linear(
            "merged",
            family=self.policy.linear,
            quantization_config=self.quant_config,
            scale_tensor=scale_tensor,
            input_size=input_size,
            output_sizes=output_sizes,
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
        from dlengine.runtime.layers.backends.selector import create_linear

        return create_linear(
            "qkv",
            family=self.policy.linear,
            quantization_config=self.quant_config,
            scale_tensor=scale_tensor,
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
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
        from dlengine.runtime.layers.backends.selector import create_linear

        return create_linear(
            "replicated",
            family=self.policy.linear,
            quantization_config=self.quant_config,
            scale_tensor=scale_tensor,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
        )

    # ------------------------------------------------------------------
    # Routed experts
    # ------------------------------------------------------------------

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
        from dlengine.runtime.layers.backends.selector import create_experts

        quantization_config = kwargs.pop("quantization_config", self.quant_config)
        return create_experts(
            family=self.policy.experts,
            quantization_config=quantization_config,
            experts_quant_override=self.policy.experts_quant_override,
            ref_fallback_allowed=self.ref_fallback_allowed,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            ep_size=ep_size,
            tp_size=tp_size,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

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
        from dlengine.runtime.layers.backends import create_attention

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

    # ------------------------------------------------------------------
    # GatedDeltaNet (Linear Attention)
    # ------------------------------------------------------------------

    def get_gated_delta_net(
        self,
        layer_idx: int,
        config,
        quantization_config=None,
        **kwargs,
    ) -> GatedDeltaNetBase:
        from dlengine.runtime.layers.backends import create_gdn

        quant_config = quantization_config or self.quant_config
        return create_gdn(
            requested=self.gdn_backend,
            layer_idx=layer_idx,
            config=config,
            quantization_config=quant_config,
            **kwargs,
        )

    def get_kimi_delta_attention(
        self,
        layer_idx: int,
        state_layer_idx: int,
        config,
        **kwargs,
    ):
        from dlengine.runtime.layers.backends.selector import create_kda

        return create_kda(
            layer_idx=layer_idx,
            state_layer_idx=state_layer_idx,
            config=config,
            **kwargs,
        )


__all__ = ["PolicyBackendFactory"]
