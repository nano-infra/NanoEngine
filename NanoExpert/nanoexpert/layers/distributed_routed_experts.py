from typing import Any, Dict, Optional, Tuple

import torch

from nanoexpert.context.expert_context import ExpertContext
from torch import nn


class DistributedRoutedExperts(nn.Module):
    """
    Unified MoE Layer handling both Expert Parallel (EP) and Tensor Parallel (TP).
    Uses DeepEP for cross-node/cross-GPU expert routing when ep_size > 1.
    Uses DeepGEMM for FP8/BF16 high-performance inner compute.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        tp_size: int,
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        norm_topk_prob: bool = False,
        routed_scaling_factor: float = 1.0,
        scoring_func: str = "softmax",
        quantization_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size

        self.ep_size = ep_size
        self.tp_size = tp_size
        self.ep_group = ep_group
        self.tp_group = tp_group

        # Expert parallelism properties
        assert (
            num_experts % ep_size == 0
        ), f"num_experts {num_experts} must be perfectly divisible by ep_size {ep_size}"
        self.num_local_experts = num_experts // ep_size
        self.ep_rank = (
            torch.distributed.get_rank(ep_group) if ep_group is not None else 0
        )

        # Tensor parallelism properties
        assert (
            intermediate_size * 2
        ) % tp_size == 0, "intermediate_size * 2 must be divisible by tp_size"
        self.tp_rank = (
            torch.distributed.get_rank(tp_group) if tp_group is not None else 0
        )
        self.local_intermediate_size = intermediate_size // tp_size

        self.quantization_config = quantization_config
        self.is_fp8 = False
        if quantization_config is not None:
            config_group = getattr(quantization_config, "quant_method", "")
            self.is_fp8 = config_group == "fp8"

        # Unified parameters: No matter whether TP/EP, the shape loaded is exactly what this rank covers.
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.local_intermediate_size * 2,
                hidden_size,
                dtype=torch.float8_e4m3fn if self.is_fp8 else torch.bfloat16,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                hidden_size,
                self.local_intermediate_size,
                dtype=torch.float8_e4m3fn if self.is_fp8 else torch.bfloat16,
            )
        )

        # 权重缩放 (FP8 Only)
        if self.is_fp8:
            # Note: The exact block size can be (out_dims // 128, in_dims // 128)
            # Loader will dynamically overwrite these empty params with correct shape.
            self.gate_up_scale_inv = nn.Parameter(torch.empty(0, dtype=torch.float32))
            self.down_scale_inv = nn.Parameter(torch.empty(0, dtype=torch.float32))
        else:
            self.gate_up_scale_inv = None
            self.down_scale_inv = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for the unified MoE logic.
        """
        # Handle EPLB mapping if logically enabled globally
        from nanodeploy.worker.runner_config import get_runner_config

        # 0. EPLB (Expert Parallel Load Balancing) Interception
        # If EPLB is globally enabled and mapped, we intercept the logical topk_ids
        # and randomly assign them to physical topk_ids mapped copies.
        from nanoexpert.layers.eplb import topk_ids_logical_to_physical

        if get_runner_config().perfect_eplb:
            import nanoexpert.layers.eplb as eplb

            # Assuming info is pre-initialized, or we can fetch if needed
            topk_ids = eplb.topk_ids_logical_to_physical(topk_ids, info=None)

        ctx = ExpertContext.get_instance()
        buffer = ctx.get_buffer()

        if self.ep_size <= 1:
            # Single-rank (or pure TP) local compute bypasses DeepEP logic
            return self._compute_local(
                hidden_states, topk_ids, topk_weights, is_prefill
            )

        # Distributed EP dispatch logic
        if is_prefill:
            return self._compute_prefill_ep(hidden_states, topk_ids, topk_weights)
        else:
            return self._compute_decode_ep(hidden_states, topk_ids, topk_weights)

    def _compute_local(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        is_prefill: bool,
    ):
        # 1. Quantize if needed and compute
        if self.is_fp8:
            from nanoexpert.kernels.fp8 import per_token_group_quant_fp8
            from nanoexpert.kernels.fused_moe_v3 import fused_moe_v3

            x_fp8, x_scales = per_token_group_quant_fp8(hidden_states, 128)
            x_to_compute = (x_fp8, x_scales)
            gate_up_weight_tup = (self.gate_up_proj, self.gate_up_scale_inv)
            down_weight_tup = (self.down_proj, self.down_scale_inv)
            out_states = fused_moe_v3(
                x_to_compute,
                topk_ids,
                topk_weights,
                gate_up_weight_tup,
                down_weight_tup,
                None,  # Let fused_moe_v3 count local tokens
            )
        else:
            from nanoexpert.kernels.fused_moe_v3 import fused_moe_v3_bf16

            out_states = fused_moe_v3_bf16(
                hidden_states,
                topk_ids,
                topk_weights,
                self.gate_up_proj,
                self.down_proj,
                None,  # Let fused_moe_v3_bf16 count local tokens
            )

        # 2. TP All-Reduce
        if self.tp_size > 1 and self.tp_group is not None:
            torch.distributed.all_reduce(out_states, group=self.tp_group)

        return out_states

    def _compute_prefill_ep(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        from nanoexpert.layers.token_dispatcher import DeepEPTokenDispatcherNormal

        ctx = ExpertContext.get_instance()
        ctx.transition_to_normal()
        dispatcher = DeepEPTokenDispatcherNormal(
            group=self.ep_group,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            params_dtype=self.gate_up_proj.dtype,
        )

        # 1. Dispatch
        if self.is_fp8:
            from nanoexpert.kernels.fp8 import per_token_group_quant_fp8

            x_fp8, x_scales = per_token_group_quant_fp8(hidden_states, 128)
            x_to_dispatch = (x_fp8, x_scales)
        else:
            x_to_dispatch = hidden_states

        recv_x, recv_topk_idx, recv_topk_weights, recv_expert_count, handle, event = (
            dispatcher.dispatch(x_to_dispatch, topk_ids, topk_weights)
        )

        # 2. Compute
        if self.is_fp8:
            from nanoexpert.kernels.fused_moe_v3 import fused_moe_v3

            gate_up_weight_tup = (self.gate_up_proj, self.gate_up_scale_inv)
            down_weight_tup = (self.down_proj, self.down_scale_inv)
            down_output = fused_moe_v3(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                gate_up_weight_tup,
                down_weight_tup,
                recv_expert_count,
            )
        else:
            from nanoexpert.kernels.fused_moe_v3 import fused_moe_v3_bf16

            down_output = fused_moe_v3_bf16(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                self.gate_up_proj,
                self.down_proj,
                recv_expert_count,
            )

        # 3. Combine
        out_states = dispatcher.combine(down_output)

        return out_states

    def _compute_decode_ep(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        import deep_gemm
        from nanoexpert.layers.token_dispatcher import DeepEPTokenDispatcherLowLatency

        ctx = ExpertContext.get_instance()
        ctx.transition_to_low_latency()
        dispatcher = DeepEPTokenDispatcherLowLatency(
            group=self.ep_group,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            params_dtype=self.gate_up_proj.dtype,
        )

        # 1. Low Latency Dispatch
        packed_recv_hidden, recv_topk_idx, recv_topk_weights, masked_m, expected_m = (
            dispatcher.dispatch(
                hidden_states,
                topk_ids,
                topk_weights,
                self.num_experts,
                use_fp8=self.is_fp8,
            )
        )

        # 2. Compute
        # CRITICAL: m must come from the actual dispatched tensor shape, NOT from expected_m.
        # The dispatch buffer has shape (num_local_experts, num_max_dispatch_tokens_per_rank, hidden_size).
        # expected_m is just a hint for how many rows are actually valid (via masked_m).
        if self.is_fp8:
            gate_up_weight_fp8 = (self.gate_up_proj, self.gate_up_scale_inv)
            recv_x, recv_x_scale = packed_recv_hidden[0], packed_recv_hidden[1]

            num_groups, m, k = recv_x.shape
            n = self.gate_up_proj.size(1)  # local_intermediate_size * 2
            expected_m = min(expected_m, m)

            recv_x_fp8 = (recv_x, recv_x_scale)

            gateup_output = torch.empty(
                (num_groups, m, n), device=hidden_states.device, dtype=torch.bfloat16
            )

            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                recv_x_fp8, gate_up_weight_fp8, gateup_output, masked_m, expected_m
            )

            # silu_and_mul and re-quantize activation for the next gemm
            from nanoexpert.kernels.fp8 import silu_and_mul_masked_post_quant_fwd

            block_size = 128
            down_input = torch.empty(
                (num_groups, m, n // 2),
                device=hidden_states.device,
                dtype=torch.float8_e4m3fn,
            )
            down_input_scale = torch.empty(
                (num_groups, m, n // 2 // block_size),
                device=hidden_states.device,
                dtype=torch.float32,
            )

            silu_and_mul_masked_post_quant_fwd(
                gateup_output,
                down_input,
                down_input_scale,
                block_size,
                masked_m,
            )
            del gateup_output

            down_n = self.down_proj.size(1)  # hidden_size
            down_input_fp8 = (down_input, down_input_scale)
            down_weight_fp8 = (self.down_proj, self.down_scale_inv)
            down_output = torch.empty(
                (num_groups, m, down_n),
                device=hidden_states.device,
                dtype=torch.bfloat16,
            )

            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                down_input_fp8, down_weight_fp8, down_output, masked_m, expected_m
            )

        else:
            # BF16 path
            # When use_fp8=False, DeepEP low_latency_dispatch returns a single 3D tensor
            # (not a tuple like FP8 which returns (tensor, scales))
            recv_x = packed_recv_hidden

            num_groups, m, k = recv_x.shape
            n = self.gate_up_proj.size(1)  # local_intermediate_size * 2
            expected_m = min(expected_m, m)

            gateup_output = torch.empty(
                (num_groups, m, n), device=hidden_states.device, dtype=torch.bfloat16
            )

            deep_gemm.m_grouped_bf16_gemm_nt_masked(
                recv_x, self.gate_up_proj, gateup_output, masked_m, expected_m
            )

            import torch.nn.functional as F

            # In-place/fused silu and mul equivalent
            gateup_output_unbound = gateup_output.chunk(2, dim=-1)
            down_input = F.silu(gateup_output_unbound[0]) * gateup_output_unbound[1]

            down_n = self.down_proj.size(1)  # hidden_size
            down_output = torch.empty(
                (num_groups, m, down_n),
                device=hidden_states.device,
                dtype=torch.bfloat16,
            )

            deep_gemm.m_grouped_bf16_gemm_nt_masked(
                down_input, self.down_proj, down_output, masked_m, expected_m
            )

        # 3. Combine
        final_hidden_states = dispatcher.combine(
            down_output, recv_topk_idx, recv_topk_weights
        )

        del packed_recv_hidden

        if self.tp_size > 1 and self.tp_group is not None:
            torch.distributed.all_reduce(final_hidden_states, group=self.tp_group)

        return final_hidden_states
