"""Hopper distributed routed experts (FP8 + DeepEP).

This is the full-featured MoE implementation for NVIDIA Hopper GPUs.
It supports:
  - FP8 block-wise quantization via DeepGEMM
  - Expert parallelism via DeepEP (normal and low-latency dispatchers)
  - Tensor parallelism via all-reduce
"""

from typing import Any, Dict, Optional

import torch
from torch import nn

from nanodeploy.backends.base_backend import DistributedRoutedExpertsBase
from nanodeploy.context.expert_context import ExpertContext
from nanodeploy.layers.local_dispatch import LocalPaddedDispatcher
from nanodeploy.worker.runner_config import get_runner_config


def compute_topk_ids(topk_ids, ranks, num_experts):
    """Optimized version: compute expert IDs for perfect load balancing.

    This function redistributes expert IDs to ensure perfect load balancing
    across expert parallel ranks. Optimized to use a single torch.arange call.
    """
    shape = topk_ids.shape
    numel = topk_ids.numel()
    step = num_experts // ranks

    # Single arange call instead of two
    indices = torch.arange(0, numel, dtype=topk_ids.dtype, device=topk_ids.device)

    # Compute both components from the same indices
    div_ranks = indices // ranks
    mod_ranks = indices % ranks

    # Compute the remapped expert IDs
    topk_ids = (div_ranks % step + mod_ranks * step) % num_experts
    topk_ids = topk_ids.reshape(shape)
    return topk_ids


class HopperDistributedRoutedExperts(DistributedRoutedExpertsBase):
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
        quantization_config=None,
        layer_idx: int = -1,
    ):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.top_k = top_k

        self.ep_size = ep_size
        self.tp_size = tp_size
        self.ep_group = ep_group
        self.tp_group = tp_group

        assert (
            num_experts % ep_size == 0
        ), f"num_experts {num_experts} must be perfectly divisible by ep_size {ep_size}"
        self.num_local_experts = num_experts // ep_size
        self.ep_rank = (
            torch.distributed.get_rank(ep_group) if ep_group is not None else 0
        )

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

        if self.is_fp8 and tp_size > 1:
            assert self.local_intermediate_size % 128 == 0, (
                f"FP8 MoE requires local_intermediate_size ({self.local_intermediate_size}) "
                f"to be divisible by 128 (FP8 block size). "
                f"intermediate_size={intermediate_size}, tp_size={tp_size}. "
                f"Please choose a tp_size that divides intermediate_size into 128-aligned chunks."
            )

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

        if self.is_fp8:
            block_size = 128
            self.gate_up_scale_inv = nn.Parameter(
                torch.ones(
                    self.num_local_experts,
                    self.local_intermediate_size * 2 // block_size,
                    hidden_size // block_size,
                    dtype=torch.float32,
                )
            )
            self.down_scale_inv = nn.Parameter(
                torch.ones(
                    self.num_local_experts,
                    hidden_size // block_size,
                    self.local_intermediate_size // block_size,
                    dtype=torch.float32,
                )
            )
        else:
            self.gate_up_scale_inv = None
            self.down_scale_inv = None

        # Lazy-init dispatcher for CUDA-Graph-safe EP==1 decode
        self._local_dispatcher: Optional[LocalPaddedDispatcher] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        from nanodeploy.layers.eplb import topk_ids_logical_to_physical
        from nanodeploy.worker.runner_config import get_runner_config

        ctx = ExpertContext.get_instance()
        buffer = ctx.get_buffer()

        if get_runner_config().dummy_eplb:
            ranks = torch.distributed.get_world_size(self.ep_group)
            topk_ids = compute_topk_ids(topk_ids, ranks, self.num_experts)

        # Use EPLB dispatch if layer_idx != -1
        runner_config = get_runner_config()
        if getattr(runner_config, "enable_eplb", False) and self.layer_idx != -1:
            import nanodeploy.layers.eplb as eplb

            info = eplb.EPLBDispatchInfo.init_new(
                ep_rank=self.ep_rank, layer_idx=self.layer_idx
            )
            topk_ids = eplb.topk_ids_logical_to_physical(topk_ids, info=info)
        if self.ep_size <= 1:
            return self._compute_local(
                hidden_states, topk_ids, topk_weights, is_prefill
            )

        # use_low_latency_ep: force decode (low-latency) EP path even during
        # prefill.  Used by MTP which needs prefill attention mode but must
        # avoid multi-stream DeepEP dispatch for CUDAGraph compatibility.
        from nanodeploy.context.context import get_context

        use_low_latency = getattr(get_context(), "use_low_latency_ep", False)

        if is_prefill and not use_low_latency:
            return self._compute_prefill_ep(hidden_states, topk_ids, topk_weights)
        else:
            return self._compute_decode_ep(hidden_states, topk_ids, topk_weights)

    def _get_or_create_local_dispatcher(self) -> LocalPaddedDispatcher:
        if self._local_dispatcher is None:
            self._local_dispatcher = LocalPaddedDispatcher.from_experts(
                num_local_experts=self.num_local_experts,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                device=self.gate_up_proj.device,
            )
        return self._local_dispatcher

    def _compute_local(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        is_prefill: bool,
    ):
        if is_prefill:
            return self._compute_local_prefill(hidden_states, topk_ids, topk_weights)
        return self._compute_local_decode(hidden_states, topk_ids, topk_weights)

    def _compute_local_prefill(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Prefill path – uses fused_moe_v3 (no CUDA Graph needed)."""
        valid_topk_ids = topk_ids[topk_ids >= 0]
        expert_counts = torch.bincount(
            valid_topk_ids, minlength=self.num_local_experts
        ).tolist()

        BLOCK_E = 128
        padded_expert_counts = [
            (count + BLOCK_E - 1) // BLOCK_E * BLOCK_E for count in expert_counts
        ]

        if self.is_fp8:
            from nanodeploy.backends.hopper.kernels.fp8 import per_token_group_quant_fp8
            from nanodeploy.backends.hopper.kernels.fused_moe_v3 import fused_moe_v3

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
                padded_expert_counts,
            )
        else:
            from nanodeploy.backends.hopper.kernels.fused_moe_v3 import (
                fused_moe_v3_bf16,
            )

            out_states = fused_moe_v3_bf16(
                hidden_states,
                topk_ids,
                topk_weights,
                self.gate_up_proj,
                self.down_proj,
                padded_expert_counts,
            )

        if self.tp_size > 1 and self.tp_group is not None:
            torch.distributed.all_reduce(out_states, group=self.tp_group)

        return out_states

    def _compute_local_decode(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Decode path – CUDA-Graph safe, uses LocalPaddedDispatcher + masked GEMM."""
        import deep_gemm

        disp = self._get_or_create_local_dispatcher()
        T = hidden_states.shape[0]
        padded_buf, masked_m, expected_m = disp.dispatch(hidden_states, topk_ids)

        E = self.num_local_experts
        max_m = disp.max_m
        N = self.gate_up_proj.size(1)  # local_intermediate_size * 2
        H = self.hidden_size

        if self.is_fp8:
            from nanodeploy.backends.hopper.kernels.fp8 import (
                per_token_group_quant_fp8,
                silu_and_mul_masked_post_quant_fwd,
            )

            block_size = 128
            # Quantize padded input to FP8 (static shape, Graph-safe)
            padded_flat = padded_buf.reshape(E * max_m, H)
            padded_fp8, padded_scale = per_token_group_quant_fp8(
                padded_flat, block_size
            )
            padded_fp8 = padded_fp8.view(E, max_m, H)
            padded_scale = padded_scale.view(E, max_m, H // block_size)

            # Gate-Up masked GEMM
            gateup_output = torch.empty(
                (E, max_m, N), device=hidden_states.device, dtype=torch.bfloat16
            )
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                (padded_fp8, padded_scale),
                (self.gate_up_proj, self.gate_up_scale_inv),
                gateup_output,
                masked_m,
                expected_m,
            )

            # SiLU + Mul + FP8 post-quant (masked-aware)
            down_input = torch.empty(
                (E, max_m, N // 2),
                device=hidden_states.device,
                dtype=torch.float8_e4m3fn,
            )
            down_input_scale = torch.empty(
                (E, max_m, N // 2 // block_size),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            silu_and_mul_masked_post_quant_fwd(
                gateup_output, down_input, down_input_scale, block_size, masked_m
            )

            # Down masked GEMM
            down_output = torch.empty(
                (E, max_m, H), device=hidden_states.device, dtype=torch.bfloat16
            )
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                (down_input, down_input_scale),
                (self.down_proj, self.down_scale_inv),
                down_output,
                masked_m,
                expected_m,
            )
        else:
            import torch.nn.functional as F

            # Gate-Up masked GEMM (BF16)
            gateup_output = torch.empty(
                (E, max_m, N), device=hidden_states.device, dtype=torch.bfloat16
            )
            deep_gemm.m_grouped_bf16_gemm_nt_masked(
                padded_buf, self.gate_up_proj, gateup_output, masked_m, expected_m
            )

            # SiLU + Mul
            gate, up = gateup_output.chunk(2, dim=-1)
            down_input = F.silu(gate) * up

            # Down masked GEMM (BF16)
            down_output = torch.empty(
                (E, max_m, H), device=hidden_states.device, dtype=torch.bfloat16
            )
            deep_gemm.m_grouped_bf16_gemm_nt_masked(
                down_input, self.down_proj, down_output, masked_m, expected_m
            )

        out = disp.combine(down_output, topk_ids, topk_weights, T)

        if self.tp_size > 1 and self.tp_group is not None:
            torch.distributed.all_reduce(out, group=self.tp_group)

        return out

    def _compute_prefill_ep(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        from nanodeploy.layers.token_dispatcher import DeepEPTokenDispatcherNormal

        ctx = ExpertContext.get_instance()
        ctx.transition_to_normal()
        dispatcher = DeepEPTokenDispatcherNormal(
            group=self.ep_group,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            params_dtype=self.gate_up_proj.dtype,
        )

        if self.is_fp8:
            from nanodeploy.backends.hopper.kernels.fp8 import per_token_group_quant_fp8

            x_fp8, x_scales = per_token_group_quant_fp8(hidden_states, 128)
            x_to_dispatch = (x_fp8, x_scales)
        else:
            x_to_dispatch = hidden_states

        recv_x, recv_topk_idx, recv_topk_weights, recv_expert_count, handle, event = (
            dispatcher.dispatch(x_to_dispatch, topk_ids, topk_weights)
        )

        if self.is_fp8:
            from nanodeploy.backends.hopper.kernels.fused_moe_v3 import fused_moe_v3

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
            from nanodeploy.backends.hopper.kernels.fused_moe_v3 import (
                fused_moe_v3_bf16,
            )

            down_output = fused_moe_v3_bf16(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                self.gate_up_proj,
                self.down_proj,
                recv_expert_count,
            )

        out_states = dispatcher.combine(down_output)
        return out_states

    def _compute_decode_ep(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        import deep_gemm

        from nanodeploy.layers.token_dispatcher import DeepEPTokenDispatcherLowLatency

        ctx = ExpertContext.get_instance()
        ctx.transition_to_low_latency()
        dispatcher = DeepEPTokenDispatcherLowLatency(
            group=self.ep_group,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            params_dtype=self.gate_up_proj.dtype,
        )

        packed_recv_hidden, recv_topk_idx, recv_topk_weights, masked_m, expected_m = (
            dispatcher.dispatch(
                hidden_states,
                topk_ids,
                topk_weights,
                self.num_experts,
                use_fp8=self.is_fp8,
            )
        )

        if self.is_fp8:
            gate_up_weight_fp8 = (self.gate_up_proj, self.gate_up_scale_inv)
            recv_x, recv_x_scale = packed_recv_hidden[0], packed_recv_hidden[1]

            num_groups, m, k = recv_x.shape
            n = self.gate_up_proj.size(1)
            expected_m = min(expected_m, m)

            recv_x_fp8 = (recv_x, recv_x_scale)
            gateup_output = torch.empty(
                (num_groups, m, n), device=hidden_states.device, dtype=torch.bfloat16
            )
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                recv_x_fp8, gate_up_weight_fp8, gateup_output, masked_m, expected_m
            )

            from nanodeploy.backends.hopper.kernels.fp8 import (
                silu_and_mul_masked_post_quant_fwd,
            )

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
                gateup_output, down_input, down_input_scale, block_size, masked_m
            )
            del gateup_output

            down_n = self.down_proj.size(1)
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
            recv_x = packed_recv_hidden
            num_groups, m, k = recv_x.shape
            n = self.gate_up_proj.size(1)
            expected_m = min(expected_m, m)

            gateup_output = torch.empty(
                (num_groups, m, n), device=hidden_states.device, dtype=torch.bfloat16
            )
            deep_gemm.m_grouped_bf16_gemm_nt_masked(
                recv_x, self.gate_up_proj, gateup_output, masked_m, expected_m
            )

            import torch.nn.functional as F

            gateup_output_unbound = gateup_output.chunk(2, dim=-1)
            down_input = F.silu(gateup_output_unbound[0]) * gateup_output_unbound[1]

            down_n = self.down_proj.size(1)
            down_output = torch.empty(
                (num_groups, m, down_n),
                device=hidden_states.device,
                dtype=torch.bfloat16,
            )
            deep_gemm.m_grouped_bf16_gemm_nt_masked(
                down_input, self.down_proj, down_output, masked_m, expected_m
            )

        final_hidden_states = dispatcher.combine(
            down_output, recv_topk_idx, recv_topk_weights
        )
        del packed_recv_hidden

        if self.tp_size > 1 and self.tp_group is not None:
            torch.distributed.all_reduce(final_hidden_states, group=self.tp_group)

        return final_hidden_states
