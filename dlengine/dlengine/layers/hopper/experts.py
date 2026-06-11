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

from dlengine.context.expert_context import ExpertContext
from dlengine.layers.base_backend import DistributedRoutedExpertsBase
from dlengine.layers.local_dispatch import LocalPaddedDispatcher
from dlengine.worker.runner_config import get_runner_config


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
        swiglu_limit: Optional[float] = None,
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

        self.routed_scaling_factor = routed_scaling_factor

        # SwiGLU clamp (DSV4 ships swiglu_limit=10.0). ``None`` = no
        # clamp (matches V2/V3). The SwiGLU triton kernels accept
        # ``+inf`` as a runtime no-op; we cache that float here so
        # we don't branch in hot code.
        self._swiglu_limit_runtime: float = (
            float("inf") if swiglu_limit is None else float(swiglu_limit)
        )

        # Lazy-init dispatcher for CUDA-Graph-safe EP==1 decode
        self._local_dispatcher: Optional[LocalPaddedDispatcher] = None

        # Mega-MoE state. Populated by ``prepare_mega_weights`` once
        # the FP8 expert weights have finished loading. Until then,
        # ``forward`` will skip the mega path and use the legacy
        # deep_ep low-latency dispatch even when ``use_mega_moe`` is on.
        # See deepseek_v4_loader.load_weights for the post-load hook.
        self.mega_l1_weights: Optional[tuple] = None
        self.mega_l2_weights: Optional[tuple] = None
        # Per-layer SymmBuffer (lazy; allocated on first decode-EP call
        # because we need a reachable ProcessGroup at that point).
        self._mega_moe_buf = None

    def _runner_use_mega_moe(self) -> bool:
        return bool(getattr(get_runner_config(), "use_mega_moe", False))

    def _runner_mega_moe_max_tokens(self) -> int:
        return int(getattr(get_runner_config(), "mega_moe_max_tokens_per_rank", 256))

    def prepare_mega_weights(self) -> None:
        """Transform expert weights for ``deep_gemm.fp8_fp4_mega_moe``.

        Called once after weight loading. Two-step pipeline matching
        sglang's ``build_mega_moe_experts_weights``:

          1. ``transform_sf_into_required_layout`` casts the per-128
             block float32 scales (DSV3 quant format) to per-32 UE8M0
             packed int32 scales (mega-MoE's required layout).
          2. ``transform_weights_for_mega_moe`` interleaves L1 weights
             and applies UTCCP scale transposition.

        Idempotent; no-op when mega-MoE is disabled or the layer is
        BF16-only.
        """
        if not self._runner_use_mega_moe():
            return
        if not self.is_fp8:
            return
        if self.mega_l1_weights is not None:
            return  # already done
        try:
            import deep_gemm
        except ImportError:
            return

        # Hard arch gate. ``deep_gemm.fp8_fp4_mega_moe`` is sm100-only
        # (Blackwell): see DeepGEMM csrc/apis/mega.hpp where the dispatch
        # is ``if (arch_major == 10) sm100_fp8_fp4_mega_moe(...) else
        # DG_HOST_UNREACHABLE``. ``transform_sf_into_required_layout``
        # likewise refuses to cast FP32→UE8M0-int32 unless the device
        # is sm100. On Hopper (sm90) the whole path is a dead-end, so
        # bail early with a clear message instead of chasing the
        # ``is_sfa`` / SF-layout assertions that are downstream of the
        # same arch gate.
        if not getattr(type(self), "_warned_arch", False):
            cc = torch.cuda.get_device_capability(self.gate_up_proj.device)
            if cc[0] != 10:
                from dlengine.logging import get_logger

                get_logger().warning(
                    "mega-MoE: deep_gemm.fp8_fp4_mega_moe is sm100-only "
                    "(Blackwell); current device is sm%d%d. Falling back "
                    "to the legacy deep_ep low-latency path.",
                    cc[0],
                    cc[1],
                )
                type(self)._warned_arch = True
                return
            type(self)._warned_arch = True

        gu = self.gate_up_proj
        gu_sf = self.gate_up_scale_inv
        dn = self.down_proj
        dn_sf = self.down_scale_inv
        if gu_sf is None or dn_sf is None:
            return

        # Step 1: cast scales to the kernel's required layout. The
        # ``recipe`` describes the *input* sf granularity:
        # ``(sfa_gran_mn, gran_mn, gran_k)``. Two cases:
        #   - sglang's DSV4 checkpoint ships per-element-M, per-32-K
        #     fp32 scales → recipe ``(1, 1, 32)``.
        #   - The lovedheart-FP8-SGlang variant (and other DSV3-style
        #     FP8 checkpoints) ship per-128 × per-128 blocked fp32
        #     scales → recipe ``(1, 128, 128)``.
        # Auto-detect from the scale shape so both layouts work.
        try:
            num_groups, n1, k1 = gu.shape
            _, n2, k2 = dn.shape

            def _detect_recipe(sf, mn, k):
                """Pick (gran_mn, gran_k) from the on-disk scale shape.

                We use the 2-tuple form of ``recipe`` so we don't need
                to set ``is_sfa`` (which 3-tuple recipes require).
                """
                if sf.dtype == torch.int32:
                    # Already int32 packed (1D1D Blackwell): each int32
                    # packs 4 UE8M0 scales × 32 elements = 128 along K.
                    return (1, 128)
                sf_mn = sf.size(-2)
                sf_k = sf.size(-1)
                if mn % sf_mn != 0 or k % sf_k != 0:
                    raise ValueError(
                        f"scale shape {tuple(sf.shape)} doesn't tile cleanly "
                        f"into mn={mn}, k={k}"
                    )
                return (mn // sf_mn, k // sf_k)

            recipe_gu = _detect_recipe(gu_sf, n1, k1)
            recipe_dn = _detect_recipe(dn_sf, n2, k2)
            if gu_sf.dtype != torch.int32:
                gu_sf = deep_gemm.transform_sf_into_required_layout(
                    gu_sf,
                    mn=n1,
                    k=k1,
                    recipe=recipe_gu,
                    num_groups=num_groups,
                    disable_ue8m0_cast=False,
                )
            if dn_sf.dtype != torch.int32:
                dn_sf = deep_gemm.transform_sf_into_required_layout(
                    dn_sf,
                    mn=n2,
                    k=k2,
                    recipe=recipe_dn,
                    num_groups=num_groups,
                    disable_ue8m0_cast=False,
                )
        except Exception:
            from dlengine.logging import get_logger

            if not getattr(type(self), "_warned_sf_layout", False):
                import traceback as _tb

                get_logger().warning(
                    "mega-MoE: transform_sf_into_required_layout failed "
                    "(gu_sf.shape=%s dtype=%s, dn_sf.shape=%s dtype=%s). "
                    "Falling back to legacy path. Traceback:\n%s",
                    tuple(self.gate_up_scale_inv.shape),
                    self.gate_up_scale_inv.dtype,
                    tuple(self.down_scale_inv.shape),
                    self.down_scale_inv.dtype,
                    _tb.format_exc(),
                )
                type(self)._warned_sf_layout = True
            return

        # Step 2: interleave L1 + UTCCP-transpose both scales.
        l1 = (gu, gu_sf)
        l2 = (dn, dn_sf)
        self.mega_l1_weights, self.mega_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(l1, l2)
        )

    def _get_mega_moe_buf(self):
        if self._mega_moe_buf is None:
            import deep_gemm

            self._mega_moe_buf = deep_gemm.get_symm_buffer_for_mega_moe(
                self.ep_group,
                self.num_experts,
                self._runner_mega_moe_max_tokens(),
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                use_fp8_dispatch=True,
                activation="swiglu",
            )
        return self._mega_moe_buf

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        from dlengine.layers.eplb import topk_ids_logical_to_physical
        from dlengine.worker.runner_config import get_runner_config

        ctx = ExpertContext.get_instance()
        buffer = ctx.get_buffer()

        if get_runner_config().dummy_eplb:
            ranks = torch.distributed.get_world_size(self.ep_group)
            topk_ids = compute_topk_ids(topk_ids, ranks, self.num_experts)

        # Use EPLB dispatch if layer_idx != -1
        runner_config = get_runner_config()
        if getattr(runner_config, "enable_eplb", False) and self.layer_idx != -1:
            import dlengine.layers.eplb as eplb

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
        from dlengine.context.context import get_context

        use_low_latency = getattr(get_context(), "use_low_latency_ep", False)

        if is_prefill and not use_low_latency:
            return self._compute_prefill_ep(hidden_states, topk_ids, topk_weights)
        # Decode-EP path. Prefer the mega-MoE kernel when it's enabled,
        # FP8 weights are loaded, and the weight transform has already
        # completed. Falls through to the legacy deep_ep low-latency
        # dispatch + per-expert GEMMs otherwise.
        if (
            self._runner_use_mega_moe()
            and self.is_fp8
            and self.mega_l1_weights is not None
            and self.mega_l2_weights is not None
        ):
            return self._compute_decode_ep_mega(hidden_states, topk_ids, topk_weights)
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
            from dlengine.kernel.triton.hopper.fp8 import per_token_group_quant_fp8
            from dlengine.kernel.triton.hopper.fused_moe_v3 import fused_moe_v3

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
                swiglu_limit=self._swiglu_limit_runtime,
            )
        else:
            from dlengine.kernel.triton.hopper.fused_moe_v3 import fused_moe_v3_bf16

            out_states = fused_moe_v3_bf16(
                hidden_states,
                topk_ids,
                topk_weights,
                self.gate_up_proj,
                self.down_proj,
                padded_expert_counts,
                swiglu_limit=self._swiglu_limit_runtime,
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
            from dlengine.kernel.triton.hopper.fp8 import (
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
                gateup_output,
                down_input,
                down_input_scale,
                block_size,
                masked_m,
                swiglu_limit=self._swiglu_limit_runtime,
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

            # SiLU + Mul (+ optional asymmetric clamp matching DSV4
            # reference: up clamped at ±L, gate at upper-bound L only,
            # both before silu*up).
            gate, up = gateup_output.chunk(2, dim=-1)
            if self._swiglu_limit_runtime != float("inf"):
                up = up.clamp(-self._swiglu_limit_runtime, self._swiglu_limit_runtime)
                gate = gate.clamp(max=self._swiglu_limit_runtime)
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
        from dlengine.layers.token_dispatcher import DeepEPTokenDispatcherNormal

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
            from dlengine.kernel.triton.hopper.fp8 import per_token_group_quant_fp8

            x_fp8, x_scales = per_token_group_quant_fp8(hidden_states, 128)
            x_to_dispatch = (x_fp8, x_scales)
        else:
            x_to_dispatch = hidden_states

        recv_x, recv_topk_idx, recv_topk_weights, recv_expert_count, handle, event = (
            dispatcher.dispatch(x_to_dispatch, topk_ids, topk_weights)
        )

        if self.is_fp8:
            from dlengine.kernel.triton.hopper.fused_moe_v3 import fused_moe_v3

            gate_up_weight_tup = (self.gate_up_proj, self.gate_up_scale_inv)
            down_weight_tup = (self.down_proj, self.down_scale_inv)
            down_output = fused_moe_v3(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                gate_up_weight_tup,
                down_weight_tup,
                recv_expert_count,
                swiglu_limit=self._swiglu_limit_runtime,
            )
        else:
            from dlengine.kernel.triton.hopper.fused_moe_v3 import fused_moe_v3_bf16

            down_output = fused_moe_v3_bf16(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                self.gate_up_proj,
                self.down_proj,
                recv_expert_count,
                swiglu_limit=self._swiglu_limit_runtime,
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

        from dlengine.layers.token_dispatcher import DeepEPTokenDispatcherLowLatency

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
            # Tilelang fast path: sglang's vendored silu+mul+UE8M0-quant
            # kernel collapses what was the Triton silu_and_mul_post_quant
            # + a downstream per-token-group quant into a single CUDA
            # launch. ~250 µs sgl vs 538 µs dlengine in v24 trace.
            _used_tilelang_silu = False
            try:
                from dlengine.kernel.jit.sgl.deepseek_v4 import silu_mul_quant_masked

                # DLEngine's eager Triton path produces NATURAL scales
                # (``s = absmax / fp8_max``), so use scale_ue8m0=False to
                # match — UE8M0 scales would diverge bit-wise from what
                # downstream deep_gemm expects.
                silu_mul_quant_masked(
                    input=gateup_output,
                    output=down_input,
                    output_scale=down_input_scale,
                    masked_m=masked_m.to(torch.int32),
                    topk=self.top_k,
                    swiglu_limit=float(self._swiglu_limit_runtime),
                    quant_group_size=block_size,
                    scale_ue8m0=False,
                    swizzle=False,
                )
                _used_tilelang_silu = True
            except Exception:
                pass
            if not _used_tilelang_silu:
                from dlengine.kernel.triton.hopper.fp8 import (
                    silu_and_mul_masked_post_quant_fwd,
                )

                silu_and_mul_masked_post_quant_fwd(
                    gateup_output,
                    down_input,
                    down_input_scale,
                    block_size,
                    masked_m,
                    swiglu_limit=self._swiglu_limit_runtime,
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

            gate, up = gateup_output.chunk(2, dim=-1)
            if self._swiglu_limit_runtime != float("inf"):
                up = up.clamp(-self._swiglu_limit_runtime, self._swiglu_limit_runtime)
                gate = gate.clamp(max=self._swiglu_limit_runtime)
            down_input = F.silu(gate) * up

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

    def _compute_decode_ep_mega(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Decode-EP path via ``deep_gemm.fp8_fp4_mega_moe``.

        Replaces the legacy ``DeepEPTokenDispatcherLowLatency.dispatch
        → m_grouped_fp8_gemm × 2 + silu_and_mul → combine`` chain with a
        single end-to-end MoE kernel. The dispatcher's symmetric buffer
        and per-expert routing are subsumed by ``SymmBuffer``.

        Numerically equivalent to ``_compute_decode_ep`` within FP8 ULP
        noise; gated behind ``Config.use_mega_moe`` so production can
        A/B against the legacy path.

        Adapted from sglang's DeepseekV2MoE forward (deepseek_v2.py:1180+).
        """
        import deep_gemm

        from dlengine.kernel.triton.hopper.fp8 import per_token_group_quant_fp8

        num_tokens = hidden_states.shape[0]
        buf = self._get_mega_moe_buf()
        padded_max = buf.topk_idx.shape[0]
        if num_tokens > padded_max:
            raise RuntimeError(
                f"mega-MoE: num_tokens={num_tokens} exceeds the per-rank cap "
                f"({padded_max}). Raise Config.mega_moe_max_tokens_per_rank "
                f"or shrink the decode batch."
            )

        # Quantise activations + populate the symmetric buffer. Sglang
        # has a fused mega_moe_pre_dispatch kernel for this; we use the
        # eager 4-step variant for now (still better than the legacy
        # path's full dispatch + per-expert GEMM chain).
        if num_tokens > 0:
            x_fp8, x_sf = per_token_group_quant_fp8(hidden_states, 32)
            buf.x[:num_tokens].copy_(x_fp8)
            buf.x_sf[:num_tokens].copy_(x_sf)
            buf.topk_idx[:num_tokens].copy_(topk_ids)
            buf.topk_weights[:num_tokens].copy_(topk_weights)
        if num_tokens < padded_max:
            buf.topk_idx[num_tokens:].fill_(-1)
            buf.topk_weights[num_tokens:].zero_()

        # mega_moe writes into a caller-provided output. Per-call empty
        # is the simplest scope; PyTorch's caching allocator amortises
        # the cost. Switch to a BumpAllocator if profiling shows alloc
        # pressure.
        y = torch.empty(
            (num_tokens, self.hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            self.mega_l1_weights,
            self.mega_l2_weights,
            buf,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=None,  # DSV4 has no swiglu_limit by default
            fast_math=True,
        )

        # Routed scaling factor — sglang has a should_fuse_in_topk
        # toggle. Nanodeploy's topk path doesn't fold it in, so apply
        # in-place here. Matches the existing legacy-decode behaviour.
        if self.routed_scaling_factor != 1.0:
            y.mul_(self.routed_scaling_factor)

        if self.tp_size > 1 and self.tp_group is not None:
            torch.distributed.all_reduce(y, group=self.tp_group)

        return y
