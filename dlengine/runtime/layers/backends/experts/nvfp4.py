"""ModelOpt NVFP4/group-16 routed experts using DeepEP + FlashInfer CuteDSL."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from dlengine.runtime.layers.base_backend import DistributedRoutedExpertsBase


def _swizzle_blockscale(scale):
    """Match SGLang CuteDSL v1 block-scale layout exactly."""
    assert scale.dtype == torch.float8_e4m3fn and scale.ndim == 3
    b, m, k = scale.shape
    mp = (m + 127) // 128 * 128
    kp = (k + 3) // 4 * 4
    padded = torch.zeros((b, mp, kp), dtype=scale.dtype, device=scale.device)
    padded[:, :m, :k] = scale
    return (
        padded.reshape(b, mp // 128, 4, 32, kp // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(b, mp, kp)
    )


def _interleave_w13_halves(tensor: torch.Tensor) -> torch.Tensor:
    """Convert checkpoint [Gate, Up] halves to CuteDSL v2 [Up, Gate] tiles."""
    split = tensor.shape[1] // 2
    gate = tensor[:, :split]
    up = tensor[:, split:]
    chunks = []
    for up_chunk, gate_chunk in zip(up.split(64, dim=1), gate.split(64, dim=1)):
        chunks.extend((up_chunk, gate_chunk))
    return torch.cat(chunks, dim=1).contiguous()


class ModelOptNvFp4Experts(DistributedRoutedExpertsBase):
    """ModelOpt NVFP4 experts sharded over the FFN EP axis."""

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
        quantization_config=None,
        activation: str = "silu",
        **_: object,
    ) -> None:
        super().__init__()
        if tp_size != 1:
            raise ValueError(f"ModelOpt NVFP4 requires ffn_tp=1, got {tp_size}")
        if num_experts % ep_size:
            raise ValueError(f"num_experts={num_experts} not divisible by ep={ep_size}")
        if not getattr(quantization_config, "is_modelopt_nvfp4", False):
            raise ValueError("ModelOptNvFp4Experts requires quant_algo=NVFP4")
        if activation not in ("silu", "swiglu"):
            raise ValueError(f"NVFP4 experts require SwiGLU, got {activation!r}")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_group = ep_group
        device = torch.get_default_device()
        e, h, n = self.num_local_experts, hidden_size, intermediate_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(e, 2 * n, h // 2, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(e, h, n // 2, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.gate_up_scale = nn.Parameter(
            torch.empty(e, 2 * n, h // 16, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.down_scale = nn.Parameter(
            torch.empty(e, h, n // 16, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.gate_up_scale_2 = nn.Parameter(
            torch.empty(e, 2, dtype=torch.float32, device=device), requires_grad=False
        )
        self.down_scale_2 = nn.Parameter(
            torch.empty(e, dtype=torch.float32, device=device), requires_grad=False
        )
        self.gate_up_input_scale = nn.Parameter(
            torch.empty(e, 2, dtype=torch.float32, device=device), requires_grad=False
        )
        self.down_input_scale = nn.Parameter(
            torch.empty(e, dtype=torch.float32, device=device), requires_grad=False
        )
        self._prepared = False

    def load_expert_weight(self, expert_idx, projection, kind, tensor, *, ep_rank):
        start = ep_rank * self.num_local_experts
        if not start <= expert_idx < start + self.num_local_experts:
            return True
        local = expert_idx - start
        if projection not in ("gate_proj", "up_proj", "down_proj"):
            return False
        half = 0 if projection == "gate_proj" else 1
        if projection == "down_proj":
            targets = {
                "weight": self.down_proj[local],
                "weight_scale": self.down_scale[local],
                "weight_scale_2": self.down_scale_2[local],
                "input_scale": self.down_input_scale[local],
            }
        else:
            row = slice(
                half * self.intermediate_size, (half + 1) * self.intermediate_size
            )
            targets = {
                "weight": self.gate_up_proj[local, row],
                "weight_scale": self.gate_up_scale[local, row],
                "weight_scale_2": self.gate_up_scale_2[local, half],
                "input_scale": self.gate_up_input_scale[local, half],
            }
        target = targets.get(kind)
        if target is None:
            return False
        # copy_ performs CPU->GPU transfer and dtype conversion directly.
        # Avoid a temporary CUDA allocation for every expert tensor.
        target.copy_(tensor.reshape(target.shape))
        return True

    @torch.no_grad()
    def process_weights_after_loading(self):
        if self._prepared:
            return

        # Match SGLang CuteDSL v1 scale formulas exactly. GEMM1 has one
        # global activation scale; GEMM2 keeps one scale per local expert.
        input1 = self.gate_up_input_scale.max().float()
        if self.ep_group is not None and self.ep_size > 1:
            torch.distributed.all_reduce(
                input1, op=torch.distributed.ReduceOp.MAX, group=self.ep_group
            )
        input2 = self.down_input_scale.float()

        self.input1_quant = (1.0 / input1).repeat(self.num_local_experts).contiguous()
        self.g1_alphas = (input1 * self.gate_up_scale_2[:, 0]).float().contiguous()
        self.input2_quant = (1.0 / input2).float().contiguous()
        self.g2_alphas = (input2 * self.down_scale_2).float().contiguous()

        if self.ep_size == 1:
            # SGLang's non-DeepEP path uses CuteDSL v2. It has a different
            # contract from the masked v1 kernel used by DeepEP: W13 is
            # [Up, Gate] interleaved in 64-row tiles and blockscales use the
            # MMA layout. Keep this conversion isolated to EP1.
            from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

            self.gate_up_proj.data = _interleave_w13_halves(self.gate_up_proj.data)
            self.gate_up_scale.data = _interleave_w13_halves(
                self.gate_up_scale.data
            )
            self.gate_up_scale.data = convert_sf_to_mma_layout(
                _swizzle_blockscale(self.gate_up_scale.data)
                .contiguous()
                .view(torch.uint8)
                .reshape(-1),
                m=self.gate_up_proj.shape[1],
                k=self.hidden_size,
                num_groups=self.num_local_experts,
                sf_vec_size=16,
            )
            self.down_scale.data = convert_sf_to_mma_layout(
                _swizzle_blockscale(self.down_scale.data)
                .contiguous()
                .view(torch.uint8)
                .reshape(-1),
                m=self.down_proj.shape[1],
                k=self.intermediate_size,
                num_groups=self.num_local_experts,
                sf_vec_size=16,
            )
            self.local_input1_quant = (1.0 / input1).reshape(1).float()
            input2_scalar = self.down_input_scale.max().float()
            self.local_input2_quant = (1.0 / input2_scalar).reshape(1).float()
            self.local_g1_alphas = (
                self.gate_up_scale_2[:, 0] / self.local_input1_quant
            ).float().contiguous()
            self.local_g2_alphas = (
                self.down_scale_2 / self.local_input2_quant
            ).float().contiguous()
            self._local_wrapper = None
            self._prepared = True
            return

        # CuteDSL v1 uses original [Gate, Up] weights. Only block scales
        # are swizzled; TRT-LLM row permutation would corrupt this path.
        self.gate_up_scale.data = _swizzle_blockscale(self.gate_up_scale.data)
        self.down_scale.data = _swizzle_blockscale(self.down_scale.data)
        self._prepared = True

    def _run_masked_experts(self, hidden_states, masked_m, input_global_scale):
        from dlengine.runtime.kernel.cutedsl_nvfp4_moe import (
            flashinfer_cutedsl_moe_masked,
        )

        return flashinfer_cutedsl_moe_masked(
            hidden_states=hidden_states,
            input_global_scale=input_global_scale,
            w1=self.gate_up_proj,
            w1_blockscale=self.gate_up_scale,
            w1_alpha=self.g1_alphas,
            w2=self.down_proj,
            a2_global_scale=self.input2_quant,
            w2_blockscale=self.down_scale,
            w2_alpha=self.g2_alphas,
            masked_m=masked_m.to(torch.int32),
            activation="silu",
        )

    def _compute_prefill_ep(self, hidden_states, topk_ids, topk_weights):
        """Run prefill with DeepEP normal dispatch and padded local experts."""
        from dlengine.runtime.context.expert import ExpertContext
        from dlengine.runtime.layers.local_dispatch import LocalPaddedDispatcher
        from dlengine.runtime.layers.token_dispatcher import DeepEPTokenDispatcherNormal

        ExpertContext.get_instance().transition_to_normal()
        dispatcher = DeepEPTokenDispatcherNormal(
            group=self.ep_group,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            expert_alignment=1,
        )
        recv_x, recv_ids, recv_weights, _, _, _ = dispatcher.dispatch(
            hidden_states, topk_ids, topk_weights
        )
        if recv_x.shape[0] == 0:
            return dispatcher.combine(recv_x)

        recv_tokens = recv_x.shape[0]
        # A/B against the last known-good NVFP4 implementation.  The later
        # worst-case sizing changes the masked-GEMM M bucket even for small,
        # normally distributed batches.
        max_m = max(
            128,
            2
            * (recv_tokens * self.top_k + self.num_local_experts - 1)
            // self.num_local_experts,
        )
        local_dispatcher = LocalPaddedDispatcher(
            num_local_experts=self.num_local_experts,
            max_m=max_m,
            hidden_size=self.hidden_size,
            top_k=self.top_k,
            max_num_tokens=recv_tokens,
            device=self.gate_up_proj.device,
        )
        padded_x, masked_m, _ = local_dispatcher.dispatch(recv_x, recv_ids)
        expert_out = self._run_masked_experts(
            (padded_x, None), masked_m, self.input1_quant
        )
        recv_out = local_dispatcher.combine(
            expert_out, recv_ids, recv_weights, recv_tokens
        )
        return dispatcher.combine(recv_out)

    def _compute_decode_ep(self, hidden_states, topk_ids, topk_weights):
        """Run decode with DeepEP low-latency native NVFP4 dispatch."""
        from dlengine.runtime.context.expert import ExpertContext
        from dlengine.runtime.layers.token_dispatcher import (
            DeepEPTokenDispatcherLowLatency,
        )

        ExpertContext.get_instance().transition_to_low_latency()
        dispatcher = DeepEPTokenDispatcherLowLatency(
            group=self.ep_group,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
        )
        recv_x, recv_ids, recv_weights, masked_m, _ = dispatcher.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            self.num_experts,
            use_fp8=False,
            use_nvfp4=True,
            x_global_scale=self.input1_quant,
        )
        recv_hidden, recv_scale = (
            recv_x if isinstance(recv_x, tuple) else (recv_x, None)
        )
        # DeepEP has already quantized the activation when it returns NVFP4 scales.
        # Passing input1_quant again would apply the global scale twice.
        input_global_scale = None if recv_scale is not None else self.input1_quant
        expert_out = self._run_masked_experts(
            (recv_hidden, recv_scale), masked_m, input_global_scale
        )
        return dispatcher.combine(expert_out, recv_ids, recv_weights)

    def _compute_local(self, hidden_states, topk_ids, topk_weights):
        """Run the SGLang-aligned CuteDSL v2 standard path for EP1."""
        from flashinfer import ActivationType, CuteDslMoEWrapper, fp4_quantize

        if self._local_wrapper is None:
            from dlengine.runtime.runner.runner_config import get_runner_config

            max_tokens = get_runner_config().max_num_batched_tokens or 4096
            with torch.inference_mode(False):
                self._local_wrapper = CuteDslMoEWrapper(
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    use_cuda_graph=False,
                    max_num_tokens=max_tokens,
                    num_local_experts=self.num_local_experts,
                    local_expert_offset=0,
                    output_dtype=torch.bfloat16,
                    device=str(self.gate_up_proj.device),
                    activation_type=ActivationType.Swiglu,
                )

        x_fp4, x_sf = fp4_quantize(
            input=hidden_states,
            global_scale=self.local_input1_quant,
            sf_vec_size=16,
            is_sf_swizzled_layout=False,
            backend="cute-dsl",
        )
        return self._local_wrapper.run(
            x=x_fp4,
            x_sf=x_sf,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            w1_weight=self.gate_up_proj,
            w1_weight_sf=self.gate_up_scale,
            w1_alpha=self.local_g1_alphas,
            fc2_input_scale=self.local_input2_quant,
            w2_weight=self.down_proj,
            w2_weight_sf=self.down_scale,
            w2_alpha=self.local_g2_alphas,
        )

    def forward(self, hidden_states, topk_ids, topk_weights, is_prefill=True):
        if not self._prepared:
            raise RuntimeError("NVFP4 expert weights have not been prepared")
        if self.ep_size <= 1:
            return self._compute_local(hidden_states, topk_ids, topk_weights)
        if is_prefill:
            return self._compute_prefill_ep(hidden_states, topk_ids, topk_weights)
        return self._compute_decode_ep(hidden_states, topk_ids, topk_weights)
