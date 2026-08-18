"""ModelOpt NVFP4/group-16 routed experts using DeepEP + FlashInfer CuteDSL."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from dlengine.layers.base_backend import DistributedRoutedExpertsBase


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

        # CuteDSL v1 uses original [Gate, Up] weights. Only block scales
        # are swizzled; TRT-LLM row permutation would corrupt this path.
        self.gate_up_scale.data = _swizzle_blockscale(self.gate_up_scale.data)
        self.down_scale.data = _swizzle_blockscale(self.down_scale.data)
        self._prepared = True

    def forward(self, hidden_states, topk_ids, topk_weights, is_prefill=True):
        if not self._prepared:
            raise RuntimeError("NVFP4 expert weights have not been prepared")
        if self.ep_size <= 1:
            raise NotImplementedError(
                "ModelOpt NVFP4 CuteDSL currently requires EP > 1"
            )

        # Match the verified SGLang recipe for both prefill and decode:
        # DeepEP low-latency dispatch -> masked CuteDSL grouped GEMMs -> combine.
        from dlengine.context.expert import ExpertContext
        from dlengine.kernel.cutedsl_nvfp4_moe import flashinfer_cutedsl_moe_masked
        from dlengine.layers.token_dispatcher import DeepEPTokenDispatcherLowLatency

        ctx = ExpertContext.get_instance()
        ctx.transition_to_low_latency()
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
        )
        if isinstance(recv_x, tuple):
            recv_hidden, recv_scale = recv_x
        else:
            recv_hidden, recv_scale = recv_x, None

        expert_out = flashinfer_cutedsl_moe_masked(
            hidden_states=(recv_hidden, recv_scale),
            input_global_scale=self.input1_quant,
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
        return dispatcher.combine(expert_out, recv_ids, recv_weights)
