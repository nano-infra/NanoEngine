"""Packed-MXFP4 routed experts using DeepGEMM MegaMoE.

MegaMoE owns only the FFN expert-parallel axis. It neither inspects attention
TP nor pads the request batch to a TP multiple. Missing kernels and unsupported
layouts are hard errors; this backend never falls back to DeepEP or torch.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from dlengine.layers.base_backend import DistributedRoutedExpertsBase
from dlengine.worker.runner_config import get_runner_config

KIMI_K3_SITU_SENTINEL_CLAMP = 0.03125
_SYMM_BUFFER_CACHE = {}


def require_deep_gemm():
    try:
        import deep_gemm
    except ImportError as exc:  # pragma: no cover - runtime image dependent
        raise RuntimeError(
            "MegaMoE requires a DeepGEMM build with the fp8_fp4 MegaMoE API."
        ) from exc
    required = (
        "fp8_fp4_mega_moe",
        "get_symm_buffer_for_mega_moe",
        "mega_moe_pre_dispatch",
        "transform_sf_into_required_layout",
        "transform_weights_for_mega_moe",
    )
    missing = [name for name in required if not hasattr(deep_gemm, name)]
    if missing:
        raise RuntimeError(
            "DeepGEMM is missing required MegaMoE APIs: " + ", ".join(missing)
        )
    return deep_gemm


class MegaMoEExperts(DistributedRoutedExpertsBase):
    """Kimi-K3 MXFP4 + SiTU expert backend for Blackwell."""

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
        layer_idx: int = -1,
        activation: str = "situ",
        activation_situ_beta: float = 4.0,
        activation_situ_linear_beta: float = 25.0,
        routed_scaling_factor: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        if ep_size <= 1:
            raise ValueError("MegaMoE requires expert parallelism (ffn_ep > 1).")
        if tp_size != 1:
            raise ValueError(
                "MegaMoE uses the FFN EP axis and requires ffn_tp=1; "
                f"got tp_size={tp_size}. Attention TP is independent."
            )
        if num_experts % ep_size:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by ep_size={ep_size}"
            )
        if not bool(getattr(quantization_config, "is_mxfp4", False)):
            raise ValueError("MegaMoE requires packed MXFP4 expert weights.")
        if activation != "situ":
            raise ValueError(f"K3 MegaMoE requires activation='situ', got {activation!r}")
        if (float(activation_situ_beta), float(activation_situ_linear_beta)) != (
            4.0,
            25.0,
        ):
            raise ValueError(
                "The K3 SiTU MegaMoE kernel requires beta=4 and linear_beta=25."
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.top_k = top_k
        self.ep_size = ep_size
        self.tp_size = tp_size
        self.ep_group = ep_group
        self.tp_group = tp_group
        self.layer_idx = layer_idx
        self.routed_scaling_factor = float(routed_scaling_factor)

        # K3 stores two E2M1 values per byte. Experts are sharded by EP only.
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                intermediate_size * 2,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.gate_up_scale = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                intermediate_size * 2,
                hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.down_scale = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                hidden_size,
                intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.mega_l1_weights = None
        self.mega_l2_weights = None
        self._mega_moe_buf = None

    def load_expert_weight(
        self,
        expert_idx: int,
        projection: str,
        kind: str,
        tensor: torch.Tensor,
        *,
        ep_rank: int,
    ) -> bool:
        """Load one K3 expert tensor from the packed checkpoint.

        K3 names the projections ``w1`` (gate), ``w3`` (up), and ``w2``
        (down). Expert ownership is contiguous on the FFN EP axis; attention
        TP is deliberately absent from this mapping.

        Returns ``True`` for both locally loaded and non-local (consumed but
        skipped) tensors, and raises for malformed local tensors.
        """
        experts_per_rank = self.num_local_experts
        expert_start = ep_rank * experts_per_rank
        expert_end = expert_start + experts_per_rank
        if not expert_start <= expert_idx < expert_end:
            return True
        local_idx = expert_idx - expert_start

        projection_map = {
            "w1": (self.gate_up_proj, self.gate_up_scale, 0),
            "w3": (
                self.gate_up_proj,
                self.gate_up_scale,
                self.intermediate_size,
            ),
            "w2": (self.down_proj, self.down_scale, 0),
        }
        if projection not in projection_map:
            raise ValueError(f"Unsupported K3 expert projection {projection!r}")
        weight, scale, row_start = projection_map[projection]
        if kind == "weight_packed":
            target = weight[local_idx]
        elif kind == "weight_scale":
            target = scale[local_idx]
        else:
            raise ValueError(f"Unsupported K3 expert tensor kind {kind!r}")

        if projection in ("w1", "w3"):
            target = target[row_start : row_start + self.intermediate_size]
        if target.shape != tensor.shape:
            raise ValueError(
                f"K3 expert {expert_idx} {projection}.{kind} has shape "
                f"{tuple(tensor.shape)}, expected {tuple(target.shape)}"
            )
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        return True

    @property
    def max_tokens_per_rank(self) -> int:
        return int(get_runner_config().mega_moe_max_tokens_per_rank)

    def prepare_mega_weights(self) -> None:
        """Convert checkpoint E2M1/E8M0 tensors to MegaMoE's layout."""
        if self.mega_l1_weights is not None:
            return
        deep_gemm = require_deep_gemm()

        def transform(weight: torch.Tensor, scale: torch.Tensor):
            # E8M0 bytes are biased exponents; conversion through fp32 is exact.
            scale_f32 = scale.view(torch.float8_e8m0fnu).to(torch.float32)
            transformed_scale = deep_gemm.transform_sf_into_required_layout(
                scale_f32,
                mn=weight.shape[1],
                k=weight.shape[2] * 2,
                recipe=(1, 32),
                num_groups=weight.shape[0],
                disable_ue8m0_cast=False,
            )
            return weight.view(torch.int8), transformed_scale

        l1 = transform(self.gate_up_proj.data, self.gate_up_scale.data)
        l2 = transform(self.down_proj.data, self.down_scale.data)
        self.mega_l1_weights, self.mega_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(l1, l2)
        )

        # MegaMoE is the only path, so do not retain a second expert layout.
        self.gate_up_proj.data = self.mega_l1_weights[0]
        self.gate_up_scale.data = self.mega_l1_weights[1]
        self.down_proj.data = self.mega_l2_weights[0]
        self.down_scale.data = self.mega_l2_weights[1]

    process_weights_after_loading = prepare_mega_weights

    def _get_buffer(self):
        if self._mega_moe_buf is None:
            deep_gemm = require_deep_gemm()
            key = (
                id(self.ep_group), self.num_experts, self.max_tokens_per_rank,
                self.top_k, self.hidden_size, self.intermediate_size,
            )
            if key not in _SYMM_BUFFER_CACHE:
                _SYMM_BUFFER_CACHE[key] = deep_gemm.get_symm_buffer_for_mega_moe(
                    self.ep_group,
                    self.num_experts,
                    self.max_tokens_per_rank,
                    self.top_k,
                    self.hidden_size,
                    self.intermediate_size,
                    use_fp8_dispatch=True,
                    activation="swiglu",
                )
            self._mega_moe_buf = _SYMM_BUFFER_CACHE[key]
        return self._mega_moe_buf

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        del is_prefill
        if self.mega_l1_weights is None or self.mega_l2_weights is None:
            raise RuntimeError(
                "MegaMoE weights are not prepared; call prepare_mega_weights() "
                "after checkpoint loading."
            )
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_tokens_per_rank:
            raise RuntimeError(
                f"MegaMoE received {num_tokens} tokens; per-rank limit is "
                f"{self.max_tokens_per_rank}."
            )

        buf = self._get_buffer()
        deep_gemm = require_deep_gemm()
        # DeepGEMM packs four group-32 FP8 scales into each int32 x_sf
        # element. A regular quant helper returns one float32 per group and is
        # therefore layout-incompatible with SymmBuffer.x_sf.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights.to(torch.float32),
            buf.x.view(torch.float8_e4m3fn),
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=32,
            use_fp4_acts=False,
        )
        output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        deep_gemm.fp8_fp4_mega_moe(
            output,
            self.mega_l1_weights,
            self.mega_l2_weights,
            buf,
            recipe=(1, 1, 32),
            activation="swiglu",
            # Patched DeepGEMM interprets this sentinel as K3 SiTU.
            activation_clamp=KIMI_K3_SITU_SENTINEL_CLAMP,
            fast_math=True,
        )
        if self.routed_scaling_factor != 1.0:
            output.mul_(self.routed_scaling_factor)
        return output
