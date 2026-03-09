"""Generic GPU (BF16) distributed routed experts.

This is a minimal MoE implementation for non-Hopper GPUs.
Expert parallelism (ep_size > 1) is not supported — raise NotImplementedError
if attempted.  Single-rank BF16 MoE (SwiGLU) is fully functional.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from nanodeploy.backends.base_backend import DistributedRoutedExpertsBase
from nanodeploy.layers.local_dispatch import LocalPaddedDispatcher


class GenericDistributedRoutedExperts(DistributedRoutedExpertsBase):
    """BF16-only MoE experts for generic GPUs.

    Supports ep_size=1 (local compute) with optional TP all-reduce.
    Raises ``NotImplementedError`` when ep_size > 1.
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
        self.is_fp8 = False  # BF16-only

        assert (
            num_experts % ep_size == 0
        ), f"num_experts {num_experts} must be divisible by ep_size {ep_size}"
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

        # BF16 weight tensors
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.local_intermediate_size * 2,
                hidden_size,
                dtype=torch.bfloat16,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                hidden_size,
                self.local_intermediate_size,
                dtype=torch.bfloat16,
            )
        )
        # No scale tensors for BF16
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
        if self.ep_size > 1:
            raise NotImplementedError(
                "Expert parallelism (ep_size > 1) requires the Hopper backend. "
                "Use NANO_BACKEND=hopper or run on an H100/H200 GPU."
            )
        return self._compute_local(hidden_states, topk_ids, topk_weights, is_prefill)

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
    ) -> torch.Tensor:
        if is_prefill:
            return self._compute_local_prefill(hidden_states, topk_ids, topk_weights)
        return self._compute_local_decode(hidden_states, topk_ids, topk_weights)

    def _compute_local_prefill(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch SwiGLU MoE forward (BF16, single-rank)."""
        num_tokens, hidden_size = hidden_states.shape
        top_k = topk_ids.shape[1]

        # Flatten tokens × top_k
        flat_expert_ids = topk_ids.flatten()  # [num_tokens * top_k]
        flat_weights = topk_weights.flatten()  # [num_tokens * top_k]

        # Gather inputs for each selected expert slot
        # expand: [num_tokens, hidden] → [num_tokens * top_k, hidden]
        repeated_hidden = (
            hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
        )

        output = torch.zeros_like(repeated_hidden)

        for expert_id in range(self.num_local_experts):
            mask = flat_expert_ids == (
                self.ep_rank * self.num_local_experts + expert_id
            )
            if not mask.any():
                continue
            expert_input = repeated_hidden[mask]  # [n_sel, hidden]
            # gate_up_proj: [local_intermediate * 2, hidden]
            gate_up = F.linear(expert_input, self.gate_up_proj[expert_id])
            # SwiGLU
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            # down_proj: [hidden, local_intermediate]
            expert_out = F.linear(act, self.down_proj[expert_id])
            output[mask] = expert_out

        # Weight and sum over top_k dimension
        output = output * flat_weights.unsqueeze(-1)
        output = output.reshape(num_tokens, top_k, hidden_size).sum(dim=1)

        if self.tp_size > 1 and self.tp_group is not None:
            torch.distributed.all_reduce(output, group=self.tp_group)

        return output

    def _compute_local_decode(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Decode path – CUDA-Graph safe, uses LocalPaddedDispatcher + bmm."""
        disp = self._get_or_create_local_dispatcher()
        T = hidden_states.shape[0]
        padded_buf, masked_m, _ = disp.dispatch(
            hidden_states, topk_ids
        )  # [E, max_m, H]

        # Gate-Up: [E, max_m, 2*inter]
        gateup = torch.bmm(padded_buf, self.gate_up_proj.transpose(-1, -2))
        gate, up = gateup.chunk(2, dim=-1)
        down_input = F.silu(gate) * up  # [E, max_m, inter]

        # Down: [E, max_m, H]
        down_output = torch.bmm(down_input, self.down_proj.transpose(-1, -2))

        out = disp.combine(down_output, topk_ids, topk_weights, T)

        if self.tp_size > 1 and self.tp_group is not None:
            torch.distributed.all_reduce(out, group=self.tp_group)

        return out
