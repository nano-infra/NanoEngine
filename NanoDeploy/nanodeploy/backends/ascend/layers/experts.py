"""Ascend NPU distributed routed experts (BF16, HCCL, torch_npu MoE ops).

Supports:
  - ep_size == 1: local compute (same as GenericDistributedRoutedExperts)
  - ep_size > 1: Ascend-native EP via AscendTokenDispatcherNormal (prefill)
                 and AscendTokenDispatcherLowLatency (decode)
  - TP via all-reduce (HCCL)
No FP8 support — BF16 only.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanodeploy.backends.base_backend import DistributedRoutedExpertsBase
from nanodeploy.context.expert_context import ExpertContext
from nanodeploy.layers.local_dispatch import LocalPaddedDispatcher


class AscendDistributedRoutedExperts(DistributedRoutedExpertsBase):
    """BF16 MoE experts for Ascend NPU with optional EP via HCCL + torch_npu ops."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        tp_size: int,
        ep_group: Optional[dist.ProcessGroup] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
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
        self.ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0

        assert (
            intermediate_size * 2
        ) % tp_size == 0, "intermediate_size * 2 must be divisible by tp_size"
        self.tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        self.local_intermediate_size = intermediate_size // tp_size

        # BF16 weight tensors: [num_local_experts, local_intermediate*2, hidden]
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
        self.gate_up_scale_inv = None
        self.down_scale_inv = None

        # Lazy-init local dispatcher for ep==1 decode path
        self._local_dispatcher: Optional[LocalPaddedDispatcher] = None
        # Lazy-init decode EP dispatcher (cached across forward calls)
        self._decode_dispatcher = None

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        if self.ep_size > 1:
            if is_prefill:
                return self._compute_prefill_ep(
                    hidden_states, topk_ids, topk_weights
                )
            else:
                return self._compute_decode_ep(hidden_states, topk_ids, topk_weights)
        else:
            return self._compute_local(hidden_states, topk_ids, topk_weights, is_prefill)

    # ------------------------------------------------------------------
    # EP == 1 paths (reuse GenericDistributedRoutedExperts logic)
    # ------------------------------------------------------------------

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
        num_tokens, hidden_size = hidden_states.shape
        top_k = topk_ids.shape[1]

        flat_expert_ids = topk_ids.flatten()
        flat_weights = topk_weights.flatten()
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
            expert_input = repeated_hidden[mask]
            gate_up = F.linear(expert_input, self.gate_up_proj[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            expert_out = F.linear(act, self.down_proj[expert_id])
            output[mask] = expert_out

        output = output * flat_weights.unsqueeze(-1)
        output = output.reshape(num_tokens, top_k, hidden_size).sum(dim=1)

        if self.tp_size > 1 and self.tp_group is not None:
            dist.all_reduce(output, group=self.tp_group)

        return output

    def _compute_local_decode(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        disp = self._get_or_create_local_dispatcher()
        T = hidden_states.shape[0]
        padded_buf, masked_m, _ = disp.dispatch(hidden_states, topk_ids)

        gateup = torch.bmm(padded_buf, self.gate_up_proj.transpose(-1, -2))
        gate, up = gateup.chunk(2, dim=-1)
        down_input = F.silu(gate) * up
        down_output = torch.bmm(down_input, self.down_proj.transpose(-1, -2))

        out = disp.combine(down_output, topk_ids, topk_weights, T)

        if self.tp_size > 1 and self.tp_group is not None:
            dist.all_reduce(out, group=self.tp_group)

        return out

    # ------------------------------------------------------------------
    # EP > 1: Prefill (AllGather path via AscendTokenDispatcherNormal)
    # ------------------------------------------------------------------

    def _compute_prefill_ep(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        from nanodeploy.layers.token_dispatcher import AscendTokenDispatcherNormal

        ctx = ExpertContext.get_instance()
        assert ctx.warmup_called, "ExpertContext must be warmed up before EP compute"

        dispatcher = AscendTokenDispatcherNormal(
            group=self.ep_group,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            params_dtype=self.gate_up_proj.dtype,
        )

        (
            recv_hidden,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            _event,
        ) = dispatcher.dispatch(hidden_states, topk_ids, topk_weights)

        # Local compute on received tokens
        output = self._compute_local_prefill_recv(
            recv_hidden, recv_topk_idx, recv_tokens_per_expert
        )

        return dispatcher.combine(output)

    def _compute_local_prefill_recv(
        self,
        recv_hidden: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        recv_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Compute on tokens received after EP dispatch.

        recv_topk_idx contains global expert ids; tokens are in (source_rank, expert)
        order — NOT sorted by local expert. Use masking to select the right tokens.
        """
        output = torch.zeros_like(recv_hidden)
        # recv_topk_idx: [total_recv, 1] with global expert ids
        expert_ids = recv_topk_idx.squeeze(-1)
        for local_expert_id in range(self.num_local_experts):
            global_expert_id = self.ep_rank * self.num_local_experts + local_expert_id
            mask = (expert_ids == global_expert_id)
            if not mask.any():
                continue
            expert_input = recv_hidden[mask]
            gate_up = F.linear(expert_input, self.gate_up_proj[local_expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            output[mask] = F.linear(act, self.down_proj[local_expert_id])

        if self.tp_size > 1 and self.tp_group is not None:
            dist.all_reduce(output, group=self.tp_group)

        return output

    # ------------------------------------------------------------------
    # EP > 1: Decode (AllGather + npu_grouped_matmul + ReduceScatter)
    # ------------------------------------------------------------------

    def _get_decode_dispatcher(self):
        """Lazy-init and cache the decode dispatcher (one per layer lifetime)."""
        if self._decode_dispatcher is None:
            from nanodeploy.layers.token_dispatcher import AscendTokenDispatcherAllGather

            ctx = ExpertContext.get_instance()
            assert ctx.warmup_called, "ExpertContext must be warmed up before EP compute"

            self._decode_dispatcher = AscendTokenDispatcherAllGather(
                group=self.ep_group,
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                hidden_size=self.hidden_size,
                params_dtype=self.gate_up_proj.dtype,
                top_k=self.top_k,
            )
        return self._decode_dispatcher

    def _compute_decode_ep(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        dispatcher = self._get_decode_dispatcher()

        (
            recv_hidden,
            recv_topk_idx,
            recv_topk_weights,
            expert_token_nums,  # [num_local_experts] padded counts, sum=active_num
            group_list_type,    # 1 = count mode
        ) = dispatcher.dispatch(hidden_states, topk_ids, topk_weights)

        # npu_grouped_matmul routes rows to experts by group_list counts.
        # Output is always [active_num, ...] (fixed shape from padded counts).
        gate_up_out = torch_npu.npu_grouped_matmul(
            x=[recv_hidden],
            weight=[self.gate_up_proj.transpose(1, 2)],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=expert_token_nums,
        )[0]

        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

        down_output = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[self.down_proj.transpose(1, 2)],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=expert_token_nums,
        )[0]

        if self.tp_size > 1 and self.tp_group is not None:
            dist.all_reduce(down_output, group=self.tp_group)

        result = dispatcher.combine(down_output, recv_topk_idx, recv_topk_weights)

        return result
