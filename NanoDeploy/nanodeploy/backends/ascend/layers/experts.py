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

        # DP context for fused DP-slice-before-allreduce optimisation
        from nanodeploy.context.distributed import get_dist_context
        _dctx = get_dist_context()
        self.dp_rank = _dctx.attn_dp_rank
        self.dp_world_size = _dctx.attn_dp_world_size

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

        # Pre-transposed weight views (cached lazily after weight loading).
        # npu_grouped_matmul expects [E, hidden, intermediate] layout.
        self._gate_up_proj_t: torch.Tensor | None = None
        self._down_proj_t: torch.Tensor | None = None

        # Lazy-init decode EP dispatcher (cached across forward calls)
        self._decode_dispatcher = None

    def _get_gate_up_t(self) -> torch.Tensor:
        if self._gate_up_proj_t is None:
            # View only — no .contiguous() to avoid doubling memory per layer.
            # npu_grouped_matmul handles strided weights internally.
            self._gate_up_proj_t = self.gate_up_proj.data.transpose(1, 2)
        return self._gate_up_proj_t

    def _get_down_t(self) -> torch.Tensor:
        if self._down_proj_t is None:
            self._down_proj_t = self.down_proj.data.transpose(1, 2)
        return self._down_proj_t

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

    def _npu_grouped_matmul_moe(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        dp_slice: bool = False,
    ) -> torch.Tensor:
        """Fused npu_moe_init_routing + npu_grouped_matmul + npu_moe_token_unpermute.

        Uses fused NPU MoE routing ops for token permutation, expert counting,
        and weighted unpermutation. Falls back to manual argsort path if the
        fused ops are unavailable.

        Args:
            dp_slice: When True and dp_world_size > 1, slice output by dp_rank
                before allreduce to halve communication volume.  Only safe when
                batch sizes are uniform across DP ranks (decode).
        """
        import torch_npu

        T, H = hidden_states.shape
        K = topk_ids.shape[1]
        active_num = T * K

        try:
            return self._npu_fused_moe(torch_npu, hidden_states, topk_ids, topk_weights, T, H, K, active_num, dp_slice)
        except (AttributeError, RuntimeError):
            return self._npu_manual_moe(torch_npu, hidden_states, topk_ids, topk_weights, T, H, K, active_num, dp_slice)

    def _npu_fused_moe(
        self,
        torch_npu,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        T: int, H: int, K: int, active_num: int,
        dp_slice: bool = False,
    ) -> torch.Tensor:
        """Fast path: fused npu_moe_init_routing_v2 + npu_moe_token_unpermute."""
        # 1. Fused sort + expand + expert counting
        sorted_hidden, expanded_row_idx, expert_tokens, _ = (
            torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids.to(torch.int32),
                active_num=active_num,
                expert_num=self.num_local_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.num_local_experts],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)

        # 2. npu_grouped_matmul: gate_up
        gate_up_out = torch_npu.npu_grouped_matmul(
            x=[sorted_hidden],
            weight=[self._get_gate_up_t()],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
        )[0]

        # 3. SwiGLU activation
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

        # 4. npu_grouped_matmul: down
        down_output = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[self._get_down_t()],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
        )[0]

        # 5. Fused unpermute + weighted combine (moved before allreduce —
        #    unpermute is linear, so unpermute(allreduce(x)) == allreduce(unpermute(x)))
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=down_output,
            sorted_indices=torch.abs(expanded_row_idx),
            probs=topk_weights,
        )

        # 6. DP slice (halves allreduce volume when attention_dp > 1, decode only)
        if dp_slice and self.dp_world_size > 1:
            tokens_per_rank = output.shape[0] // self.dp_world_size
            start = self.dp_rank * tokens_per_rank
            output = output[start : start + tokens_per_rank].contiguous()

        # 7. TP AllReduce (async for graph-captured overlap)
        if self.tp_size > 1 and self.tp_group is not None:
            work = dist.all_reduce(output, group=self.tp_group, async_op=True)
            work.wait()

        return output

    def _npu_manual_moe(
        self,
        torch_npu,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        T: int, H: int, K: int, active_num: int,
        dp_slice: bool = False,
    ) -> torch.Tensor:
        """Fallback: manual argsort + scatter_add + IndexPutV2."""
        # 1. Expand: [T, H] -> [T*K, H]
        x_expanded = hidden_states.unsqueeze(1).expand(-1, K, -1).reshape(-1, H)
        flat_experts = topk_ids.reshape(-1)  # [T*K]

        # 2. Argsort by expert (float32 keeps it on AiCore)
        perm = flat_experts.to(torch.float32).argsort(stable=True)
        sorted_hidden = x_expanded[perm]

        # 3. Per-expert token counts
        ones = torch.ones(active_num, device=hidden_states.device, dtype=torch.float32)
        expert_counts = torch.zeros(
            self.num_local_experts, device=hidden_states.device, dtype=torch.float32
        )
        expert_counts.scatter_add_(0, flat_experts.to(torch.int64), ones)
        expert_token_nums = expert_counts.to(torch.int64)

        # 4. npu_grouped_matmul: gate_up
        gate_up_out = torch_npu.npu_grouped_matmul(
            x=[sorted_hidden],
            weight=[self._get_gate_up_t()],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_token_nums,
        )[0]

        # 5. SwiGLU activation
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

        # 6. npu_grouped_matmul: down
        down_output = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[self._get_down_t()],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_token_nums,
        )[0]

        # 7. Unsort and weight-sum (moved before allreduce — linear ops commute)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(active_num, device=perm.device, dtype=perm.dtype)
        unsorted = down_output[inv_perm]

        flat_weights = topk_weights.reshape(-1).unsqueeze(-1).to(unsorted.dtype)
        weighted = unsorted * flat_weights
        output = weighted.reshape(T, K, H).sum(dim=1)

        # 8. DP slice (halves allreduce volume when attention_dp > 1, decode only)
        if dp_slice and self.dp_world_size > 1:
            tokens_per_rank = output.shape[0] // self.dp_world_size
            start = self.dp_rank * tokens_per_rank
            output = output[start : start + tokens_per_rank].contiguous()

        # 9. TP AllReduce (async for graph-captured overlap)
        if self.tp_size > 1 and self.tp_group is not None:
            work = dist.all_reduce(output, group=self.tp_group, async_op=True)
            work.wait()

        return output

    def _compute_local_prefill(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._npu_grouped_matmul_moe(hidden_states, topk_ids, topk_weights)

    def _compute_local_decode(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._npu_grouped_matmul_moe(hidden_states, topk_ids, topk_weights, dp_slice=True)

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

        # TP AllReduce (async for graph-captured overlap)
        if self.tp_size > 1 and self.tp_group is not None:
            work = dist.all_reduce(output, group=self.tp_group, async_op=True)
            work.wait()

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
            weight=[self._get_gate_up_t()],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=expert_token_nums,
        )[0]

        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

        down_output = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[self._get_down_t()],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=expert_token_nums,
        )[0]

        # Unpermute/combine before allreduce (linear ops commute)
        result = dispatcher.combine(down_output, recv_topk_idx, recv_topk_weights)

        # DP slice (halves allreduce volume when attention_dp > 1)
        if self.dp_world_size > 1:
            tokens_per_rank = result.shape[0] // self.dp_world_size
            start = self.dp_rank * tokens_per_rank
            result = result[start : start + tokens_per_rank].contiguous()

        # TP AllReduce (async for graph-captured overlap)
        if self.tp_size > 1 and self.tp_group is not None:
            work = dist.all_reduce(result, group=self.tp_group, async_op=True)
            work.wait()

        return result
