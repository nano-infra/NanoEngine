# Copyright (c) 2025, DeepLink.
try:
    # deepep 1.2.1+9af0e0d
    from deep_ep import Buffer

    use_deepep = True
except ImportError:
    use_deepep = False

import os
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from nanodeploy.context.expert_context import ExpertContext


class DeepEPMode(Enum):
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"


class DeepEPTokenDispatcherNormal:
    """Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
    https://github.com/NVIDIA/Megatron-
    LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        expert_alignment: int = 128,
    ):
        self.dispatch_count = 0
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_bytes = params_dtype.itemsize if params_dtype is not None else 2

        ctx = ExpertContext.get_instance()
        assert (
            ctx.warmup_called
        ), "ExpertContext must be warmed up before instantiating dispatchers"

        self.buffer_normal = ctx.get_buffer()
        if self.group.size() > 1 and self.buffer_normal is None:
            raise RuntimeError("DeepEP Buffer is None but ep_size > 1")

        self.expert_alignment = expert_alignment

        # In Normal Mode, DeepEP does not explicitly bound `num_max_dispatch_tokens_per_rank` in dispatch layout calculation.
        self.num_max_dispatch_tokens_per_rank = -1
        self.handle = None

    def get_buffer(self):
        return self.buffer_normal

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: Optional[int] = None,
        previous_event=None,
    ):
        hidden_states, x_scales = x if isinstance(x, tuple) else (x, None)
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        (
            x,
            topk_idx,
            topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        ) = self.dispatch_normal(
            x, topk_idx, topk_weights, self.num_experts, previous_event
        )

        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        return x, topk_idx, topk_weights, recv_tokens_per_expert, handle, event

    def dispatch_normal(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        previous_event=None,
    ):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.get_buffer().get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        ) = self.get_buffer().dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights.to(torch.float32),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
            expert_alignment=self.expert_alignment,
        )  # Note: expert_alignment = 128 if deepgemm else 1

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        )

    def dispatch_normal_async(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: Optional[int] = None,
        previous_event=None,
        async_finish=True,
    ):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.get_buffer().get_dispatch_layout(
            topk_idx,
            num_experts=self.num_experts if num_experts is None else num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        ) = self.get_buffer().dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
            expert_alignment=self.expert_alignment,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            event,
        )

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, event = self.combine_normal(hidden_states, self.handle)
        self.handle = None
        return hidden_states.view(self.hidden_shape)

    def combine_normal(self, x: torch.Tensor, handle: Tuple, previous_event=None):
        combined_x, _, event = self.get_buffer().combine(
            x,
            handle,
            async_finish=False,
            previous_event=previous_event,
            allocate_on_comm_stream=False,
        )
        return combined_x, event

    def combine_normal_async(
        self, x: torch.Tensor, handle: Tuple, previous_event=None, async_finish=True
    ):
        combined_x, _, event = self.get_buffer().combine(
            x,
            handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )
        return combined_x, event

    def release(self):
        self.handle = None
        self.topk_idx = None
        self.topk_weights = None
        return True


class DeepEPTokenDispatcherLowLatency:

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        return_recv_hook: bool = False,
    ):
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_bytes = params_dtype.itemsize if params_dtype is not None else 2

        ctx = ExpertContext.get_instance()
        assert (
            ctx.warmup_called
        ), "ExpertContext must be warmed up before instantiating dispatchers"

        self.buffer_low_latency = ctx.get_buffer()
        if self.group.size() > 1 and self.buffer_low_latency is None:
            raise RuntimeError("DeepEP Buffer is None but ep_size > 1")

        # Read from ExpertContext to align buffer sizing with dispatch token count
        self.num_max_dispatch_tokens_per_rank = ctx.num_max_dispatch_tokens_per_rank
        self.return_recv_hook = return_recv_hook
        self.handle = None

    def get_buffer(self):
        return self.buffer_low_latency

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: Optional[int] = None,
        use_fp8: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_experts is None:
            num_experts = self.num_experts
        if num_experts is not None and self.num_experts is not None:
            assert self.num_experts == num_experts
        topk_idx = topk_idx.to(torch.int64)
        expected_m = (
            hidden_states.shape[0] * self.get_buffer().group_size * topk_idx.shape[1]
            + num_experts
        ) // num_experts

        (
            packed_recv_hidden,
            masked_m,
            self.handle,
            event,
            hook,
        ) = self.get_buffer().low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8=use_fp8,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        )
        hook() if self.return_recv_hook else event.current_stream_wait()
        return (
            packed_recv_hidden,
            topk_idx,
            topk_weights,
            masked_m,
            expected_m,
        )

    # TODO: add use_ue8m0 and use_nvfp4 with round_scale support
    def dispatch_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: Optional[int] = None,
        use_fp8: bool = True,
        async_finish: bool = True,
    ):
        assert topk_idx.dtype == torch.int64
        if num_experts is not None and self.num_experts is not None:
            assert self.num_experts == num_experts
        (
            recv_hidden_states,
            recv_expert_count,
            handle,
            event,
            hook,
        ) = self.get_buffer().low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            num_experts=self.num_experts,
            use_fp8=use_fp8,
            async_finish=async_finish,
            return_recv_hook=not async_finish,
        )
        return recv_hidden_states, recv_expert_count, handle, event, hook

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        combined_hidden_states, event, hook = self.get_buffer().low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights.to(torch.float32),
            self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        )
        hook() if self.return_recv_hook else event.current_stream_wait()
        return combined_hidden_states

    def combine_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Tuple,
        async_finish: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert topk_idx.dtype == torch.int64
        assert topk_weights.dtype == torch.float32
        combined_hidden_states, event, hook = self.get_buffer().low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            async_finish=async_finish,
            return_recv_hook=not async_finish,
        )
        return combined_hidden_states, event, hook


# ---------------------------------------------------------------------------
# Ascend NPU Token Dispatchers
# ---------------------------------------------------------------------------


class AscendTokenDispatcherNormal:
    """Prefill EP dispatcher for Ascend NPU.

    Uses ``torch_npu.npu_moe_init_routing`` for local token sorting by expert
    and ``torch.distributed.all_to_all`` (HCCL) for cross-rank exchange.

    Interface mirrors DeepEPTokenDispatcherNormal.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype = None,
        expert_alignment: int = 1,
    ):
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.ep_size = group.size() if group is not None else 1
        self.ep_rank = dist.get_rank(group) if group is not None else 0

        ctx = ExpertContext.get_instance()
        assert ctx.warmup_called, (
            "ExpertContext must be warmed up (ascend_warmup) before creating dispatchers"
        )

        self.hidden_shape: Optional[torch.Size] = None
        self._dispatch_result = None  # store for combine

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: Optional[int] = None,
        previous_event=None,
    ) -> Tuple:
        """Dispatch tokens to expert ranks via HCCL all_to_all.

        Uses pure PyTorch sorting (no NPU-specific routing ops) so it works
        across all CANN versions.

        Returns:
            (recv_x, recv_topk_idx, recv_topk_weights,
             recv_tokens_per_expert, handle=None, event=None)
        """
        self.hidden_shape = x.shape
        num_tokens = x.shape[0]
        top_k = topk_idx.shape[1]
        topk_idx = topk_idx.to(torch.int64)

        # 1. Expand tokens for all top_k routing slots and sort by expert
        # flat_expert_idx[i] = which expert slot i goes to
        # flat_row_idx[i]    = which original token slot i came from
        flat_expert_idx = topk_idx.reshape(-1)                          # [N*K]
        flat_row_idx = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1).expand(-1, top_k).reshape(-1)
        )                                                                # [N*K]
        flat_weights = topk_weights.reshape(-1)                         # [N*K]

        sort_perm = torch.argsort(flat_expert_idx)
        sorted_expert_idx = flat_expert_idx[sort_perm]
        sorted_row_idx    = flat_row_idx[sort_perm]
        sorted_weights    = flat_weights[sort_perm]
        sorted_x          = x[sorted_row_idx]                           # [N*K, H]

        # expert_token_counts[e] = number of dispatch slots for expert e
        expert_token_counts = torch.bincount(
            flat_expert_idx, minlength=self.num_experts
        )                                                                # [num_experts]

        # 2. Build per-rank send counts and exchange to get recv counts
        L = self.num_local_experts
        send_counts = torch.stack(
            [expert_token_counts[r * L : (r + 1) * L].sum()
             for r in range(self.ep_size)]
        ).to(torch.int64)

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)

        # 3. Exchange tokens via all_to_all
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        total_recv  = sum(recv_splits)

        recv_x = torch.empty(
            total_recv, self.hidden_size, dtype=x.dtype, device=x.device
        )
        recv_sorted_expert_flat = torch.empty(
            total_recv, dtype=torch.int64, device=x.device
        )
        recv_sorted_weights_flat = torch.empty(
            total_recv, dtype=torch.float32, device=x.device
        )

        dist.all_to_all(
            list(recv_x.split(recv_splits, dim=0)),
            list(sorted_x.split(send_splits, dim=0)),
            group=self.group,
        )
        dist.all_to_all(
            list(recv_sorted_expert_flat.split(recv_splits)),
            list(sorted_expert_idx.split(send_splits)),
            group=self.group,
        )
        dist.all_to_all(
            list(recv_sorted_weights_flat.split(recv_splits)),
            list(sorted_weights.float().split(send_splits)),
            group=self.group,
        )

        # recv_tokens_per_expert: how many tokens we received per local expert
        # Count from the received expert indices
        local_expert_start = self.ep_rank * self.num_local_experts
        local_expert_end   = local_expert_start + self.num_local_experts
        recv_tokens_per_expert = torch.bincount(
            recv_sorted_expert_flat,
            minlength=self.num_experts,
        )[local_expert_start:local_expert_end].clone()

        # Store state for combine (sorted_row_idx maps back to original tokens)
        self._dispatch_result = {
            "hidden_shape":   self.hidden_shape,
            "send_splits":    send_splits,
            "recv_splits":    recv_splits,
            "sorted_row_idx": sorted_row_idx,      # [N*K] original token indices
            "sorted_weights": sorted_weights,      # [N*K] topk weights (pre-alltoall order)
            "recv_weights":   recv_sorted_weights_flat,
        }

        return (
            recv_x,
            recv_sorted_expert_flat.unsqueeze(1),  # recv_topk_idx [total_recv, 1]
            recv_sorted_weights_flat.unsqueeze(1), # recv_topk_weights [total_recv, 1]
            recv_tokens_per_expert,
            None,  # handle
            None,  # event
        )

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reverse all_to_all and scatter-add weighted outputs back to original tokens."""
        state = self._dispatch_result
        send_splits = state["send_splits"]
        recv_splits = state["recv_splits"]
        orig_tokens = state["hidden_shape"][0]

        # Reverse all_to_all: send expert outputs back to source ranks
        recv_back = torch.empty(
            sum(send_splits),
            self.hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dist.all_to_all(
            list(recv_back.split(send_splits, dim=0)),
            list(hidden_states.split(recv_splits, dim=0)),
            group=self.group,
        )

        # Scatter-add back to original token positions weighted by topk weights
        # sorted_row_idx[i] = original token index for dispatch slot i
        # sorted_weights[i] = routing weight for dispatch slot i
        sorted_row_idx = state["sorted_row_idx"]
        sorted_weights = state["sorted_weights"].to(hidden_states.dtype)

        output = torch.zeros(
            orig_tokens, self.hidden_size,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        output.scatter_add_(
            0,
            sorted_row_idx.unsqueeze(-1).expand_as(recv_back),
            recv_back * sorted_weights.unsqueeze(-1),
        )

        self._dispatch_result = None
        return output.view(self.hidden_shape)


def _allgather(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """AllGather along dim=0. Output: [tensor.shape[0] * world_size, ...]"""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    output = torch.empty(
        [tensor.shape[0] * world_size] + list(tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(output, tensor, group=group)
    return output


def _reduce_scatter(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """ReduceScatter(sum) along dim=0. Output: [tensor.shape[0] / world_size, ...]"""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    chunk_size = tensor.shape[0] // world_size
    output = torch.empty(
        [chunk_size] + list(tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.reduce_scatter_tensor(output, tensor, group=group)
    return output


class AscendTokenDispatcherAllGather:
    """AllGather-based MoE dispatcher for Ascend NPU decode.

    Fixed-shape collectives (AllGather/ReduceScatter) enable ACL graph capture.
    Uses argsort to sort tokens by expert, then each EP rank computes only its
    local experts.  Inverse permutation restores original token order with
    weighted aggregation, and ReduceScatter sums contributions from all ranks.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype = None,
        top_k: int = 1,
    ):
        self.group = group
        self.ep_size = dist.get_world_size(group)
        self.ep_rank = dist.get_rank(group)
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.top_k = top_k

        # Expert index range for this EP rank
        self.first_expert = self.ep_rank * self.num_local_experts
        self.last_expert = self.first_expert + self.num_local_experts

        # State saved between dispatch and combine
        self._perm: Optional[torch.Tensor] = None
        self._inv_perm: Optional[torch.Tensor] = None
        self._topk_weights: Optional[torch.Tensor] = None
        self._num_local_tokens: int = 0
        self._active_num: int = 0
        self._total_tokens: int = 0
        self._top_k: int = 0
        self._local_start: int = 0

        # Pre-allocated buffers (lazily sized on first dispatch, reused after).
        # Eliminates per-call Range / OnesLike / ZerosLike kernel launches.
        self._buf_active_num: int = 0
        self._arange_buf: Optional[torch.Tensor] = None
        self._ones_buf: Optional[torch.Tensor] = None
        self._expert_count_buf: Optional[torch.Tensor] = None
        self._full_sorted_buf: Optional[torch.Tensor] = None

    def _ensure_buffers(self, active_num: int, device: torch.device):
        """Allocate reusable buffers on first call; reuse on subsequent calls."""
        if self._buf_active_num == active_num:
            return
        self._buf_active_num = active_num
        self._arange_buf = torch.arange(
            active_num, device=device, dtype=torch.int64,
        )
        self._ones_buf = torch.ones(
            active_num, device=device, dtype=torch.float32,
        )
        self._expert_count_buf = torch.zeros(
            self.num_experts, device=device, dtype=torch.float32,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Dispatch tokens via AllGather + argsort-by-expert.

        Output ``sorted_hidden`` is always ``[active_num, H]`` (fixed shape);
        only the first ``sum(local_expert_tokens)`` rows contain real data.

        Returns:
            (sorted_hidden, topk_ids, topk_weights, local_expert_tokens,
             group_list_type=1)
        """
        num_local_tokens = hidden_states.shape[0]
        self._num_local_tokens = num_local_tokens

        # 1. AllGather across EP group — fixed shapes
        gathered_hidden = _allgather(hidden_states, self.group)      # [B*E, H]
        gathered_topk_ids = _allgather(topk_ids, self.group)         # [B*E, K]
        gathered_topk_weights = _allgather(topk_weights, self.group) # [B*E, K]

        total_tokens = gathered_hidden.shape[0]
        top_k = gathered_topk_ids.shape[1]
        active_num = total_tokens * top_k
        H = gathered_hidden.shape[1]
        self._active_num = active_num
        self._total_tokens = total_tokens
        self._top_k = top_k

        # Pre-allocated buffers (no Range / OnesLike / ZerosLike per call)
        self._ensure_buffers(active_num, hidden_states.device)
        arange_buf = self._arange_buf      # [active_num] int64

        # 2. Pre-expand hidden: [T, H] -> [T*K, H]
        x_expanded = (
            gathered_hidden.unsqueeze(1)
            .expand(-1, top_k, -1)
            .reshape(-1, H)
        )  # [T*K, H]

        # 3. Sort by expert via argsort on flattened expert ids
        #    Cast to float32 so argsort runs on AiCore (int32/int64 falls
        #    back to AiCpu which is slower and may block graph capture).
        flat_experts = gathered_topk_ids.reshape(-1)  # [T*K]
        perm = flat_experts.to(torch.float32).argsort(stable=True)  # [T*K]
        sorted_hidden_full = x_expanded[perm]          # [T*K, H] sorted by expert
        sorted_experts = flat_experts[perm]            # [T*K] sorted experts

        # Build inverse permutation for combine (reuse arange_buf)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = arange_buf

        # 4. Per-expert token counts (graph-capture-safe, float32 → AiCore)
        self._expert_count_buf.zero_()
        self._expert_count_buf.scatter_add_(
            0, sorted_experts.to(torch.int64), self._ones_buf,
        )
        expert_tokens_all = self._expert_count_buf.to(torch.int64)

        # 5. Extract local expert range
        local_expert_tokens = expert_tokens_all[
            self.first_expert : self.last_expert
        ]
        if self.first_expert > 0:
            local_start_t = expert_tokens_all[: self.first_expert].sum()
        else:
            local_start_t = expert_tokens_all.new_zeros(())
        self._local_start = local_start_t

        # 6. Fixed-size extraction: always [active_num, H]
        gather_idx = (arange_buf + local_start_t).clamp(max=active_num - 1)
        sorted_hidden = sorted_hidden_full.index_select(0, gather_idx)

        # 7. Zero out padding rows beyond local count
        local_count_t = local_expert_tokens.sum()
        data_mask = (arange_buf < local_count_t).unsqueeze(-1).to(sorted_hidden.dtype)
        sorted_hidden = sorted_hidden * data_mask

        # 8. Pad group_list so sum = active_num
        local_expert_tokens_padded = local_expert_tokens.clone()
        padding_needed = active_num - local_count_t
        local_expert_tokens_padded[-1] = (
            local_expert_tokens_padded[-1] + padding_needed
        )

        # Save for combine
        self._perm = perm
        self._inv_perm = inv_perm
        self._topk_weights = gathered_topk_weights

        return (
            sorted_hidden,
            topk_ids,
            topk_weights,
            local_expert_tokens_padded,
            1,  # group_list_type = 1 (count mode for npu_grouped_matmul)
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor = None,
        topk_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Unpermute expert outputs and ReduceScatter back to local tokens.

        Args:
            hidden_states: expert compute output [active_num, H]
        Returns:
            [num_local_tokens, H]
        """
        arange_buf = self._arange_buf      # [active_num] int64

        # 1. Place local expert output back into full sorted array
        if self._full_sorted_buf is None or self._full_sorted_buf.shape != (
            self._active_num, self.hidden_size,
        ):
            self._full_sorted_buf = torch.zeros(
                self._active_num, self.hidden_size,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
        full_sorted = self._full_sorted_buf
        full_sorted.zero_()

        scatter_idx = (arange_buf + self._local_start).clamp(
            max=self._active_num - 1,
        )
        full_sorted.scatter_add_(
            0,
            scatter_idx.unsqueeze(-1).expand_as(hidden_states),
            hidden_states,
        )

        # 2. Unsort: reverse the argsort permutation → [T*K, H] in original
        #    (i*K+k) order
        unsorted = full_sorted[self._inv_perm]  # [T*K, H]

        # 3. Apply weights and sum per original token
        flat_weights = self._topk_weights.reshape(-1)  # [T*K]
        weighted = unsorted * flat_weights.unsqueeze(-1).to(unsorted.dtype)

        # Reshape to [T, K, H] and sum over K
        per_token = weighted.reshape(
            self._total_tokens, self._top_k, self.hidden_size
        ).sum(dim=1)  # [T, H]

        # 4. ReduceScatter across EP group
        result = _reduce_scatter(per_token, self.group)  # [B, H]

        return result[: self._num_local_tokens]


class AscendTokenDispatcherLowLatency:
    """Decode EP dispatcher for Ascend NPU.

    Uses ``torch_npu.npu_moe_distribute_dispatch`` /
    ``torch_npu.npu_moe_distribute_combine`` (MC2 fused AllToAll) when
    available, with a fallback to standard ``dist.all_to_all`` (HCCL).

    Interface mirrors DeepEPTokenDispatcherLowLatency.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype = None,
        return_recv_hook: bool = False,
    ):
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.ep_size = group.size() if group is not None else 1
        self.ep_rank = dist.get_rank(group) if group is not None else 0
        self.return_recv_hook = return_recv_hook

        ctx = ExpertContext.get_instance()
        assert ctx.warmup_called, (
            "ExpertContext must be warmed up (ascend_warmup) before creating dispatchers"
        )
        self.num_max_dispatch_tokens_per_rank = ctx.num_max_dispatch_tokens_per_rank
        self.global_bs = ctx.num_max_dispatch_tokens_per_rank * self.ep_size

        # MC2 (npu_moe_distribute_dispatch/combine) requires >= 16 cards on A2
        # (910B).  vllm-ascend falls back to AllGather when
        # world_size_across_dp < 16 — see ascend_forward_context.py.
        # Disable MC2 for now; use dist.all_to_all fallback instead.
        self.moe_all_to_all_group_name: Optional[str] = None
        # TODO: re-enable MC2 for >= 16 card setups (A2) or A3/A5 devices.

        self.enable_dispatch_v2 = False
        self.need_extra_args = True  # A2/A3 need extra tp args; safe default

        self._dispatch_result = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: Optional[int] = None,
        use_fp8: bool = False,
    ) -> Tuple:
        """Dispatch tokens for decode EP.

        Returns:
            (expand_x, topk_idx, topk_weights, expert_token_nums, group_list_type)

            MC2 path:
                expand_x:           FLAT [total_expanded, H]
                expert_token_nums:  [num_local_experts] cumsum offsets (group_list)
                group_list_type:    0 (cumsum offsets for npu_grouped_matmul)

            Fallback path:
                expand_x:           [num_local_experts, max_m, H] (padded)
                expert_token_nums:  [num_local_experts] int32 actual counts
                group_list_type:    -1 (sentinel: use bmm, not grouped_matmul)
        """
        topk_idx = topk_idx.to(torch.int64)
        num_tokens = hidden_states.shape[0]
        top_k = topk_idx.shape[1]
        ne = num_experts or self.num_experts

        # Try MC2 fused dispatch
        use_mc2 = False
        assist_info = ep_recv_counts = tp_recv_counts = expand_scales = None

        if self.moe_all_to_all_group_name is not None:
            try:
                import torch_npu

                expert_ids_i32 = topk_idx.to(torch.int32)

                # --- Pre-dispatch diagnostics ---
                # Sync NPU and barrier to ensure all ranks arrive together.
                # If barrier hangs, some rank is stuck before MoE layer.
                import sys as _sys
                _rank = dist.get_rank() if dist.is_initialized() else -1
                torch.npu.synchronize()
                print(f"[MC2-DBG][R{_rank}] pre-barrier OK, "
                      f"x.shape={hidden_states.shape} x.dtype={hidden_states.dtype} "
                      f"expert_ids={expert_ids_i32.flatten().tolist()} "
                      f"ne={ne} global_bs={self.global_bs} "
                      f"ep_rank={self.ep_rank} ep_size={self.ep_size} "
                      f"comm={self.moe_all_to_all_group_name} "
                      f"v2={self.enable_dispatch_v2}",
                      file=_sys.stderr, flush=True)
                dist.barrier(group=self.group)
                print(f"[MC2-DBG][R{_rank}] post-barrier OK, calling dispatch",
                      file=_sys.stderr, flush=True)

                kwargs_mc2 = {
                    "x": hidden_states,
                    "expert_ids": expert_ids_i32,
                    "expert_shard_type": 0,
                    "shared_expert_rank_num": 0,
                    "moe_expert_num": ne,
                    "global_bs": self.global_bs,
                    "expert_token_nums_type": 0,
                    "scales": None,
                    "quant_mode": 0,
                    "group_ep": self.moe_all_to_all_group_name,
                    "ep_world_size": self.ep_size,
                    "ep_rank_id": self.ep_rank,
                }
                if self.need_extra_args:
                    kwargs_mc2.update({
                        "group_tp": self.moe_all_to_all_group_name,
                        "tp_world_size": 1,
                        "tp_rank_id": 0,
                    })

                output_mc2 = (
                    torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)
                    if self.enable_dispatch_v2
                    else torch_npu.npu_moe_distribute_dispatch(**kwargs_mc2)
                )
                (expand_x, _, assist_info, expert_token_nums,
                 ep_recv_counts, tp_recv_counts, expand_scales) = output_mc2[0:7]

                # expand_x is FLAT [total_expanded, H] — do NOT reshape to [L, m, H].
                # expert_token_nums is cumsum offsets (group_list_type=0) for
                # npu_grouped_matmul routing.
                use_mc2 = True
            except Exception as _mc2_err:
                # Do NOT silently fall back to dist.all_to_all.
                # If MC2 dispatch fails on one rank but succeeds on others, the
                # fallback would call a different collective → permanent deadlock.
                # Raise visibly so we can diagnose and fix the MC2 kwargs.
                import logging as _logging
                _logging.getLogger(__name__).error(
                    "MC2 dispatch failed (%s: %s) — raising to prevent collective mismatch",
                    type(_mc2_err).__name__, _mc2_err,
                )
                raise RuntimeError(
                    f"MC2 dispatch failed; silent fallback would deadlock: {_mc2_err}"
                ) from _mc2_err

        # Initialize fallback-only variables so they are always defined for the state dict
        send_splits = recv_splits = masked_m = None
        flat_row_idx = flat_weights = sort_perm = all_pack_indices = None
        total_recv = 0

        if not use_mc2:
            # Fallback: all_to_all on expanded token-expert pairs.
            # hidden_states is [N, H]; we need to send one copy per expert assignment,
            # so we expand to [N*K, H] sorted by destination rank/expert.
            L = self.num_local_experts
            local_expert_start = self.ep_rank * L

            # Expand: one entry per (token, expert) pair
            flat_row_idx = (
                torch.arange(num_tokens, device=hidden_states.device)
                .unsqueeze(1).expand(-1, top_k).reshape(-1)
            )  # [N*K]
            flat_expert_idx = topk_idx.flatten().to(torch.int64)  # [N*K]
            flat_weights = topk_weights.flatten()                  # [N*K]

            # Sort by expert so assignments for each dest rank are contiguous
            sort_perm = torch.argsort(flat_expert_idx)
            sorted_expert_idx = flat_expert_idx[sort_perm]   # [N*K]
            sorted_row_idx    = flat_row_idx[sort_perm]
            sorted_weights    = flat_weights[sort_perm]
            sorted_x          = hidden_states[sorted_row_idx]  # [N*K, H]

            # Count assignments per rank (send side)
            send_counts = torch.tensor(
                [
                    int(((sorted_expert_idx >= r * L) & (sorted_expert_idx < (r + 1) * L)).sum().item())
                    for r in range(self.ep_size)
                ],
                dtype=torch.int64,
                device=hidden_states.device,
            )
            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=self.group)

            send_splits = send_counts.tolist()
            recv_splits = recv_counts.tolist()
            total_recv = int(recv_counts.sum().item())

            # Exchange hidden states and expert indices together
            recv_hidden_flat = torch.zeros(
                total_recv, self.hidden_size,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            recv_expert_flat = torch.zeros(
                total_recv, dtype=torch.int64, device=hidden_states.device,
            )
            dist.all_to_all(
                list(recv_hidden_flat.split(recv_splits, dim=0)),
                list(sorted_x.split(send_splits, dim=0)),
                group=self.group,
            )
            dist.all_to_all(
                list(recv_expert_flat.split(recv_splits, dim=0)),
                list(sorted_expert_idx.split(send_splits, dim=0)),
                group=self.group,
            )

            # Compute per-local-expert receive counts to get exact max_m
            # (using average would truncate experts with above-average load)
            all_pack_indices = []
            masked_m_list = []
            for le in range(L):
                ge = local_expert_start + le
                indices = (recv_expert_flat == ge).nonzero(as_tuple=False).squeeze(-1)
                all_pack_indices.append(indices)
                masked_m_list.append(indices.shape[0])
            max_m = max(max(masked_m_list), 1)

            # Pack into [L, max_m, H] by local expert; record pack permutation
            # so combine can reverse it.
            masked_m = torch.zeros(L, dtype=torch.int32, device=hidden_states.device)
            packed_recv_hidden = torch.zeros(
                L, max_m, self.hidden_size,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            for le in range(L):
                indices = all_pack_indices[le]
                n = indices.shape[0]
                if n > 0:
                    packed_recv_hidden[le, :n] = recv_hidden_flat[indices]
                masked_m[le] = n
            all_pack_indices = torch.cat(all_pack_indices) if total_recv > 0 else torch.zeros(0, dtype=torch.int64, device=hidden_states.device)

        self._dispatch_result = {
            "hidden_states": hidden_states,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "use_mc2": use_mc2,
            # MC2-only fields:
            "assist_info": assist_info,
            "ep_recv_counts": ep_recv_counts,
            "tp_recv_counts": tp_recv_counts,
            "expand_scales": expand_scales,
            # fallback-only fields:
            "send_splits": send_splits if not use_mc2 else None,
            "recv_splits": recv_splits if not use_mc2 else None,
            "masked_m": masked_m if not use_mc2 else None,
            "flat_row_idx": flat_row_idx if not use_mc2 else None,
            "flat_weights": flat_weights if not use_mc2 else None,
            "sort_perm": sort_perm if not use_mc2 else None,
            "all_pack_indices": all_pack_indices if not use_mc2 else None,
            "total_recv": total_recv if not use_mc2 else None,
        }

        if use_mc2:
            # MC2 path: return flat expand_x + cumsum group_list
            return (
                expand_x,               # flat [total_expanded, H]
                topk_idx,
                topk_weights,
                expert_token_nums,       # [L] cumsum offsets (group_list)
                0,                       # group_list_type = 0 (cumsum)
            )
        else:
            # Fallback path: return padded [L, max_m, H]
            return (
                packed_recv_hidden,
                topk_idx,
                topk_weights,
                masked_m,                # [L] actual counts
                -1,                      # sentinel: use bmm path
            )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Combine expert outputs back to original token order."""
        state = self._dispatch_result
        use_mc2 = state["use_mc2"]

        if use_mc2 and self.moe_all_to_all_group_name is not None:
            try:
                import torch_npu

                ep_recv_counts = state["ep_recv_counts"]
                tp_recv_counts = state["tp_recv_counts"]
                expand_scales  = state["expand_scales"]
                assist_info    = state["assist_info"]

                # hidden_states is already flat [total_expanded, H] from
                # npu_grouped_matmul — pass directly to combine (no reshape).
                kwargs_mc2 = {
                    "expand_x": hidden_states,
                    "expert_ids": topk_idx.to(torch.int32),
                    "expert_scales": topk_weights.to(torch.float32),
                    "expert_shard_type": 0,
                    "shared_expert_rank_num": 0,
                    "moe_expert_num": self.num_experts,
                    "global_bs": self.global_bs,
                    "ep_send_counts": ep_recv_counts,
                    "group_ep": self.moe_all_to_all_group_name,
                    "ep_world_size": self.ep_size,
                    "ep_rank_id": self.ep_rank,
                    "expand_scales": expand_scales,
                }
                if self.enable_dispatch_v2:
                    kwargs_mc2["assist_info_for_combine"] = assist_info
                else:
                    kwargs_mc2["expand_idx"] = assist_info
                if self.need_extra_args:
                    kwargs_mc2.update({
                        "tp_send_counts": tp_recv_counts,
                        "group_tp": self.moe_all_to_all_group_name,
                        "tp_world_size": 1,
                        "tp_rank_id": 0,
                    })

                combined = (
                    torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2)
                    if self.enable_dispatch_v2
                    else torch_npu.npu_moe_distribute_combine(**kwargs_mc2)
                )
                self._dispatch_result = None
                return combined
            except Exception as _mc2_err:
                # Do NOT silently fall back to dist.all_to_all.
                # If MC2 combine fails on one rank after AllToAll already started on
                # others, the fallback would call a different collective → deadlock.
                import logging as _logging
                _logging.getLogger(__name__).error(
                    "MC2 combine failed (%s: %s) — raising to prevent collective mismatch",
                    type(_mc2_err).__name__, _mc2_err,
                )
                raise RuntimeError(
                    f"MC2 combine failed; silent fallback would deadlock: {_mc2_err}"
                ) from _mc2_err

        # Fallback: reverse all_to_all and weighted scatter-add back to [N, H]
        orig_hidden   = state["hidden_states"]
        send_splits   = state["send_splits"]
        recv_splits   = state["recv_splits"]
        masked_m      = state["masked_m"]
        flat_row_idx  = state["flat_row_idx"]
        flat_weights  = state["flat_weights"]
        sort_perm     = state["sort_perm"]
        all_pack_indices = state["all_pack_indices"]
        total_recv    = state["total_recv"]

        num_tokens = orig_hidden.shape[0]
        L, max_m, H = hidden_states.shape

        # Unpack expert outputs [L, max_m, H] -> [total_recv, H] in recv order
        # (reverse of the pack step which grouped by local expert)
        parts = [hidden_states[le, :int(masked_m[le].item())] for le in range(L)]
        packed_flat = torch.cat(parts, dim=0) if total_recv > 0 else \
            torch.zeros(0, H, dtype=hidden_states.dtype, device=hidden_states.device)

        # Scatter packed_flat back into recv_ordered (indexed by all_pack_indices)
        recv_ordered = torch.zeros(
            total_recv, H, dtype=hidden_states.dtype, device=hidden_states.device,
        )
        if total_recv > 0:
            recv_ordered[all_pack_indices] = packed_flat

        # Reverse all_to_all: send recv_ordered back; receive sorted_x results
        total_send_back = int(sum(send_splits))
        recv_back = torch.zeros(
            total_send_back, H, dtype=hidden_states.dtype, device=hidden_states.device,
        )
        dist.all_to_all(
            list(recv_back.split(send_splits, dim=0)),
            list(recv_ordered.split(recv_splits, dim=0)),
            group=self.group,
        )

        # recv_back is in sort_perm order; unsort to flat (token, expert) order
        unsort_perm = torch.argsort(sort_perm)
        recv_back_unsorted = recv_back[unsort_perm]  # [N*K, H]

        # Weighted scatter-add back to [N, H]
        output = torch.zeros(
            num_tokens, H, dtype=hidden_states.dtype, device=hidden_states.device,
        )
        output.scatter_add_(
            0,
            flat_row_idx.unsqueeze(-1).expand_as(recv_back_unsorted),
            recv_back_unsorted * flat_weights.unsqueeze(-1).to(recv_back_unsorted.dtype),
        )

        self._dispatch_result = None
        return output
