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
