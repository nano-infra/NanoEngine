from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch

from nanodeploy.worker.ep_context import get_ep_context


class DeepEPTokenDispatcherNormal:
    """Synchronous native DeepEP dispatcher for normal/prefill MoE."""

    def __init__(
        self,
        *,
        group: torch.distributed.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        expert_alignment: int = 128,
    ) -> None:
        del params_dtype
        context = get_ep_context()
        if context.num_experts != num_experts:
            raise ValueError(
                f"dispatcher num_experts={num_experts} does not match "
                f"DeepEP context num_experts={context.num_experts}"
            )
        if context.num_local_experts != num_local_experts:
            raise ValueError("local expert count does not match DeepEP context")
        if context.hidden_size != hidden_size:
            raise ValueError("hidden size does not match DeepEP context")
        self.group = group
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self._context = context
        self._buffer = context.get_buffer()
        self._handle: Optional[Tuple[Any, ...]] = None
        self._input_shape: Optional[torch.Size] = None

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_list=None,
    ):
        del expert_list
        if self._handle is not None:
            raise RuntimeError("normal DeepEP dispatch already has an in-flight handle")
        self._context.mark_normal()
        hidden_states = x[0] if isinstance(x, tuple) else x
        self._input_shape = hidden_states.shape
        topk_idx = topk_idx.to(dtype=self._context.topk_idx_dtype)
        topk_weights = topk_weights.to(dtype=torch.float32)
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self._buffer.get_dispatch_layout(
            topk_idx,
            self.num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_tokens_per_expert,
            handle,
            _event,
        ) = self._buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
            expert_alignment=self.expert_alignment,
        )
        self._handle = handle
        return recv_x, recv_topk_idx, recv_topk_weights, recv_tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._handle is None or self._input_shape is None:
            raise RuntimeError("normal DeepEP combine called without dispatch")
        handle = self._handle
        self._handle = None
        combined, _weights, _event = self._buffer.combine(
            hidden_states,
            handle,
            async_finish=False,
            previous_event=None,
            allocate_on_comm_stream=False,
        )
        return combined.view(self._input_shape)


class DeepEPTokenDispatcherLowLatency:
    """Synchronous native DeepEP low-latency dispatcher."""

    def __init__(
        self,
        *,
        group: torch.distributed.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        del params_dtype
        context = get_ep_context()
        if (
            context.num_experts != num_experts
            or context.num_local_experts != num_local_experts
            or context.hidden_size != hidden_size
        ):
            raise ValueError("low-latency dispatcher does not match DeepEP context")
        self.group = group
        self.num_experts = num_experts
        self._context = context
        self._buffer = context.get_buffer()
        self._handle: Optional[Tuple[Any, ...]] = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: Optional[int] = None,
    ):
        if self._handle is not None:
            raise RuntimeError(
                "low-latency DeepEP dispatch already has an in-flight handle"
            )
        if num_experts is not None and num_experts != self.num_experts:
            raise ValueError("num_experts does not match dispatcher configuration")
        self._context.prepare_low_latency()
        topk_idx = topk_idx.to(dtype=self._context.topk_idx_dtype)
        topk_weights = topk_weights.to(dtype=torch.float32)
        expected_m = max(
            1,
            (
                hidden_states.shape[0]
                * self._context.ep_size
                * topk_idx.shape[1]
                + self.num_experts
                - 1
            )
            // self.num_experts,
        )
        packed, masked_m, handle, _event, hook = (
            self._buffer.low_latency_dispatch(
                hidden_states,
                topk_idx,
                self._context.max_tokens_per_rank,
                self.num_experts,
                use_fp8=True,
                round_scale=False,
                use_ue8m0=False,
                async_finish=False,
                return_recv_hook=True,
            )
        )
        if hook is None:
            raise RuntimeError("DeepEP did not return the requested dispatch hook")
        hook()
        self._handle = handle
        expected_m = min(expected_m, packed[0].shape[1])
        return packed, topk_idx, topk_weights, masked_m, expected_m

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._handle is None:
            raise RuntimeError("low-latency DeepEP combine called without dispatch")
        handle = self._handle
        self._handle = None
        combined, _event, hook = self._buffer.low_latency_combine(
            hidden_states,
            topk_idx.to(dtype=self._context.topk_idx_dtype),
            topk_weights.to(dtype=torch.float32),
            handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=True,
        )
        if hook is None:
            raise RuntimeError("DeepEP did not return the requested combine hook")
        hook()
        return combined


__all__ = [
    "DeepEPTokenDispatcherLowLatency",
    "DeepEPTokenDispatcherNormal",
]
