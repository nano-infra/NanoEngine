"""CUDA-Graph-safe local token dispatcher for single-card & TP MoE decode.

``LocalPaddedDispatcher`` replaces the CPU-sync ``bincount().tolist()`` path
(EP==1) with a fully GPU-resident padded routing scheme whose output format
``[E, max_m, H] + masked_m[E]`` is identical to DeepEP ``low_latency_dispatch``,
so the same masked grouped GEMMs work for both EP==1 and EP>1.

Triton kernels live in:
    nanodeploy/backends/gpu_generic/kernels/local_dispatch.py
"""

from typing import Union

import torch
import triton

from nanodeploy.backends.gpu_generic.kernels.local_dispatch import (
    ALIGNMENT,
    local_combine_kernel,
    local_dispatch_kernel,
)


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


class LocalPaddedDispatcher:
    """GPU-resident padded token dispatcher for EP==1 decode.

    All buffers are pre-allocated at init time so that ``dispatch()`` and
    ``combine()`` are purely kernel launches — no memory allocation, no CPU
    sync, fully CUDA-Graph compatible.

    Parameters
    ----------
    num_local_experts : int
        Number of local experts (``E``).
    max_m : int
        Per-expert token capacity; rounded up to ``ALIGNMENT`` (128).
    hidden_size : int
        Model hidden dimension.
    top_k : int
        Number of experts each token is routed to.
    max_num_tokens : int
        Maximum batch size (number of tokens in one decode step).
    device : str | torch.device
        CUDA device.
    """

    def __init__(
        self,
        num_local_experts: int,
        max_m: int,
        hidden_size: int,
        top_k: int,
        max_num_tokens: int,
        device: Union[str, torch.device] = "cuda",
    ):
        self.E = num_local_experts
        self.max_m = _align_up(max(max_m, ALIGNMENT), ALIGNMENT)
        self.H = hidden_size
        self.K = top_k
        self.max_T = max_num_tokens

        # Pre-allocated working buffers (reused every step)
        self.padded_buf = torch.zeros(
            self.E, self.max_m, hidden_size, dtype=torch.bfloat16, device=device
        )
        self.masked_m = torch.zeros(self.E, dtype=torch.int32, device=device)
        self.slot_map = torch.full(
            (max_num_tokens, top_k), -1, dtype=torch.int32, device=device
        )
        self.out_buf = torch.zeros(
            max_num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )

    @classmethod
    def from_experts(
        cls,
        num_local_experts: int,
        top_k: int,
        hidden_size: int,
        device: torch.device,
    ) -> "LocalPaddedDispatcher":
        """Create a dispatcher sized for the current RunnerConfig.

        Shared factory used by both Hopper and Generic expert backends.
        """
        from nanodeploy.worker.runner_config import get_runner_config

        max_num_seqs = get_runner_config().max_num_seqs or 512
        capacity_factor = 2
        raw_max_m = max_num_seqs * top_k // num_local_experts * capacity_factor
        max_m = max(raw_max_m, 128)
        return cls(
            num_local_experts=num_local_experts,
            max_m=max_m,
            hidden_size=hidden_size,
            top_k=top_k,
            max_num_tokens=max_num_seqs,
            device=device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Scatter *hidden_states* into the padded expert buffer.

        Returns
        -------
        padded_buf : Tensor  ``[E, max_m, H]``  bf16
        masked_m   : Tensor  ``[E]``  int32 – valid token count per expert
        expected_m : int  – average load hint for DeepGEMM
        """
        T, K = hidden_states.shape[0], topk_ids.shape[1]
        assert K == self.K
        assert T <= self.max_T, (
            f"dispatch: T={T} exceeds pre-allocated max_num_tokens={self.max_T}. "
            "Increase max_num_seqs in RunnerConfig."
        )

        # In-place reset (CUDA-Graph safe – no allocation)
        self.masked_m.zero_()
        self.padded_buf.zero_()
        self.slot_map[:T].fill_(-1)

        BLOCK_H = min(triton.next_power_of_2(self.H), 1024)

        local_dispatch_kernel[(T * K,)](
            hidden_states,
            topk_ids,
            self.padded_buf,
            self.masked_m,
            self.slot_map,
            self.E,
            self.max_m,
            self.H,
            hidden_states.stride(0),
            hidden_states.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            self.padded_buf.stride(0),
            self.padded_buf.stride(1),
            self.padded_buf.stride(2),
            self.slot_map.stride(0),
            self.slot_map.stride(1),
            BLOCK_H=BLOCK_H,
            TOP_K=K,
        )

        # Clamp to pre-allocated capacity to prevent out-of-bounds access in
        # subsequent kernels (e.g. DeepGEMM) when a single expert receives
        # more tokens than max_m.
        self.masked_m.clamp_(max=self.max_m)

        expected_m = min((T * K + self.E - 1) // self.E, self.max_m)
        return self.padded_buf, self.masked_m, expected_m

    def combine(
        self,
        expert_out: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Gather and weight-sum expert outputs back to token order.

        Parameters
        ----------
        expert_out   : ``[E, max_m, H_out]``
        topk_ids     : ``[T, K]``  (same tensor passed to ``dispatch``)
        topk_weights : ``[T, K]``
        num_tokens   : actual T (≤ max_T)

        Returns
        -------
        out : ``[T, H_out]``  bf16
        """
        assert num_tokens <= self.max_T, (
            f"combine: num_tokens={num_tokens} exceeds pre-allocated "
            f"max_num_tokens={self.max_T}."
        )
        H_out = expert_out.shape[-1]
        out = self.out_buf[:num_tokens]
        out.zero_()

        BLOCK_H = min(triton.next_power_of_2(H_out), 1024)
        num_h_blocks = (H_out + BLOCK_H - 1) // BLOCK_H

        local_combine_kernel[(num_tokens, num_h_blocks)](
            expert_out,
            topk_ids,
            topk_weights,
            self.slot_map,
            out,
            H_out,
            expert_out.stride(0),
            expert_out.stride(1),
            expert_out.stride(2),
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            self.slot_map.stride(0),
            self.slot_map.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_H=BLOCK_H,
            TOP_K=self.K,
        )
        return out
