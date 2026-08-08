from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import deep_ep

from nanodeploy.logging import get_logger


logger = get_logger()


@dataclass
class EPContext:
    """Own the process-wide native DeepEP buffer.

    DeepEP creates process-global NVSHMEM state while constructing a Buffer, so
    a worker must create one shared instance and explicitly release it before
    tearing down the process groups used by that instance.
    """

    buffer: Any | None = field(default=None, init=False)
    topk_idx_t: Any | None = field(default=None, init=False)
    topk_idx_dtype: Any | None = field(default=None, init=False)
    ep_size: int = field(default=1, init=False)
    num_experts: int = field(default=0, init=False)
    num_local_experts: int = field(default=0, init=False)
    hidden_size: int = field(default=0, init=False)
    max_tokens_per_rank: int = field(default=0, init=False)
    num_sms: int = field(default=0, init=False)
    num_qps_per_rank: int = field(default=0, init=False)
    nvshmem_qp_depth: int = field(default=0, init=False)
    num_nvl_bytes: int = field(default=0, init=False)
    num_rdma_bytes: int = field(default=0, init=False)
    _latest_mode: str | None = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)
    _destroyed: bool = field(default=False, init=False)

    def initialize(
        self,
        *,
        ep_group: Any,
        ep_size: int,
        num_experts: int,
        hidden_size: int,
        max_tokens_per_rank: int,
        num_sms: int,
        allow_mnnvl: bool,
        nvshmem_qp_depth: int,
    ) -> None:
        """Construct the shared DeepEP buffer exactly once."""
        if self._initialized:
            raise RuntimeError("DeepEP context is already initialized")
        if self._destroyed:
            raise RuntimeError("destroyed DeepEP context cannot be initialized")
        if ep_size <= 1:
            raise ValueError(f"DeepEP requires ep_size > 1; got {ep_size}")
        if num_experts <= 0 or num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts={num_experts} must be positive and divisible "
                f"by ep_size={ep_size}"
            )
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive; got {hidden_size}")
        if max_tokens_per_rank <= 0:
            raise ValueError(
                "max_tokens_per_rank must be positive; "
                f"got {max_tokens_per_rank}"
            )
        if num_sms <= 0 or num_sms % 2 != 0:
            raise ValueError(
                f"num_sms must be a positive even integer; got {num_sms}"
            )

        minimum_qp_depth = 2 * (max_tokens_per_rank + 1)
        if nvshmem_qp_depth < minimum_qp_depth:
            raise ValueError(
                "NVSHMEM_QP_DEPTH must be at least "
                "2 * (max_tokens_per_rank + 1); "
                f"got {nvshmem_qp_depth} < {minimum_qp_depth}"
            )
        # DeepEP reads this environment variable during Buffer construction.
        os.environ["NVSHMEM_QP_DEPTH"] = str(nvshmem_qp_depth)

        buffer_type = deep_ep.Buffer
        buffer_type.set_num_sms(num_sms)

        # Normal dispatch may carry FP8 activations, but combine always carries
        # BF16. DeepEP therefore sizes normal buffers with at least two bytes
        # per hidden element.
        hidden_bytes = hidden_size * 2
        num_nvl_bytes = 0
        normal_rdma_bytes = 0
        for config in (
            buffer_type.get_dispatch_config(ep_size),
            buffer_type.get_combine_config(ep_size),
        ):
            num_nvl_bytes = max(
                num_nvl_bytes,
                config.get_nvl_buffer_size_hint(hidden_bytes, ep_size),
            )
            normal_rdma_bytes = max(
                normal_rdma_bytes,
                config.get_rdma_buffer_size_hint(hidden_bytes, ep_size),
            )

        low_latency_rdma_bytes = buffer_type.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_tokens_per_rank,
            hidden=hidden_size,
            num_ranks=ep_size,
            num_experts=num_experts,
        )
        num_rdma_bytes = max(normal_rdma_bytes, low_latency_rdma_bytes)

        num_local_experts = num_experts // ep_size
        # Preserve the existing communication tuning while keeping the QP
        # count distinct from the SMs used by normal-mode kernels.
        num_qps_per_rank = max(num_sms, num_local_experts)

        self.buffer = buffer_type(
            ep_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=allow_mnnvl,
            explicitly_destroy=True,
        )
        self.topk_idx_t = deep_ep.topk_idx_t
        self.topk_idx_dtype = self.topk_idx_t
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_sms = num_sms
        self.num_qps_per_rank = num_qps_per_rank
        self.nvshmem_qp_depth = nvshmem_qp_depth
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self._initialized = True

    def get_buffer(self) -> Any:
        if not self._initialized or self.buffer is None:
            raise RuntimeError("DeepEP context is not initialized")
        return self.buffer

    def mark_normal(self) -> None:
        """Record that the shared buffer was used by normal dispatch."""
        self.get_buffer()
        self._latest_mode = "normal"

    def prepare_low_latency(self) -> None:
        """Prepare the shared buffer for a low-latency dispatch.

        Model forwards construct short-lived MoE facades, so this transition
        state must live beside the process-wide Buffer. NanoDeploy currently
        executes these calls sequentially on one stream; no concurrent mode
        transitions are supported here.
        """
        buffer = self.get_buffer()
        if self._latest_mode == "normal":
            buffer.clean_low_latency_buffer(
                self.max_tokens_per_rank,
                self.hidden_size,
                self.num_experts,
            )
        self._latest_mode = "low_latency"

    def destroy(self) -> bool:
        """Release the native buffer once; repeated calls are no-ops."""
        if self._destroyed or self.buffer is None:
            self._destroyed = True
            return False

        buffer = self.buffer
        # Clear the owner reference first so a repeated cleanup after an error
        # cannot invoke native destruction twice.
        self.buffer = None
        self._destroyed = True
        buffer.destroy()
        return True


_EP_CONTEXT: EPContext | None = None


def get_ep_context() -> EPContext:
    if _EP_CONTEXT is None:
        raise RuntimeError("DeepEP context has not been configured")
    return _EP_CONTEXT


def set_ep_context(**kwargs: Any) -> EPContext:
    global _EP_CONTEXT
    if _EP_CONTEXT is not None:
        raise RuntimeError("DeepEP context is already configured")
    context = EPContext()
    context.initialize(**kwargs)
    _EP_CONTEXT = context
    return context


def destroy_ep_context() -> bool:
    if _EP_CONTEXT is None:
        return False
    return _EP_CONTEXT.destroy()


def get_sp_context() -> EPContext:
    """Compatibility alias for the original, misnamed EP getter."""
    return get_ep_context()


def reset_sp_context() -> None:
    raise AttributeError("EP Buffer Context is immutable")
