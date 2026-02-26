from typing import Optional

import deep_ep
import torch


class ExpertContext:
    _instance: Optional["ExpertContext"] = None

    def __init__(self):
        self.buffer: Optional[deep_ep.Buffer] = None
        self.ep_size: int = 1
        self.is_fp8: bool = False
        self.num_sms: int = deep_ep.Buffer.num_sms  # Default from DeepEP (20)
        self.warmup_called: bool = False
        self.num_max_dispatch_tokens_per_rank: int = 128  # DLBlas default
        self.num_local_experts: int = 0
        self.num_experts: int = 0
        self.hidden_size: int = 0
        self._latest_mode: Optional[str] = None  # "normal" or "low_latency"

    @classmethod
    def get_instance(cls) -> "ExpertContext":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warmup(
        self,
        ep_group: torch.distributed.ProcessGroup,
        ep_rank: int,
        ep_size: int,
        num_local_experts: int,
        hidden_size: int,
        max_num_sequence: int,
        is_fp8: bool = False,
        num_sms: Optional[int] = None,
        num_max_dispatch_tokens_per_rank: Optional[int] = None,
    ) -> None:
        """
        初始化 DeepEP Buffer。必须且仅能在模型加载前调用一次。
        """
        if self.warmup_called:
            return

        self.ep_group = ep_group
        self.ep_size = ep_size
        self.is_fp8 = is_fp8
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts * ep_size
        self.hidden_size = hidden_size

        # 单机模式（含纯TP），无需DeepEP
        if ep_size <= 1:
            self.warmup_called = True
            return

        assert deep_ep is not None, "DeepEP library is required when ep_size > 1"
        assert torch.cuda.is_available(), "CUDA must be available for DeepEP"

        # num_max_dispatch_tokens_per_rank: align with DLBlas default of 128
        if num_max_dispatch_tokens_per_rank is not None:
            self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        # else keep default 128

        # num_sms: use DeepEP default (Buffer.num_sms = 20), matching DLBlas behavior
        if num_sms is not None:
            self.num_sms = num_sms
        # else keep default Buffer.num_sms (20)

        # Normal mode buffer sizing: always use at least BF16 sizing (2 bytes/element)
        # because combine() always operates on BF16 outputs, even when dispatch uses FP8.
        # This matches DeepEP's get_hidden_bytes: t.size(1) * max(t.element_size(), 2)
        hidden_size_bytes = hidden_size * 2

        num_nvl_bytes, normal_rdma_bytes = 0, 0
        for config in (
            deep_ep.Buffer.get_dispatch_config(ep_size),
            deep_ep.Buffer.get_combine_config(ep_size),
        ):
            num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_size_bytes, ep_size),
                num_nvl_bytes,
            )
            normal_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_size_bytes, ep_size),
                normal_rdma_bytes,
            )

        # Low latency buffer sizing:
        # get_low_latency_rdma_size_hint expects `hidden` as the DIMENSION (not bytes),
        # as documented: "The size calculation will be done with BF16."
        total_num_experts = num_local_experts * ep_size
        ll_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            self.num_max_dispatch_tokens_per_rank,  # use proper dispatch token count, NOT max_num_sequence
            hidden_size,  # raw hidden dimension, NOT bytes
            ep_size,
            total_num_experts,
        )

        num_rdma_bytes = max(normal_rdma_bytes, ll_rdma_bytes)

        # num_qps_per_rank: DLBlas uses max(num_sms, num_local_experts) for combined buffer
        num_qps_per_rank = max(self.num_sms, num_local_experts)

        self.buffer = deep_ep.Buffer(
            ep_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
        )
        self.buffer.set_num_sms(self.num_sms)

        self.warmup_called = True

    def get_buffer(self) -> Optional[deep_ep.Buffer]:
        """获取当前实例的 DeepEP Buffer，如果未初始化或单卡则返回 None"""
        return self.buffer

    def transition_to_low_latency(self):
        """
        Call before low-latency (decode) dispatch. Cleans buffer if previously
        used in normal (prefill) mode, as required by DeepEP.
        """
        if self.buffer is not None and self._latest_mode == "normal":
            self.buffer.clean_low_latency_buffer(
                self.num_max_dispatch_tokens_per_rank,
                self.hidden_size,
                self.num_experts,
            )
        self._latest_mode = "low_latency"

    def transition_to_normal(self):
        """Call before normal (prefill) dispatch."""
        self._latest_mode = "normal"

    @classmethod
    def reset(cls) -> None:
        """用于测试等场景重置上下文"""
        cls._instance = None
