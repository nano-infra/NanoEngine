"""
Metrics module for NanoDeploy.

This module provides two types of metrics:
1. SequenceMetric: Per-sequence metrics (ITL, TTFT, E2E latency, etc.)
2. ServerMetric: Server-level metrics (throughput, token usage, running requests, etc.)
"""

import time
from typing import Optional

# 引入 C++ 定义的类
from nanodeploy.engine._core import (
    SequenceMetric as CppSequenceMetric,
    ServerMetric as CppServerMetric,
)
from nanodeploy.logging import get_logger

logger = get_logger()


class SequenceMetric(CppSequenceMetric):
    """
    Wrapper around C++ SequenceMetric to add Python-specific logging.
    Most data and calculation logic is handled in C++.
    """

    def __init__(self, seq_id: str, num_prompt_tokens: int):
        # 初始化 C++ 父类
        super().__init__(seq_id, num_prompt_tokens)
        # 在创建时自动记录到达时间
        self.record_arrival()

    @property
    def p50_itl(self) -> Optional[float]:
        # 调用 C++ 的 get_itl_stats (返回 tuple 或 None)
        stats = self.get_itl_stats()
        return stats[0] if stats else None

    @property
    def p99_itl(self) -> Optional[float]:
        stats = self.get_itl_stats()
        return stats[1] if stats else None

    def log_metrics(self):
        """Log all metrics using Python logger."""
        # 直接访问 C++ property (我们在 binding 中定义了这些 readonly property)
        ttft_val = self.ttft
        e2e_val = self.e2e_latency

        ttft_str = f"{ttft_val:.2f}ms" if ttft_val is not None else "N/A"
        e2e_str = f"{e2e_val:.2f}ms" if e2e_val is not None else "N/A"

        avg_itl_val = self.avg_itl
        if avg_itl_val is not None:
            stats = self.get_itl_stats()
            if stats:
                p50, p99 = stats
                itl_str = f"{avg_itl_val:.2f}/{p50:.2f}/{p99:.2f}ms"
            else:
                itl_str = f"{avg_itl_val:.2f}/N/A/N/A ms"
        else:
            itl_str = "N/A"

        tpot_wo = self.avg_tpot_wo_queueing
        tpot_w = self.avg_tpot_with_queueing

        tpot_wo_str = f"{tpot_wo:.2f}ms" if tpot_wo is not None else "N/A"
        tpot_w_str = f"{tpot_w:.2f}ms" if tpot_w is not None else "N/A"

        # 计算排队时间 (属性在 C++ 中暴露为 readwrite)
        if (
            self.decode_first_scheduled_time is not None
            and self.arrival_time is not None
        ):
            queueing_time = (
                self.decode_first_scheduled_time - self.arrival_time
            ) * 1000
            queueing_time_str = f"{queueing_time:.5f}ms"
        else:
            queueing_time_str = "N/A"

        logger.info(
            f"SequenceMetric [{self.seq_id[:8]}...] - "
            f"TTFT: {ttft_str}, "
            f"E2E: {e2e_str}, "
            f"Prompt: {self.num_prompt_tokens}, Gen: {self.num_generated_tokens}, "
            f"Queue: {queueing_time_str}, "
            f"ITL Wo/Q: {tpot_wo_str}, "
            f"ITL W/Q: {tpot_w_str}, "
            f"ITL (Avg/P50/P99): {itl_str}"
        )


class ServerMetric(CppServerMetric):
    """
    Wrapper around C++ ServerMetric.

    Calculation logic (throughput, uptime, tokens) is inherited from C++.
    This class only handles Python-specific logging formatting.
    """

    def log_metrics(self, include_detailed: bool = False):
        """Log current server metrics using Python logger."""
        # 直接使用 C++ 计算好的属性
        prefill_tput = self.current_prefill_throughput or 0
        decode_tput = self.current_decode_throughput or 0

        # 直接访问 C++ 字段
        logger.info(
            f"ServerMetric - "
            f"Run/Wait/Mig: {self.num_running_requests}/{self.num_waiting_requests}/{self.num_waiting_migration_requests}, "
            f"Completed: {self.num_completed_requests}, "
            f"Tokens: {self.total_tokens} (P: {self.total_prompt_tokens}, G: {self.total_generated_tokens}), "
            f"Tput: Pre {prefill_tput:.0f}, Dec {decode_tput:.0f} tok/s"
        )

        if include_detailed and self.token_usage_by_dp:
            for dp_idx, tokens in self.token_usage_by_dp.items():
                logger.debug(f"  DP[{dp_idx}] token usage: {tokens}")


class MetricsManager:
    """
    Manager for tracking both sequence and server metrics.
    """

    def __init__(self):
        self.server_metric = ServerMetric()
        self.sequence_metrics: dict[str, SequenceMetric] = {}

    def create_sequence_metric(
        self, seq_id: str, num_prompt_tokens: int
    ) -> SequenceMetric:
        """
        Create a new sequence metric.
        """
        metric = SequenceMetric(seq_id=seq_id, num_prompt_tokens=num_prompt_tokens)
        self.sequence_metrics[seq_id] = metric

        # 调用 C++ ServerMetric 的 add_tokens 方法
        # 我们在 pybind11 中配置了 kwargs 支持: .def("add_tokens", ..., py::arg("num_prompt")=0, ...)
        self.server_metric.add_tokens(num_prompt=num_prompt_tokens)
        return metric

    def get_sequence_metric(self, seq_id: str) -> Optional[SequenceMetric]:
        """Get sequence metric by ID."""
        return self.sequence_metrics.get(seq_id)

    def complete_sequence(self, seq_id: str):
        """Mark a sequence as completed and log its metrics."""
        metric = self.sequence_metrics.get(seq_id)
        if metric:
            metric.record_completion()  # 调用 C++ 方法
            # Only log if the sequence has meaningful metrics
            if metric.first_token_time is not None or metric.num_generated_tokens > 0:
                metric.log_metrics()  # 调用 Python Wrapper 方法
            self.server_metric.add_completed_request()  # 调用 C++ 方法

    def remove_sequence_metric(self, seq_id: str):
        """Remove a sequence metric (e.g., after logging)."""
        self.sequence_metrics.pop(seq_id, None)

    def log_server_metrics(self, include_detailed: bool = False):
        """Log current server metrics."""
        self.server_metric.log_metrics(include_detailed=include_detailed)

    def get_server_summary(self) -> dict:
        """Get server metrics summary."""
        # 直接调用 C++ 的 get_summary，它已经返回一个 dict
        return self.server_metric.get_summary()
