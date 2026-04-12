"""
Metrics module for NanoDeploy.

This module provides two types of metrics:
1. SequenceMetric: Per-sequence metrics (ITL, TTFT, E2E latency, etc.)
2. ServerMetric: Server-level metrics (throughput, token usage, running requests, etc.)
"""

import time
from collections import defaultdict
from typing import Optional

import numpy as np

from nanodeploy._cpp import (
    SequenceMetric as _CppSequenceMetric,
    ServerMetric as _CppServerMetric,
)
from nanodeploy.logging import get_logger

logger = get_logger()


class SequenceMetric(_CppSequenceMetric):
    def log_metrics(self):
        """Log all metrics for this sequence."""

        ttft_str = f"{self.ttft:.2f}ms" if self.ttft is not None else "N/A"
        e2e_str = f"{self.e2e_latency:.2f}ms" if self.e2e_latency is not None else "N/A"

        tpot_wo_queue_str = (
            f"{self.avg_tpot_wo_queueing:.2f}ms"
            if self.avg_tpot_wo_queueing is not None
            else "N/A"
        )
        tpot_with_queue_str = (
            f"{self.avg_tpot_with_queueing:.2f}ms"
            if self.avg_tpot_with_queueing is not None
            else "N/A"
        )
        queueing_time_str = (
            f"{self.queueing_time_ms:.2f}ms"
            if self.queueing_time_ms is not None
            else "N/A"
        )
        decode_queueing_time_str = (
            f"{self.decode_queue_time_ms:.2f}ms"
            if self.decode_queue_time_ms is not None
            else "N/A"
        )

        logger.info(
            f"SequenceMetric [{str(self.seq_id)[:8]}...] - "
            f"TTFT: {ttft_str}, "
            f"E2E: {e2e_str}, "
            f"Prompt Length: {self.num_prompt_tokens}, Output Length: {self.num_generated_tokens}, "
            f"Queueing Time: {queueing_time_str}, "
            f"Decode Queueing Time: {decode_queueing_time_str}, "
            f"ITL Wo Queue: {tpot_wo_queue_str}, "
            f"ITL With Queue: {tpot_with_queue_str}"
        )


ServerMetric = _CppServerMetric


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

        Args:
            seq_id: Unique sequence identifier
            num_prompt_tokens: Number of tokens in the prompt

        Returns:
            The created SequenceMetric instance
        """
        metric = SequenceMetric(seq_id=seq_id, num_prompt_tokens=num_prompt_tokens)
        self.sequence_metrics[seq_id] = metric
        self.server_metric.add_tokens(num_prompt=num_prompt_tokens)
        return metric

    def get_sequence_metric(self, seq_id: str) -> Optional[SequenceMetric]:
        """Get sequence metric by ID."""
        return self.sequence_metrics.get(seq_id)

    def complete_sequence(self, seq_id: str):
        """Mark a sequence as completed and log its metrics."""
        metric = self.sequence_metrics.get(seq_id)
        if metric:
            metric.record_completion()
            # Only log if the sequence has meaningful metrics
            if metric.first_token_time is not None or metric.num_generated_tokens > 0:
                metric.log_metrics()
            self.server_metric.add_tokens(num_generated=metric.num_generated_tokens)
            self.server_metric.add_completed_request()

    def remove_sequence_metric(self, seq_id: str):
        """Remove a sequence metric (e.g., after logging)."""
        self.sequence_metrics.pop(seq_id, None)

    def log_server_metrics(self, include_detailed: bool = False):
        """Log current server metrics using the Python logger."""
        report_str = self.server_metric.get_metric_report(include_detailed)
        logger.info(report_str)

    def get_server_summary(self) -> dict:
        """Get server metrics summary."""
        return self.server_metric.get_summary()

    def log_final_summary(self):
        """Log final server + sequence metrics summary at end of generation."""
        sep = "=" * 60
        logger.info(sep)
        logger.info("Final Server Metrics Summary")
        logger.info(sep)
        self.log_server_metrics(include_detailed=True)

        summary = self.get_server_summary()
        for key, value in summary.items():
            if value is not None:
                logger.info(f"  {key}: {value}")

        # Per-sequence ITL (from start/end timestamps)
        itl_values = [
            m.avg_tpot_wo_queueing
            for m in self.sequence_metrics.values()
            if m.avg_tpot_wo_queueing is not None
        ]
        if itl_values:
            arr = np.array(itl_values)
            logger.info(
                f"  Per-Sequence ITL (from timestamps): "
                f"mean={np.mean(arr):.2f}ms, "
                f"median={np.median(arr):.2f}ms, "
                f"p99={np.percentile(arr, 99):.2f}ms, "
                f"n={len(arr)}"
            )

        # Per-token ITL w/o first token (from itl_samples collected in C++)
        all_itl = []
        for m in self.sequence_metrics.values():
            if m.itl_samples:
                all_itl.extend(m.itl_samples)
        if all_itl:
            arr = np.array(all_itl)
            logger.info(
                f"  ITL w/o first token (per-token samples): "
                f"mean={np.mean(arr):.2f}ms, "
                f"median={np.median(arr):.2f}ms, "
                f"p99={np.percentile(arr, 99):.2f}ms, "
                f"n={len(arr)}"
            )

        # Effective wall-clock throughput
        total_uptime = summary.get("uptime_seconds", 0)
        total_gen = summary.get("total_generated_tokens", 0)
        if total_uptime and total_gen:
            logger.info(
                f"  Effective decode throughput (wall-clock): "
                f"{total_gen / total_uptime:.0f} tok/s "
                f"({total_gen} tokens / {total_uptime:.1f}s)"
            )
        logger.info(sep)
