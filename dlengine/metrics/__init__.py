"""
Metrics module for DLEngine.

Runtime aggregation lives in Rust. This module keeps the public Python surface
small and stable, plus the Python-only logging helpers.
"""

from typing import Optional

from dlengine._rust.core import (
    RuntimeMetrics as _RustRuntimeMetrics,
    SequenceMetric as _RustSequenceMetric,
    ServerMetric as _RustServerMetric,
)
from dlengine.logging import get_logger

logger = get_logger()


class SequenceMetric(_RustSequenceMetric):
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


ServerMetric = _RustServerMetric


class MetricsManager:
    """
    Manager for sequence metrics plus Rust-owned server/runtime aggregation.
    """

    def __init__(self, report_interval_s: float = 5.0):
        self.server_metric = ServerMetric()
        self.sequence_metrics: dict[str, SequenceMetric] = {}
        self._runtime = _RustRuntimeMetrics(report_interval_s)

    def __getattr__(self, name: str):
        return getattr(self._runtime, name)

    def create_sequence_metric(
        self, seq_id: str, num_prompt_tokens: int
    ) -> SequenceMetric:
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
            should_log = self._runtime.record_sequence_completion(
                metric, self.server_metric
            )
            if should_log:
                metric.log_metrics()

    def remove_sequence_metric(self, seq_id: str):
        """Remove a sequence metric (e.g., after logging)."""
        self.sequence_metrics.pop(seq_id, None)

    def maybe_report_engine_status(self, **kwargs):
        """Accumulate per-step tokens and emit a status line every interval."""
        message = self._runtime.maybe_report_engine_status(
            self.server_metric,
            kwargs["engine_id"],
            kwargs["mode"],
            kwargs["running_per_dp"],
            kwargs["waiting"],
            kwargs["waiting_migration"],
            kwargs["used_blocks_per_dp"],
            kwargs["total_blocks"],
            kwargs["prefill_tokens_per_dp"],
            kwargs["decode_tokens_per_dp"],
            kwargs.get("prefix_cached_tokens_per_dp"),
            kwargs.get("prefix_prompt_tokens_per_dp"),
            kwargs.get("schedule_ms", 0.0),
            kwargs.get("forward_ms", 0.0),
            kwargs.get("postprocess_ms", 0.0),
            kwargs.get("forward_tx_bytes", 0),
            kwargs.get("forward_rx_bytes", 0),
            kwargs.get("transfer_ms", 0.0),
            kwargs.get("wwi_ms", 0.0),
            kwargs.get("immrecv_ms", 0.0),
            kwargs.get("net_ms", 0.0),
            kwargs.get("serialize_ms", 0.0),
        )
        if message:
            logger.info(message)

    def log_engine_status(self, **kwargs):
        """Record a throughput window and log a one-line engine status report."""
        message = self._runtime.log_engine_status(
            self.server_metric,
            kwargs["engine_id"],
            kwargs["mode"],
            kwargs["running_per_dp"],
            kwargs["waiting"],
            kwargs["waiting_migration"],
            kwargs["used_blocks_per_dp"],
            kwargs["total_blocks"],
            kwargs["prefill_tokens_per_dp"],
            kwargs["decode_tokens_per_dp"],
            kwargs["elapsed"],
            kwargs.get("avg_schedule_ms", 0.0),
            kwargs.get("avg_forward_ms", 0.0),
            kwargs.get("avg_postprocess_ms", 0.0),
            kwargs.get("steps", 0),
            kwargs.get("fwd_tx_bytes", 0),
            kwargs.get("fwd_rx_bytes", 0),
            kwargs.get("avg_transfer_ms", 0.0),
            kwargs.get("avg_wwi_ms", 0.0),
            kwargs.get("avg_immrecv_ms", 0.0),
            kwargs.get("avg_net_ms", 0.0),
            kwargs.get("avg_serialize_ms", 0.0),
            kwargs.get("prefix_cached_tokens_per_dp"),
            kwargs.get("prefix_prompt_tokens_per_dp"),
        )
        logger.info(message)

    def log_server_metrics(self, include_detailed: bool = False):
        """Log current server metrics using the Python logger."""
        logger.info(self.server_metric.get_metric_report(include_detailed))

    def get_server_summary(self) -> dict:
        """Get server metrics summary."""
        return self.server_metric.get_summary()

    def to_prometheus(self) -> str:
        """Return a Prometheus text-format snapshot for Grafana dashboards."""
        return self._runtime.to_prometheus(self.server_metric)

    def log_final_summary(self):
        """Log final server + sequence metrics summary at end of generation."""
        import numpy as np

        sep = "=" * 60
        logger.info(sep)
        logger.info("Final Server Metrics Summary")
        logger.info(sep)
        self.log_server_metrics(include_detailed=True)

        summary = self.get_server_summary()
        for key, value in summary.items():
            if value is not None:
                logger.info(f"  {key}: {value}")

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

        total_uptime = summary.get("uptime_seconds", 0)
        total_gen = summary.get("total_generated_tokens", 0)
        if total_uptime and total_gen:
            logger.info(
                f"  Effective decode throughput (wall-clock): "
                f"{total_gen / total_uptime:.0f} tok/s "
                f"({total_gen} tokens / {total_uptime:.1f}s)"
            )
        logger.info(sep)
