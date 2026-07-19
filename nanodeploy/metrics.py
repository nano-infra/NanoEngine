"""
Metrics module for NanoDeploy.

This module provides two types of metrics:
1. SequenceMetric: Per-sequence metrics (ITL, TTFT, E2E latency, etc.)
2. ServerMetric: Server-level metrics (throughput, token usage, running requests, etc.)
"""

from dataclasses import dataclass
from typing import Optional

from nanodeploy.logging import get_logger
from nanodeploy._cpp import (
    SequenceMetric as _CppSequenceMetric,
    ServerMetric as _CppServerMetric,
)

logger = get_logger()


@dataclass(frozen=True, slots=True)
class SequenceMetricTicket:
    """A provisional metric entry owned by exactly one MetricsManager."""

    seq_id: int
    metric: "SequenceMetric"
    _owner: object
    _ticket_id: int


class SequenceMetric(_CppSequenceMetric):
    def log_metrics(self):
        """Log all metrics for this sequence."""
        return 
        # ttft_str = f"{self.ttft:.2f}ms" if self.ttft is not None else "N/A"
        # e2e_str = (
        #     f"{self.e2e_latency:.2f}ms" if self.e2e_latency is not None else "N/A"
        # )

        # queueing_time_str = (
        #     f"{self.queueing_time_ms:.2f}ms"
        #     if self.queueing_time_ms is not None
        #     else "N/A"
        # )
        # decode_queueing_time_str = (
        #     f"{self.decode_queue_time_ms:.2f}ms"
        #     if self.decode_queue_time_ms is not None
        #     else "N/A"
        # )
        
        # # Format ITL samples (up to first 32)
        # itl_samples = self.itl_samples
        # num_samples_to_show = min(len(itl_samples), 32)
        # samples_str = ", ".join([f"{s:.2f}" for s in itl_samples[:num_samples_to_show]])
        # itl_samples_log = f"ITL Samples(First {num_samples_to_show}): [{samples_str}]"

        # itl_with_dq_str = (
        #     f"{self.avg_itl_with_decode_queue:.2f}ms"
        #     if self.avg_itl_with_decode_queue is not None
        #     else "N/A"
        # )

        # logger.info(
        #     f"SequenceMetric [{str(self.seq_id)[:8]}...] - "
        #     f"TTFT: {ttft_str}, "
        #     f"E2E: {e2e_str}, "
        #     f"Prompt Length: {self.num_prompt_tokens}, Output Length: {self.num_generated_tokens}, "
        #     f"Queueing Time: {queueing_time_str}, "
        #     f"Decode Queueing Time: {decode_queueing_time_str}, "
        #     f"ITL: {self.avg_itl:.2f}ms, "
        #     f"ITL with DQ: {itl_with_dq_str}, "
        #     f"{itl_samples_log}"
        # )

ServerMetric = _CppServerMetric

class MetricsManager:
    """
    Manager for tracking both sequence and server metrics.
    """

    def __init__(self):
        self.server_metric = ServerMetric()
        self.sequence_metrics: dict[int, SequenceMetric] = {}
        self._provisional_sequence_metrics: dict[int, SequenceMetricTicket] = {}
        self._metric_ticket_owner = object()
        self._next_metric_ticket_id = 0
        self.ls_rejected_requests_by_reason: dict[str, int] = {}

    def record_ls_rejected_request(self, error) -> None:
        reason = error.name
        self.ls_rejected_requests_by_reason[reason] = (
            self.ls_rejected_requests_by_reason.get(reason, 0) + 1
        )

    def prepare_sequence_metric(
        self, seq_id: int, num_prompt_tokens: int
    ) -> SequenceMetricTicket:
        """Create an unpublished Decode-ingress metric ticket.

        Arrival timestamps are captured at prepare time, while prompt/server
        counters remain unchanged until commit.  Insert-if-absent semantics
        prevent a rejected or duplicate request from overwriting an existing
        request's metric.
        """
        if (
            seq_id in self.sequence_metrics
            or seq_id in self._provisional_sequence_metrics
        ):
            raise ValueError(f"sequence metric already exists for seq_id={seq_id}")

        metric = SequenceMetric(seq_id=seq_id, num_prompt_tokens=num_prompt_tokens)
        metric.record_arrival()
        metric.record_decode_arrival()
        ticket = SequenceMetricTicket(
            seq_id=seq_id,
            metric=metric,
            _owner=self._metric_ticket_owner,
            _ticket_id=self._next_metric_ticket_id,
        )
        self._next_metric_ticket_id += 1
        self._provisional_sequence_metrics[seq_id] = ticket
        return ticket

    def commit_sequence_metric(
        self, ticket: SequenceMetricTicket
    ) -> SequenceMetric:
        """Publish a prepared ticket and account its prompt exactly once."""
        self._validate_metric_ticket(ticket)
        del self._provisional_sequence_metrics[ticket.seq_id]
        self.sequence_metrics[ticket.seq_id] = ticket.metric
        self.server_metric.add_tokens(num_prompt=ticket.metric.num_prompt_tokens)
        return ticket.metric

    def abort_sequence_metric(self, ticket: SequenceMetricTicket) -> bool:
        """Discard this ticket without touching committed or newer metrics."""
        if ticket._owner is not self._metric_ticket_owner:
            raise ValueError("sequence metric ticket belongs to another manager")
        current = self._provisional_sequence_metrics.get(ticket.seq_id)
        if current is not ticket:
            return False
        del self._provisional_sequence_metrics[ticket.seq_id]
        return True

    def _validate_metric_ticket(self, ticket: SequenceMetricTicket) -> None:
        if ticket._owner is not self._metric_ticket_owner:
            raise ValueError("sequence metric ticket belongs to another manager")
        current = self._provisional_sequence_metrics.get(ticket.seq_id)
        if current is not ticket:
            raise ValueError(
                f"sequence metric ticket is not active for seq_id={ticket.seq_id}"
            )

    def create_sequence_metric(
        self, seq_id: int, num_prompt_tokens: int
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

    def get_sequence_metric(self, seq_id: int) -> Optional[SequenceMetric]:
        """Get sequence metric by ID."""
        return self.sequence_metrics.get(seq_id)

    def complete_sequence(self, seq_id: int):
        """Mark a sequence as completed and log its metrics."""
        metric = self.sequence_metrics.get(seq_id)
        if metric:
            metric.record_completion()
            # Only log if the sequence has meaningful metrics
            if metric.first_token_time is not None or metric.num_generated_tokens > 0:
                metric.log_metrics()
            self.server_metric.add_tokens(num_generated=metric.num_generated_tokens)
            self.server_metric.add_completed_request()

    def remove_sequence_metric(self, seq_id: int):
        """Remove a sequence metric (e.g., after logging)."""
        self.sequence_metrics.pop(seq_id, None)

    def log_server_metrics(self, include_detailed: bool = False):
        """Log current server metrics using the Python logger."""
        report_str = self.server_metric.get_metric_report(include_detailed)
        logger.info(report_str)
    def get_server_summary(self) -> dict:
        """Get server metrics summary."""
        summary = self.server_metric.get_summary()
        summary["ls_rejected_requests"] = sum(
            self.ls_rejected_requests_by_reason.values()
        )
        summary["ls_rejected_requests_by_reason"] = dict(
            self.ls_rejected_requests_by_reason
        )
        return summary
