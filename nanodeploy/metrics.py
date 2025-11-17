"""
Metrics module for NanoDeploy.

This module provides two types of metrics:
1. SequenceMetric: Per-sequence metrics (ITL, TTFT, E2E latency, etc.)
2. ServerMetric: Server-level metrics (throughput, token usage, running requests, etc.)
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import numpy as np

from nanodeploy.logging import get_logger


logger = get_logger()


@dataclass
class SequenceMetric:
    """
    Metrics for individual sequence processing.
    
    Attributes:
        seq_id: Unique identifier for the sequence
        arrival_time: Timestamp when the request arrived
        first_token_time: Timestamp when the first token was generated (TTFT)
        completion_time: Timestamp when the sequence completed
        num_prompt_tokens: Number of tokens in the prompt
        num_generated_tokens: Number of tokens generated
        itl_samples: List of inter-token latencies (ms)
    """
    seq_id: str
    arrival_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    # without queueing time
    itl_samples: list[float] = field(default_factory=list)
    last_token_time: Optional[float] = None
    
    def record_first_token(self):
        """Record the timestamp of the first generated token."""
        if self.first_token_time is None:
            self.first_token_time = time.time()
            self.last_token_time = self.first_token_time
    
    def record_token(self):
        """Record a new token generation and calculate ITL."""
        current_time = time.time()
        if self.last_token_time is not None:
            itl = (current_time - self.last_token_time) * 1000  # Convert to ms
            self.itl_samples.append(itl)
        self.last_token_time = current_time
        self.num_generated_tokens += 1
    
    def record_completion(self):
        """Record the completion timestamp."""
        self.completion_time = time.time()
    
    @property
    def ttft(self) -> Optional[float]:
        """
        Time to first token (TTFT) in milliseconds. Includes queueing time.
        Returns None if first token hasn't been generated yet.
        """
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.arrival_time) * 1000
    
    @property
    def e2e_latency(self) -> Optional[float]:
        """
        End-to-end latency in milliseconds.
        Returns None if sequence hasn't completed yet.
        """
        if self.completion_time is None:
            return None
        return (self.completion_time - self.arrival_time) * 1000
    
    @property
    def avg_tpot_with_queueing(self) -> Optional[float]:
        """
        Time per output token (TPOT) in milliseconds. Includes queueing time.
        Returns None if no tokens have been generated yet.
        """
        if self.num_generated_tokens == 0 or self.completion_time is None:
            return None
        total_decode_time = (self.completion_time - self.arrival_time) * 1000  # ms
        return total_decode_time / self.num_generated_tokens
    
    @property
    def avg_itl(self) -> Optional[float]:
        """
        Average inter-token latency in milliseconds. Not include queueing time.
        Returns None if no tokens have been generated yet.
        """
        if not self.itl_samples:
            return None
        return np.mean(self.itl_samples)
    
    @property
    def p50_itl(self) -> Optional[float]:
        """P50 (median) inter-token latency in milliseconds."""
        if not self.itl_samples:
            return None
        return np.median(self.itl_samples)
    
    @property
    def p99_itl(self) -> Optional[float]:
        """P99 inter-token latency in milliseconds."""
        if not self.itl_samples:
            return None
        return np.percentile(self.itl_samples, 99)
    
    def log_metrics(self):
        """Log all metrics for this sequence."""
        ttft_str = f"{self.ttft:.2f}ms" if self.ttft is not None else "N/A"
        e2e_str = f"{self.e2e_latency:.2f}ms" if self.e2e_latency is not None else "N/A"
        
        # Format ITL metrics, handling None values
        if self.avg_itl is not None and self.p50_itl is not None and self.p99_itl is not None:
            itl_str = f"{self.avg_itl:.2f}/{self.p50_itl:.2f}/{self.p99_itl:.2f}ms"
        else:
            itl_str = "N/A"
        
        logger.info(
            f"SequenceMetric [{self.seq_id[:8]}...] - "
            f"TTFT: {ttft_str}, "
            f"E2E: {e2e_str}, "
            f"Tokens: {self.num_prompt_tokens}→{self.num_generated_tokens}, "
            f"ITL (avg/p50/p99): {itl_str}"
        )


@dataclass
class ServerMetric:
    """
    Server-level metrics for tracking overall performance.
    
    Attributes:
        total_tokens: Total number of tokens processed (prompt + generated)
        total_prompt_tokens: Total number of prompt tokens
        total_generated_tokens: Total number of generated tokens
        num_running_requests: Current number of running requests
        num_waiting_requests: Current number of waiting requests
        num_completed_requests: Total number of completed requests
        prefill_throughput_samples: List of prefill throughput samples (tokens/s)
        decode_throughput_samples: List of decode throughput samples (tokens/s)
        token_usage_by_dp: Token usage per data parallel rank
        start_time: Server start timestamp
    """
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_generated_tokens: int = 0
    num_running_requests: int = 0
    num_waiting_requests: int = 0
    num_completed_requests: int = 0
    prefill_throughput_samples: list[float] = field(default_factory=list)
    decode_throughput_samples: list[float] = field(default_factory=list)
    token_usage_by_dp: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    start_time: float = field(default_factory=time.time)
    
    def update_running_requests(self, count: int):
        """Update the number of running requests."""
        self.num_running_requests = count
    
    def update_waiting_requests(self, count: int):
        """Update the number of waiting requests."""
        self.num_waiting_requests = count
    
    def add_completed_request(self):
        """Increment the completed request counter."""
        self.num_completed_requests += 1
    
    def add_tokens(self, num_prompt: int = 0, num_generated: int = 0):
        """Add tokens to the total count."""
        self.total_prompt_tokens += num_prompt
        self.total_generated_tokens += num_generated
        self.total_tokens += num_prompt + num_generated
    
    def record_prefill_throughput(self, num_tokens: int, duration: float):
        """
        Record prefill throughput.
        
        Args:
            num_tokens: Number of tokens processed
            duration: Duration in seconds
        """
        if duration > 0:
            throughput = num_tokens / duration
            self.prefill_throughput_samples.append(throughput)
    
    def record_decode_throughput(self, num_tokens: int, duration: float):
        """
        Record decode throughput.
        
        Args:
            num_tokens: Number of tokens generated
            duration: Duration in seconds
        """
        if duration > 0:
            throughput = num_tokens / duration
            self.decode_throughput_samples.append(throughput)
    
    def update_token_usage(self, dp_idx: int, num_tokens: int):
        """Update token usage for a specific data parallel rank."""
        self.token_usage_by_dp[dp_idx] = num_tokens
    
    @property
    def avg_prefill_throughput(self) -> Optional[float]:
        """Average prefill throughput in tokens/s."""
        if not self.prefill_throughput_samples:
            return None
        return sum(self.prefill_throughput_samples) / len(self.prefill_throughput_samples)
    
    @property
    def avg_decode_throughput(self) -> Optional[float]:
        """Average decode throughput in tokens/s."""
        if not self.decode_throughput_samples:
            return None
        return sum(self.decode_throughput_samples) / len(self.decode_throughput_samples)
    
    @property
    def current_prefill_throughput(self) -> Optional[float]:
        """Most recent prefill throughput in tokens/s."""
        if not self.prefill_throughput_samples:
            return None
        return self.prefill_throughput_samples[-1]
    
    @property
    def current_decode_throughput(self) -> Optional[float]:
        """Most recent decode throughput in tokens/s."""
        if not self.decode_throughput_samples:
            return None
        return self.decode_throughput_samples[-1]
    
    @property
    def total_token_usage(self) -> int:
        """Total token usage across all DP ranks."""
        return sum(self.token_usage_by_dp.values())
    
    @property
    def uptime(self) -> float:
        """Server uptime in seconds."""
        return time.time() - self.start_time
    
    def log_metrics(self, include_detailed: bool = False):
        """
        Log server metrics.
        
        Args:
            include_detailed: Whether to include detailed per-DP metrics
        """
        prefill_tput = self.current_prefill_throughput or 0
        decode_tput = self.current_decode_throughput or 0
        
        logger.info(
            f"ServerMetric - "
            f"Running/Waiting: {self.num_running_requests}/{self.num_waiting_requests}, "
            f"Completed: {self.num_completed_requests}, "
            f"Tokens: {self.total_tokens} (prompt: {self.total_prompt_tokens}, gen: {self.total_generated_tokens}), "
            f"Throughput: Prefill {prefill_tput:.0f} tok/s, "
            f"Decode {decode_tput:.0f} tok/s"
        )
        
        if include_detailed and self.token_usage_by_dp:
            for dp_idx, tokens in self.token_usage_by_dp.items():
                logger.debug(f"  DP[{dp_idx}] token usage: {tokens}")
    
    def get_summary(self) -> dict:
        """Get a summary dictionary of all metrics."""
        return {
            "uptime_seconds": self.uptime,
            "total_requests": self.num_completed_requests,
            "running_requests": self.num_running_requests,
            "waiting_requests": self.num_waiting_requests,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generated_tokens": self.total_generated_tokens,
            "avg_prefill_throughput": self.avg_prefill_throughput,
            "avg_decode_throughput": self.avg_decode_throughput,
            "current_prefill_throughput": self.current_prefill_throughput,
            "current_decode_throughput": self.current_decode_throughput,
            "total_token_usage": self.total_token_usage,
        }


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
        metric = SequenceMetric(
            seq_id=seq_id,
            num_prompt_tokens=num_prompt_tokens
        )
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
            self.server_metric.add_completed_request()
    
    def remove_sequence_metric(self, seq_id: str):
        """Remove a sequence metric (e.g., after logging)."""
        self.sequence_metrics.pop(seq_id, None)
    
    def log_server_metrics(self, include_detailed: bool = False):
        """Log current server metrics."""
        self.server_metric.log_metrics(include_detailed=include_detailed)
    
    def get_server_summary(self) -> dict:
        """Get server metrics summary."""
        return self.server_metric.get_summary()
