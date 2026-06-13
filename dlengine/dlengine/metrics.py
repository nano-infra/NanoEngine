"""
Metrics module for DLEngine.

This module provides two types of metrics:
1. SequenceMetric: Per-sequence metrics (ITL, TTFT, E2E latency, etc.)
2. ServerMetric: Server-level metrics (throughput, token usage, running requests, etc.)
"""

import time
from collections import defaultdict
from typing import Optional

import numpy as np

from dlengine._cpp import (
    SequenceMetric as _CppSequenceMetric,
    ServerMetric as _CppServerMetric,
)
from dlengine.logging import get_logger

logger = get_logger()


def _prom_value(value: object, default: float = 0.0) -> float:
    """Convert optional metric values to Prometheus-friendly floats."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _human_bytes(n: float) -> str:
    """Format a byte count as a compact human-readable string."""
    step = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if step < 1024.0 or unit == "GiB":
            return f"{step:.1f}{unit}" if unit != "B" else f"{int(step)}B"
        step /= 1024.0
    return f"{step:.1f}GiB"


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

    def __init__(self, report_interval_s: float = 5.0):
        self.server_metric = ServerMetric()
        self.sequence_metrics: dict[str, SequenceMetric] = {}
        self.engine_id: str = ""
        self.engine_mode: str = ""
        self.running_per_dp: list[int] = []
        self.used_blocks_per_dp: list[int] = []
        self.total_blocks_per_dp: int = 0
        self.last_schedule_ms: float = 0.0
        self.last_forward_ms: float = 0.0
        self.last_postprocess_ms: float = 0.0
        self.last_step_count: int = 0
        self.last_forward_tx_bytes: int = 0
        self.last_forward_rx_bytes: int = 0
        self.last_transfer_ms: float = 0.0
        self.last_wwi_ms: float = 0.0
        self.last_immrecv_ms: float = 0.0
        self.last_net_ms: float = 0.0
        self.last_serialize_ms: float = 0.0
        self.last_prefix_cache_hit_rate: float = 0.0
        # Periodic engine-status reporting state, driven from the engine's
        # step loop via maybe_report_engine_status().
        self._report_interval_s = report_interval_s
        self._last_report_time = time.time()
        self._report_prefill_tokens_per_dp: list[int] | None = None
        self._report_decode_tokens_per_dp: list[int] | None = None
        # Per-window step-latency breakdown accumulators (schedule / forward /
        # postprocess), averaged per step in the heartbeat line.
        self._report_step_count = 0
        self._report_sched_ms = 0.0
        self._report_forward_ms = 0.0
        self._report_post_ms = 0.0
        # Bytes sent to / received from the runners during forward (per window).
        self._report_fwd_tx_bytes = 0
        self._report_fwd_rx_bytes = 0
        # DLSlime transfer time (submit + wait round trip) accumulator.
        self._report_transfer_ms = 0.0
        # DLSlime RPC verb probes (writeWithImm / immRecv) accumulators.
        self._report_wwi_ms = 0.0
        self._report_immrecv_ms = 0.0
        # Pure network latency (round trip - remote handler) accumulator.
        self._report_net_ms = 0.0
        # Driver-side request serialization time (serialize_run_batch).
        self._report_serialize_ms = 0.0
        # Prefix-cache accounting over the window: cached prompt tokens served
        # from the KV cache vs. total prompt tokens of newly admitted prefills.
        self._report_prefix_cached_tokens = 0
        self._report_prefix_prompt_tokens = 0

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

    def maybe_report_engine_status(
        self,
        *,
        engine_id: str,
        mode: str,
        running_per_dp: list[int],
        waiting: int,
        waiting_migration: int,
        used_blocks_per_dp: list[int] | None,
        total_blocks: int,
        prefill_tokens_per_dp: list[int],
        decode_tokens_per_dp: list[int],
        prefix_cached_tokens: int = 0,
        prefix_prompt_tokens: int = 0,
        schedule_ms: float = 0.0,
        forward_ms: float = 0.0,
        postprocess_ms: float = 0.0,
        forward_tx_bytes: int = 0,
        forward_rx_bytes: int = 0,
        transfer_ms: float = 0.0,
        wwi_ms: float = 0.0,
        immrecv_ms: float = 0.0,
        net_ms: float = 0.0,
        serialize_ms: float = 0.0,
    ):
        """Accumulate per-step tokens and emit a status line every interval.

        Called once per engine step. Throttling and token accounting live here
        so the engine just forwards its current runtime state each step.
        ``schedule_ms`` / ``forward_ms`` / ``postprocess_ms`` are this step's
        latency breakdown; they are accumulated and reported as per-step
        averages over the window.
        """
        if self._report_prefill_tokens_per_dp is None:
            self._report_prefill_tokens_per_dp = [0] * len(prefill_tokens_per_dp)
            self._report_decode_tokens_per_dp = [0] * len(decode_tokens_per_dp)
        for i, t in enumerate(prefill_tokens_per_dp):
            self._report_prefill_tokens_per_dp[i] += t
        for i, t in enumerate(decode_tokens_per_dp):
            self._report_decode_tokens_per_dp[i] += t
        self._report_step_count += 1
        self._report_sched_ms += schedule_ms
        self._report_forward_ms += forward_ms
        self._report_post_ms += postprocess_ms
        self._report_fwd_tx_bytes += forward_tx_bytes
        self._report_fwd_rx_bytes += forward_rx_bytes
        self._report_transfer_ms += transfer_ms
        self._report_wwi_ms += wwi_ms
        self._report_immrecv_ms += immrecv_ms
        self._report_net_ms += net_ms
        self._report_serialize_ms += serialize_ms
        self._report_prefix_cached_tokens += prefix_cached_tokens
        self._report_prefix_prompt_tokens += prefix_prompt_tokens
        now = time.time()
        elapsed = now - self._last_report_time
        if elapsed < self._report_interval_s:
            return
        steps = max(self._report_step_count, 1)
        self.log_engine_status(
            engine_id=engine_id,
            mode=mode,
            running_per_dp=running_per_dp,
            waiting=waiting,
            waiting_migration=waiting_migration,
            used_blocks_per_dp=used_blocks_per_dp,
            total_blocks=total_blocks,
            prefill_tokens_per_dp=self._report_prefill_tokens_per_dp,
            decode_tokens_per_dp=self._report_decode_tokens_per_dp,
            elapsed=elapsed,
            avg_schedule_ms=self._report_sched_ms / steps,
            avg_forward_ms=self._report_forward_ms / steps,
            avg_postprocess_ms=self._report_post_ms / steps,
            steps=self._report_step_count,
            fwd_tx_bytes=self._report_fwd_tx_bytes,
            fwd_rx_bytes=self._report_fwd_rx_bytes,
            avg_transfer_ms=self._report_transfer_ms / steps,
            avg_wwi_ms=self._report_wwi_ms / steps,
            avg_immrecv_ms=self._report_immrecv_ms / steps,
            avg_net_ms=self._report_net_ms / steps,
            avg_serialize_ms=self._report_serialize_ms / steps,
            prefix_cached_tokens=self._report_prefix_cached_tokens,
            prefix_prompt_tokens=self._report_prefix_prompt_tokens,
        )
        self._last_report_time = now
        self._report_prefill_tokens_per_dp = None
        self._report_decode_tokens_per_dp = None
        self._report_step_count = 0
        self._report_sched_ms = 0.0
        self._report_forward_ms = 0.0
        self._report_post_ms = 0.0
        self._report_fwd_tx_bytes = 0
        self._report_fwd_rx_bytes = 0
        self._report_transfer_ms = 0.0
        self._report_wwi_ms = 0.0
        self._report_immrecv_ms = 0.0
        self._report_net_ms = 0.0
        self._report_serialize_ms = 0.0
        self._report_prefix_cached_tokens = 0
        self._report_prefix_prompt_tokens = 0

    def log_engine_status(
        self,
        *,
        engine_id: str,
        mode: str,
        running_per_dp: list[int],
        waiting: int,
        waiting_migration: int,
        used_blocks_per_dp: list[int] | None,
        total_blocks: int,
        prefill_tokens_per_dp: list[int],
        decode_tokens_per_dp: list[int],
        elapsed: float,
        avg_schedule_ms: float | None = None,
        avg_forward_ms: float | None = None,
        avg_postprocess_ms: float | None = None,
        steps: int | None = None,
        fwd_tx_bytes: int | None = None,
        fwd_rx_bytes: int | None = None,
        avg_transfer_ms: float | None = None,
        avg_wwi_ms: float | None = None,
        avg_immrecv_ms: float | None = None,
        avg_net_ms: float | None = None,
        avg_serialize_ms: float | None = None,
        prefix_cached_tokens: int = 0,
        prefix_prompt_tokens: int = 0,
    ):
        """Record windowed throughput and log a one-line engine status report.

        The engine passes in its current runtime state (queue depths, KV usage)
        plus the tokens produced since the last report; this method owns the
        throughput accounting and the (single-line) formatting/logging.
        """
        prefill_tokens = sum(prefill_tokens_per_dp)
        decode_tokens = sum(decode_tokens_per_dp)
        if prefill_tokens > 0:
            self.server_metric.record_prefill_throughput(prefill_tokens, elapsed)
        if decode_tokens > 0:
            self.server_metric.record_decode_throughput(decode_tokens, elapsed)

        self.engine_id = engine_id
        self.engine_mode = mode
        self.running_per_dp = list(running_per_dp)
        self.used_blocks_per_dp = list(used_blocks_per_dp or [])
        self.total_blocks_per_dp = total_blocks
        self.last_schedule_ms = _prom_value(avg_schedule_ms)
        self.last_forward_ms = _prom_value(avg_forward_ms)
        self.last_postprocess_ms = _prom_value(avg_postprocess_ms)
        self.last_step_count = int(steps or 0)
        self.last_forward_tx_bytes = int(fwd_tx_bytes or 0)
        self.last_forward_rx_bytes = int(fwd_rx_bytes or 0)
        self.last_transfer_ms = _prom_value(avg_transfer_ms)
        self.last_wwi_ms = _prom_value(avg_wwi_ms)
        self.last_immrecv_ms = _prom_value(avg_immrecv_ms)
        self.last_net_ms = _prom_value(avg_net_ms)
        self.last_serialize_ms = _prom_value(avg_serialize_ms)
        if prefix_prompt_tokens > 0:
            self.last_prefix_cache_hit_rate = (
                prefix_cached_tokens / prefix_prompt_tokens
            )

        if used_blocks_per_dp is not None:
            used_str = "|".join(str(u) for u in used_blocks_per_dp)
        else:
            used_str = "?"
        kv = f"{used_str}/{total_blocks}"
        run_str = "|".join(str(r) for r in running_per_dp)
        if elapsed > 0:
            pf_str = "|".join(f"{t / elapsed:.0f}" for t in prefill_tokens_per_dp)
            dec_str = "|".join(f"{t / elapsed:.0f}" for t in decode_tokens_per_dp)
        else:
            pf_str = "|".join("0" for _ in prefill_tokens_per_dp)
            dec_str = "|".join("0" for _ in decode_tokens_per_dp)
        sm = self.server_metric
        # Prefix-cache hit rate over the window: cached prompt tokens reused
        # from the KV cache divided by the total prompt tokens of newly
        # admitted prefills. Only shown when at least one prefill was admitted.
        cache_str = ""
        if prefix_prompt_tokens > 0:
            hit_rate = prefix_cached_tokens / prefix_prompt_tokens
            cache_str = (
                f" | prefix-cache {hit_rate * 100:.1f}% "
                f"({prefix_cached_tokens}/{prefix_prompt_tokens} tok)"
            )
        # Per-step latency breakdown (schedule / forward / postprocess),
        # averaged over the reporting window. ``total`` is their sum, useful for
        # spotting which stage dominates step time.
        lat_str = ""
        if avg_schedule_ms is not None:
            total_ms = avg_schedule_ms + avg_forward_ms + avg_postprocess_ms
            steps_str = f" n={steps}" if steps is not None else ""
            lat_str = (
                f" | lat sch={avg_schedule_ms:.2f} fwd={avg_forward_ms:.2f} "
                f"post={avg_postprocess_ms:.2f} total={total_ms:.2f} ms/step{steps_str}"
            )

        # Forward transfer volume to/from the runners over the window (plus the
        # per-step average, comparable to batch size).
        xfer_str = ""
        if fwd_tx_bytes:
            n = max(steps or 1, 1)
            xfer_str = (
                f" | xfer fwd_tx={_human_bytes(fwd_tx_bytes)} "
                f"fwd_rx={_human_bytes(fwd_rx_bytes or 0)} "
                f"(tx {_human_bytes(fwd_tx_bytes / n)}/step)"
            )
            if avg_serialize_ms:
                xfer_str += f" serialize={avg_serialize_ms:.2f} ms/step"
            if avg_transfer_ms:
                xfer_str += f" transfer={avg_transfer_ms:.2f} ms/step"
            # Breakdown of the RDMA verbs inside transfer (DLSlime only):
            # wwi = writeWithImm send wait, immrecv = inbound imm completion
            # wait (covers reply in-flight + remote compute). Summed across DP
            # shards, averaged per step.
            if avg_wwi_ms or avg_immrecv_ms:
                xfer_str += (
                    f" [wwi={avg_wwi_ms or 0.0:.2f} "
                    f"immrecv={avg_immrecv_ms or 0.0:.2f} ms/step]"
                )
            # Pure network latency: round trip minus remote handler time,
            # excluding GPU compute and pump idle. This is the clean signal
            # for diagnosing whether the RDMA transport itself is slow.
            if avg_net_ms is not None:
                xfer_str += f" net={avg_net_ms:.2f} ms/step"
        logger.info(
            f"[engine {engine_id[:8]} {mode}] "
            f"run={run_str} wait={waiting} mig={waiting_migration} | "
            f"kv={kv} blk | "
            f"tput pf={pf_str} dec={dec_str} tok/s ({elapsed:.0f}s) | "
            f"done={sm.num_completed_requests} "
            f"tok={sm.total_prompt_tokens}p/{sm.total_generated_tokens}g"
            f"{cache_str}"
            f"{lat_str}"
            f"{xfer_str}"
        )

    def log_server_metrics(self, include_detailed: bool = False):
        """Log current server metrics using the Python logger."""
        report_str = self.server_metric.get_metric_report(include_detailed)
        logger.info(report_str)

    def get_server_summary(self) -> dict:
        """Get server metrics summary."""
        return self.server_metric.get_summary()

    def to_prometheus(self) -> str:
        """Return a Prometheus text-format snapshot for Grafana dashboards."""
        sm = self.server_metric
        lines = [
            "# HELP dlengine_up DLEngine metrics exporter health.",
            "# TYPE dlengine_up gauge",
            "dlengine_up 1",
            "# HELP dlengine_uptime_seconds DLEngine process uptime.",
            "# TYPE dlengine_uptime_seconds gauge",
            f"dlengine_uptime_seconds {_prom_value(sm.uptime)}",
            "# HELP dlengine_requests Number of requests by state.",
            "# TYPE dlengine_requests gauge",
            f'dlengine_requests{{state="running"}} {sm.num_running_requests}',
            f'dlengine_requests{{state="waiting"}} {sm.num_waiting_requests}',
            (
                'dlengine_requests{state="waiting_migration"} '
                f"{sm.num_waiting_migration_requests}"
            ),
            f'dlengine_requests{{state="completed"}} {sm.num_completed_requests}',
            "# HELP dlengine_tokens Total token counters.",
            "# TYPE dlengine_tokens counter",
            f'dlengine_tokens{{type="prompt"}} {sm.total_prompt_tokens}',
            f'dlengine_tokens{{type="generated"}} {sm.total_generated_tokens}',
            f'dlengine_tokens{{type="all"}} {sm.total_tokens}',
            "# HELP dlengine_throughput_tokens_per_second Current token throughput.",
            "# TYPE dlengine_throughput_tokens_per_second gauge",
            (
                'dlengine_throughput_tokens_per_second{phase="prefill"} '
                f"{_prom_value(sm.current_prefill_throughput)}"
            ),
            (
                'dlengine_throughput_tokens_per_second{phase="decode"} '
                f"{_prom_value(sm.current_decode_throughput)}"
            ),
            "# HELP dlengine_avg_throughput_tokens_per_second Average token throughput.",
            "# TYPE dlengine_avg_throughput_tokens_per_second gauge",
            (
                'dlengine_avg_throughput_tokens_per_second{phase="prefill"} '
                f"{_prom_value(sm.avg_prefill_throughput)}"
            ),
            (
                'dlengine_avg_throughput_tokens_per_second{phase="decode"} '
                f"{_prom_value(sm.avg_decode_throughput)}"
            ),
            "# HELP dlengine_waiting_blocks Waiting queue KV block demand.",
            "# TYPE dlengine_waiting_blocks gauge",
            f'dlengine_waiting_blocks{{type="head"}} {sm.num_waiting_head_blocks}',
            f'dlengine_waiting_blocks{{type="total"}} {sm.num_waiting_total_blocks}',
            "# HELP dlengine_step_latency_ms Last reported average step latency.",
            "# TYPE dlengine_step_latency_ms gauge",
            f'dlengine_step_latency_ms{{phase="schedule"}} {self.last_schedule_ms}',
            f'dlengine_step_latency_ms{{phase="forward"}} {self.last_forward_ms}',
            (
                'dlengine_step_latency_ms{phase="postprocess"} '
                f"{self.last_postprocess_ms}"
            ),
            "# HELP dlengine_step_count Last reported heartbeat window step count.",
            "# TYPE dlengine_step_count gauge",
            f"dlengine_step_count {self.last_step_count}",
            "# HELP dlengine_forward_bytes Last reported forward transfer bytes.",
            "# TYPE dlengine_forward_bytes gauge",
            f'dlengine_forward_bytes{{direction="tx"}} {self.last_forward_tx_bytes}',
            f'dlengine_forward_bytes{{direction="rx"}} {self.last_forward_rx_bytes}',
            "# HELP dlengine_transfer_latency_ms Last reported transfer latency.",
            "# TYPE dlengine_transfer_latency_ms gauge",
            f'dlengine_transfer_latency_ms{{phase="transfer"}} {self.last_transfer_ms}',
            f'dlengine_transfer_latency_ms{{phase="write_with_imm"}} {self.last_wwi_ms}',
            f'dlengine_transfer_latency_ms{{phase="imm_recv"}} {self.last_immrecv_ms}',
            f'dlengine_transfer_latency_ms{{phase="network"}} {self.last_net_ms}',
            f'dlengine_transfer_latency_ms{{phase="serialize"}} {self.last_serialize_ms}',
            "# HELP dlengine_prefix_cache_hit_rate Prefix-cache hit rate of admitted prefills.",
            "# TYPE dlengine_prefix_cache_hit_rate gauge",
            f"dlengine_prefix_cache_hit_rate {_prom_value(self.last_prefix_cache_hit_rate)}",
        ]
        for dp_idx, running in enumerate(self.running_per_dp):
            lines.append(
                "# HELP dlengine_running_requests_per_dp Running requests per DP rank."
                if dp_idx == 0
                else ""
            )
            lines.append(
                "# TYPE dlengine_running_requests_per_dp gauge" if dp_idx == 0 else ""
            )
            lines.append(f'dlengine_running_requests_per_dp{{dp="{dp_idx}"}} {running}')
        for dp_idx, used in enumerate(self.used_blocks_per_dp):
            lines.append(
                "# HELP dlengine_kv_blocks KV cache blocks per DP rank."
                if dp_idx == 0
                else ""
            )
            lines.append("# TYPE dlengine_kv_blocks gauge" if dp_idx == 0 else "")
            lines.append(f'dlengine_kv_blocks{{dp="{dp_idx}",state="used"}} {used}')
            lines.append(
                f'dlengine_kv_blocks{{dp="{dp_idx}",state="total"}} '
                f"{self.total_blocks_per_dp}"
            )
        return "\n".join(line for line in lines if line) + "\n"

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
