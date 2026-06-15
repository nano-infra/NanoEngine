"""Engine-side request/metric dumper to Redis (works offline *and* online).

This lives in the engine (``LLMEngine``) rather than the HTTP server so that it
captures every request regardless of how the engine is driven:

- ``dlengine serve`` (the engine runs in its own process), and
- offline scripts that construct ``LLM`` / ``LLMEngine`` directly and call
  ``generate()`` (no HTTP server, no asyncio event loop).

Because the engine step loop is synchronous and on the latency-critical path,
writes are handed to a background daemon thread via a bounded queue and never
block stepping (entries are dropped if the queue backs up, so a slow/unavailable
Redis can never stall serving).

Enable via the ``--dump_requests_redis`` serve flag / ``Config.dump_requests_redis``
(or the ``DLENGINE_DUMP_REQUESTS_REDIS`` env var as a fallback):

- unset / empty / ``0`` / ``false`` -> disabled (zero overhead)
- ``1`` / ``true``                  -> ``redis://127.0.0.1:6379/0``
- any other value                   -> used verbatim as the Redis URL

Two record kinds are appended to the stream, joinable on ``seq_id``:

- ``kind="request"``  (on admission): ``ts``, ``seq_id``, ``model``,
  ``affinity_key``, ``prompt_len``, ``token_ids`` (JSON int list), ``prompt_text``.
- ``kind="complete"`` (on finish): ``ts``, ``seq_id``, ``model``,
  ``affinity_key``, ``prompt_len``, ``cached_len``, ``output_len`` and
  engine-measured latency (ms): ``ttft_ms``, ``tpot_ms``, ``e2e_ms``,
  ``queue_ms``, ``prefill_ms`` (first_scheduled -> first token), ``avg_itl_ms`` /
  ``p50_itl_ms`` / ``p99_itl_ms``, ``num_prefill_chunks`` and ``chunk_prefill_ms``
  (JSON float list of per-chunk prefill latencies), plus ``itl_ms`` (JSON float
  list of per-token inter-token latencies).

Inspect with, e.g.::

    redis-cli XREVRANGE dlengine:requests + - COUNT 5
    redis-cli XLEN dlengine:requests
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Optional

from dlengine.logging import get_logger

logger = get_logger("dlengine")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


def _resolve_url(setting: Optional[str]) -> Optional[str]:
    # Explicit config value takes precedence over the env var fallback.
    raw = (
        setting
        if setting is not None
        else os.environ.get("DLENGINE_DUMP_REQUESTS_REDIS", "")
    )
    raw = (raw or "").strip()
    if raw.lower() in _FALSE:
        return None
    if raw.lower() in _TRUE:
        return "redis://127.0.0.1:6379/0"
    return raw


def _round_ms(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def _rounded_list(values: Any) -> list[float]:
    try:
        return [round(float(x), 3) for x in values]
    except Exception:  # noqa: BLE001 - debug output should never break serving
        return []


def _prompt_token_ids(seq: Any) -> list[int]:
    try:
        return list(seq.prompt_token_ids)
    except Exception:  # noqa: BLE001 - tolerate older Sequence wrappers
        return list(seq.token_ids)[: seq.num_prompt_tokens]


def _decode_prompt(tokenizer: Any, prompt_ids: list[int]) -> str:
    try:
        return tokenizer.decode(prompt_ids, skip_special_tokens=False)
    except Exception:  # noqa: BLE001 - prompt text is best-effort debug data
        return ""


class MetricDumper:
    """Fire-and-forget Redis stream writer driven from the engine step loop.

    Synchronous API (``dump``) that enqueues to a background thread; safe to call
    from the engine thread with no event loop. Never raises and never blocks the
    caller for more than a queue ``put_nowait``.
    """

    def __init__(
        self,
        setting: Optional[str] = None,
        stream: Optional[str] = None,
        maxlen: Optional[int] = None,
        max_pending: int = 20000,
    ) -> None:
        self._url = _resolve_url(setting)
        self.enabled = self._url is not None
        self._stream = stream or os.environ.get(
            "DLENGINE_DUMP_REQUESTS_STREAM", "dlengine:requests"
        )
        if maxlen is not None:
            self._maxlen = maxlen
        else:
            try:
                self._maxlen = int(
                    os.environ.get("DLENGINE_DUMP_REQUESTS_MAXLEN", "200000")
                )
            except ValueError:
                self._maxlen = 200000
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=max_pending)
        self._dropped = 0
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._run, name="dlengine-metric-dump", daemon=True
            )
            self._thread.start()
            logger.info(
                "Engine request/metric dump to Redis ENABLED: url=%s stream=%s maxlen~=%d",
                self._url,
                self._stream,
                self._maxlen,
            )

    def dump(self, **fields: Any) -> None:
        """Enqueue ``fields`` for a non-blocking push to the Redis stream.

        Drops the entry (and logs once per 1000 drops) if the background writer
        cannot keep up, so the engine step loop is never blocked.
        """
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(fields)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 1000 == 1:
                logger.warning(
                    "MetricDumper queue full; dropped %d entries (Redis too slow?)",
                    self._dropped,
                )

    def _run(self) -> None:
        try:
            import redis

            client = redis.from_url(self._url, decode_responses=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("MetricDumper disabled (cannot connect to Redis): %s", e)
            self.enabled = False
            return
        while True:
            fields = self._queue.get()
            if fields is None:  # shutdown sentinel
                break
            try:
                entry = {
                    k: (
                        v
                        if isinstance(v, (str, int, float))
                        else json.dumps(v, ensure_ascii=False)
                    )
                    for k, v in fields.items()
                    if v is not None
                }
                client.xadd(self._stream, entry, maxlen=self._maxlen, approximate=True)
            except Exception as e:  # noqa: BLE001 - debug tool must never break serving
                logger.warning("MetricDumper xadd failed (will keep trying): %s", e)
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                # Force the sentinel through even under backpressure.
                self._queue.put(None)
            self._thread.join(timeout=2.0)
            self._thread = None


class EngineMetricDumper:
    """Build engine request/completion records and send them to Redis."""

    def __init__(
        self,
        model: str,
        setting: Optional[str] = None,
        stream: Optional[str] = None,
        maxlen: Optional[int] = None,
    ) -> None:
        self._model = os.path.basename(str(model).rstrip("/"))
        self._writer = MetricDumper(setting=setting, stream=stream, maxlen=maxlen)

    @property
    def enabled(self) -> bool:
        return self._writer.enabled

    def close(self) -> None:
        self._writer.close()

    def record_request(self, seq: Any, tokenizer: Any) -> None:
        """Dump the admitted request's tokenized prompt."""
        if not self.enabled:
            return
        prompt_ids = _prompt_token_ids(seq)
        self._writer.dump(
            kind="request",
            ts=time.time(),
            seq_id=seq.seq_id,
            model=self._model,
            affinity_key=str(getattr(seq, "affinity_key", 0)),
            prompt_len=seq.num_prompt_tokens,
            token_ids=prompt_ids,
            prompt_text=_decode_prompt(tokenizer, prompt_ids),
        )

    def record_completion(self, seq: Any, cached_len: int) -> None:
        """Dump per-request latency metrics from ``SequenceMetric``."""
        if not self.enabled:
            return
        metric = getattr(seq, "metric", None)
        if metric is None:
            return

        self._writer.dump(
            kind="complete",
            ts=time.time(),
            seq_id=seq.seq_id,
            model=self._model,
            affinity_key=str(getattr(seq, "affinity_key", 0)),
            prompt_len=seq.num_prompt_tokens,
            cached_len=cached_len,
            output_len=metric.num_generated_tokens,
            ttft_ms=_round_ms(metric.ttft),
            tpot_ms=_round_ms(metric.avg_tpot_wo_queueing),
            e2e_ms=_round_ms(metric.e2e_latency),
            queue_ms=_round_ms(metric.queueing_time_ms),
            prefill_ms=_round_ms(getattr(metric, "prefill_time_ms", None)),
            num_prefill_chunks=getattr(metric, "num_prefill_chunks", 0),
            chunk_prefill_ms=_rounded_list(
                getattr(metric, "prefill_chunk_samples", [])
            ),
            avg_itl_ms=_round_ms(metric.avg_itl),
            p50_itl_ms=_round_ms(metric.p50_itl),
            p99_itl_ms=_round_ms(metric.p99_itl),
            itl_ms=_rounded_list(metric.itl_samples),
        )
