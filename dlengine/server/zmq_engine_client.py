"""ZMQ client driver for the ``dlengine serve`` multiprocess engine path.

The OpenAI HTTP server always runs the engine in a separate process (see
:func:`dlengine.server.engine_server.run_engine_server`) that exposes a zmq
DEALER socket over an ``ipc://`` endpoint. This module provides
:class:`ZmqEngineWorker` (``start``/``stop``/``submit``/``free_sequences``
surface, consumed by :class:`~dlengine.server.openai_server.OpenAIServer`) that
talks to that engine process as a zmq client.

This worker lives entirely on the
FastAPI asyncio event loop: an outbound queue serializes sends and a single recv
task fans incoming StepOut/Migration packets back into each request's asyncio
queue with ``put_nowait``.

Wire protocol (JSON packet, see ``dlengine.server.wire``):
- client -> engine: action 1 = ADD (Rust bincode RequestIn/RequestMigrate), 2 = GET_INFO,
  3 = FREE (FreeSequences).
- engine -> client: action 0 = StepOut, 1 = Migration (Rust bincode RequestMigrate),
  2 = engine_info (JSON).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any, Optional

import zmq
import zmq.asyncio

from dlengine.logging import get_logger
from dlengine.server import pd
from dlengine.server.wire import (
    decode_packet,
    decode_stepout,
    encode_free_sequences,
    encode_packet,
    SequenceStatus,
)

logger = get_logger("dlengine.server")
_TTFT_DEBUG = os.environ.get("DLENGINE_TTFT_DEBUG", "0") == "1"

# Action codes (must match engine_server.BackendService).
_ACTION_STEPOUT = 0
_ACTION_MIGRATION = 1
_ACTION_ADD = 1
_ACTION_GET_INFO = 2
_ACTION_FREE = 3
_ACTION_ABORT = 5
_ACTION_GET_METRICS = 6


class ZmqEngineWorker:
    """Drives a separate engine process over zmq, mirroring ``EngineWorker``.

    HTTP handlers call :meth:`submit` (sync); requests are serialized and queued
    for the send loop. The recv loop decodes engine packets and pushes newly
    produced tokens / migration payloads back to each request's asyncio queue.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._ctx: Optional[zmq.asyncio.Context] = None
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._active: dict[int, Any] = {}
        self._outbox: "asyncio.Queue[tuple[int, bytes]]" = None  # type: ignore[assignment]
        self._recv_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self.engine_id: Optional[str] = None
        self._metrics_future: Optional[asyncio.Future] = None

    async def start(self, info_timeout: Optional[float] = None) -> None:
        """Connect to the engine process and wait until it is ready.

        Sends a GET_INFO request and awaits the reply; the engine only answers
        once its (heavy) model load has finished and its step loop is draining
        the request queue, so this doubles as a readiness barrier. Populates
        :attr:`engine_id` from the engine info.
        """
        self._ctx = zmq.asyncio.Context.instance()
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._endpoint)

        self._outbox = asyncio.Queue()
        self._recv_task = asyncio.create_task(self._recv_loop(), name="zmq-engine-recv")
        self._send_task = asyncio.create_task(self._send_loop(), name="zmq-engine-send")

        logger.info(f"ZmqEngineWorker connecting to engine at {self._endpoint}")
        self._outbox.put_nowait((_ACTION_GET_INFO, b""))
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=info_timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"Timed out waiting for engine info from {self._endpoint}; "
                "engine may still be loading"
            )
        logger.info(f"ZmqEngineWorker ready (engine_id={self.engine_id})")

    def stop(self) -> None:
        for task in (self._recv_task, self._send_task):
            if task is not None:
                task.cancel()
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:  # noqa: BLE001
                pass

    # -- outbound --------------------------------------------------------

    def submit(self, req: Any) -> None:
        """Queue an ADD request for the engine socket."""
        if getattr(req, "migration_payload", None):
            payload = pd.RequestMigrate.from_bytes(
                base64.b64decode(req.migration_payload)
            )
        else:
            payload = pd.RequestIn(
                req.seq_id,
                req.prompt_ids or [],
                req.sampling_params,
                req.affinity_key,
            )
        self._active[req.seq_id] = req
        self._outbox.put_nowait((_ACTION_ADD, payload))

    def free_sequences(self, seq_ids: list[int]) -> None:
        """Queue a FREE packet (PD: release migrated prefill-side KV blocks)."""
        if not seq_ids:
            return
        payload = self._build_free_payload(seq_ids)
        self._outbox.put_nowait((_ACTION_FREE, payload))

    def abort(self, seq_id: int) -> None:
        """Stop generating for ``seq_id`` (server-side stop string / cancel).

        Finalizes the request locally so the caller's stream ends immediately
        (without a round-trip), then asks the engine to drop the sequence and
        free its KV blocks so it stops running out to ``max_tokens``.
        """
        req = self._active.pop(seq_id, None)
        if req is not None:
            self._push(req, {"finish": True})
            self._push(req, None)
        if self._outbox is not None:
            payload = self._build_free_payload([seq_id])
            self._outbox.put_nowait((_ACTION_ABORT, payload))

    async def get_metrics(self) -> str:
        """Fetch Prometheus-format metrics from the engine process.

        Sends a GET_METRICS request and waits for the response.
        Returns the metrics text or empty string on error.
        """
        if self._socket is None:
            return ""
        loop = asyncio.get_event_loop()
        self._metrics_future = loop.create_future()
        self._outbox.put_nowait((_ACTION_GET_METRICS, b""))
        try:
            return await asyncio.wait_for(self._metrics_future, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for metrics response")
            return ""
        except Exception as e:
            logger.warning(f"Error fetching metrics: {e}")
            return ""
        finally:
            self._metrics_future = None

    def _build_free_payload(self, seq_ids: list[int]) -> bytes:
        return encode_free_sequences(seq_ids, self.engine_id or "")

    async def _send_loop(self) -> None:
        assert self._socket is not None
        while True:
            action, item = await self._outbox.get()
            try:
                seq_id = getattr(item, "seq_id", None)
                payload = item.to_bytes() if isinstance(item, pd.RequestIn) else item
                if isinstance(item, pd.RequestMigrate):
                    payload = item.to_bytes()
                await self._socket.send(encode_packet(action, payload))
                if action == _ACTION_ADD and seq_id is not None:
                    req = self._active.get(int(seq_id))
                    if req is not None:
                        now = time.perf_counter()
                        setattr(req, "zmq_sent_at", now)
                        if _TTFT_DEBUG:
                            logger.info(
                                "[ttft] seq_id=%s zmq_add_sent submit_to_send=%.2fms",
                                seq_id,
                                (now - getattr(req, "submitted_at", now)) * 1000,
                            )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"ZmqEngineWorker send failed (action={action}): {e}")

    # -- inbound ---------------------------------------------------------

    async def _recv_loop(self) -> None:
        assert self._socket is not None
        logger.info("ZmqEngineWorker recv loop started")
        while True:
            try:
                data = await self._socket.recv()
            except asyncio.CancelledError:
                raise
            except zmq.ZMQError as e:
                if e.errno != zmq.ETERM:
                    logger.error(f"ZmqEngineWorker recv error: {e}")
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"ZmqEngineWorker recv loop error: {e}")
                break
            try:
                action, payload = decode_packet(bytes(data))
                if action == _ACTION_STEPOUT:
                    self._handle_stepout(payload)
                elif action == _ACTION_MIGRATION:
                    self._handle_migration(payload)
                elif action == _ACTION_GET_INFO:
                    self._handle_engine_info(payload)
                elif action == _ACTION_GET_METRICS:
                    self._handle_engine_metrics(payload)
                else:
                    logger.warning(f"ZmqEngineWorker unknown action: {action}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"ZmqEngineWorker failed to process packet: {e}")

    def _push(self, req: Any, item: Optional[dict]) -> None:
        # recv loop runs on the same event loop that owns req.aqueue.
        req.aqueue.put_nowait(item)

    def _handle_stepout(self, payload: bytes) -> None:
        step = decode_stepout(payload)
        seq_id = step.seq_id
        req = self._active.get(seq_id)
        if req is None:
            return

        tokens = step.token_ids or [step.token_id]
        if tokens:
            if getattr(req, "zmq_first_stepout_at", 0.0) == 0.0:
                now = time.perf_counter()
                setattr(req, "zmq_first_stepout_at", now)
                sent_at = getattr(req, "zmq_sent_at", 0.0)
                if _TTFT_DEBUG:
                    logger.info(
                        "[ttft] seq_id=%s zmq_first_stepout send_to_recv=%.2fms "
                        "submit_to_recv=%.2fms tokens=%s status=%s",
                        seq_id,
                        (now - sent_at) * 1000 if sent_at else -1.0,
                        (now - getattr(req, "submitted_at", now)) * 1000,
                        len(tokens),
                        step.status,
                    )
            self._push(req, {"tokens": tokens})

        if step.status == SequenceStatus.FINISHED:
            self._push(req, {"finish": True})
            self._push(req, None)
            self._active.pop(seq_id, None)
            logger.info(f"Request finished: seq_id={seq_id}")

    def _handle_migration(self, payload: bytes) -> None:
        try:
            migration = pd.RequestMigrate.from_bytes(payload)
            seq_id, first_token = migration.metadata
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to decode migration metadata: {e}")
            return
        req = self._active.get(seq_id)
        if req is None:
            return
        b64 = base64.b64encode(payload).decode("ascii")
        self._push(
            req,
            {
                "migration": b64,
                "first_token": first_token,
                "seq_id": seq_id,
            },
        )
        self._push(req, None)
        self._active.pop(seq_id, None)
        logger.info(f"Request handed off for migration: seq_id={seq_id}")

    def _handle_engine_info(self, payload: bytes) -> None:
        try:
            info = json.loads(payload.decode("utf-8"))
            self.engine_id = info.get("id")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to parse engine info: {e}")
        finally:
            self._ready.set()

    def _handle_engine_metrics(self, payload: bytes) -> None:
        """Handle metrics response from engine."""
        if self._metrics_future is not None and not self._metrics_future.done():
            try:
                metrics = payload.decode("utf-8")
                self._metrics_future.set_result(metrics)
            except Exception as e:  # noqa: BLE001
                self._metrics_future.set_exception(e)
