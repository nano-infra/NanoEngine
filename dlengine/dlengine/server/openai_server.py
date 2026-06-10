"""Single-process OpenAI-compatible HTTP server for DLEngine.

This is the ``dlengine serve`` entry point. It runs a *hybrid* engine
(prefill + decode in the same process) directly in-process and exposes an
OpenAI-compatible HTTP API, in the spirit of ``vllm serve``::

    dlengine serve /path/to/model \
        --host 0.0.0.0 --port 8100 \
        --served-model-name Qwen3-4B \
        --ctrl-address 127.0.0.1:4479

Unlike the disaggregated stack (dlengine-router + ZMQ engine servers), this path
talks to the engine through in-process queues, with no ZMQ and no Rust
front-end. When ``--ctrl-address`` is given, the server registers its own
HTTP endpoint with dlslime-ctrl so a router (e.g. DLRouter) can discover it.

Endpoints:
- ``GET  /health``
- ``GET  /v1/models``
- ``POST /v1/completions``
- ``POST /v1/chat/completions``  (streaming + non-streaming)
"""

from __future__ import annotations

import asyncio
import itertools
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from dlengine.config import Config
from dlengine.logging import get_logger

logger = get_logger("dlengine.server")

# Engine-assigned sequence ids below this are reserved for system/dummy
# sequences (see engine step loop), so user requests start well above it.
_SEQ_ID_BASE = 1024
_seq_id_counter = itertools.count(_SEQ_ID_BASE)


@dataclass
class _Request:
    """In-flight request state shared between the HTTP and engine threads."""

    seq: Any
    aqueue: "asyncio.Queue[Optional[dict]]"
    loop: asyncio.AbstractEventLoop
    emitted: int = 0


@dataclass
class _Generation:
    """Result of consuming a request's token stream."""

    token_ids: list = field(default_factory=list)
    finish_reason: str = "stop"


class EngineWorker:
    """Drives ``engine.step()`` on a background thread.

    HTTP handlers submit requests via :meth:`submit`; the worker feeds them to
    the engine, then after every step pushes newly produced completion tokens
    back to each request's asyncio queue (thread-safe via ``call_soon_threadsafe``).
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._inbox: "queue.Queue[_Request]" = queue.Queue()
        # PD: seq_ids whose prefill-side MIGRATE KV blocks can be freed once a
        # decode engine has pulled them. Drained on the engine thread.
        self._free_inbox: "queue.Queue[list[int]]" = queue.Queue()
        self._active: dict[int, _Request] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="dlengine-engine", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def submit(self, req: _Request) -> None:
        self._inbox.put(req)

    def free_sequences(self, seq_ids: list[int]) -> None:
        """Queue prefill-side MIGRATE KV blocks for release (PD)."""
        if seq_ids:
            self._free_inbox.put(list(seq_ids))

    def _push(self, req: _Request, item: Optional[dict]) -> None:
        req.loop.call_soon_threadsafe(req.aqueue.put_nowait, item)

    def _run(self) -> None:
        engine = self.engine
        logger.info("Engine worker loop started")
        while not self._stop.is_set():
            # Admit any newly submitted requests.
            while True:
                try:
                    req = self._inbox.get_nowait()
                except queue.Empty:
                    break
                try:
                    engine.add_request(req.seq)
                    self._active[req.seq.seq_id] = req
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to add request {req.seq.seq_id}: {e}")
                    self._push(req, {"error": str(e)})
                    self._push(req, None)

            # PD: release prefill-side MIGRATE KV blocks once a decode engine
            # has confirmed it pulled them (driven via /pd/free).
            while True:
                try:
                    free_ids = self._free_inbox.get_nowait()
                except queue.Empty:
                    break
                try:
                    from dlengine import Sequence

                    stubs = []
                    for sid in free_ids:
                        stub = Sequence([])
                        stub.seq_id = sid
                        stubs.append(stub)
                    engine.free_to_be_migrated(stubs)
                    logger.info(f"Freed migrated sequences: {free_ids}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"free_to_be_migrated failed for {free_ids}: {e}")

            if engine.is_finished():
                time.sleep(0.001)
                continue

            try:
                engine.step()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Engine step failed: {e}", exc_info=True)
                for req in list(self._active.values()):
                    self._push(req, {"error": str(e)})
                    self._push(req, None)
                self._active.clear()
                time.sleep(0.1)
                continue

            for seq_id, req in list(self._active.items()):
                seq = req.seq
                # PD prefill handoff: a ``mode="prefill"`` engine marks the
                # sequence TO_BE_MIGRATED after the first generated token. Ship
                # the serialized sequence (with its MIGRATE BlockContext) to the
                # caller and stop driving it locally.
                if seq.is_to_be_migrated:
                    try:
                        from dlengine.server.pd import encode_migration

                        comp = seq.completion_token_ids
                        payload = encode_migration(seq)
                        self._push(
                            req,
                            {
                                "migration": payload,
                                "first_token": comp[-1] if comp else None,
                                "seq_id": seq_id,
                            },
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Failed to serialize migration for {seq_id}: {e}")
                        self._push(req, {"error": f"migration serialize failed: {e}"})
                    self._push(req, None)
                    self._active.pop(seq_id, None)
                    continue

                comp = seq.completion_token_ids
                if len(comp) > req.emitted:
                    new_tokens = comp[req.emitted :]
                    req.emitted = len(comp)
                    self._push(req, {"tokens": new_tokens})
                if seq.is_finished:
                    self._push(req, {"finish": True})
                    self._push(req, None)
                    self._active.pop(seq_id, None)


class OpenAIServer:
    """Holds the engine worker, tokenizer and serving metadata."""

    def __init__(
        self,
        worker: EngineWorker,
        tokenizer: Any,
        served_model_name: str,
        model_path: str,
        default_max_tokens: int = 512,
    ) -> None:
        self.worker = worker
        self.tokenizer = tokenizer
        self.served_model_name = served_model_name
        self.model_path = model_path.rstrip("/")
        self.default_max_tokens = default_max_tokens
        self._model_aliases = self._build_model_aliases()

    def _build_model_aliases(self) -> set[str]:
        """OpenAI ``model`` values accepted on this server (alias + path)."""
        aliases: set[str] = {self.served_model_name}
        for raw in (self.model_path, os.path.expanduser(self.model_path)):
            aliases.add(raw.rstrip("/"))
            try:
                aliases.add(os.path.realpath(raw).rstrip("/"))
            except OSError:
                pass
        base = self.model_path.split("/")[-1]
        if base:
            aliases.add(base)
        return aliases

    def resolve_request_model(self, requested: str | None) -> str | None:
        """Map client ``model`` to canonical served name, or None if unknown."""
        if not requested:
            return None
        key = requested.strip().rstrip("/")
        if key in self._model_aliases:
            return self.served_model_name
        expanded = os.path.expanduser(key)
        if expanded.rstrip("/") in self._model_aliases:
            return self.served_model_name
        try:
            if os.path.realpath(expanded).rstrip("/") in self._model_aliases:
                return self.served_model_name
        except OSError:
            pass
        return None

    # -- prompt construction --

    def _encode_chat(self, messages: list[dict]) -> list[int]:
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Minimal fallback when the tokenizer ships no chat template.
            parts = [
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            ]
            prompt = "\n".join(parts) + "\nassistant:"
        return tok.encode(prompt)

    # -- request submission + streaming --

    def _build_sampling_params(self, body: dict) -> Any:
        from dlengine.sampling_params import SamplingParams

        max_tokens = (
            body.get("max_tokens")
            or body.get("max_completion_tokens")
            or self.default_max_tokens
        )
        temperature = body.get("temperature")
        if temperature is None:
            temperature = 1.0
        return SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            ignore_eos=bool(body.get("ignore_eos", False)),
        )

    def submit(self, prompt_ids: list[int], sampling_params: Any) -> _Request:
        from dlengine import Sequence

        seq = Sequence(prompt_ids, sampling_params=sampling_params)
        seq.seq_id = next(_seq_id_counter)
        loop = asyncio.get_running_loop()
        req = _Request(seq=seq, aqueue=asyncio.Queue(), loop=loop)
        self.worker.submit(req)
        return req

    def submit_migrated(self, seq: Any) -> _Request:
        """Submit a deserialized prefilled sequence to a decode engine (PD).

        The sequence keeps its original seq_id and its MIGRATE BlockContext, so
        the decode engine routes it to ``waiting_migration`` and RDMA-pulls the
        KV cache from the prefill engine on the next step. Already-generated
        completion tokens (the prefill token) are streamed back too, so the
        decode response carries the full answer.
        """
        loop = asyncio.get_running_loop()
        req = _Request(seq=seq, aqueue=asyncio.Queue(), loop=loop)
        self.worker.submit(req)
        return req

    async def await_migration(self, req: _Request) -> dict:
        """Drain a prefill request until it hands off or finishes locally.

        Returns one of:
        - ``{"migration": ..., "first_token": ..., "seq_id": ...}`` when the
          sequence was marked TO_BE_MIGRATED and should resume on a decode
          engine (the normal PD path), or
        - ``{"finished": True, "tokens": [...]}`` when the prefill engine fully
          finished the request locally (e.g. the first sampled token is EOS, so
          the scheduler marks it FINISHED instead of TO_BE_MIGRATED). In that
          case there is no KV to migrate and no decode handoff is needed.
        """
        tokens: list[int] = []
        while True:
            item = await req.aqueue.get()
            if item is None:
                return {"finished": True, "tokens": tokens}
            if "error" in item:
                raise RuntimeError(item["error"])
            if "migration" in item:
                return item
            if "tokens" in item:
                tokens.extend(item["tokens"])

    async def stream_text(
        self, req: _Request, max_tokens: int
    ) -> AsyncGenerator[tuple[str, _Generation], None]:
        """Yield ``(delta_text, generation)`` as tokens arrive."""
        gen = _Generation()
        decoded = ""
        while True:
            item = await req.aqueue.get()
            if item is None:
                break
            if "error" in item:
                raise RuntimeError(item["error"])
            if "tokens" in item:
                gen.token_ids.extend(item["tokens"])
                full = self.tokenizer.decode(gen.token_ids, skip_special_tokens=True)
                # A multi-byte UTF-8 character (e.g. an emoji) can be split
                # across several byte-level BPE tokens. Decoding before all of
                # its bytes have arrived yields a trailing U+FFFD replacement
                # char. Hold the delta back until the character completes so we
                # never emit (and lock in) a broken "\ufffd".
                if full.endswith("\ufffd"):
                    continue
                delta = full[len(decoded) :]
                decoded = full
                if delta:
                    yield delta, gen
        gen.finish_reason = "length" if len(gen.token_ids) >= max_tokens else "stop"


# ----------------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------------


def build_app(server: OpenAIServer):
    app = FastAPI(title="DLEngine OpenAI Server")

    @app.get("/health")
    async def health() -> PlainTextResponse:  # noqa: ANN202
        return PlainTextResponse("OK")

    @app.get("/v1/models")
    async def models() -> JSONResponse:  # noqa: ANN202
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": server.served_model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "dlengine",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):  # noqa: ANN202
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON body: {e}",
                        "type": "invalid_request_error",
                    }
                },
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "JSON body must be an object",
                        "type": "invalid_request_error",
                    }
                },
            )
        if server.resolve_request_model(body.get("model")) is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": (
                            f"Model '{body.get('model')}' not found. "
                            f"Use one of: {sorted(server._model_aliases)}"
                        ),
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )
        messages = body.get("messages") or []
        sampling_params = server._build_sampling_params(body)
        max_tokens = sampling_params.max_tokens
        prompt_ids = server._encode_chat(messages)

        created = int(time.time())
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = server.served_model_name
        stream = bool(body.get("stream", False))

        kv_transfer = body.get("kv_transfer_params") or {}

        # PD prefill stage: run prefill and return the serialized migration
        # payload instead of generating a full completion.
        #
        # NOTE: do NOT clamp max_tokens to 1 here. A ``mode="prefill"`` engine
        # already stops after the first generated token by marking the sequence
        # TO_BE_MIGRATED (see scheduler postprocess). Clamping to 1 would instead
        # make ``num_completed_tokens >= max_tokens`` true, marking the sequence
        # FINISHED (no migration), and would also serialize max_tokens=1 into the
        # migrated sequence so the decode engine generates nothing. We keep the
        # user's max_tokens so the decode engine resumes with the correct budget.
        if kv_transfer.get("do_remote_decode"):
            preq = server.submit(prompt_ids, sampling_params)
            try:
                mig = await server.await_migration(preq)
            except RuntimeError as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": str(e), "type": "engine_error"}},
                )
            # Prefill finished the request locally (e.g. first token is EOS):
            # there is nothing to migrate, so return the completion directly.
            if not mig.get("migration"):
                done_tokens = mig.get("tokens") or []
                text = server.tokenizer.decode(done_tokens, skip_special_tokens=True)
                return JSONResponse(
                    {
                        "id": cmpl_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt_ids),
                            "completion_tokens": len(done_tokens),
                            "total_tokens": len(prompt_ids) + len(done_tokens),
                        },
                    }
                )
            return JSONResponse(
                {
                    "id": cmpl_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": 0,
                        "total_tokens": len(prompt_ids),
                    },
                    "kv_transfer_params": {
                        "migration": mig["migration"],
                        "first_token": mig.get("first_token"),
                        "seq_id": mig.get("seq_id"),
                    },
                }
            )

        # PD decode stage: resume a prefilled sequence pulled from a prefill node.
        if kv_transfer.get("migration"):
            from dlengine.server.pd import decode_migration

            try:
                migrated_seq = decode_migration(kv_transfer["migration"])
            except Exception as e:  # noqa: BLE001
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": f"invalid migration payload: {e}",
                            "type": "invalid_request_error",
                        }
                    },
                )
            req = server.submit_migrated(migrated_seq)
        else:
            req = server.submit(prompt_ids, sampling_params)

        if stream:

            async def event_stream() -> AsyncGenerator[bytes, None]:
                import json

                first = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first)}\n\n".encode()
                gen = _Generation()
                try:
                    async for delta, gen in server.stream_text(req, max_tokens):
                        chunk = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                except RuntimeError as e:
                    err = {"error": {"message": str(e), "type": "engine_error"}}
                    yield f"data: {json.dumps(err)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
                final = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": gen.finish_reason}
                    ],
                }
                yield f"data: {json.dumps(final)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Non-streaming: drain the whole generation.
        text = ""
        gen = _Generation()
        try:
            async for delta, gen in server.stream_text(req, max_tokens):
                text += delta
        except RuntimeError as e:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "engine_error"}},
            )
        return JSONResponse(
            {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": gen.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(gen.token_ids),
                    "total_tokens": len(prompt_ids) + len(gen.token_ids),
                },
            }
        )

    @app.post("/v1/completions")
    async def completions(request: Request):  # noqa: ANN202
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON body: {e}",
                        "type": "invalid_request_error",
                    }
                },
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "JSON body must be an object",
                        "type": "invalid_request_error",
                    }
                },
            )
        if server.resolve_request_model(body.get("model")) is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": (
                            f"Model '{body.get('model')}' not found. "
                            f"Use one of: {sorted(server._model_aliases)}"
                        ),
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        sampling_params = server._build_sampling_params(body)
        max_tokens = sampling_params.max_tokens
        prompt_ids = server.tokenizer.encode(prompt)

        created = int(time.time())
        cmpl_id = f"cmpl-{uuid.uuid4().hex}"
        model = server.served_model_name
        stream = bool(body.get("stream", False))

        kv_transfer = body.get("kv_transfer_params") or {}

        # PD prefill stage: run prefill and return a migration payload. Do NOT
        # clamp max_tokens to 1 -- the mode="prefill" engine already stops after
        # the first token via TO_BE_MIGRATED, and the user's max_tokens must be
        # preserved into the migrated sequence for the decode engine. See the
        # chat handler above for the full rationale.
        if kv_transfer.get("do_remote_decode"):
            preq = server.submit(prompt_ids, sampling_params)
            try:
                mig = await server.await_migration(preq)
            except RuntimeError as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": str(e), "type": "engine_error"}},
                )
            # Prefill finished the request locally (e.g. first token is EOS):
            # there is nothing to migrate, so return the completion directly.
            if not mig.get("migration"):
                done_tokens = mig.get("tokens") or []
                text = server.tokenizer.decode(done_tokens, skip_special_tokens=True)
                return JSONResponse(
                    {
                        "id": cmpl_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "text": text, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt_ids),
                            "completion_tokens": len(done_tokens),
                            "total_tokens": len(prompt_ids) + len(done_tokens),
                        },
                    }
                )
            return JSONResponse(
                {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": 0,
                        "total_tokens": len(prompt_ids),
                    },
                    "kv_transfer_params": {
                        "migration": mig["migration"],
                        "first_token": mig.get("first_token"),
                        "seq_id": mig.get("seq_id"),
                    },
                }
            )

        # PD decode stage: resume a prefilled sequence.
        if kv_transfer.get("migration"):
            from dlengine.server.pd import decode_migration

            try:
                migrated_seq = decode_migration(kv_transfer["migration"])
            except Exception as e:  # noqa: BLE001
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": f"invalid migration payload: {e}",
                            "type": "invalid_request_error",
                        }
                    },
                )
            req = server.submit_migrated(migrated_seq)
        else:
            req = server.submit(prompt_ids, sampling_params)

        if stream:

            async def event_stream() -> AsyncGenerator[bytes, None]:
                import json

                gen = _Generation()
                try:
                    async for delta, gen in server.stream_text(req, max_tokens):
                        chunk = {
                            "id": cmpl_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "text": delta, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                except RuntimeError as e:
                    err = {"error": {"message": str(e), "type": "engine_error"}}
                    yield f"data: {json.dumps(err)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
                final = {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "text": "", "finish_reason": gen.finish_reason}
                    ],
                }
                yield f"data: {json.dumps(final)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        text = ""
        gen = _Generation()
        try:
            async for delta, gen in server.stream_text(req, max_tokens):
                text += delta
        except RuntimeError as e:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "engine_error"}},
            )
        return JSONResponse(
            {
                "id": cmpl_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "text": text, "finish_reason": gen.finish_reason}
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(gen.token_ids),
                    "total_tokens": len(prompt_ids) + len(gen.token_ids),
                },
            }
        )

    @app.post("/pd/free")
    async def pd_free(request: Request):  # noqa: ANN202
        """Release prefill-side MIGRATE KV blocks after a decode pull (PD).

        Called by the router (or decode node) once the decode engine has pulled
        the KV cache, so the prefill engine can reclaim the migrated blocks.
        """
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON body: {e}",
                        "type": "invalid_request_error",
                    }
                },
            )
        seq_ids = body.get("seq_ids") if isinstance(body, dict) else None
        if not isinstance(seq_ids, list):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "'seq_ids' must be a list of integers",
                        "type": "invalid_request_error",
                    }
                },
            )
        try:
            seq_ids = [int(s) for s in seq_ids]
        except (TypeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"invalid seq_ids: {e}",
                        "type": "invalid_request_error",
                    }
                },
            )
        server.worker.free_sequences(seq_ids)
        return JSONResponse({"freed": seq_ids})

    return app


# ----------------------------------------------------------------------------
# dlslime-ctrl self-registration
# ----------------------------------------------------------------------------

CTRL_ENTITY_KIND = "dlengine"


def _advertise_host(host: str) -> str:
    if host in ("0.0.0.0", ""):
        from dlengine.context.distributed import get_local_ip

        return get_local_ip()
    return host


def register_with_ctrl(
    *,
    ctrl_address: str,
    ctrl_scope: Optional[str],
    host: str,
    port: int,
    served_model_name: str,
    model_path: str,
    role: str = "hybrid",
    engine_id: Optional[str] = None,
):
    """Register this HTTP endpoint with dlslime-ctrl and start heartbeat.

    ``role`` (hybrid|prefill|decode) lets a router (DLRouter) assign the node to
    the right PD pool; ``engine_id`` maps the HTTP node to its in-engine
    NanoCtrl entity (used for KV migration peer resolution).

    Returns the ``NanoCtrlClient`` (call ``.stop()`` on shutdown) or ``None``.
    """
    from dlslime.ctrl import NanoCtrlClient

    advertise_host = _advertise_host(host)
    entity_id = f"dlengine-{served_model_name}-{advertise_host}:{port}"
    endpoint = {"host": advertise_host, "port": port, "protocol": "http"}
    metadata = {
        "served_model_name": served_model_name,
        "model_path": model_path,
        "role": role,
        "host": advertise_host,
        "port": port,
    }
    if engine_id is not None:
        metadata["engine_id"] = engine_id

    client = NanoCtrlClient(ctrl_address, ctrl_scope)
    client.check_connection()

    def _do_register() -> bool:
        return client.register(
            entity_id,
            kind=CTRL_ENTITY_KIND,
            endpoint=endpoint,
            metadata=metadata,
        )

    if _do_register():
        client.start_heartbeat(on_not_found=_do_register, name=f"heartbeat-{entity_id}")
        logger.info(
            f"Registered with dlslime-ctrl: {entity_id} -> http://{advertise_host}:{port}"
        )
    else:
        logger.error("dlslime-ctrl registration failed; node will not be discoverable")
    return client


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def run_server(
    config: Config,
    *,
    served_model_name: str,
    ctrl_address: Optional[str] = None,
    ctrl_scope: Optional[str] = None,
) -> None:
    """Build the engine, start the HTTP server, and optionally register to ctrl."""
    import uvicorn
    from transformers import PreTrainedTokenizerFast

    from dlengine.llm_component import LLM, LLMComponent

    host = config.host
    port = config.port

    logger.info("=" * 72)
    logger.info("DLEngine OpenAI Server")
    logger.info(f"  model:             {config.model}")
    logger.info(f"  served-model-name: {served_model_name}")
    logger.info(f"  http:              {host}:{port}")
    logger.info(f"  mode:              {config.mode}")
    logger.info(f"  ctrl-address:      {ctrl_address or '(disabled)'}")
    logger.info("=" * 72)

    # Build the in-process engine.
    #
    # hybrid: use the bare LLM. The engine does not self-register with
    #   dlslime-ctrl; this HTTP server registers its own endpoint instead.
    # prefill/decode (PD disaggregation): use LLMComponent so the engine
    #   registers under its engine_id with peer_addrs / pool metadata, which the
    #   peer decode engine needs to RDMA-pull KV. Workers also register their KV
    #   memory-regions (the mode != "hybrid" path). This requires ctrl_address.
    if config.mode != "hybrid":
        if not config.ctrl_address:
            raise ValueError(
                f"mode={config.mode!r} (PD disaggregation) requires --ctrl_address "
                "so engines can register peer agents and resolve KV migration peers"
            )
        engine = LLMComponent(config)
    else:
        engine = LLM(config)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model)

    worker = EngineWorker(engine)
    worker.start()

    server = OpenAIServer(
        worker=worker,
        tokenizer=tokenizer,
        served_model_name=served_model_name,
        model_path=config.model,
    )
    app = build_app(server)

    ctrl_client = None

    @app.on_event("startup")
    async def _on_startup() -> None:  # noqa: ANN202
        nonlocal ctrl_client
        if ctrl_address:
            try:
                ctrl_client = register_with_ctrl(
                    ctrl_address=ctrl_address,
                    ctrl_scope=ctrl_scope,
                    host=host,
                    port=port,
                    served_model_name=served_model_name,
                    model_path=config.model,
                    role=config.mode,
                    engine_id=getattr(engine, "engine_id", None),
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Could not register with dlslime-ctrl: {e}")

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:  # noqa: ANN202
        if ctrl_client is not None:
            ctrl_client.stop()
        worker.stop()

    uvicorn.run(app, host=host, port=port)
