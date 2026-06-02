"""Single-process OpenAI-compatible HTTP server for NanoDeploy.

This is the ``nanodeploy serve`` entry point. It runs a *hybrid* engine
(prefill + decode in the same process) directly in-process and exposes an
OpenAI-compatible HTTP API, in the spirit of ``vllm serve``::

    nanodeploy serve /path/to/model \
        --host 0.0.0.0 --port 8100 \
        --served-model-name Qwen3-4B \
        --ctrl-address 127.0.0.1:4479

Unlike the disaggregated stack (NanoRoute + ZMQ engine servers), this path
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

from nanodeploy.config import Config
from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy.server")

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
        self._active: dict[int, _Request] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="nanodeploy-engine", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def submit(self, req: _Request) -> None:
        self._inbox.put(req)

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
                comp = req.seq.completion_token_ids
                if len(comp) > req.emitted:
                    new_tokens = comp[req.emitted :]
                    req.emitted = len(comp)
                    self._push(req, {"tokens": new_tokens})
                if req.seq.is_finished:
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
        from nanodeploy.sampling_params import SamplingParams

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
        from nanodeploy import Sequence

        seq = Sequence(prompt_ids, sampling_params=sampling_params)
        seq.seq_id = next(_seq_id_counter)
        loop = asyncio.get_running_loop()
        req = _Request(seq=seq, aqueue=asyncio.Queue(), loop=loop)
        self.worker.submit(req)
        return req

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
                delta = full[len(decoded) :]
                decoded = full
                if delta:
                    yield delta, gen
        gen.finish_reason = "length" if len(gen.token_ids) >= max_tokens else "stop"


# ----------------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------------


def build_app(server: OpenAIServer):
    app = FastAPI(title="NanoDeploy OpenAI Server")

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
                        "owned_by": "nanodeploy",
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
        req = server.submit(prompt_ids, sampling_params)

        created = int(time.time())
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = server.served_model_name
        stream = bool(body.get("stream", False))

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
        req = server.submit(prompt_ids, sampling_params)

        created = int(time.time())
        cmpl_id = f"cmpl-{uuid.uuid4().hex}"
        model = server.served_model_name
        stream = bool(body.get("stream", False))

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

    return app


# ----------------------------------------------------------------------------
# dlslime-ctrl self-registration
# ----------------------------------------------------------------------------

CTRL_ENTITY_KIND = "nanodeploy"


def _advertise_host(host: str) -> str:
    if host in ("0.0.0.0", ""):
        from nanodeploy.context.distributed import get_local_ip

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
):
    """Register this HTTP endpoint with dlslime-ctrl and start heartbeat.

    Returns the ``NanoCtrlClient`` (call ``.stop()`` on shutdown) or ``None``.
    """
    from dlslime.ctrl import NanoCtrlClient

    advertise_host = _advertise_host(host)
    entity_id = f"nanodeploy-{served_model_name}-{advertise_host}:{port}"
    endpoint = {"host": advertise_host, "port": port, "protocol": "http"}
    metadata = {
        "served_model_name": served_model_name,
        "model_path": model_path,
        "role": "hybrid",
        "host": advertise_host,
        "port": port,
    }

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

    from nanodeploy.llm_component import LLM

    host = config.host
    port = config.port

    logger.info("=" * 72)
    logger.info("NanoDeploy OpenAI Server")
    logger.info(f"  model:             {config.model}")
    logger.info(f"  served-model-name: {served_model_name}")
    logger.info(f"  http:              {host}:{port}")
    logger.info(f"  mode:              {config.mode}")
    logger.info(f"  ctrl-address:      {ctrl_address or '(disabled)'}")
    logger.info("=" * 72)

    # Build the in-process engine. We deliberately use LLM (not LLMComponent)
    # and leave config.ctrl_address unset so the *engine* does not register its
    # ZMQ endpoint; this server registers its own HTTP endpoint instead.
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
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Could not register with dlslime-ctrl: {e}")

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:  # noqa: ANN202
        if ctrl_client is not None:
            ctrl_client.stop()
        worker.stop()

    uvicorn.run(app, host=host, port=port)
