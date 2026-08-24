"""OpenAI-compatible HTTP server for DLEngine.

This is the ``dlengine serve`` entry point. It exposes an OpenAI-compatible
HTTP API, in the spirit of ``vllm serve``::

    dlengine serve /path/to/model \
        --host 0.0.0.0 --port 8100 \
        --served-model-name Qwen3-4B \
        --ctrl-address 127.0.0.1:4479

The engine always runs in a separate process (``EngineServer``) exposing a zmq
DEALER over an ipc:// socket; this HTTP process connects as a zmq client
(``ZmqEngineWorker``) and never initializes CUDA/Ray itself. When
``--ctrl-address`` is given, the server registers its own HTTP endpoint with
dlslime-ctrl so a router (e.g. DLRouter) can discover it.

Endpoints:
- ``GET  /health``
- ``POST /start_profiler``
- ``POST /stop_profiler``
- ``GET  /v1/models``
- ``POST /v1/completions``
- ``POST /v1/chat/completions``  (streaming + non-streaming)
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)

from dlengine.config import Config
from dlengine.logging import get_logger
from dlengine.utils.trace_merger import merge_trace_jsons_to_gzip

logger = get_logger("dlengine.server")

# Engine-assigned sequence ids below this are reserved for system/dummy
# sequences (see engine step loop), so user requests start well above it.
_SEQ_ID_BASE = 1024
_seq_id_counter = itertools.count(_SEQ_ID_BASE)


def _is_token_id_list(value: Any) -> bool:
    return isinstance(value, list) and all(type(item) is int for item in value)


def _stream_includes_usage(body: dict) -> bool:
    """Whether an OpenAI stream requested a final usage-only chunk."""
    options = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True


def _migration_metadata(kv_transfer: dict) -> tuple[int, int | None]:
    seq_id = int(kv_transfer.get("seq_id") or 0)
    if seq_id <= 0:
        raise ValueError("missing positive seq_id")
    first_token = kv_transfer.get("first_token")
    return seq_id, None if first_token is None else int(first_token)


@dataclass
class _Request:
    """In-flight request state shared between the HTTP and engine threads."""

    seq_id: int
    aqueue: "asyncio.Queue[Optional[dict]]"
    loop: asyncio.AbstractEventLoop
    prompt_ids: Optional[list[int]] = None
    sampling_params: Any = None
    affinity_key: int = 0
    migration_payload: Optional[str] = None
    emitted: int = 0


@dataclass
class _Generation:
    """Result of consuming a request's token stream.

    ``prefix_offset`` / ``read_offset`` track the incremental detokenizer window
    (vLLM-style): instead of re-decoding the whole token list on every step
    (O(n^2) and event-loop blocking), only a small trailing slice is decoded to
    compute each delta. See :meth:`OpenAIServer._incremental_detokenize`.
    """

    token_ids: list = field(default_factory=list)
    finish_reason: str = "stop"
    prefix_offset: int = 0
    read_offset: int = 0
    tool_calls: list = field(default_factory=list)
    reasoning: Optional[str] = None
    # True while the current streamed delta belongs to the model's reasoning
    # ("thinking") region rather than the user-visible answer. Consumers route
    # the delta to ``reasoning_content`` instead of ``content`` accordingly.
    in_reasoning: bool = False


def _earliest_stop(text: str, stops: list[str]) -> Optional[int]:
    """Return the index of the earliest stop-string occurrence, or ``None``.

    Matches OpenAI/SGLang semantics: generation is cut just before the first
    stop string and the stop string itself is excluded from the output.
    """
    best: Optional[int] = None
    for s in stops:
        idx = text.find(s)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


def _partial_stop_holdback(text: str, stops: list[str]) -> int:
    """Length of the trailing run of ``text`` that may be a partial stop match.

    A stop string can be split across decode/delta boundaries (e.g. ``"Quest"``
    then ``"ion:"`` for stop ``"Question:"``). To avoid emitting a prefix of a
    stop string that later completes, we withhold the longest suffix of ``text``
    that equals a proper prefix of some stop string. The held bytes are emitted
    later, either after the match completes (and is cut) or once the stream ends.
    """
    max_hold = 0
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:k]):
                max_hold = max(max_hold, k)
                break
    return max_hold


def _skip_redundant_think_openers(text: str, start: int) -> tuple[int, bool]:
    """Skip generated ``<think>`` tags when the prompt already opened thinking.

    Returns the next unread offset and whether more text is needed to decide if
    the remaining suffix is another (possibly partial) opening tag.
    """
    marker = "<think>"
    pos = start
    while text.startswith(marker, pos):
        pos += len(marker)
    suffix = text[pos:]
    return pos, not suffix or marker.startswith(suffix)


class OpenAIServer:
    """Holds the engine worker, tokenizer and serving metadata."""

    def __init__(
        self,
        worker: Any,
        tokenizer: Any,
        served_model_name: str,
        model_path: str,
        default_max_tokens: int = 512,
        max_model_len: int = 16384,
        tool_call_parser: Optional[str] = None,
        reasoning_parser: Optional[str] = None,
    ) -> None:
        self.worker = worker
        self.tokenizer = tokenizer
        self.served_model_name = served_model_name
        self.model_path = model_path.rstrip("/")
        self.default_max_tokens = default_max_tokens
        self.max_model_len = int(max_model_len)
        self._model_aliases = self._build_model_aliases()
        from dlengine.server.tool_parser import detect_parser_name, get_tool_parser

        detected_parser = detect_parser_name(
            model_path,
            served_model_name,
            getattr(tokenizer, "chat_template", None),
        )
        self.tool_parser_name = tool_call_parser or detected_parser
        self.reasoning_parser_name = reasoning_parser or self.tool_parser_name
        self.tool_parser = get_tool_parser(self.tool_parser_name)
        logger.info(f"Tool-call parser: {self.tool_parser_name}")
        # NOTE: request/metric dumping (``--dump_requests_redis``) is performed
        # engine-side (see ``LLMEngine`` / ``dlengine.metrics.dump``), so it works
        # for offline ``generate()`` too and avoids double-writing here.

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

    @staticmethod
    def _normalize_messages(messages: list[dict]) -> list[dict]:
        """Reshape OpenAI/Anthropic-proxy messages into what chat templates expect.

        Two incompatibilities break Jinja chat templates (e.g. Qwen3.5):
          1. ``content`` arrives as a list of content parts
             (``[{"type": "text", "text": ...}]``) instead of a plain string.
          2. assistant ``tool_calls[].function.arguments`` arrives as a JSON
             string (per the OpenAI spec), but templates iterate it with the
             ``|items`` filter, which requires a mapping.
        """

        def flatten_content(content: Any) -> Any:
            if not isinstance(content, list):
                return content
            texts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            return "".join(texts)

        normalized: list[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                normalized.append(message)
                continue
            new_message = dict(message)

            if "content" in new_message:
                new_message["content"] = flatten_content(new_message["content"])

            tool_calls = new_message.get("tool_calls")
            if isinstance(tool_calls, list):
                new_tool_calls = []
                for call in tool_calls:
                    if not isinstance(call, dict):
                        new_tool_calls.append(call)
                        continue
                    call = dict(call)
                    fn = call.get("function")
                    if isinstance(fn, dict):
                        fn = dict(fn)
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args) if args.strip() else {}
                            except (ValueError, TypeError):
                                parsed = {}
                            fn["arguments"] = parsed if isinstance(parsed, dict) else {}
                        elif args is None:
                            fn["arguments"] = {}
                        call["function"] = fn
                    new_tool_calls.append(call)
                new_message["tool_calls"] = new_tool_calls

            normalized.append(new_message)
        return normalized

    @staticmethod
    def _prompt_opens_thinking(prompt: str) -> bool:
        """True if the rendered prompt leaves a ``<think>`` block open.

        Thinking templates (e.g. Qwen3.5) append a bare ``<think>`` to the
        generation prompt, so the model's output starts inside the reasoning
        region and emits only the closing ``</think>``. When thinking is
        disabled the template instead appends a self-closed ``<think></think>``,
        which this correctly reports as not open.
        """
        pairs = (
            ("<think>", "</think>"),
            ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
        )
        for opener, closer in pairs:
            open_idx = prompt.rfind(opener)
            if open_idx >= 0 and prompt.rfind(closer) < open_idx:
                return True
        return False

    def _encode_chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
    ) -> tuple[list[int], bool]:
        """Return ``(prompt_token_ids, reasoning_open)``.

        ``reasoning_open`` indicates the chat template started a ``<think>``
        block the model is expected to close, so the streaming layer can route
        the leading reasoning to ``reasoning_content`` instead of ``content``.
        """
        tok = self.tokenizer
        messages = self._normalize_messages(messages)
        prompt = None
        try:
            template_kwargs: dict[str, Any] = {}
            if tools:
                # Qwen3/Hermes templates render tool schemas into a system
                # preamble when ``tools`` is provided. Unsupported templates
                # silently ignore the kwarg, so this is safe across models.
                template_kwargs["tools"] = tools
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except (ValueError, TypeError, NotImplementedError):
            pass
        if prompt is None:
            # Minimal fallback when the tokenizer ships no chat template.
            parts = [
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            ]
            prompt = "\n".join(parts) + "\nassistant:"
        return tok.encode(prompt), self._prompt_opens_thinking(prompt)

    # -- request submission + streaming --

    def _build_sampling_params(self, body: dict) -> Any:
        from dlengine._rust.proto import SamplingParams

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

    @staticmethod
    def _parse_stop(body: dict) -> list[str]:
        """Extract OpenAI-style stop sequences from a request body.

        ``stop`` may be a single string or a list of strings (lm_eval sends its
        ``until`` list here, e.g. ``["Question:", "</s>", "<|im_end|>"]``). The
        engine has no native stop-string support, so generation is truncated in
        the serving layer (see :meth:`stream_text`).
        """
        raw = body.get("stop")
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [s for s in raw if isinstance(s, str) and s]

    # Headers a client/proxy can use to pin a conversation to a DP rank under
    # the Affinity routing strategy. Checked in order; first non-empty wins.
    _SESSION_HEADERS = ("x-session-id", "x-dlengine-session", "anthropic-session-id")

    @staticmethod
    def session_affinity_key(request: Any, body: dict) -> int:
        """Derive a 64-bit session affinity key for Affinity routing.

        Priority: an explicit session header, then the OpenAI ``user`` field,
        then the Anthropic ``metadata.user_id`` field. Returns 0 ("no explicit
        session") when none is present, in which case the scheduler falls back to
        content-derived cache-aware routing. The key only matters under
        ``--routing_strategy Affinity``; it is carried on the serialized
        Sequence so it reaches the engine process.
        """
        raw = None
        try:
            for h in OpenAIServer._SESSION_HEADERS:
                v = request.headers.get(h)
                if v:
                    raw = v
                    break
        except Exception:  # noqa: BLE001 - headers may be absent in some callers
            raw = None
        if not raw:
            user = body.get("user")
            if isinstance(user, str) and user:
                raw = user
        if not raw:
            meta = body.get("metadata")
            if isinstance(meta, dict):
                uid = meta.get("user_id")
                if isinstance(uid, str) and uid:
                    raw = uid
        if not raw:
            return 0
        digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest()
        key = int.from_bytes(digest, "big") & 0xFFFFFFFFFFFFFFFF
        return key or 1  # 0 is reserved for "no session"

    def submit(
        self, prompt_ids: list[int], sampling_params: Any, affinity_key: int = 0
    ) -> _Request:
        seq_id = next(_seq_id_counter)
        loop = asyncio.get_running_loop()
        req = _Request(
            seq_id=seq_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            affinity_key=affinity_key,
            aqueue=asyncio.Queue(),
            loop=loop,
        )
        self.worker.submit(req)
        logger.info(
            f"Submitted request to engine: seq_id={seq_id} "
            f"prompt_len={len(prompt_ids)} max_tokens={sampling_params.max_tokens}"
        )
        return req

    def validate_request_length(
        self, prompt_ids: list[int], sampling_params: Any
    ) -> Optional[JSONResponse]:
        prompt_len = len(prompt_ids)
        max_tokens = int(getattr(sampling_params, "max_tokens", 0) or 0)
        max_model_len = self.max_model_len
        total_len = prompt_len + max_tokens
        if prompt_len <= max_model_len and total_len <= max_model_len:
            return None

        message = (
            f"Requested token length exceeds max_model_len={max_model_len}: "
            f"prompt_tokens={prompt_len}, max_tokens={max_tokens}, "
            f"prompt_tokens+max_tokens={total_len}."
        )
        logger.warning("Rejected request (400): " + message)
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                }
            },
        )

    def submit_migrated(
        self, migration_payload: str, seq_id: int, first_token: int | None = None
    ) -> _Request:
        """Submit an opaque prefilled migration payload to a decode engine (PD).

        The engine process decodes the Rust-owned migration protocol into its
        scheduler compatibility state. The HTTP process deliberately does not
        materialize ``Sequence`` for migration.
        """
        loop = asyncio.get_running_loop()
        req = _Request(
            seq_id=seq_id,
            migration_payload=migration_payload,
            aqueue=asyncio.Queue(),
            loop=loop,
        )
        # The prefill engine has already sampled ``first_token`` and serialized
        # it into the migrated Sequence. Decode consumes that token as the
        # input to its first step, so it only reports tokens sampled *after*
        # it. Seed the serving queue here to emit the prefill token exactly
        # once, before any decode StepOut packets. Besides fixing streaming,
        # this also keeps non-streaming output and completion-token usage exact.
        if first_token is not None:
            req.aqueue.put_nowait({"tokens": [int(first_token)]})
        self.worker.submit(req)
        logger.info(f"Submitted migrated request to engine: seq_id={seq_id}")
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

    def _incremental_detokenize(self, gen: _Generation) -> str:
        """Decode only the newly produced text since the last delta.

        Decodes a trailing slice (``[prefix_offset:]``) rather than the whole
        token list, so per-token work is ~O(1) instead of O(n) (the old
        full-sequence decode made each stream O(n^2) and saturated the single
        FastAPI event loop, starving request admission). This is the vLLM
        incremental-detokenization scheme.

        A multi-byte UTF-8 character (e.g. an emoji) can be split across several
        byte-level BPE tokens; decoding before all bytes have arrived yields a
        trailing U+FFFD. In that case we return "" and hold the delta back until
        the character completes, so we never emit a broken "\ufffd".
        """
        ids = gen.token_ids
        prefix_text = self.tokenizer.decode(
            ids[gen.prefix_offset : gen.read_offset], skip_special_tokens=True
        )
        new_text = self.tokenizer.decode(
            ids[gen.prefix_offset :], skip_special_tokens=True
        )
        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            delta = new_text[len(prefix_text) :]
            gen.prefix_offset = gen.read_offset
            gen.read_offset = len(ids)
            return delta
        return ""

    async def stream_text(
        self,
        req: _Request,
        max_tokens: int,
        stop: Optional[list[str]] = None,
        hold_markers: Optional[list[str]] = None,
        reasoning_open: bool = False,
    ) -> AsyncGenerator[tuple[str, _Generation], None]:
        """Yield ``(delta_text, generation)`` as tokens arrive.

        When ``stop`` sequences are given, generation is truncated at the
        earliest stop string (matched against the cumulative decoded text, so a
        stop string split across token/delta boundaries is still caught). The
        stop string itself is not emitted, matching OpenAI/SGLang semantics.

        On a stop hit we ask the engine to abort the sequence (free its KV and
        stop generating) instead of letting it run out to ``max_tokens``, so we
        get SGLang-style early-stop throughput.

        ``hold_markers`` (e.g. ``<tool_call>`` / ``<think>``) are *soft* cuts: as
        soon as one appears, content emission freezes at the marker and the rest
        of the generation is buffered (not streamed as content). Generation is
        NOT aborted so the tool call completes; the caller parses the full text
        afterwards (see :meth:`OpenAIServer.tool_parser`). Trailing partial
        matches of either ``stop`` or ``hold_markers`` are withheld so a tag
        split across delta boundaries never leaks a fragment.
        """
        stops = stop or []
        markers = hold_markers or []
        gen = _Generation()
        text = ""  # cumulative decoded text (emitted + pending)
        emitted = 0  # number of chars already yielded
        finish_reason: Optional[str] = None
        holding_marker = False
        # Thinking models open ``<think>`` in the prompt; the generated text is
        # reasoning until the closing ``</think>``. While active we route the
        # text to a reasoning channel (gen.in_reasoning) instead of content.
        reasoning_active = reasoning_open
        think_close = self.tool_parser.reasoning_close_marker
        drop_redundant_think_openers = reasoning_open
        while True:
            item = await req.aqueue.get()
            if item is None:
                break
            if "error" in item:
                raise RuntimeError(item["error"])
            if "tokens" not in item:
                continue
            # A distributed step can deliver a small token bundle.  Do not let
            # the serving-side accumulator cross the request limit even if the
            # final engine packet contains more than the remaining allowance.
            remaining = max_tokens - len(gen.token_ids)
            if remaining <= 0:
                continue
            gen.token_ids.extend(item["tokens"][:remaining])
            delta = self._incremental_detokenize(gen)
            if not delta:
                continue
            text += delta
            if reasoning_active:
                if drop_redundant_think_openers:
                    emitted, need_more = _skip_redundant_think_openers(text, emitted)
                    if need_more:
                        continue
                    drop_redundant_think_openers = False
                close = text.find(think_close, emitted)
                if close == -1:
                    # No closing tag yet: stream reasoning, but hold back a
                    # trailing partial "</think>" so the tag never leaks.
                    holdback = _partial_stop_holdback(text, [think_close])
                    safe = len(text) - holdback
                    if safe > emitted:
                        gen.in_reasoning = True
                        chunk = text[emitted:safe]
                        gen.reasoning = (gen.reasoning or "") + chunk
                        yield chunk, gen
                        emitted = safe
                    continue
                # Closing tag found: flush remaining reasoning, drop the tag,
                # then fall through to normal content handling for the tail.
                if close > emitted:
                    gen.in_reasoning = True
                    chunk = text[emitted:close]
                    gen.reasoning = (gen.reasoning or "") + chunk
                    yield chunk, gen
                emitted = close + len(think_close)
                reasoning_active = False
                gen.in_reasoning = False
            if stops:
                idx = _earliest_stop(text, stops)
                if idx is not None:
                    if idx > emitted:
                        yield text[emitted:idx], gen
                        emitted = idx
                    finish_reason = "stop"
                    self._abort_request(req)
                    break
            # Freeze emission at the first tool/think marker; buffer the rest.
            if markers and not holding_marker:
                midx = _earliest_stop(text, markers)
                if midx is not None:
                    holding_marker = True
                    if midx > emitted:
                        yield text[emitted:midx], gen
                        emitted = midx
            if holding_marker:
                continue
            # Hold back a trailing partial stop/marker match until it completes
            # (and gets cut) or is proven not to be a real tag.
            holdback = _partial_stop_holdback(text, stops + markers)
            safe = len(text) - holdback
            if safe > emitted:
                yield text[emitted:safe], gen
                emitted = safe
        if finish_reason is None:
            # Stream ended (EOS / max_tokens): flush any held-back partial-stop
            # tail, since it never completed into a real stop string. Do NOT
            # flush a buffered tool/think region; the caller parses it instead.
            if reasoning_active and emitted < len(text):
                # Model produced only reasoning and never closed </think>:
                # emit the tail on the reasoning channel, not as content.
                gen.in_reasoning = True
                chunk = text[emitted:]
                gen.reasoning = (gen.reasoning or "") + chunk
                yield chunk, gen
            elif not holding_marker and not reasoning_active and emitted < len(text):
                gen.in_reasoning = False
                yield text[emitted:], gen
            finish_reason = "length" if len(gen.token_ids) >= max_tokens else "stop"
        gen.finish_reason = finish_reason

    def _abort_request(self, req: _Request) -> None:
        """Best-effort engine-side abort for a request that hit a stop string."""
        try:
            self.worker.abort(req.seq_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"abort failed for seq_id={req.seq_id}: {e}")

    def _spawn_disconnect_monitor(self, request: "Request", req: _Request):
        """Abort the engine sequence if the HTTP client disconnects mid-generation.

        Without this, killing the client (e.g. ``curl`` Ctrl+C) leaves the
        sequence running on the GPU until it hits EOS/``max_tokens``, wasting
        compute and KV. Non-streaming requests in particular are never cancelled
        by the ASGI server on disconnect, so we poll ``is_disconnected()`` here
        and trigger the existing abort path (engine frees KV and stops decoding).

        Returns an ``asyncio.Task`` the caller must cancel once generation
        finishes normally.
        """

        async def _monitor() -> None:
            try:
                while True:
                    if await request.is_disconnected():
                        logger.info(
                            f"client disconnected; aborting seq_id={req.seq_id}"
                        )
                        self._abort_request(req)
                        return
                    await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001 - monitor must never crash
                logger.debug(f"disconnect monitor error for seq_id={req.seq_id}: {e}")
                return

        return asyncio.ensure_future(_monitor())


# ----------------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------------


def _profiler_trace_files(result: dict[str, Any]) -> list[Path]:
    """Return the unique, finalized worker traces from a stop response."""
    paths = {
        Path(path).expanduser().resolve()
        for worker in result.get("workers", [])
        for path in worker.get("trace_files", [])
        if str(path).endswith(".pt.trace.json")
    }
    traces = sorted(paths)
    if not traces:
        raise ValueError("profiler stop returned no finalized trace files")
    missing = [path for path in traces if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "profiler traces are not visible to the HTTP server: "
            + ", ".join(str(path) for path in missing)
        )
    return traces


def _profiler_merged_trace_path(
    result: dict[str, Any],
    traces: list[Path],
    *,
    compact_streams: bool = False,
) -> Path:
    trace_dirs = {
        Path(worker["trace_dir"]).expanduser().resolve()
        for worker in result.get("workers", [])
        if worker.get("trace_dir")
    }
    if len(trace_dirs) > 1:
        raise ValueError("profiler workers reported different trace directories")
    trace_dir = trace_dirs.pop() if trace_dirs else traces[0].parent
    suffix = (
        "_compact_merged.trace.json.gz" if compact_streams else "_merged.trace.json.gz"
    )
    return trace_dir / f"{trace_dir.name}{suffix}"


def build_app(server: OpenAIServer):
    app = FastAPI(title="DLEngine OpenAI Server")

    @app.get("/health")
    async def health() -> PlainTextResponse:  # noqa: ANN202
        return PlainTextResponse("OK")

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:  # noqa: ANN202
        # Fetch metrics from the engine process via ZmqEngineWorker.
        if hasattr(server.worker, "get_metrics"):
            try:
                metrics_text = await server.worker.get_metrics()
                if metrics_text:
                    return PlainTextResponse(
                        metrics_text,
                        media_type="text/plain; version=0.0.4; charset=utf-8",
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch metrics from engine: {e}")

        # Fallback for in-process engine
        engine = getattr(server.worker, "engine", None)
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is None:
            return PlainTextResponse(
                "# HELP dlengine_up DLEngine metrics exporter health.\n"
                "# TYPE dlengine_up gauge\n"
                "dlengine_up 0\n",
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )
        return PlainTextResponse(
            scheduler.metrics_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/start_profiler")
    async def start_profiler(request: Request) -> JSONResponse:  # noqa: ANN202
        try:
            raw_body = await request.body()
            body = json.loads(raw_body) if raw_body else {}
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            trace_name = body.get("trace_name")
            if trace_name is not None and not isinstance(trace_name, str):
                raise ValueError("trace_name must be a string")
            result = await server.worker.start_profiler(trace_name)
            return JSONResponse(
                status_code=200 if result.get("ok") else 400,
                content=result,
            )
        except (ValueError, json.JSONDecodeError) as e:
            return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"ok": False, "error": "profiler start timed out"},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to start profiler")
            return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

    @app.post("/stop_profiler")
    async def stop_profiler(request: Request):  # noqa: ANN202
        try:
            raw_body = await request.body()
            body = json.loads(raw_body) if raw_body else {}
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            merge = body.get("merge", False)
            if type(merge) is not bool:
                raise ValueError("merge must be a boolean")
            compact_streams = body.get("compact_streams", False)
            if type(compact_streams) is not bool:
                raise ValueError("compact_streams must be a boolean")
            if compact_streams and not merge:
                raise ValueError("compact_streams requires merge=true")

            result = await server.worker.stop_profiler()
            if not result.get("ok") or not merge:
                return JSONResponse(
                    status_code=200 if result.get("ok") else 500,
                    content=result,
                )

            traces = _profiler_trace_files(result)
            merged_trace = _profiler_merged_trace_path(
                result, traces, compact_streams=compact_streams
            )
            await asyncio.to_thread(
                merge_trace_jsons_to_gzip,
                traces,
                merged_trace,
                compact_streams=compact_streams,
            )
            return FileResponse(
                merged_trace,
                media_type="application/gzip",
                filename=merged_trace.name,
                headers={
                    "X-DLEngine-Profiler-Trace-Count": str(len(traces)),
                    "X-DLEngine-Profiler-Compact-Streams": str(compact_streams).lower(),
                },
            )
        except (ValueError, json.JSONDecodeError) as e:
            return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"ok": False, "error": "profiler stop timed out"},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to stop profiler")
            return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

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
        client = request.client.host if request.client else "unknown"
        logger.info(f"Received request: POST /v1/chat/completions from {client}")
        try:
            body = await request.json()
        except Exception as e:
            logger.warning(f"Rejected request (400): invalid JSON body: {e}")
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
            logger.warning("Rejected request (400): JSON body must be an object")
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
            logger.warning(
                f"Rejected request (404): model {body.get('model')!r} not found"
            )
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
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        use_tools = bool(tools) and tool_choice != "none"
        sampling_params = server._build_sampling_params(body)
        max_tokens = sampling_params.max_tokens
        stop = server._parse_stop(body)
        prompt_ids, reasoning_open = server._encode_chat(
            messages, tools=tools, tool_choice=tool_choice
        )
        length_error = server.validate_request_length(prompt_ids, sampling_params)
        if length_error is not None:
            return length_error
        affinity_key = server.session_affinity_key(request, body)

        created = int(time.time())
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = server.served_model_name
        stream = bool(body.get("stream", False))
        include_usage = _stream_includes_usage(body)

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
            preq = server.submit(prompt_ids, sampling_params, affinity_key=affinity_key)
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
            try:
                migrated_seq_id, first_token = _migration_metadata(kv_transfer)
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
            req = server.submit_migrated(
                kv_transfer["migration"], migrated_seq_id, first_token
            )
        else:
            req = server.submit(prompt_ids, sampling_params, affinity_key=affinity_key)

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
                finished_normally = False
                monitor = server._spawn_disconnect_monitor(request, req)
                hold_markers = (
                    list(server.tool_parser.open_markers) if use_tools else None
                )
                streamed_reasoning = False
                try:
                    async for delta, gen in server.stream_text(
                        req,
                        max_tokens,
                        stop=stop,
                        hold_markers=hold_markers,
                        reasoning_open=reasoning_open,
                    ):
                        # Reasoning ("thinking") tokens go to reasoning_content;
                        # everything else is the user-visible answer.
                        if gen.in_reasoning:
                            streamed_reasoning = True
                            delta_obj = {"reasoning_content": delta}
                        else:
                            delta_obj = {"content": delta}
                        chunk = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta_obj,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                    final_finish_reason = gen.finish_reason
                    if use_tools:
                        full_text = server.tokenizer.decode(
                            gen.token_ids, skip_special_tokens=True
                        )
                        parsed = server.tool_parser.parse_full(full_text)
                        tool_delta: dict[str, Any] = {}
                        # Only attach reasoning here if it was not already
                        # streamed live (avoids duplicating the think region).
                        if parsed.reasoning is not None and not streamed_reasoning:
                            tool_delta["reasoning_content"] = parsed.reasoning
                        if parsed.tool_calls:
                            tool_delta["tool_calls"] = [
                                {"index": i, **tc.to_dict()}
                                for i, tc in enumerate(parsed.tool_calls)
                            ]
                            final_finish_reason = "tool_calls"
                        if tool_delta:
                            tool_chunk = {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": tool_delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(tool_chunk)}\n\n".encode()
                    final = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": final_finish_reason,
                            }
                        ],
                    }
                    finished_normally = True
                    yield f"data: {json.dumps(final)}\n\n".encode()
                    if include_usage:
                        usage = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": len(prompt_ids),
                                "completion_tokens": len(gen.token_ids),
                                "total_tokens": len(prompt_ids) + len(gen.token_ids),
                            },
                        }
                        yield f"data: {json.dumps(usage)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                except RuntimeError as e:
                    err = {"error": {"message": str(e), "type": "engine_error"}}
                    finished_normally = True
                    yield f"data: {json.dumps(err)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                finally:
                    monitor.cancel()
                    # Client dropped mid-stream (GeneratorExit before the loop
                    # finished): make sure the engine stops generating.
                    if not finished_normally:
                        server._abort_request(req)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Non-streaming: drain the whole generation.
        text = ""
        reasoning_text = ""
        gen = _Generation()
        monitor = server._spawn_disconnect_monitor(request, req)
        try:
            async for delta, gen in server.stream_text(
                req, max_tokens, stop=stop, reasoning_open=reasoning_open
            ):
                if gen.in_reasoning:
                    reasoning_text += delta
                else:
                    text += delta
        except RuntimeError as e:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "engine_error"}},
            )
        finally:
            monitor.cancel()
        # Always split off the reasoning region and any tool-call markup so the
        # think text never leaks into ``content`` (the template opens <think>
        # in the prompt, so the answer is preceded by reasoning + </think>).
        parsed = server.tool_parser.parse_full(text)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": parsed.content if parsed.content is not None else "",
        }
        reasoning = reasoning_text or parsed.reasoning
        if reasoning:
            message["reasoning_content"] = reasoning
        finish_reason = gen.finish_reason
        if use_tools and parsed.tool_calls:
            message["tool_calls"] = [tc.to_dict() for tc in parsed.tool_calls]
            finish_reason = "tool_calls"
        return JSONResponse(
            {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(gen.token_ids),
                    "total_tokens": len(prompt_ids) + len(gen.token_ids),
                },
            }
        )

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):  # noqa: ANN202
        from dlengine.server.anthropic_api import handle_messages

        client = request.client.host if request.client else "unknown"
        logger.info(f"Received request: POST /v1/messages from {client}")
        try:
            body = await request.json()
        except Exception as e:
            logger.warning(f"Rejected request (400): invalid JSON body: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid JSON body: {e}",
                    },
                },
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "JSON body must be an object",
                    },
                },
            )
        return await handle_messages(server, request, body)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):  # noqa: ANN202
        from dlengine.server.anthropic_api import handle_count_tokens

        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid JSON body: {e}",
                    },
                },
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "JSON body must be an object",
                    },
                },
            )
        return await handle_count_tokens(server, body)

    @app.post("/v1/completions")
    async def completions(request: Request):  # noqa: ANN202
        client = request.client.host if request.client else "unknown"
        logger.info(f"Received request: POST /v1/completions from {client}")
        try:
            body = await request.json()
        except Exception as e:
            logger.warning(f"Rejected request (400): invalid JSON body: {e}")
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
            logger.warning("Rejected request (400): JSON body must be an object")
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
            logger.warning(
                f"Rejected request (404): model {body.get('model')!r} not found"
            )
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
        sampling_params = server._build_sampling_params(body)
        max_tokens = sampling_params.max_tokens
        stop = server._parse_stop(body)
        input_ids = body.get("input_ids")
        if input_ids is not None:
            if not _is_token_id_list(input_ids):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "input_ids must be a list of integer token ids",
                            "type": "invalid_request_error",
                        }
                    },
                )
            prompt_ids = list(input_ids)
        else:
            prompt = body.get("prompt", "")
            if _is_token_id_list(prompt):
                prompt_ids = list(prompt)
            elif isinstance(prompt, list):
                # Batched completions are not implemented here; preserve the
                # previous behavior of serving the first prompt.
                prompt = prompt[0] if prompt else ""
                prompt_ids = server.tokenizer.encode(prompt)
            else:
                prompt_ids = server.tokenizer.encode(prompt)
        length_error = server.validate_request_length(prompt_ids, sampling_params)
        if length_error is not None:
            return length_error
        affinity_key = server.session_affinity_key(request, body)

        created = int(time.time())
        cmpl_id = f"cmpl-{uuid.uuid4().hex}"
        model = server.served_model_name
        stream = bool(body.get("stream", False))
        include_usage = _stream_includes_usage(body)

        kv_transfer = body.get("kv_transfer_params") or {}

        # PD prefill stage: run prefill and return a migration payload. Do NOT
        # clamp max_tokens to 1 -- the mode="prefill" engine already stops after
        # the first token via TO_BE_MIGRATED, and the user's max_tokens must be
        # preserved into the migrated sequence for the decode engine. See the
        # chat handler above for the full rationale.
        if kv_transfer.get("do_remote_decode"):
            preq = server.submit(prompt_ids, sampling_params, affinity_key=affinity_key)
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
            try:
                migrated_seq_id, first_token = _migration_metadata(kv_transfer)
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
            req = server.submit_migrated(
                kv_transfer["migration"], migrated_seq_id, first_token
            )
        else:
            req = server.submit(prompt_ids, sampling_params, affinity_key=affinity_key)

        if stream:

            async def event_stream() -> AsyncGenerator[bytes, None]:
                import json

                gen = _Generation()
                finished_normally = False
                monitor = server._spawn_disconnect_monitor(request, req)
                try:
                    async for delta, gen in server.stream_text(
                        req, max_tokens, stop=stop
                    ):
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
                    final = {
                        "id": cmpl_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "text": "", "finish_reason": gen.finish_reason}
                        ],
                    }
                    finished_normally = True
                    yield f"data: {json.dumps(final)}\n\n".encode()
                    if include_usage:
                        usage = {
                            "id": cmpl_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": len(prompt_ids),
                                "completion_tokens": len(gen.token_ids),
                                "total_tokens": len(prompt_ids) + len(gen.token_ids),
                            },
                        }
                        yield f"data: {json.dumps(usage)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                except RuntimeError as e:
                    err = {"error": {"message": str(e), "type": "engine_error"}}
                    finished_normally = True
                    yield f"data: {json.dumps(err)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                finally:
                    monitor.cancel()
                    if not finished_normally:
                        server._abort_request(req)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        text = ""
        gen = _Generation()
        monitor = server._spawn_disconnect_monitor(request, req)
        try:
            async for delta, gen in server.stream_text(req, max_tokens, stop=stop):
                text += delta
        except RuntimeError as e:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "engine_error"}},
            )
        finally:
            monitor.cancel()
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
DEFAULT_CTRL_ADDRESS = "http://127.0.0.1:4479"


def _advertise_host(host: str) -> str:
    from dlengine.utils.network import get_advertise_host

    return get_advertise_host(host)


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
        return None
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
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from dlengine.utils.network import get_bind_host

    host = get_bind_host(config.host)
    # Bind before starting the engine so port=0 is resolved exactly once and
    # the already-reserved socket can be handed to Uvicorn without a
    # probe-then-bind race.
    uvicorn_config = uvicorn.Config(app=None, host=host, port=config.port)
    http_socket = uvicorn_config.bind_socket()
    port = int(http_socket.getsockname()[1])
    config.port = port
    advertise_host = _advertise_host(host)
    discovery_ctrl_address = (
        ctrl_address or os.environ.get("DLSLIME_CTRL_ADDRESS") or DEFAULT_CTRL_ADDRESS
    )
    discovery_ctrl_scope = ctrl_scope or os.environ.get("DLSLIME_CTRL_SCOPE")
    ctrl_is_implicit = ctrl_address is None and not os.environ.get(
        "DLSLIME_CTRL_ADDRESS"
    )

    logger.info("=" * 72)
    logger.info("DLEngine OpenAI Server")
    logger.info(f"  model:             {config.model}")
    logger.info(f"  served-model-name: {served_model_name}")
    logger.info(f"  http bind:         {host}:{port}")
    logger.info(f"  http advertise:    {advertise_host}:{port}")
    logger.info(f"  mode:              {config.mode}")
    logger.info(
        f"  ctrl-address:      {discovery_ctrl_address}"
        f"{' (auto)' if ctrl_is_implicit else ''}"
    )
    logger.info(
        f"  monitor:           {'enabled' if config.enable_monitor else 'disabled'}"
    )
    logger.info("=" * 72)

    # Build the engine + worker.
    #
    # The engine always runs in a separate process (EngineServer) exposing a zmq
    # DEALER over an ipc:// socket; this HTTP server talks to it via
    # ZmqEngineWorker and never initializes CUDA/Ray itself.
    import multiprocessing

    from dlengine.server.engine_server import run_engine_server
    from dlengine.server.zmq_engine_client import ZmqEngineWorker

    engine_endpoint = f"ipc:///tmp/dlengine-{uuid.uuid4().hex}.sock"
    # NOT daemon: EngineServer.serve() itself spawns a child backend process
    # (run_engine_backend), and daemonic processes cannot have children. Cleaned
    # up explicitly in the shutdown hook below.
    engine_proc = multiprocessing.Process(
        target=run_engine_server,
        args=(config, engine_endpoint),
        daemon=False,
        name="dlengine-engine",
    )
    engine_proc.start()
    logger.info(f"Started engine process (pid={engine_proc.pid}) at {engine_endpoint}")
    worker: Any = ZmqEngineWorker(engine_endpoint)

    model_type = getattr(config.hf_config, "model_type", "")
    if model_type == "kimi_k3":
        # K3's remote TikToken class overrides apply_chat_template without a
        # ``chat_template`` attribute. Forcing PreTrainedTokenizerFast drops
        # the thinking-effort/system preamble required by the model.
        tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True, fix_mistral_regex=True
        )
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            config.model, fix_mistral_regex=True
        )

    server = OpenAIServer(
        worker=worker,
        tokenizer=tokenizer,
        served_model_name=served_model_name,
        model_path=config.model,
        max_model_len=config.max_model_len,
        tool_call_parser=config.tool_call_parser,
        reasoning_parser=config.reasoning_parser,
    )
    app = build_app(server)

    ctrl_client = None

    @app.on_event("startup")
    async def _on_startup() -> None:  # noqa: ANN202
        nonlocal ctrl_client
        # Register signal handlers on the *running* loop so Ctrl+C / SIGTERM are
        # dispatched promptly even under uvloop and even while we are parked in
        # the engine-readiness ``await`` below (plain signal.signal handlers are
        # unreliable there). _signal_cleanup is defined later in run_server but
        # resolved via closure at call time.
        try:
            loop = asyncio.get_running_loop()
            for _sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(_sig, _signal_cleanup, _sig)
        except (NotImplementedError, RuntimeError):
            pass
        # Connect to the engine process and block until it is ready (also
        # resolves the engine_id needed for PD peer registration).
        #
        # Race readiness against engine-process liveness: if the engine
        # subprocess dies during startup (no GPUs / Ray PG unavailable /
        # CUDA OOM), the readiness handshake would otherwise hang forever.
        # Surface it as a startup failure so the server cleans up and exits
        # instead of wedging with an unkillable HTTP loop.
        start_task = asyncio.ensure_future(worker.start())
        while not start_task.done():
            if engine_proc is not None and not engine_proc.is_alive():
                start_task.cancel()
                raise RuntimeError(
                    "Engine process exited during startup "
                    f"(exitcode={engine_proc.exitcode}); aborting server start"
                )
            done, _ = await asyncio.wait({start_task}, timeout=0.5)
        await start_task  # propagate any exception raised by worker.start()
        engine_id = worker.engine_id
        if config.enable_monitor:
            try:
                import shutil

                from dlengine.monitor import (
                    create_monitor_stack,
                    default_dlengine_target,
                    GRAFANA_PORT,
                    PROMETHEUS_PORT,
                    start_monitor_stack,
                )

                docker_available = shutil.which("docker") is not None
                target = (
                    f"host.docker.internal:{port}"
                    if docker_available
                    else default_dlengine_target(port)
                )
                monitor_root = create_monitor_stack(
                    config.monitor_dir,
                    dlengine_target=target,
                    scrape_interval=config.monitor_scrape_interval,
                )
                if docker_available:
                    start_monitor_stack(
                        config.monitor_dir,
                        dlengine_target=target,
                        scrape_interval=config.monitor_scrape_interval,
                    )
                    logger.info(
                        "Started DLEngine monitor stack at %s "
                        "(Prometheus http://localhost:%s, Grafana http://localhost:%s, "
                        "target %s)",
                        monitor_root,
                        PROMETHEUS_PORT,
                        GRAFANA_PORT,
                        target,
                    )
                else:
                    logger.info(
                        "DLEngine monitor config written to %s; docker CLI not "
                        "found, so Prometheus/Grafana were not started and no "
                        "running Prometheus config was changed. Configure your "
                        "Prometheus to scrape %s. Metrics endpoint from inside "
                        "this container: http://localhost:%s/metrics",
                        monitor_root,
                        target,
                        port,
                    )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Could not start monitor stack. /metrics is still available "
                    "for Prometheus scrape: %s",
                    e,
                )
        try:
            ctrl_client = register_with_ctrl(
                ctrl_address=discovery_ctrl_address,
                ctrl_scope=discovery_ctrl_scope,
                host=host,
                port=port,
                served_model_name=served_model_name,
                model_path=config.model,
                role=config.mode,
                engine_id=engine_id,
            )
        except Exception as e:  # noqa: BLE001
            if ctrl_is_implicit:
                logger.warning(
                    "Automatic dlslime-ctrl discovery at %s is unavailable: %s",
                    discovery_ctrl_address,
                    e,
                )
            else:
                logger.error(f"Could not register with dlslime-ctrl: {e}")

        logger.info("=" * 72)
        logger.info("DLEngine READY: http://%s:%s", advertise_host, port)
        if ctrl_client is not None:
            logger.info(
                "Router discovery: registered with %s (scope=%s)",
                discovery_ctrl_address,
                discovery_ctrl_scope or "default",
            )
        else:
            logger.warning(
                "Router discovery: NOT REGISTERED. Direct endpoint remains "
                "available at http://%s:%s; start dlslime-ctrl at %s or pass "
                "--ctrl_address explicitly.",
                advertise_host,
                port,
                discovery_ctrl_address,
            )
        logger.info("=" * 72)

    _cleanup_done = {"v": False}

    def _cleanup() -> None:
        """Tear down the ctrl registration, zmq worker, and engine subprocess.

        Idempotent and safe to call from a signal handler, the FastAPI shutdown
        hook, or the post-serve ``finally``. Crucially this runs even when the
        ASGI lifespan never finished startup (e.g. Ctrl+C during a slow model
        load), which the old shutdown-event-only path did not — that left the
        engine process and its Ray actors orphaned.
        """
        if _cleanup_done["v"]:
            return
        _cleanup_done["v"] = True
        try:
            if ctrl_client is not None:
                ctrl_client.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            worker.stop()
        except Exception:  # noqa: BLE001
            pass
        if engine_proc is not None:
            try:
                engine_proc.terminate()
                engine_proc.join(timeout=5)
                if engine_proc.is_alive():
                    engine_proc.kill()
                    engine_proc.join(timeout=3)
            except Exception:  # noqa: BLE001
                pass
        if engine_endpoint and engine_endpoint.startswith("ipc://"):
            sock_path = engine_endpoint[len("ipc://") :]
            try:
                os.unlink(sock_path)
            except OSError:
                pass
        try:
            http_socket.close()
        except OSError:
            pass

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:  # noqa: ANN202
        _cleanup()

    # Own the signal handlers instead of uvicorn: uvicorn does not cancel the
    # lifespan-startup task on SIGINT, so Ctrl+C during a slow startup would
    # hang the process (and never clean up the engine subprocess). Our handler
    # stays installed throughout startup, tears everything down, and hard-exits.
    _signal_state = {"in_progress": False}

    def _signal_cleanup(signum, _frame=None) -> None:
        # A second signal while we're already tearing down means "I don't care
        # about graceful, kill it now" — hard-exit immediately.
        if _signal_state["in_progress"]:
            os._exit(1)
        _signal_state["in_progress"] = True
        logger.info(f"Received signal {signum}; shutting down engine and exiting")
        _cleanup()
        os._exit(0)

    uvicorn_config.app = app
    server_uv = uvicorn.Server(uvicorn_config)
    server_uv.install_signal_handlers = lambda: None
    # Plain handlers cover the window before the event loop is running. Once the
    # loop is up we ALSO register via loop.add_signal_handler in the startup
    # hook below: uvicorn runs on uvloop, which does not reliably dispatch plain
    # signal.signal handlers while the main thread is parked in an ``await``
    # (e.g. the engine readiness wait), so Ctrl+C would otherwise be ignored.
    signal.signal(signal.SIGINT, _signal_cleanup)
    signal.signal(signal.SIGTERM, _signal_cleanup)
    try:
        server_uv.run(sockets=[http_socket])
    finally:
        _cleanup()
