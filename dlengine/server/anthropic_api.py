"""Native Anthropic Messages API (``POST /v1/messages``) for DLEngine.

Lets Claude Code (and other Anthropic-SDK clients) point ``ANTHROPIC_BASE_URL``
straight at DLEngine with no translation proxy. Because we own the request,
HTTP headers the Anthropic SDK adds (e.g. ``x-anthropic-billing-header`` with a
volatile ``cch=<random>`` token) stay HTTP headers and never get folded into the
prompt -- which is exactly the prefix-cache-buster the request dump surfaced.

This module is pure request/response shaping. It converts the Anthropic schema
into the OpenAI-style messages/tools the existing chat template + tool parser
already understand (see ``OpenAIServer._encode_chat`` /
``OpenAIServer.tool_parser``), then reuses ``OpenAIServer.stream_text`` for
generation. Only ``temperature`` / ``max_tokens`` reach the sampler (DLEngine's
sampler is temperature-only); ``top_p`` / ``top_k`` are accepted but ignored and
``stop_sequences`` are enforced in the serving layer.
"""

from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Optional

# Some clients/proxies fold the Anthropic ``x-anthropic-billing-header`` HTTP
# header into the system prompt text. It carries a volatile ``cch=<random>``
# token that changes every request, so it shifts the entire prompt and destroys
# prefix-cache reuse. Strip the whole ``key=value;`` header run wherever it
# appears at the start of a system block.
_BILLING_HEADER_RE = re.compile(
    r"x-anthropic-billing-header:\s*(?:[\w.\-]+=[^;]*;\s*)+",
    re.IGNORECASE,
)


def _strip_volatile_headers(text: str) -> str:
    return _BILLING_HEADER_RE.sub("", text)


class AnthropicError(Exception):
    """A 4xx that should be returned in the Anthropic error envelope."""

    def __init__(
        self, status_code: int, message: str, err_type: str = "invalid_request_error"
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type

    def to_response(self):  # noqa: ANN201
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=self.status_code,
            content={
                "type": "error",
                "error": {"type": self.err_type, "message": self.message},
            },
        )


def _text_from_blocks(content: Any) -> str:
    """Join the text of a string or a list of Anthropic content blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def _convert_tools(tools: Any) -> Optional[list[dict]]:
    """Anthropic tools -> OpenAI function-tool schema the template renders."""
    if not isinstance(tools, list) or not tools:
        return None
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return out or None


def anthropic_to_openai_messages(
    body: dict,
) -> tuple[list[dict], Optional[list[dict]], Any]:
    """Convert an Anthropic Messages request body to ``(messages, tools, tool_choice)``.

    The returned ``messages`` are in the OpenAI shape consumed by
    ``OpenAIServer._encode_chat`` (which also runs ``_normalize_messages``):
    a leading ``system`` message, ``user`` / ``assistant`` turns, assistant
    ``tool_calls`` (from ``tool_use`` blocks) and ``tool`` messages (from
    ``tool_result`` blocks).
    """
    messages: list[dict] = []

    # Collect ALL system text into a single leading system message. Strict Qwen
    # chat templates strip messages[0] when it is a system message and then
    # raise "System message must be at the beginning." on any later system role,
    # so we must never emit more than one system message and it must be first.
    system_texts: list[str] = []
    top_system = body.get("system")
    if top_system is not None:
        t = _text_from_blocks(top_system)
        if t:
            system_texts.append(t)

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise AnthropicError(400, "'messages' must be a list")

    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")

        # Defensive: fold any stray system-role turn into the leading system.
        if role == "system":
            t = _text_from_blocks(content)
            if t:
                system_texts.append(t)
            continue

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": ""})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[tuple[Any, str]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                t = block.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
            elif btype == "tool_use":  # assistant requested a tool
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif btype == "tool_result":  # user returning a tool's output
                tool_results.append(
                    (block.get("tool_use_id"), _text_from_blocks(block.get("content")))
                )
            elif btype in ("thinking", "redacted_thinking"):
                # Never feed prior-turn reasoning back into the prompt: it is
                # long and unique per turn, so it would only bloat the history
                # and break prefix-cache reuse. Drop it intentionally.
                continue
            elif btype == "image":
                raise AnthropicError(
                    400,
                    "image content blocks are not supported by this text-only model",
                )

        if role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
        else:
            # tool_result blocks become standalone tool-role messages that the
            # chat template renders before any accompanying user text.
            for tool_use_id, result_text in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": result_text,
                    }
                )
            joined = "".join(text_parts)
            if joined or not tool_results:
                messages.append({"role": "user", "content": joined})

    if system_texts:
        system_content = _strip_volatile_headers("\n\n".join(system_texts))
        messages.insert(0, {"role": "system", "content": system_content})

    tools = _convert_tools(body.get("tools"))
    return messages, tools, body.get("tool_choice")


def uses_tools(tools: Optional[list[dict]], tool_choice: Any) -> bool:
    if not tools:
        return False
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "none":
        return False
    return True


def build_sampling_params(body: dict) -> Any:
    """Anthropic params -> DLEngine SamplingParams (temperature + max_tokens)."""
    from dlengine._rust.proto import SamplingParams

    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise AnthropicError(
            400, "'max_tokens' is required and must be a positive integer"
        )
    temperature = body.get("temperature")
    if temperature is None:
        temperature = 1.0
    return SamplingParams(
        temperature=float(temperature), max_tokens=int(max_tokens), ignore_eos=False
    )


def _migration_metadata(kv_transfer: dict) -> tuple[int, int | None]:
    seq_id = int(kv_transfer.get("seq_id") or 0)
    if seq_id <= 0:
        raise AnthropicError(400, "missing positive seq_id")
    first_token = kv_transfer.get("first_token")
    return seq_id, None if first_token is None else int(first_token)


def parse_stop_sequences(body: dict) -> list[str]:
    raw = body.get("stop_sequences")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str) and s]
    return []


def _tool_input(tc: Any) -> dict:
    try:
        args = tc.function.arguments
        return json.loads(args) if args else {}
    except (ValueError, TypeError):
        return {}


def format_messages_response(
    server: Any,
    *,
    model: str,
    text: str,
    gen: Any,
    use_tools: bool,
    tools: Optional[list[dict]],
    input_tokens: int,
) -> dict:
    """Build the non-streaming Anthropic ``message`` response body."""
    parsed = server.tool_parser.parse_full(text, tools=tools if use_tools else None)
    content_blocks: list[dict] = []
    if parsed.content:
        content_blocks.append({"type": "text", "text": parsed.content})

    stop_reason = "end_turn"
    if gen is not None and gen.finish_reason == "length":
        stop_reason = "max_tokens"

    tool_calls = parsed.tool_calls if use_tools else []
    for tc in tool_calls:
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": _tool_input(tc),
            }
        )
    if tool_calls:
        stop_reason = "tool_use"

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    output_tokens = len(gen.token_ids) if gen is not None else 0
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def event_stream(
    server: Any,
    request: Any,
    req: Any,
    *,
    model: str,
    max_tokens: int,
    stop: list[str],
    reasoning_open: bool,
    use_tools: bool,
    tools: Optional[list[dict]],
    input_tokens: int,
) -> AsyncGenerator[bytes, None]:
    """Emit the Anthropic streaming SSE event sequence for one request.

    Order: ``message_start`` -> a ``text`` content block (start / ``text_delta``
    per chunk / stop) -> one ``tool_use`` block per parsed tool call (start /
    single ``input_json_delta`` / stop) -> ``message_delta`` (stop_reason +
    output token count) -> ``message_stop``. The model's ``<think>`` reasoning is
    dropped (not surfaced as Anthropic thinking blocks).
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )

    index = -1
    text_block_open = False
    finished_normally = False
    gen = None
    monitor = server._spawn_disconnect_monitor(request, req)
    hold_markers = list(server.tool_parser.open_markers) if use_tools else None

    try:
        async for delta, gen in server.stream_text(
            req,
            max_tokens,
            stop=stop,
            hold_markers=hold_markers,
            reasoning_open=reasoning_open,
        ):
            # Drop reasoning ("thinking") deltas; only stream the visible answer.
            if gen.in_reasoning or not delta:
                continue
            if not text_block_open:
                index += 1
                text_block_open = True
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": delta},
                },
            )

        stop_reason = "end_turn"
        if gen is not None and gen.finish_reason == "length":
            stop_reason = "max_tokens"

        tool_calls = []
        if use_tools and gen is not None:
            full_text = server.tokenizer.decode(gen.token_ids, skip_special_tokens=True)
            tool_calls = server.tool_parser.parse_full(
                full_text, tools=tools
            ).tool_calls

        # Anthropic messages must carry at least one content block.
        if not text_block_open and not tool_calls:
            index += 1
            text_block_open = True
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        if text_block_open:
            yield _sse(
                "content_block_stop", {"type": "content_block_stop", "index": index}
            )
            text_block_open = False

        if tool_calls:
            stop_reason = "tool_use"
            for tc in tool_calls:
                index += 1
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": {},
                        },
                    },
                )
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(_tool_input(tc)),
                        },
                    },
                )
                yield _sse(
                    "content_block_stop", {"type": "content_block_stop", "index": index}
                )

        output_tokens = len(gen.token_ids) if gen is not None else 0
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
        finished_normally = True
    except RuntimeError as e:
        finished_normally = True
        yield _sse(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": str(e)}},
        )
    finally:
        monitor.cancel()
        # Client dropped mid-stream: make sure the engine stops generating.
        if not finished_normally:
            server._abort_request(req)


async def handle_messages(server: Any, request: Any, body: dict):  # noqa: ANN201
    """Top-level handler for ``POST /v1/messages`` (streaming + non-streaming)."""
    from fastapi.responses import JSONResponse, StreamingResponse

    try:
        messages, tools, tool_choice = anthropic_to_openai_messages(body)
        sampling_params = build_sampling_params(body)
    except AnthropicError as e:
        return e.to_response()

    use_tools = uses_tools(tools, tool_choice)
    stop = parse_stop_sequences(body)
    model = body.get("model") or server.served_model_name

    prompt_ids, reasoning_open = server._encode_chat(
        messages, tools=tools if use_tools else None, tool_choice=tool_choice
    )
    affinity_key = server.session_affinity_key(request, body)
    input_tokens = len(prompt_ids)
    kv_transfer = body.get("kv_transfer_params") or {}
    if not isinstance(kv_transfer, dict):
        kv_transfer = {}

    if kv_transfer.get("do_remote_decode"):
        preq = server.submit(prompt_ids, sampling_params, affinity_key=affinity_key)
        try:
            mig = await server.await_migration(preq)
        except RuntimeError as e:
            return JSONResponse(
                status_code=500,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                },
            )
        if not mig.get("migration"):
            done_tokens = mig.get("tokens") or []
            text = server.tokenizer.decode(done_tokens, skip_special_tokens=True)
            gen = SimpleNamespace(token_ids=done_tokens, finish_reason="stop")
            return JSONResponse(
                format_messages_response(
                    server,
                    model=model,
                    text=text,
                    gen=gen,
                    use_tools=use_tools,
                    tools=tools,
                    input_tokens=input_tokens,
                )
            )

        response = format_messages_response(
            server,
            model=model,
            text="",
            gen=None,
            use_tools=use_tools,
            tools=tools,
            input_tokens=input_tokens,
        )
        response["kv_transfer_params"] = {
            "migration": mig["migration"],
            "first_token": mig.get("first_token"),
            "seq_id": mig.get("seq_id"),
        }
        return JSONResponse(response)

    if kv_transfer.get("migration"):
        try:
            migrated_seq_id, first_token = _migration_metadata(kv_transfer)
        except AnthropicError as e:
            return e.to_response()
        req = server.submit_migrated(
            kv_transfer["migration"], migrated_seq_id, first_token
        )
    else:
        req = server.submit(prompt_ids, sampling_params, affinity_key=affinity_key)

    if bool(body.get("stream", False)):
        return StreamingResponse(
            event_stream(
                server,
                request,
                req,
                model=model,
                max_tokens=sampling_params.max_tokens,
                stop=stop,
                reasoning_open=reasoning_open,
                use_tools=use_tools,
                tools=tools,
                input_tokens=input_tokens,
            ),
            media_type="text/event-stream",
        )

    text = ""
    gen = None
    monitor = server._spawn_disconnect_monitor(request, req)
    try:
        async for delta, gen in server.stream_text(
            req,
            sampling_params.max_tokens,
            stop=stop,
            reasoning_open=reasoning_open,
        ):
            if not gen.in_reasoning:
                text += delta
    except RuntimeError as e:
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            },
        )
    finally:
        monitor.cancel()

    return JSONResponse(
        format_messages_response(
            server,
            model=model,
            text=text,
            gen=gen,
            use_tools=use_tools,
            tools=tools,
            input_tokens=input_tokens,
        )
    )


async def handle_count_tokens(server: Any, body: dict):  # noqa: ANN201
    """Top-level handler for ``POST /v1/messages/count_tokens``."""
    from fastapi.responses import JSONResponse

    try:
        messages, tools, tool_choice = anthropic_to_openai_messages(body)
    except AnthropicError as e:
        return e.to_response()

    use_tools = uses_tools(tools, tool_choice)
    prompt_ids, _ = server._encode_chat(
        messages, tools=tools if use_tools else None, tool_choice=tool_choice
    )
    return JSONResponse({"input_tokens": len(prompt_ids)})
