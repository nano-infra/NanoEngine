import json
from types import SimpleNamespace

import pytest
from dlengine.server.openai_server import build_app, OpenAIServer
from dlengine.server.tool_parser import get_tool_parser
from fastapi.testclient import TestClient

PARSER = get_tool_parser("kimi_k3")
RESPONSE = (
    PARSER.RESPONSE_OPEN
    + "答案 99 < 100"
    + PARSER.RESPONSE_CLOSE
    + PARSER.MESSAGE_CLOSE
    + PARSER.END_OF_MESSAGE
)


def parse_chunks(chunks, *, reasoning_open=False):
    parser = PARSER.create_stream_parser(reasoning_open=reasoning_open)
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return (
        "".join(event.content or "" for event in events),
        "".join(event.reasoning or "" for event in events),
        [call for event in events for call in event.tool_calls],
    )


@pytest.mark.parametrize("split", range(len(RESPONSE) + 1))
def test_response_framing_split_at_every_character(split):
    assert parse_chunks([RESPONSE[:split], RESPONSE[split:]]) == (
        "答案 99 < 100",
        "",
        [],
    )


def test_character_deltas_and_truncated_control_marker():
    assert parse_chunks(RESPONSE) == ("答案 99 < 100", "", [])
    assert parse_chunks([PARSER.RESPONSE_OPEN + "answer<|close|>res"])[0] == "answer"
    assert parse_chunks(["plain <"])[0] == "plain <"


class Tokenizer:
    chat_template = None

    def __init__(self, text):
        self.text = text

    def apply_chat_template(self, messages, **kwargs):
        return PARSER.THINK_OPEN

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        return "".join(self.text[i] for i in token_ids)


class Worker:
    def __init__(self, text):
        self.text = text
        self.aborted = []

    def submit(self, req):
        for i in range(len(self.text)):
            req.aqueue.put_nowait({"tokens": [i]})
        req.aqueue.put_nowait(None)

    def abort(self, seq_id):
        self.aborted.append(seq_id)


def server_for(text):
    server = OpenAIServer.__new__(OpenAIServer)
    server.worker = Worker(text)
    server.tokenizer = Tokenizer(text)
    server.served_model_name = "test-k3"
    server.model_path = "/models/test-k3"
    server.default_max_tokens = 4096
    server.max_model_len = 8192
    server._model_aliases = {"test-k3"}
    server.tool_parser = PARSER
    server._build_sampling_params = lambda body: SimpleNamespace(
        max_tokens=body.get("max_tokens", 4096)
    )
    return server


def request(text, **overrides):
    body = {
        "model": "test-k3",
        "messages": [{"role": "user", "content": "answer"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        **overrides,
    }
    server = server_for(text)
    with TestClient(build_app(server)) as client:
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    chunks = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    return chunks, server


def content(chunks, key="content"):
    return "".join(
        choice.get("delta", {}).get(key, "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )


def test_openai_sse_keeps_reasoning_finish_and_raw_token_usage():
    raw = "reasoning" + PARSER.THINK_CLOSE + RESPONSE
    chunks, _ = request(raw)
    assert content(chunks) == "答案 99 < 100"
    assert content(chunks, "reasoning_content") == "reasoning"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == len(raw)


def test_openai_sse_tool_calls_do_not_leak_or_duplicate_reasoning():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]
    raw = (
        "look it up"
        + PARSER.THINK_CLOSE
        + PARSER.TOOLS_OPEN
        + '<|open|>call tool="lookup"<|sep|><|open|>argument key="q" type="string"<|sep|>K3<|close|>argument<|sep|><|close|>call<|sep|>'
        + PARSER.TOOLS_CLOSE
        + PARSER.MESSAGE_CLOSE
        + PARSER.END_OF_MESSAGE
    )
    chunks, _ = request(raw, tools=tools)
    assert content(chunks) == ""
    assert content(chunks, "reasoning_content") == "look it up"
    calls = [
        c["delta"]["tool_calls"][0]
        for chunk in chunks
        for c in chunk.get("choices", [])
        if "tool_calls" in c.get("delta", {})
    ]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "K3"}
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["usage"]["completion_tokens"] == len(raw)


def test_stop_and_length_limits_still_use_raw_generation():
    prefix = PARSER.THINK_CLOSE + PARSER.RESPONSE_OPEN
    chunks, server = request(
        prefix + "answer STOP ignored" + PARSER.RESPONSE_CLOSE, stop=["STOP"]
    )
    assert content(chunks) == "answer "
    assert server.worker.aborted
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    chunks, _ = request(
        prefix + "answer" + PARSER.RESPONSE_CLOSE,
        max_tokens=len(prefix + "answer<|close|>res"),
    )
    assert content(chunks) == "answer"
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"


def test_full_parser_strips_end_of_message_without_response_wrapper():
    assert (
        PARSER.parse_full("99" + PARSER.MESSAGE_CLOSE + PARSER.END_OF_MESSAGE).content
        == "99"
    )


def test_anthropic_stream_uses_same_clean_content_path():
    raw = "reasoning" + PARSER.THINK_CLOSE + RESPONSE
    with TestClient(build_app(server_for(raw))) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-k3",
                "messages": [{"role": "user", "content": "answer"}],
                "max_tokens": 4096,
                "stream": True,
            },
        )
    assert response.status_code == 200
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    answer = "".join(
        event.get("delta", {}).get("text", "")
        for event in events
        if event.get("type") == "content_block_delta"
    )
    assert answer == "答案 99 < 100"
    assert events[-1]["type"] == "message_stop"


def xtml_call(name, arguments):
    from html import escape

    attrs = lambda value: escape(value, quote=True).replace("&#x27;", "'")
    body = "".join(
        '<|open|>argument key="'
        + attrs(key)
        + '" type="'
        + kind
        + '"<|sep|>'
        + value
        + "<|close|>argument<|sep|>"
        for key, kind, value in arguments
    )
    return (
        '<|open|>call tool="' + attrs(name) + '"<|sep|>' + body + "<|close|>call<|sep|>"
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 10000])
def test_state_preserves_response_text_and_raw_string_arguments(chunk_size):
    literal = "文档示例 " + PARSER.RESPONSE_OPEN + " 不应在当前 response 内被删除。"
    argument = PARSER.MESSAGE_CLOSE + PARSER.TOOLS_OPEN + '\n<&quot;> "🙂"'
    call = xtml_call(
        'look"up&',
        [('q&"', "string", argument), ("opts", "object", '{"a":[1,true,null]}')],
    )
    raw = (
        PARSER.THINK_OPEN
        + "reason"
        + PARSER.THINK_CLOSE
        + PARSER.RESPONSE_OPEN
        + literal
        + PARSER.RESPONSE_CLOSE
        + PARSER.TOOLS_OPEN
        + call
        + PARSER.TOOLS_CLOSE
        + PARSER.MESSAGE_CLOSE
    )
    content_text, reasoning, calls = parse_chunks(
        [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
    )
    assert content_text == literal
    assert reasoning == "reason"
    assert len(calls) == 1
    assert calls[0].function.name == 'look"up&'
    assert json.loads(calls[0].function.arguments) == {
        'q&"': argument,
        "opts": {"a": [1, True, None]},
    }
    full = PARSER.parse_full(raw)
    assert full.content == content_text and full.reasoning == reasoning
    assert full.tool_calls[0].function.arguments == calls[0].function.arguments


@pytest.mark.parametrize("chunk_size", [1, 5, 10000])
def test_prefix_consumption_and_plain_response_are_chunk_invariant(chunk_size):
    for text, expected, thinking in [
        ("\n  body " + PARSER.RESPONSE_OPEN, "\n  body " + PARSER.RESPONSE_OPEN, False),
        (PARSER.RESPONSE_CLOSE + PARSER.MESSAGE_CLOSE, "", False),
        (
            "reason"
            + PARSER.THINK_CLOSE
            + PARSER.RESPONSE_OPEN
            + " answer "
            + PARSER.RESPONSE_CLOSE,
            " answer ",
            True,
        ),
        (
            PARSER.THINK_OPEN
            + PARSER.THINK_OPEN
            + "reason"
            + PARSER.THINK_CLOSE
            + PARSER.RESPONSE_OPEN
            + "answer"
            + PARSER.RESPONSE_CLOSE,
            "answer",
            True,
        ),
    ]:
        parsed = parse_chunks(
            [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)],
            reasoning_open=thinking,
        )
        assert parsed[0] == expected
        if thinking:
            assert parsed[1] == "reason"


def test_completed_calls_are_emitted_once_before_end_of_message():
    parser = PARSER.create_stream_parser()
    first = xtml_call("first", [("x", "integer", "1")])
    second = xtml_call("second", [("flag", "boolean", "false")])
    before = parser.feed(PARSER.TOOLS_OPEN + first[:-1])
    assert not any(event.tool_calls for event in before)
    events = parser.feed(first[-1:])
    assert len(events) == 1 and events[0].tool_calls[0].function.name == "first"
    first_id = events[0].tool_calls[0].id
    assert parser.feed("") == []
    events = parser.feed(second + PARSER.TOOLS_CLOSE + PARSER.MESSAGE_CLOSE)
    assert len(events) == 1 and events[0].tool_calls[0].function.name == "second"
    assert events[0].tool_calls[0].id != first_id
    assert parser.finish() == []


def test_incomplete_calls_are_not_emitted_as_tools_or_content():
    first = xtml_call("complete", [("x", "integer", "1")])
    incomplete = '<|open|>call tool="unfinished"<|sep|><|open|>argument key="q" type="string"<|sep|>secret tool argument'
    text, reasoning, calls = parse_chunks([PARSER.TOOLS_OPEN + first + incomplete])
    assert text == reasoning == ""
    assert [c.function.name for c in calls] == ["complete"]


def test_sse_multiple_calls_keep_indexes_and_ids_without_final_reparse():
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in ["one", "two"]
    ]
    raw = (
        PARSER.THINK_CLOSE
        + PARSER.TOOLS_OPEN
        + xtml_call("one", [])
        + xtml_call("two", [])
        + PARSER.TOOLS_CLOSE
        + PARSER.MESSAGE_CLOSE
    )
    chunks, _ = request(raw, tools=tools)
    calls = [
        call
        for chunk in chunks
        for choice in chunk.get("choices", [])
        for call in choice.get("delta", {}).get("tool_calls", [])
    ]
    assert [call["index"] for call in calls] == [0, 1]
    assert [call["function"]["name"] for call in calls] == ["one", "two"]
    assert len({call["id"] for call in calls}) == 2
    assert content(chunks) == ""
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"


def test_no_tools_and_tool_choice_none_do_not_expose_tool_protocol():
    raw = (
        PARSER.THINK_CLOSE
        + PARSER.TOOLS_OPEN
        + xtml_call("hidden", [])
        + PARSER.TOOLS_CLOSE
        + PARSER.MESSAGE_CLOSE
    )
    for extra in [
        {},
        {
            "tools": [{"type": "function", "function": {"name": "hidden"}}],
            "tool_choice": "none",
        },
    ]:
        chunks, _ = request(raw, **extra)
        assert content(chunks) == ""
        assert not any(
            "tool_calls" in choice.get("delta", {})
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        assert chunks[-1]["usage"]["completion_tokens"] == len(raw)


def test_non_streaming_reuses_structured_body_without_reinterpreting_literals():
    body = "示例 " + PARSER.RESPONSE_OPEN + " 保留。"
    raw = (
        PARSER.THINK_CLOSE
        + PARSER.RESPONSE_OPEN
        + body
        + PARSER.RESPONSE_CLOSE
        + PARSER.MESSAGE_CLOSE
    )
    with TestClient(build_app(server_for(raw))) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-k3",
                "messages": [{"role": "user", "content": "answer"}],
                "stream": False,
            },
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == body


def test_server_exposes_completed_call_while_engine_queue_is_still_open():
    import asyncio

    first = PARSER.TOOLS_OPEN + xtml_call("ready", [("x", "integer", "42")])
    raw = first + PARSER.TOOLS_CLOSE + PARSER.MESSAGE_CLOSE

    async def check():
        server = server_for(raw)
        req = SimpleNamespace(aqueue=asyncio.Queue(), seq_id=1)
        req.aqueue.put_nowait({"tokens": list(range(len(first)))})
        stream = server.stream_text(req, max_tokens=4096)
        try:
            delta, gen = await asyncio.wait_for(anext(stream), timeout=1)
            assert delta == "" and len(gen.tool_calls) == 1
            assert gen.tool_calls[0].function.name == "ready"
            assert json.loads(gen.tool_calls[0].function.arguments) == {"x": 42}
            call_id = gen.tool_calls[0].id
            # The engine has not sent the tool/message closers or EOS yet.
            assert req.aqueue.empty() and len(gen.token_ids) == len(first)
            req.aqueue.put_nowait({"tokens": list(range(len(first), len(raw)))})
            req.aqueue.put_nowait(None)
            async for _, gen in stream:
                pass
            assert [call.id for call in gen.tool_calls] == [call_id]
            assert gen.finish_reason == "stop" and len(gen.token_ids) == len(raw)
        finally:
            await stream.aclose()

    asyncio.run(check())
