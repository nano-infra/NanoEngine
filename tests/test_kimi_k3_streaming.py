import json
from types import SimpleNamespace

import pytest
from dlengine.server.openai_server import build_app, OpenAIServer
from dlengine.server.tool_parser import get_tool_parser, StreamingContentFilter
from fastapi.testclient import TestClient

PARSER = get_tool_parser("kimi_k3")
RESPONSE = (
    PARSER.RESPONSE_OPEN
    + "答案 99 < 100"
    + PARSER.RESPONSE_CLOSE
    + PARSER.MESSAGE_CLOSE
    + PARSER.END_OF_MESSAGE
)


@pytest.mark.parametrize("split", range(len(RESPONSE) + 1))
def test_response_framing_split_at_every_character(split):
    f = StreamingContentFilter(PARSER.content_markers)
    assert (
        f.feed(RESPONSE[:split]) + f.feed(RESPONSE[split:]) + f.finish()
        == "答案 99 < 100"
    )


def test_character_deltas_and_truncated_control_marker():
    f = StreamingContentFilter(PARSER.content_markers)
    assert "".join(f.feed(c) for c in RESPONSE) + f.finish() == "答案 99 < 100"
    assert f.feed(PARSER.RESPONSE_OPEN + "answer<|close|>res") + f.finish() == "answer"
    assert f.feed("plain <") + f.finish() == "plain <"


class Tokenizer:
    chat_template = None

    def __init__(self, text):
        self.text = text

    def apply_chat_template(self, messages, **kwargs):
        return PARSER.THINK_OPEN

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, token_ids, skip_special_tokens=True):
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
