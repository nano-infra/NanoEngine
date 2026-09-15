import json
from types import SimpleNamespace

from dlengine.server.openai_server import build_app, OpenAIServer
from dlengine.server.tool_parser import get_tool_parser
from fastapi.testclient import TestClient

_TOOL_TEXT = (
    "<tool_call>TaskUpdate"
    "<arg_key>taskId</arg_key><arg_value>1</arg_value>"
    "<arg_key>status</arg_key><arg_value>completed</arg_value>"
    "</tool_call>"
)
_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "TaskUpdate",
            "parameters": {
                "type": "object",
                "properties": {
                    "taskId": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        },
    }
]
_ANTHROPIC_TOOLS = [
    {
        "name": "TaskUpdate",
        "input_schema": {
            "type": "object",
            "properties": {
                "taskId": {"type": "string"},
                "status": {"type": "string"},
            },
        },
    }
]
_EXPECTED_INPUT = {"taskId": "1", "status": "completed"}


class _ToolTokenizer:
    chat_template = None

    def apply_chat_template(self, messages, **kwargs):
        return "encoded chat prompt"

    def encode(self, text):
        return [10, 11, 12]

    def decode(self, token_ids, skip_special_tokens=True):
        return _TOOL_TEXT if token_ids else ""


class _ToolWorker:
    def submit(self, req):
        req.aqueue.put_nowait({"tokens": [21]})
        req.aqueue.put_nowait(None)

    def abort(self, seq_id):
        return None


def _server():
    server = OpenAIServer.__new__(OpenAIServer)
    server.worker = _ToolWorker()
    server.tokenizer = _ToolTokenizer()
    server.served_model_name = "test-model"
    server.model_path = "/models/test-model"
    server.default_max_tokens = 8
    server.max_model_len = 128
    server._model_aliases = {"test-model"}
    server.tool_parser = get_tool_parser("glm")
    server._build_sampling_params = lambda body: SimpleNamespace(
        max_tokens=body.get("max_tokens", 8)
    )
    return server


def _openai_body(stream: bool) -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "update the task"}],
        "tools": _OPENAI_TOOLS,
        "max_tokens": 8,
        "stream": stream,
    }


def _anthropic_body(stream: bool) -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "update the task"}],
        "tools": _ANTHROPIC_TOOLS,
        "max_tokens": 8,
        "stream": stream,
    }


def test_openai_non_streaming_threads_tool_schema_to_glm_parser():
    with TestClient(build_app(_server())) as client:
        response = client.post("/v1/chat/completions", json=_openai_body(False))

    assert response.status_code == 200
    tool_call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(tool_call["function"]["arguments"]) == _EXPECTED_INPUT


def test_openai_streaming_threads_tool_schema_to_glm_parser():
    with TestClient(build_app(_server())) as client:
        response = client.post("/v1/chat/completions", json=_openai_body(True))

    assert response.status_code == 200
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    tool_delta = next(
        chunk["choices"][0]["delta"]
        for chunk in chunks
        if chunk["choices"] and "tool_calls" in chunk["choices"][0]["delta"]
    )
    arguments = tool_delta["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == _EXPECTED_INPUT


def test_anthropic_non_streaming_threads_tool_schema_to_glm_parser():
    with TestClient(build_app(_server())) as client:
        response = client.post("/v1/messages", json=_anthropic_body(False))

    assert response.status_code == 200
    tool_use = next(
        block for block in response.json()["content"] if block["type"] == "tool_use"
    )
    assert tool_use["input"] == _EXPECTED_INPUT


def test_anthropic_streaming_threads_tool_schema_to_glm_parser():
    with TestClient(build_app(_server())) as client:
        response = client.post("/v1/messages", json=_anthropic_body(True))

    assert response.status_code == 200
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    input_delta = next(
        payload["delta"]
        for payload in payloads
        if payload.get("type") == "content_block_delta"
        and payload["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(input_delta["partial_json"]) == _EXPECTED_INPUT
