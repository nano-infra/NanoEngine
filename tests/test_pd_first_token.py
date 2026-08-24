import asyncio
import json
from types import SimpleNamespace

from dlengine.server.openai_server import build_app, OpenAIServer
from fastapi.testclient import TestClient


class _ImmediateDecodeWorker:
    """Stand-in for decode delivering its first StepOut immediately."""

    def __init__(self, token: int):
        self.token = token
        self.submitted = []

    def submit(self, req):
        self.submitted.append(req)
        req.aqueue.put_nowait({"tokens": [self.token]})


def test_submit_migrated_emits_prefill_token_before_decode_tokens():
    async def run():
        worker = _ImmediateDecodeWorker(token=8)
        server = SimpleNamespace(worker=worker)

        req = OpenAIServer.submit_migrated(server, "migration", 42, first_token=7)

        assert worker.submitted == [req]
        assert await req.aqueue.get() == {"tokens": [7]}
        assert await req.aqueue.get() == {"tokens": [8]}

    asyncio.run(run())


def test_submit_migrated_without_prefill_token_does_not_invent_one():
    async def run():
        worker = _ImmediateDecodeWorker(token=8)
        server = SimpleNamespace(worker=worker)

        req = OpenAIServer.submit_migrated(server, "migration", 42)

        assert await req.aqueue.get() == {"tokens": [8]}
        assert req.aqueue.empty()

    asyncio.run(run())


class _StreamingTokenizer:
    chat_template = None

    def apply_chat_template(self, messages, **kwargs):
        return "encoded chat prompt"

    def encode(self, text):
        return [10, 11, 12]

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = {21: "hello", 22: " world"}
        return "".join(pieces.get(token_id, "") for token_id in token_ids)


class _StreamingWorker:
    def submit(self, req):
        req.aqueue.put_nowait({"tokens": [21, 22]})
        req.aqueue.put_nowait(None)

    def abort(self, seq_id):
        return None


def _streaming_server():
    server = OpenAIServer.__new__(OpenAIServer)
    server.worker = _StreamingWorker()
    server.tokenizer = _StreamingTokenizer()
    server.served_model_name = "test-model"
    server.model_path = "/models/test-model"
    server.default_max_tokens = 8
    server.max_model_len = 128
    server._model_aliases = {"test-model"}
    server.tool_parser = SimpleNamespace(
        open_markers=[], reasoning_close_marker="</think>"
    )
    server._build_sampling_params = lambda body: SimpleNamespace(
        max_tokens=body.get("max_tokens", 8)
    )
    return server


def _sse_chunks(response):
    payloads = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert payloads[-1] == "[DONE]"
    return [json.loads(payload) for payload in payloads[:-1]]


def test_chat_stream_emits_requested_usage_before_done():
    with TestClient(build_app(_streaming_server())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    chunks = _sse_chunks(response)
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_text_stream_emits_requested_usage_before_done():
    with TestClient(build_app(_streaming_server())) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "input_ids": [1, 2],
                "max_tokens": 8,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    chunks = _sse_chunks(response)
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }


def test_stream_omits_usage_when_not_requested():
    with TestClient(build_app(_streaming_server())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": True,
            },
        )

    chunks = _sse_chunks(response)
    assert all(chunk.get("choices") for chunk in chunks)
    assert all("usage" not in chunk for chunk in chunks)
