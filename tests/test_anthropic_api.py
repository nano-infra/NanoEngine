import asyncio
import json
from types import SimpleNamespace

from dlengine.server.anthropic_api import event_stream, format_messages_response
from dlengine.tool_parser import get_tool_parser


def test_gemma4_native_tool_call_becomes_anthropic_tool_use():
    server = SimpleNamespace(tool_parser=get_tool_parser("gemma4"))
    generated = SimpleNamespace(token_ids=[10, 11, 12], finish_reason="stop")

    response = format_messages_response(
        server,
        model="gemma-4-e2b-it",
        text=("<|tool_call>call:get_weather{" 'city:<|"|>上海<|"|>' "}<tool_call|>"),
        gen=generated,
        use_tools=True,
        input_tokens=81,
    )

    assert response["stop_reason"] == "tool_use"
    assert response["content"] == [
        {
            "type": "tool_use",
            "id": response["content"][0]["id"],
            "name": "get_weather",
            "input": {"city": "上海"},
        }
    ]
    assert response["usage"] == {"input_tokens": 81, "output_tokens": 3}


def test_anthropic_stream_emits_tool_use_when_no_visible_delta_was_streamed():
    native_call = '<|tool_call>call:get_weather{city:<|"|>上海<|"|>}<tool_call|>'

    async def stream_text(*args, **kwargs):
        yield "", SimpleNamespace(
            token_ids=[10, 11, 12], finish_reason="stop", in_reasoning=False
        )

    class Monitor:
        def cancel(self):
            return None

    server = SimpleNamespace(
        tool_parser=get_tool_parser("gemma4"),
        stream_text=stream_text,
        _decode_generated=lambda *args, **kwargs: native_call,
        _spawn_disconnect_monitor=lambda *args: Monitor(),
        _abort_request=lambda *args: None,
    )

    async def collect():
        return b"".join(
            [
                chunk
                async for chunk in event_stream(
                    server,
                    request=object(),
                    req=object(),
                    model="gemma-4-e2b-it",
                    max_tokens=96,
                    stop=[],
                    reasoning_open=False,
                    use_tools=True,
                    input_tokens=65,
                )
            ]
        ).decode()

    events = asyncio.run(collect())
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in events.splitlines()
        if line.startswith("data: ")
    ]
    tool_delta = next(
        payload for payload in payloads if payload.get("type") == "content_block_delta"
    )

    assert '"type": "tool_use"' in events
    assert '"name": "get_weather"' in events
    assert json.loads(tool_delta["delta"]["partial_json"]) == {"city": "上海"}
    assert '"stop_reason": "tool_use"' in events
    assert '"output_tokens": 3' in events
