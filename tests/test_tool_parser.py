import json

from dlengine.server.openai_server import _skip_redundant_think_openers
from dlengine.server.tool_parser import detect_parser_name, get_tool_parser


def test_kimi_k3_xtml_reasoning_response_and_tools():
    parsed = get_tool_parser("kimi_k3").parse_full(
        '<|open|>think<|sep|>reason<|close|>think<|sep|>'
        '<|open|>response<|sep|>answer<|close|>response<|sep|>'
        '<|open|>tools<|sep|>'
        '<|open|>call tool="weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>北京'
        '<|close|>argument<|sep|>'
        '<|open|>argument key="days" type="integer"<|sep|>2'
        '<|close|>argument<|sep|>'
        '<|close|>call<|sep|><|close|>tools<|sep|>'
    )
    assert parsed.reasoning == "reason"
    assert parsed.content == "answer"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].function.name == "weather"
    assert parsed.tool_calls[0].function.arguments == '{"city": "北京", "days": 2}'


def test_kimi_k3_parser_auto_detection():
    assert detect_parser_name("/models/Kimi-K3", "Kimi-K3") == "kimi_k3"


def test_kimi_k3_parser_exposes_reasoning_close_marker():
    parser = get_tool_parser("kimi_k3")
    assert parser.reasoning_close_marker == "<|close|>think<|sep|>"


def test_glm_tool_parser_arg_key_value_format():
    parser = get_tool_parser("glm")
    parsed = parser.parse_full(
        "<think>need a lookup</think>\n"
        "<tool_call>search"
        "<arg_key>query</arg_key><arg_value>GLM tool parser</arg_value>"
        "<arg_key>limit</arg_key><arg_value>5</arg_value>"
        "</tool_call>"
    )

    assert parsed.content is None
    assert parsed.reasoning == "need a lookup"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].function.name == "search"
    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "query": "GLM tool parser",
        "limit": 5,
    }


def test_glm_tool_parser_supports_empty_arguments_and_multiple_calls():
    parser = get_tool_parser("glm5")
    parsed = parser.parse_full(
        "ok\n"
        "<tool_call>list_files</tool_call>"
        "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/a.py</arg_value></tool_call>"
    )

    assert parsed.content == "ok"
    assert [tc.function.name for tc in parsed.tool_calls] == ["list_files", "read_file"]
    assert json.loads(parsed.tool_calls[0].function.arguments) == {}
    assert json.loads(parsed.tool_calls[1].function.arguments) == {"path": "/tmp/a.py"}


def test_detect_glm_parser_from_local_chat_template():
    template = (
        "<tool_call>{function-name}"
        "<arg_key>{arg-key}</arg_key><arg_value>{arg-value}</arg_value>"
        "</tool_call>"
    )

    assert (
        detect_parser_name("/mnt/h_public/GLM-5-Int4-Porvider", "glm5", template)
        == "glm"
    )


def test_skip_redundant_think_openers():
    text = "<think><think><think>reasoning"
    offset, need_more = _skip_redundant_think_openers(text, 0)

    assert text[offset:] == "reasoning"
    assert need_more is False


def test_skip_redundant_think_openers_holds_partial_marker():
    offset, need_more = _skip_redundant_think_openers("<think><thi", 0)

    assert offset == len("<think>")
    assert need_more is True


def test_skip_redundant_think_openers_preserves_reasoning_text():
    text = "The model starts reasoning directly."
    offset, need_more = _skip_redundant_think_openers(text, 0)

    assert offset == 0
    assert need_more is False


def test_glm_tool_parser_drops_redundant_think_openers():
    parsed = get_tool_parser("glm").parse_full(
        "<think><think><think>reasoning body</think>answer"
    )

    assert parsed.reasoning == "reasoning body"
    assert parsed.content == "answer"


def test_glm_tool_parser_routes_unterminated_repeated_think_as_reasoning():
    parsed = get_tool_parser("glm").parse_full(
        "<think><think><think>unfinished reasoning"
    )

    assert parsed.reasoning == "unfinished reasoning"
    assert parsed.content is None
