import json
from types import SimpleNamespace

from dlengine.server.openai_server import _skip_redundant_think_openers, OpenAIServer
from dlengine.server.tool_parser import detect_parser_name, get_tool_parser


def test_kimi_k3_xtml_reasoning_response_and_tools():
    parsed = get_tool_parser("kimi_k3").parse_full(
        "<|open|>think<|sep|>reason<|close|>think<|sep|>"
        "<|open|>response<|sep|>answer<|close|>response<|sep|>"
        "<|open|>tools<|sep|>"
        '<|open|>call tool="weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>北京'
        "<|close|>argument<|sep|>"
        '<|open|>argument key="days" type="integer"<|sep|>2'
        "<|close|>argument<|sep|>"
        "<|close|>call<|sep|><|close|>tools<|sep|>"
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


def test_glm_tool_parser_honors_declared_string_argument_type():
    parsed = get_tool_parser("glm").parse_full(
        "<tool_call>TaskUpdate"
        "<arg_key>taskId</arg_key><arg_value>1</arg_value>"
        "<arg_key>status</arg_key><arg_value>completed</arg_value>"
        "</tool_call>",
        tools=[
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
        ],
    )

    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "taskId": "1",
        "status": "completed",
    }


def test_glm_tool_parser_stringifies_json_values_for_string_schema():
    parsed = get_tool_parser("glm5").parse_full(
        "<tool_call>record"
        "<arg_key>number</arg_key><arg_value>1</arg_value>"
        "<arg_key>boolean</arg_key><arg_value>true</arg_value>"
        "<arg_key>nullable</arg_key><arg_value>null</arg_value>"
        "<arg_key>array</arg_key><arg_value>[1, 2]</arg_value>"
        '<arg_key>object</arg_key><arg_value>{"a": 1}</arg_value>'
        "</tool_call>",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "number": {"type": "string"},
                            "boolean": {"type": "string"},
                            "nullable": {"type": ["null", "string"]},
                            "array": {"anyOf": [{"type": "array"}, {"type": "string"}]},
                            "object": {"enum": ["object", 1]},
                        },
                    },
                },
            }
        ],
    )

    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "number": "1",
        "boolean": "True",
        "nullable": "None",
        "array": "[1, 2]",
        "object": '{"a": 1}',
    }


def test_glm_tool_parser_preserves_inference_for_non_string_or_missing_schema():
    text = (
        "<tool_call>record"
        "<arg_key>count</arg_key><arg_value>1</arg_value>"
        "<arg_key>enabled</arg_key><arg_value>true</arg_value>"
        "<arg_key>items</arg_key><arg_value>[1, 2]</arg_value>"
        '<arg_key>metadata</arg_key><arg_value>{"a": 1}</arg_value>'
        "<arg_key>unknown</arg_key><arg_value>2</arg_value>"
        "</tool_call>"
    )
    parsed = get_tool_parser("glm").parse_full(
        text,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "enabled": {"type": "boolean"},
                            "items": {"items": {"type": "integer"}},
                            "metadata": {"properties": {"a": {"type": "integer"}}},
                        },
                    },
                },
            }
        ],
    )

    expected = {
        "count": 1,
        "enabled": True,
        "items": [1, 2],
        "metadata": {"a": 1},
        "unknown": 2,
    }
    assert json.loads(parsed.tool_calls[0].function.arguments) == expected
    assert (
        json.loads(
            get_tool_parser("glm").parse_full(text).tool_calls[0].function.arguments
        )
        == expected
    )


def test_glm_tool_parser_accepts_native_anthropic_schema_shape():
    parsed = get_tool_parser("glm").parse_full(
        "<tool_call>TaskUpdate"
        "<arg_key>taskId</arg_key><arg_value>1</arg_value>"
        "</tool_call>",
        tools=[
            {
                "name": "TaskUpdate",
                "input_schema": {
                    "type": "object",
                    "properties": {"taskId": {"type": "string"}},
                },
            }
        ],
    )

    assert json.loads(parsed.tool_calls[0].function.arguments) == {"taskId": "1"}


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


def test_detect_glm5_parser_precedes_generic_chat_template():
    template = (
        "<tool_call>{function-name}"
        "<arg_key>{arg-key}</arg_key><arg_value>{arg-value}</arg_value>"
        "</tool_call>"
    )

    assert detect_parser_name("/models/GLM-5.2", "GLM-5.2", template) == "glm47"


def test_glm5_server_selects_tool_and_reasoning_parser_pair():
    server = OpenAIServer(
        worker=None,
        tokenizer=SimpleNamespace(chat_template="<tool_call><arg_key>"),
        served_model_name="GLM-5.2",
        model_path="/models/GLM-5.2",
    )

    assert server.tool_parser_name == "glm47"
    assert server.reasoning_parser_name == "glm45"


def test_glm5_server_preserves_explicit_parser_overrides():
    server = OpenAIServer(
        worker=None,
        tokenizer=SimpleNamespace(chat_template="<tool_call><arg_key>"),
        served_model_name="GLM-5.2",
        model_path="/models/GLM-5.2",
        tool_call_parser="hermes",
        reasoning_parser="glm5",
    )

    assert server.tool_parser_name == "hermes"
    assert server.reasoning_parser_name == "glm5"


def test_detect_glm_parser_from_local_chat_template():
    template = (
        "<tool_call>{function-name}"
        "<arg_key>{arg-key}</arg_key><arg_value>{arg-value}</arg_value>"
        "</tool_call>"
    )

    assert (
        detect_parser_name("/mnt/h_public/GLM-5-Int4-Porvider", "glm5", template)
        == "glm47"
    )


def test_skip_redundant_think_openers():
    text = "<think><think><think>reasoning"
    offset, need_more = _skip_redundant_think_openers(text, 0)

    assert text[offset:] == "reasoning"
    assert need_more is False


def test_skip_redundant_think_openers_holds_partial_marker():
    offset, need_more = _skip_redundant_think_openers("<think><thin", 0)

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
