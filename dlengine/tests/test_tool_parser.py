import json

from dlengine.server.tool_parser import detect_parser_name, get_tool_parser


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

    assert detect_parser_name("/mnt/h_public/GLM-5-Int4-Porvider", "glm5", template) == "glm"
