import json

from dlengine.server.openai_server import _skip_redundant_think_openers
from dlengine.tool_parser import detect_parser_name, get_tool_parser


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
    offset, need_more = _skip_redundant_think_openers("<think><th" + "i", 0)

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


def test_deepseek_v31_tool_parser():
    parsed = get_tool_parser("deepseek_v3_1").parse_full(
        "reasoning</think>"
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>weather<｜tool▁sep｜>"
        '{"city":"上海"}'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    assert parsed.reasoning == "reasoning"
    assert parsed.content is None
    assert parsed.tool_calls[0].function.name == "weather"
    assert json.loads(parsed.tool_calls[0].function.arguments) == {"city": "上海"}


def test_deepseek_dsml_tool_parser():
    parsed = get_tool_parser("deepseek_v4").parse_full(
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="weather">\n'
        '<｜DSML｜parameter name="city" string="true">上海</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )

    assert parsed.content is None
    assert parsed.tool_calls[0].function.name == "weather"
    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "city": "上海",
        "days": 3,
    }


def test_detect_deepseek_parsers_from_chat_template():
    assert detect_parser_name("model", "model", "<｜DSML｜tool_calls>") == "deepseek_v4"
    assert (
        detect_parser_name("model", "model", "<｜DSML｜function_calls>")
        == "deepseek_v3_2"
    )
    assert (
        detect_parser_name("model", "model", "<｜tool▁calls▁begin｜>")
        == "deepseek_v3_1"
    )


def test_qwen_and_glm_parsers_unescape_xml_values():
    qwen = get_tool_parser("qwen3_xml").parse_full(
        "<tool_call><function=search>"
        "<parameter=query>A &lt; B</parameter>"
        "</function></tool_call>"
    )
    glm = get_tool_parser("glm").parse_full(
        "<tool_call>search"
        "<arg_key>query</arg_key><arg_value>A &lt; B</arg_value>"
        "</tool_call>"
    )

    assert json.loads(qwen.tool_calls[0].function.arguments) == {"query": "A < B"}
    assert json.loads(glm.tool_calls[0].function.arguments) == {"query": "A < B"}


def test_gemma4_tool_parser_handles_native_strings_and_nested_values():
    parsed = get_tool_parser("gemma4").parse_full(
        "<|channel>thought\nneed weather<channel|>"
        "<|tool_call>call:get-weather{"
        'city:<|"|>Tokyo, {JP}<|"|>,'
        "days:3,metric:true,"
        'options:{lang:<|"|>ja<|"|>},'
        'tags:[<|"|>current<|"|>,<|"|>alerts<|"|>]'
        "}<tool_call|><turn|>"
    )

    assert parsed.reasoning == "need weather"
    assert parsed.content is None
    assert parsed.tool_calls[0].function.name == "get-weather"
    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "city": "Tokyo, {JP}",
        "days": 3,
        "metric": True,
        "options": {"lang": "ja"},
        "tags": ["current", "alerts"],
    }


def test_gemma4_tool_parser_accepts_standard_json_arguments():
    parsed = get_tool_parser("gemma4").parse_full(
        '<|tool_call>call:weather{"city":"Tokyo"}<tool_call|>'
    )

    assert json.loads(parsed.tool_calls[0].function.arguments) == {"city": "Tokyo"}


def test_detect_gemma4_parser():
    assert detect_parser_name("model", "model", "<|tool_call>call:") == "gemma4"
    assert detect_parser_name("/models/gemma-4-26b-a4b-it", "model") == "gemma4"
