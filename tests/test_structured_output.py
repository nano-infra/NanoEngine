import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from dlengine.layers.structured_output import StructuredOutputManager
from dlengine.server.openai_server import OpenAIServer
from dlengine.tool_parser import build_tool_structural_tag, normalize_tool_choice
from tokenizers import models, Tokenizer
from transformers import PreTrainedTokenizerFast


def test_parse_json_schema_response_format_is_canonical():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
                "type": "object",
            },
        },
    }

    schema = OpenAIServer._parse_json_schema(response_format)

    assert json.loads(schema) == response_format["json_schema"]["schema"]
    assert " " not in schema


def test_parse_json_object_response_format():
    assert json.loads(OpenAIServer._parse_json_schema({"type": "json_object"})) == {
        "type": "object"
    }


@pytest.mark.parametrize(
    "response_format",
    [
        "json_schema",
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {}},
        {"type": "yaml"},
    ],
)
def test_parse_json_schema_rejects_invalid_formats(response_format):
    with pytest.raises(ValueError):
        OpenAIServer._parse_json_schema(response_format)


def test_sampling_params_reject_unsupported_constraint_combinations():
    server = object.__new__(OpenAIServer)
    server.default_max_tokens = 32
    server.num_speculative_tokens = 0
    response_format = {"type": "json_object"}

    with pytest.raises(ValueError, match="ignore_eos"):
        server._build_sampling_params(
            {"response_format": response_format, "ignore_eos": True}
        )

    with pytest.raises(ValueError, match="tool calling"):
        server._build_sampling_params(
            {
                "response_format": response_format,
                "tools": [{"type": "function", "function": {"name": "f"}}],
            }
        )


def _toy_manager():
    vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "{": 2,
        "}": 3,
        '"x"': 4,
        ":": 5,
        "1": 6,
        " ": 7,
        ",": 8,
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>"
    )
    return StructuredOutputManager(tokenizer, len(vocab), [1])


def test_xgrammar_masks_and_accepts_complete_json_object():
    manager = _toy_manager()
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        }
    )

    for token_id in [2, 4, 5, 6, 3, 1]:
        logits = torch.zeros(1, 9)
        manager.apply(logits, [10], [schema], [None])
        assert torch.isfinite(logits[0, token_id])
        manager.accept([10], [schema], [None], torch.tensor([token_id]))

    assert manager._matchers == {}


def test_xgrammar_restores_first_generated_token_after_pd_migration():
    manager = _toy_manager()
    schema = json.dumps({"type": "object"})
    logits = torch.zeros(1, 9)

    manager.apply(logits, [20], [schema], [None], seed_tokens=torch.tensor([2]))

    assert torch.isfinite(logits[0, 4])
    assert not torch.isfinite(logits[0, 2])


def test_xgrammar_compiles_structural_tag_constraint():
    manager = _toy_manager()
    structural_tag = json.dumps(
        {
            "type": "structural_tag",
            "format": {"type": "const_string", "value": "{"},
        }
    )
    logits = torch.zeros(1, 9)

    manager.apply(logits, [30], [None], [structural_tag])

    assert torch.isfinite(logits[0, 2])
    assert not torch.isfinite(logits[0, 3])


def test_stream_text_returns_final_state_when_tool_marker_holds_all_text():
    async def collect():
        server = object.__new__(OpenAIServer)
        server._incremental_detokenize = lambda *args, **kwargs: (
            '<|tool_call>call:weather{city:<|"|>上海<|"|>}<tool_call|>'
        )
        queue = asyncio.Queue()
        queue.put_nowait({"tokens": [42]})
        queue.put_nowait(None)
        request = SimpleNamespace(aqueue=queue)
        return [
            item
            async for item in server.stream_text(
                request,
                max_tokens=16,
                hold_markers=["<|tool_call>"],
                preserve_special_tokens=True,
            )
        ]

    chunks = asyncio.run(collect())

    assert len(chunks) == 1
    delta, generation = chunks[0]
    assert delta == ""
    assert generation.token_ids == [42]
    assert generation.finish_reason == "stop"


def test_from_model_resolves_stop_tokens_when_runner_config_is_empty(monkeypatch):
    tokenizer = object()
    captured = {}

    monkeypatch.setattr(
        "transformers.PreTrainedTokenizerFast.from_pretrained",
        lambda model_path: tokenizer,
    )
    monkeypatch.setattr(
        "dlengine.models.trait.resolve_eos_token_ids",
        lambda model_path, loaded_tokenizer: [42, 43],
    )

    def fake_init(self, loaded_tokenizer, vocab_size, stop_token_ids):
        captured.update(
            tokenizer=loaded_tokenizer,
            vocab_size=vocab_size,
            stop_token_ids=stop_token_ids,
        )

    monkeypatch.setattr(StructuredOutputManager, "__init__", fake_init)

    StructuredOutputManager.from_model("/models/qwen", 128, [])

    assert captured == {
        "tokenizer": tokenizer,
        "vocab_size": 128,
        "stop_token_ids": [42, 43],
    }


def test_from_model_merges_runner_and_model_stop_tokens(monkeypatch):
    monkeypatch.setattr(
        "transformers.PreTrainedTokenizerFast.from_pretrained",
        lambda model_path: object(),
    )
    monkeypatch.setattr(
        "dlengine.models.trait.resolve_eos_token_ids",
        lambda model_path, tokenizer: [43, 44],
    )
    captured = {}

    def fake_init(self, tokenizer, vocab_size, stop_token_ids):
        captured["stop_token_ids"] = stop_token_ids

    monkeypatch.setattr(StructuredOutputManager, "__init__", fake_init)

    StructuredOutputManager.from_model("/models/qwen", 128, [42, 43])

    assert captured["stop_token_ids"] == [42, 43, 44]


@pytest.mark.parametrize(
    ("parser_name", "expected_fragment"),
    [
        ("hermes", "arguments"),
        ("qwen3_xml", '"style":"qwen_xml"'),
        ("deepseek_v3_1", "<｜tool▁calls▁begin｜>"),
        ("deepseek_v3_2", "<｜DSML｜function_calls>"),
        ("deepseek_v4", "<｜DSML｜tool_calls>"),
        ("glm", '"style":"glm_xml"'),
    ],
)
def test_build_builtin_tool_structural_tags(parser_name, expected_fragment):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    tag = build_tool_structural_tag(
        parser_name=parser_name,
        model_path="/models/test",
        served_model_name="test",
        tools=tools,
        tool_choice="auto",
        reasoning=False,
    )

    assert json.loads(tag)["type"] == "structural_tag"
    assert expected_fragment in tag


def test_normalize_anthropic_tool_choices():
    assert normalize_tool_choice({"type": "auto"}) == "auto"
    assert normalize_tool_choice({"type": "any"}) == "required"
    assert normalize_tool_choice({"type": "tool", "name": "weather"}) == {
        "type": "function",
        "function": {"name": "weather"},
    }


def test_named_tool_choice_constrains_to_selected_function():
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in ("weather", "calendar")
    ]

    tag = build_tool_structural_tag(
        parser_name="hermes",
        model_path="test",
        served_model_name="test",
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "weather"}},
        reasoning=False,
    )

    assert "weather" in tag
    assert "calendar" not in tag


def test_server_attaches_model_native_tool_constraint():
    from dlengine._rust.proto import SamplingParams

    server = object.__new__(OpenAIServer)
    server.num_speculative_tokens = 0
    server.tool_parser_name = "qwen3_xml"
    server.model_path = "/models/Qwen3.5"
    server.served_model_name = "qwen3.5"
    params = SamplingParams()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object"},
            },
        }
    ]

    server._attach_tool_constraint(params, tools, "auto", reasoning_open=False)

    assert "qwen_xml" in params.structural_tag


def test_server_attaches_gemma4_native_tool_constraint():
    from dlengine._rust.proto import SamplingParams

    server = object.__new__(OpenAIServer)
    server.num_speculative_tokens = 0
    server.tool_parser_name = "gemma4"
    server.model_path = "/models/gemma-4"
    server.served_model_name = "gemma-4"
    params = SamplingParams()
    tools = [
        {
            "type": "function",
            "function": {"name": "weather", "parameters": {"type": "object"}},
        }
    ]

    server._attach_tool_constraint(params, tools, "auto", reasoning_open=True)

    assert "<|tool_call>call:weather" in params.structural_tag
    parsed = json.loads(params.structural_tag)
    grammar = parsed["format"]["elements"][1]["tags"][0]["content"]["grammar"]
    assert 'native_string ::= "<|\\"|>"' in grammar


def test_gemma4_native_grammar_compiles_with_schema_and_optional_thinking():
    import xgrammar as xgr

    tag = build_tool_structural_tag(
        parser_name="gemma4",
        model_path="/models/gemma-4",
        served_model_name="gemma-4",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                            "options": {
                                "type": "object",
                                "properties": {"lang": {"type": "string"}},
                                "required": ["lang"],
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_choice="auto",
        reasoning=True,
    )

    parsed = json.loads(tag)
    thought = parsed["format"]["elements"][0]
    tool = parsed["format"]["elements"][1]["tags"][0]
    grammar = tool["content"]["grammar"]

    assert thought["type"] == "optional"
    assert tool["begin"] == "<|tool_call>call:weather"
    assert 'native_string ::= "<|\\"|>"' in grammar
    assert '"city" ws ":" ws native_string' in grammar
    assert '"days" ws ":" ws integer' in grammar
    tokenizer_info = xgr.TokenizerInfo(
        [bytes([value]) for value in range(256)],
        vocab_size=256,
        stop_token_ids=[0],
    )
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_structural_tag(tag)

    valid = xgr.GrammarMatcher(compiled)
    assert valid.accept_string(
        "<|channel>thought\ncheck weather<channel|>"
        '<|tool_call>call:weather{city:<|"|>上海<|"|>,'
        'options:{lang:<|"|>zh<|"|>},'
        'tags:[<|"|>current<|"|>,<|"|>alerts<|"|>]}<tool_call|>'
    )
    assert valid.is_completed()

    invalid = xgr.GrammarMatcher(compiled)
    assert not invalid.accept_string("<|tool_call>call:weather{city:上海}<tool_call|>")


def test_gemma4_anthropic_named_tool_choice_filters_other_tools():
    tag = build_tool_structural_tag(
        parser_name="gemma4",
        model_path="/models/gemma-4",
        served_model_name="gemma-4",
        tools=[
            {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            }
            for name in ("weather", "calendar")
        ],
        tool_choice={"type": "tool", "name": "weather"},
        reasoning=True,
    )

    assert "call:weather" in tag
    assert "call:calendar" not in tag


def test_decode_generated_preserves_tool_tokens_but_removes_eos():
    class _Tokenizer:
        eos_token_id = 9

        @staticmethod
        def decode(token_ids, skip_special_tokens):
            values = {1: "<tool_call>", 2: "payload", 9: "<eos>"}
            text = "".join(values[token] for token in token_ids)
            return text.replace("<tool_call>", "") if skip_special_tokens else text

    server = object.__new__(OpenAIServer)
    server.tokenizer = _Tokenizer()
    server._eos_token_ids = {9}

    assert (
        server._decode_generated([1, 2, 9], preserve_special_tokens=True)
        == "<tool_call>payload"
    )
