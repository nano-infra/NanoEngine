"""Build XGrammar structural tags for model-native tool-call formats."""

from __future__ import annotations

import json
from typing import Any, Optional


class UnsupportedToolGrammarError(ValueError):
    """Raised when XGrammar cannot express a model's native tool syntax."""


_PARSER_TO_XGRAMMAR_MODEL = {
    # Qwen3's JSON-in-<tool_call> format is also the Hermes format consumed by
    # HermesToolParser.
    "hermes": "qwen_3",
    "qwen3_xml": "qwen_3_5",
    "qwen3_coder": "qwen_3_coder",
    "glm": "glm_4_7",
    "glm45": "glm_4_7",
    "glm47": "glm_4_7",
    "glm5": "glm_4_7",
    "deepseek_v3_1": "deepseek_v3_1",
    "deepseek_v3_2": "deepseek_v3_2",
    "deepseek_v4": "deepseek_v4",
}


_GEMMA_STRING_DELIMITER = '<|"|>'


def _ebnf_literal(value: str) -> str:
    """Return an XGrammar EBNF string literal."""
    return json.dumps(value, ensure_ascii=False)


class _GemmaSchemaGrammar:
    """Translate JSON Schema into Gemma 4's native structured-data EBNF.

    Gemma 4 uses bare object keys and ``<|"|>`` around string values.  That
    syntax cannot be represented by XGrammar's JSONSchemaFormat, whose JSON
    renderer always uses ordinary double quotes.

    Object properties are emitted in schema declaration order. Optional
    properties remain optional through recursive suffix rules, avoiding the
    exponential expansion of every possible property combination.
    """

    _PRELUDE = r"""
ws ::= [ \n\t]*
integer ::= "-"? ("0" | [1-9] [0-9]*)
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
native_string ::= "<|\"|>" native_string_char* "<|\"|>"
native_string_char ::= [^<] | "<" [^|] | "<|" [^\"] | "<|\"" [^|] | "<|\"|" [^>]
any_value ::= native_string | number | "true" | "false" | "null" | any_array | any_object
any_array ::= "[" ws (any_value (ws "," ws any_value)*)? ws "]"
any_key ::= [^,:{}[\]\0-\x20]+
any_object ::= "{" ws (any_key ws ":" ws any_value (ws "," ws any_key ws ":" ws any_value)*)? ws "}"
""".strip()

    def __init__(self) -> None:
        self._rules: list[str] = []
        self._next_rule = 0

    def build(self, schema: Any) -> str:
        root_expr = self._schema_expr(
            schema if isinstance(schema, (dict, bool)) else {}
        )
        return f"root ::= {root_expr}\n{self._PRELUDE}\n" + "\n".join(self._rules)

    def _new_rule(self, prefix: str, expression: str) -> str:
        name = f"{prefix}_{self._next_rule}"
        self._next_rule += 1
        self._rules.append(f"{name} ::= {expression}")
        return name

    def _schema_expr(self, schema: Any) -> str:
        if schema is False:
            raise ValueError("Gemma 4 tool schema cannot be false")
        if schema is True or not isinstance(schema, dict) or not schema:
            return "any_value"

        if "const" in schema:
            return self._value_expr(schema["const"])
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return "(" + " | ".join(self._value_expr(value) for value in enum) + ")"

        variants = schema.get("anyOf") or schema.get("oneOf")
        if isinstance(variants, list) and variants:
            return "(" + " | ".join(self._schema_expr(item) for item in variants) + ")"

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            variants = [
                self._schema_expr({**schema, "type": item}) for item in schema_type
            ]
            return "(" + " | ".join(variants) + ")"
        if schema_type == "string":
            return "native_string"
        if schema_type == "integer":
            return "integer"
        if schema_type == "number":
            return "number"
        if schema_type == "boolean":
            return '("true" | "false")'
        if schema_type == "null":
            return '"null"'
        if schema_type == "array":
            return self._array_expr(schema)
        if schema_type == "object" or "properties" in schema:
            return self._object_expr(schema)
        return "any_value"

    def _value_expr(self, value: Any) -> str:
        if isinstance(value, str):
            return _ebnf_literal(
                _GEMMA_STRING_DELIMITER + value + _GEMMA_STRING_DELIMITER
            )
        if value is True:
            return '"true"'
        if value is False:
            return '"false"'
        if value is None:
            return '"null"'
        if isinstance(value, (int, float)):
            return _ebnf_literal(json.dumps(value, ensure_ascii=False))
        # Complex const/enum values are uncommon in tool schemas.  Generate
        # them from their inferred schema rather than silently dropping the
        # constraint.
        if isinstance(value, list):
            items = [self._value_expr(item) for item in value]
            body = ' ws "," ws '.join(items)
            return f'"[" ws {body} ws "]"' if body else '"[" ws "]"'
        if isinstance(value, dict):
            properties = {key: {"const": item} for key, item in value.items()}
            return self._object_expr({"type": "object", "properties": properties})
        return "any_value"

    def _array_expr(self, schema: dict) -> str:
        item_expr = self._schema_expr(schema.get("items", {}))
        minimum = schema.get("minItems", 0)
        maximum = schema.get("maxItems")
        minimum = minimum if isinstance(minimum, int) and minimum >= 0 else 0
        maximum = maximum if isinstance(maximum, int) and maximum >= minimum else None

        if maximum is not None and maximum <= 16:
            variants = []
            for count in range(minimum, maximum + 1):
                body = ' ws "," ws '.join([item_expr] * count)
                variants.append(f'"[" ws {body} ws "]"' if body else '"[" ws "]"')
            return "(" + " | ".join(variants) + ")"
        if minimum == 0:
            return f'"[" ws ({item_expr} (ws "," ws {item_expr})*)? ws "]"'
        required = ' ws "," ws '.join([item_expr] * minimum)
        return f'"[" ws {required} (ws "," ws {item_expr})* ws "]"'

    def _object_expr(self, schema: dict) -> str:
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            additional = schema.get("additionalProperties", True)
            return '"{" ws "}"' if additional is False else "any_object"

        fields = []
        for key, subschema in properties.items():
            fields.append(
                f'{_ebnf_literal(str(key))} ws ":" ws {self._schema_expr(subschema)}'
            )
        required = schema.get("required")
        required_keys = set(required) if isinstance(required, list) else set()
        keys = list(properties)
        field_rules: dict[tuple[int, bool], str] = {}

        def build_fields(index: int, has_previous: bool) -> str:
            if index == len(fields):
                return '""'
            cache_key = (index, has_previous)
            if cache_key in field_rules:
                return field_rules[cache_key]

            emitted_tail = build_fields(index + 1, True)
            separator = 'ws "," ws ' if has_previous else ""
            emit = f"{separator}{fields[index]} {emitted_tail}"
            if keys[index] in required_keys:
                expression = emit
            else:
                skipped_tail = build_fields(index + 1, has_previous)
                expression = f"({skipped_tail} | ({emit}))"
            name = self._new_rule("fields", expression)
            field_rules[cache_key] = name
            return name

        body = build_fields(0, False)
        return self._new_rule("object", f'"{{" ws {body} ws "}}"')


def _build_gemma4_structural_tag(tools: list[dict], tool_choice: Any) -> str:
    """Build a native Gemma 4 tool grammar using XGrammar public formats."""
    from xgrammar.builtin_structural_tag import normalize_tool_choice as xgr_normalize
    from xgrammar.structural_tag import (
        AnyTextFormat,
        GrammarFormat,
        OptionalFormat,
        SequenceFormat,
        StructuralTag,
        TagFormat,
        TriggeredTagsFormat,
    )

    function_tools, builtin_tools, normalized_choice = xgr_normalize(
        tools, normalize_tool_choice(tool_choice)
    )
    if builtin_tools:
        raise ValueError("Gemma 4 only supports function tools")

    tags = []
    for tool in function_tools:
        function = tool.function
        parameters = function.parameters or {
            "type": "object",
            "properties": {},
        }
        grammar = _GemmaSchemaGrammar().build(parameters)
        tags.append(
            TagFormat(
                begin=f"<|tool_call>call:{function.name}",
                content=GrammarFormat(grammar=grammar),
                end="<tool_call|>",
            )
        )

    thought = OptionalFormat(
        content=TagFormat(
            begin="<|channel>thought\n",
            content=AnyTextFormat(excludes=["<channel|>"]),
            end="<channel|>",
        )
    )
    # TriggeredTagsFormat reserves the tool trigger itself. Only keep channel
    # markers out of ordinary visible text; excluding the trigger here would
    # prevent the dispatcher from consuming a tool call in auto mode.
    excludes = ["<|channel>", "<channel|>"]

    if normalized_choice == "forced":
        if len(tags) != 1:
            raise ValueError("forced Gemma 4 tool choice must resolve to one tool")
        suffix = tags[0]
    else:
        suffix = TriggeredTagsFormat(
            triggers=["<|tool_call>"],
            tags=tags,
            excludes=excludes,
            at_least_one=normalized_choice == "required",
        )

    structural_tag = StructuralTag(format=SequenceFormat(elements=[thought, suffix]))
    return structural_tag.model_dump_json(exclude_none=True)


def normalize_tool_choice(tool_choice: Any) -> Any:
    """Convert OpenAI or Anthropic tool-choice shapes to XGrammar's shape."""
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        raise ValueError("tool_choice must be a string or object")

    choice_type = tool_choice.get("type")
    if choice_type in (None, "auto"):
        return "auto"
    if choice_type == "none":
        return "none"
    if choice_type in ("required", "any"):
        return "required"
    if choice_type == "function":
        function = tool_choice.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            raise ValueError("named function tool_choice requires function.name")
        return tool_choice
    if choice_type == "tool":  # Anthropic: {"type": "tool", "name": ...}
        name = tool_choice.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Anthropic tool_choice type=tool requires name")
        return {"type": "function", "function": {"name": name}}
    raise ValueError(f"unsupported tool_choice type: {choice_type!r}")


def xgrammar_model_for_parser(
    parser_name: str,
    model_path: str = "",
    served_model_name: str = "",
) -> Optional[str]:
    """Resolve the XGrammar builtin matching the selected output parser."""
    if parser_name == "qwen3_xml":
        ident = f"{model_path} {served_model_name}".lower()
        if "coder" in ident:
            return "qwen_3_coder"
    return _PARSER_TO_XGRAMMAR_MODEL.get(parser_name)


def build_tool_structural_tag(
    *,
    parser_name: str,
    model_path: str,
    served_model_name: str,
    tools: list[dict],
    tool_choice: Any,
    reasoning: bool,
) -> str:
    """Return a serialized XGrammar StructuralTag for one tool request."""
    if not tools:
        raise ValueError("tool-call grammar requires at least one tool")
    if parser_name == "gemma4":
        if normalize_tool_choice(tool_choice) == "none":
            raise ValueError("tool-call grammar cannot be built for tool_choice=none")
        return _build_gemma4_structural_tag(tools, tool_choice)
    model = xgrammar_model_for_parser(parser_name, model_path, served_model_name)
    if model is None:
        raise UnsupportedToolGrammarError(
            f"XGrammar tool calling is not supported for parser {parser_name!r}"
        )

    normalized_choice = normalize_tool_choice(tool_choice)
    if normalized_choice == "none":
        raise ValueError("tool-call grammar cannot be built for tool_choice=none")

    try:
        from xgrammar.builtin_structural_tag import get_model_structural_tag

        structural_tag = get_model_structural_tag(
            model,
            tools=tools,
            tool_choice=normalized_choice,
            reasoning=reasoning,
            any_order=False,
            exclude_special_tokens=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid tool schema or tool_choice: {exc}") from exc
    return structural_tag.model_dump_json(exclude_none=True)
