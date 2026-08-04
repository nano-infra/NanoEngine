"""Tool-call parsers for OpenAI function-calling compatibility.

Different model families emit tool calls in different surface syntaxes. This
module turns the raw generated text into the OpenAI ``tool_calls`` shape so the
serving layer (:mod:`dlengine.server.openai_server`) can return function calls
that agent scaffolds (SWE-bench, etc.) understand.

The supported formats are:

- *Hermes* (Qwen2.5 and earlier Qwen3 chat templates): each call is a JSON
  object wrapped in ``<tool_call> ... </tool_call>``::

      <tool_call>
      {"name": "get_weather", "arguments": {"city": "Tokyo"}}
      </tool_call>

- *Qwen3.5 / Qwen3-Coder XML*: an inner ``<function=NAME>`` block with one
  ``<parameter=NAME>`` element per argument, nested in ``<tool_call>``::

      <tool_call>
      <function=get_weather>
      <parameter=city>
      Tokyo
      </parameter>
      </function>
      </tool_call>

- *GLM-4.5+ / GLM-5 XML*: the function name immediately follows
  ``<tool_call>``, with ``<arg_key>`` / ``<arg_value>`` pairs for arguments::

      <tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>

- *DeepSeek V3.1 / V3.2 / V4*: either special-token JSON calls or DSML XML
  ``invoke`` / ``parameter`` blocks.

- *Gemma 4*: channel-style reasoning followed by a native function call whose
  string values use the reserved ``<|"|>`` delimiter::

      <|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>

Reasoning ("thinking") models wrap a chain-of-thought in ``<think> ...
</think>`` before the answer. The chat template usually injects the opening
``<think>`` into the prompt, so the generated text often contains only the
closing ``</think>``; we strip the reasoning into a separate field either way.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Function:
    name: str
    arguments: str  # JSON-encoded string, per the OpenAI schema

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolCall:
    function: Function
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: str = "function"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict(),
        }


@dataclass
class ParsedOutput:
    """Result of parsing a full model completion."""

    content: Optional[str]
    tool_calls: list[ToolCall]
    reasoning: Optional[str] = None


def _extract_reasoning(text: str) -> tuple[Optional[str], str]:
    """Split a leading <think>...</think> region off ``text``.

    Thinking templates typically inject the opening ``<think>`` into the prompt,
    so the generated text contains only the trailing ``</think>``. Handle both
    the paired form and the closing-only form: everything up to the first
    ``</think>`` is treated as reasoning, the rest is the answer.
    """
    close = text.find("</think>")
    if close == -1:
        unfinished = text.strip()
        if not unfinished.startswith("<think>"):
            return None, text
        while unfinished.startswith("<think>"):
            unfinished = unfinished[len("<think>") :].lstrip()
        return (unfinished or None), ""
    open_idx = text.find("<think>")
    if 0 <= open_idx < close:
        reasoning = text[open_idx + len("<think>") : close]
        rest = text[:open_idx] + text[close + len("</think>") :]
    else:
        reasoning = text[:close]
        rest = text[close + len("</think>") :]
    reasoning = reasoning.strip()
    while reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>") :].lstrip()
    return (reasoning or None), rest


def _coerce_value(raw: str):
    """Best-effort typing of an XML parameter value.

    Qwen3-Coder emits parameter values as raw text. Try JSON so numbers,
    booleans, null, arrays and objects become typed; otherwise keep the
    stripped string (e.g. ``Tokyo``).
    """
    s = raw.strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


class _GemmaValueParser:
    """Parse Gemma 4's JSON-like structured-data notation."""

    _STRING_DELIMITER = '<|"|>'

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def parse(self):
        value = self._parse_value()
        self._skip_whitespace()
        if self.pos != len(self.text):
            raise ValueError(f"unexpected Gemma value at offset {self.pos}")
        return value

    def _skip_whitespace(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _consume(self, expected: str) -> None:
        if not self.text.startswith(expected, self.pos):
            raise ValueError(f"expected {expected!r} at offset {self.pos}")
        self.pos += len(expected)

    def _parse_value(self):
        self._skip_whitespace()
        if self.text.startswith(self._STRING_DELIMITER, self.pos):
            return self._parse_gemma_string()
        if self.pos >= len(self.text):
            raise ValueError("unexpected end of Gemma value")
        char = self.text[self.pos]
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        if char == '"':
            return self._parse_json_string()
        return self._parse_scalar()

    def _parse_gemma_string(self) -> str:
        self._consume(self._STRING_DELIMITER)
        end = self.text.find(self._STRING_DELIMITER, self.pos)
        if end == -1:
            raise ValueError("unterminated Gemma string")
        value = self.text[self.pos : end]
        self.pos = end + len(self._STRING_DELIMITER)
        return value

    def _parse_json_string(self) -> str:
        decoder = json.JSONDecoder()
        value, consumed = decoder.raw_decode(self.text[self.pos :])
        if not isinstance(value, str):
            raise ValueError("expected a string")
        self.pos += consumed
        return value

    def _parse_key(self) -> str:
        self._skip_whitespace()
        if self.text.startswith(self._STRING_DELIMITER, self.pos):
            return self._parse_gemma_string()
        if self.pos < len(self.text) and self.text[self.pos] == '"':
            return self._parse_json_string()
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ":,}":
            self.pos += 1
        key = self.text[start : self.pos].strip()
        if not key:
            raise ValueError(f"empty Gemma object key at offset {start}")
        return key

    def _parse_object(self) -> dict:
        self._consume("{")
        result = {}
        self._skip_whitespace()
        if self.pos < len(self.text) and self.text[self.pos] == "}":
            self.pos += 1
            return result
        while True:
            key = self._parse_key()
            self._skip_whitespace()
            self._consume(":")
            result[key] = self._parse_value()
            self._skip_whitespace()
            if self.pos < len(self.text) and self.text[self.pos] == "}":
                self.pos += 1
                return result
            self._consume(",")

    def _parse_array(self) -> list:
        self._consume("[")
        result = []
        self._skip_whitespace()
        if self.pos < len(self.text) and self.text[self.pos] == "]":
            self.pos += 1
            return result
        while True:
            result.append(self._parse_value())
            self._skip_whitespace()
            if self.pos < len(self.text) and self.text[self.pos] == "]":
                self.pos += 1
                return result
            self._consume(",")

    def _parse_scalar(self):
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ",]}":
            self.pos += 1
        raw = self.text[start : self.pos].strip()
        if not raw:
            raise ValueError(f"empty Gemma value at offset {start}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


class ToolParser:
    """Base class. Subclasses implement :meth:`parse_full`."""

    # Token markers the streaming layer holds back so partial tags never leak
    # into content deltas.
    open_markers: tuple[str, ...] = ()

    def parse_full(self, text: str) -> ParsedOutput:  # pragma: no cover - abstract
        raise NotImplementedError


class HermesToolParser(ToolParser):
    """Parser for the Hermes JSON format (Qwen2.5 / older Qwen3 templates)."""

    open_markers = ("<tool_call>", "<think>")

    _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def parse_full(self, text: str) -> ParsedOutput:
        reasoning, text = _extract_reasoning(text)

        tool_calls: list[ToolCall] = []
        for m in self._TOOL_CALL_RE.finditer(text):
            raw = m.group(1).strip()
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = obj.get("arguments", {})
            arguments = args if isinstance(args, str) else json.dumps(args)
            tool_calls.append(
                ToolCall(function=Function(name=name, arguments=arguments))
            )

        content = self._TOOL_CALL_RE.sub("", text).strip()
        return ParsedOutput(
            content=content or None, tool_calls=tool_calls, reasoning=reasoning
        )


class Qwen3XMLToolParser(ToolParser):
    """Parser for the Qwen3.5 / Qwen3-Coder XML function-call format."""

    open_markers = ("<tool_call>", "<think>")

    _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    _FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
    _PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)

    def parse_full(self, text: str) -> ParsedOutput:
        reasoning, text = _extract_reasoning(text)

        tool_calls: list[ToolCall] = []
        for tc in self._TOOL_CALL_RE.finditer(text):
            fn = self._FUNC_RE.search(tc.group(1))
            if fn is None:
                continue
            name = fn.group(1).strip()
            if not name:
                continue
            args: dict = {}
            for p in self._PARAM_RE.finditer(fn.group(2)):
                args[html.unescape(p.group(1).strip())] = _coerce_value(
                    html.unescape(p.group(2))
                )
            tool_calls.append(
                ToolCall(function=Function(name=name, arguments=json.dumps(args)))
            )

        content = self._TOOL_CALL_RE.sub("", text).strip()
        return ParsedOutput(
            content=content or None, tool_calls=tool_calls, reasoning=reasoning
        )


class GLMXMLToolParser(ToolParser):
    """Parser for the GLM-4.5 / GLM-4.6 / GLM-4.7 / GLM-5 XML format."""

    open_markers = ("<tool_call>", "<think>")

    _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    _ARG_RE = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
        re.DOTALL,
    )

    def parse_full(self, text: str) -> ParsedOutput:
        reasoning, text = _extract_reasoning(text)

        tool_calls: list[ToolCall] = []
        for tc in self._TOOL_CALL_RE.finditer(text):
            block = tc.group(1).strip()
            if not block:
                continue

            first_arg = block.find("<arg_key>")
            if first_arg == -1:
                name = block.strip()
                arg_text = ""
            else:
                name = block[:first_arg].strip()
                arg_text = block[first_arg:]
            if not name:
                continue

            args: dict = {}
            for p in self._ARG_RE.finditer(arg_text):
                key = html.unescape(p.group(1).strip())
                if not key:
                    continue
                args[key] = _coerce_value(html.unescape(p.group(2)))
            tool_calls.append(
                ToolCall(
                    function=Function(
                        name=name, arguments=json.dumps(args, ensure_ascii=False)
                    )
                )
            )

        content = self._TOOL_CALL_RE.sub("", text).strip()
        return ParsedOutput(
            content=content or None, tool_calls=tool_calls, reasoning=reasoning
        )


class DeepSeekV31ToolParser(ToolParser):
    """Parser for DeepSeek-V3.1 special-token JSON tool calls."""

    _CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
    _CALLS_END = "<｜tool▁calls▁end｜>"
    _CALL_BEGIN = "<｜tool▁call▁begin｜>"
    _CALL_END = "<｜tool▁call▁end｜>"
    _TOOL_SEP = "<｜tool▁sep｜>"
    open_markers = (_CALLS_BEGIN, "<think>")

    _CALL_RE = re.compile(
        re.escape(_CALL_BEGIN)
        + r"(.*?)"
        + re.escape(_TOOL_SEP)
        + r"(.*?)"
        + re.escape(_CALL_END),
        re.DOTALL,
    )
    _CALLS_RE = re.compile(
        re.escape(_CALLS_BEGIN) + r".*?" + re.escape(_CALLS_END), re.DOTALL
    )

    def parse_full(self, text: str) -> ParsedOutput:
        reasoning, text = _extract_reasoning(text)
        tool_calls: list[ToolCall] = []
        for match in self._CALL_RE.finditer(text):
            name = match.group(1).strip()
            if not name:
                continue
            try:
                args = json.loads(match.group(2).strip())
            except json.JSONDecodeError:
                continue
            arguments = (
                args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            )
            tool_calls.append(
                ToolCall(function=Function(name=name, arguments=arguments))
            )

        content = self._CALLS_RE.sub("", text).strip()
        return ParsedOutput(
            content=content or None, tool_calls=tool_calls, reasoning=reasoning
        )


class DeepSeekDSMLToolParser(ToolParser):
    """Parser for DeepSeek-V3.2/V4 DSML XML tool calls."""

    open_markers = (
        "<｜DSML｜function_calls>",
        "<｜DSML｜tool_calls>",
        "<think>",
    )
    _INVOKE_RE = re.compile(
        r'<｜DSML｜invoke\s+name="([^"]+)">\s*(.*?)</｜DSML｜invoke>',
        re.DOTALL,
    )
    _PARAM_RE = re.compile(
        r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)">'
        r"\s*(.*?)\s*</｜DSML｜parameter>",
        re.DOTALL,
    )
    _CALLS_RE = re.compile(
        r"<｜DSML｜(?:function_calls|tool_calls)>.*?"
        r"</｜DSML｜(?:function_calls|tool_calls)>",
        re.DOTALL,
    )

    def parse_full(self, text: str) -> ParsedOutput:
        reasoning, text = _extract_reasoning(text)
        tool_calls: list[ToolCall] = []
        for invoke in self._INVOKE_RE.finditer(text):
            name = html.unescape(invoke.group(1).strip())
            if not name:
                continue
            args: dict = {}
            for param in self._PARAM_RE.finditer(invoke.group(2)):
                key = html.unescape(param.group(1).strip())
                raw = html.unescape(param.group(3).strip())
                args[key] = raw if param.group(2) == "true" else _coerce_value(raw)
            tool_calls.append(
                ToolCall(
                    function=Function(
                        name=name, arguments=json.dumps(args, ensure_ascii=False)
                    )
                )
            )

        content = self._CALLS_RE.sub("", text).strip()
        return ParsedOutput(
            content=content or None, tool_calls=tool_calls, reasoning=reasoning
        )


class Gemma4ToolParser(ToolParser):
    """Parser for Gemma 4 channel and native structured-data tool calls."""

    open_markers = ("<|tool_call>", "<|channel>")
    _REASONING_RE = re.compile(
        r"<\|channel>thought(?:\r?\n)?(.*?)<channel\|>", re.DOTALL
    )
    _CALL_RE = re.compile(r"<\|tool_call>call:(.*?)<tool_call\|>", re.DOTALL)

    def parse_full(self, text: str) -> ParsedOutput:
        reasoning_match = self._REASONING_RE.search(text)
        reasoning = None
        if reasoning_match is not None:
            reasoning = reasoning_match.group(1).strip() or None
            text = self._REASONING_RE.sub("", text, count=1)

        tool_calls: list[ToolCall] = []
        for match in self._CALL_RE.finditer(text):
            block = match.group(1).strip()
            args_begin = block.find("{")
            if args_begin == -1:
                continue
            name = block[:args_begin].strip()
            if not name:
                continue
            try:
                args = _GemmaValueParser(block[args_begin:]).parse()
            except ValueError:
                continue
            if not isinstance(args, dict):
                continue
            tool_calls.append(
                ToolCall(
                    function=Function(
                        name=name, arguments=json.dumps(args, ensure_ascii=False)
                    )
                )
            )

        content = self._CALL_RE.sub("", text).replace("<turn|>", "").strip()
        return ParsedOutput(
            content=content or None, tool_calls=tool_calls, reasoning=reasoning
        )


_REGISTRY: dict[str, type[ToolParser]] = {
    "hermes": HermesToolParser,
    "qwen3_xml": Qwen3XMLToolParser,
    "qwen3_coder": Qwen3XMLToolParser,
    "glm": GLMXMLToolParser,
    "glm45": GLMXMLToolParser,
    "glm47": GLMXMLToolParser,
    "glm5": GLMXMLToolParser,
    "deepseek_v3_1": DeepSeekV31ToolParser,
    "deepseek_v3_2": DeepSeekDSMLToolParser,
    "deepseek_v4": DeepSeekDSMLToolParser,
    "gemma4": Gemma4ToolParser,
}


def get_tool_parser(name: Optional[str]) -> ToolParser:
    """Return a parser instance for ``name`` (defaults to Hermes)."""
    key = (name or "hermes").lower()
    cls = _REGISTRY.get(key, HermesToolParser)
    return cls()


def detect_parser_name(
    model_path: str, served_model_name: str, chat_template: Optional[str] = None
) -> str:
    """Heuristically pick a parser.

    The most reliable signal is the chat template itself: Qwen3.5 / Qwen3-Coder
    templates instruct the model to emit ``<function=...>`` XML, GLM-4.5+
    templates use ``<arg_key>`` / ``<arg_value>`` pairs, while Hermes templates
    use a JSON object inside ``<tool_call>``. Fall back to the model path, then
    to Hermes.
    """
    if chat_template:
        if "<|tool_call>" in chat_template or "<|channel>thought" in chat_template:
            return "gemma4"
        if "<｜DSML｜tool_calls>" in chat_template:
            return "deepseek_v4"
        if "<｜DSML｜function_calls>" in chat_template:
            return "deepseek_v3_2"
        if "<｜tool▁calls▁begin｜>" in chat_template:
            return "deepseek_v3_1"
        if "<arg_key>" in chat_template or "<arg_value>" in chat_template:
            return "glm"
        if "<function=" in chat_template:
            return "qwen3_xml"
        if "<tool_call>" in chat_template:
            return "hermes"
    ident = f"{model_path} {served_model_name}".lower()
    if "gemma-4" in ident or "gemma4" in ident or "gemma_4" in ident:
        return "gemma4"
    if "deepseek" in ident:
        if "v4" in ident:
            return "deepseek_v4"
        if "v3.2" in ident or "v3_2" in ident:
            return "deepseek_v3_2"
        if "v3.1" in ident or "v3_1" in ident:
            return "deepseek_v3_1"
    if "glm" in ident:
        return "glm"
    if "qwen3.5" in ident or "qwen3-coder" in ident or "qwen3_coder" in ident:
        return "qwen3_xml"
    return "hermes"
