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

Reasoning ("thinking") models wrap a chain-of-thought in ``<think> ...
</think>`` before the answer. The chat template usually injects the opening
``<think>`` into the prompt, so the generated text often contains only the
closing ``</think>``; we strip the reasoning into a separate field either way.
"""

from __future__ import annotations

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


class ToolParser:
    """Base class. Subclasses implement :meth:`parse_full`."""

    # Token markers the streaming layer holds back so partial tags never leak
    # into content deltas.
    open_markers: tuple[str, ...] = ()
    reasoning_close_marker = "</think>"

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
                args[p.group(1).strip()] = _coerce_value(p.group(2))
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
                key = p.group(1).strip()
                if not key:
                    continue
                args[key] = _coerce_value(p.group(2))
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


class KimiK3ToolParser(ToolParser):
    """Parser for Kimi K3's XTML response, reasoning, and tool channels."""

    THINK_OPEN = "<|open|>think<|sep|>"
    THINK_CLOSE = "<|close|>think<|sep|>"
    RESPONSE_OPEN = "<|open|>response<|sep|>"
    RESPONSE_CLOSE = "<|close|>response<|sep|>"
    TOOLS_OPEN = "<|open|>tools<|sep|>"
    TOOLS_CLOSE = "<|close|>tools<|sep|>"
    MESSAGE_CLOSE = "<|close|>message<|sep|>"
    reasoning_close_marker = THINK_CLOSE
    open_markers = (THINK_OPEN, TOOLS_OPEN)
    _CALL_RE = re.compile(
        r'<\|open\|>call\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>'
        r'(?P<body>.*?)<\|close\|>call<\|sep\|>', re.DOTALL
    )
    _ARG_RE = re.compile(
        r'<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>'
        r'(?P<val>.*?)<\|close\|>argument<\|sep\|>', re.DOTALL
    )
    _ATTR_RE = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')

    @classmethod
    def _attrs(cls, raw: str) -> dict[str, str]:
        return {
            m["k"]: m["v"].replace("&quot;", '"').replace("&amp;", "&")
            for m in cls._ATTR_RE.finditer(raw)
        }

    @classmethod
    def _unwrap_response(cls, text: str) -> str:
        start = text.find(cls.RESPONSE_OPEN)
        if start >= 0:
            start += len(cls.RESPONSE_OPEN)
            end = text.find(cls.RESPONSE_CLOSE, start)
            text = text[start:] if end < 0 else text[start:end]
        else:
            text = text.replace(cls.RESPONSE_CLOSE, "")
        return text.replace(cls.MESSAGE_CLOSE, "")

    def parse_full(self, text: str) -> ParsedOutput:
        reasoning = None
        think_start = text.find(self.THINK_OPEN)
        content_start = think_start + len(self.THINK_OPEN) if think_start >= 0 else 0
        think_end = text.find(self.THINK_CLOSE, content_start)
        if think_end >= 0:
            reasoning = text[content_start:think_end].strip() or None
            text = text[think_end + len(self.THINK_CLOSE):]

        tools_start = text.find(self.TOOLS_OPEN)
        normal = text if tools_start < 0 else text[:tools_start]
        section = "" if tools_start < 0 else text[tools_start + len(self.TOOLS_OPEN):]
        tools_end = section.find(self.TOOLS_CLOSE)
        if tools_end >= 0:
            section = section[:tools_end]
        calls: list[ToolCall] = []
        for match in self._CALL_RE.finditer(section):
            name = self._attrs(match["attrs"]).get("tool")
            if not name:
                continue
            arguments = {}
            for arg in self._ARG_RE.finditer(match["body"]):
                attrs = self._attrs(arg["attrs"])
                key = attrs.get("key")
                if not key:
                    continue
                raw = arg["val"]
                if attrs.get("type", "string") == "string":
                    arguments[key] = raw
                else:
                    try:
                        arguments[key] = json.loads(raw)
                    except json.JSONDecodeError:
                        arguments[key] = raw
            calls.append(ToolCall(function=Function(
                name=name, arguments=json.dumps(arguments, ensure_ascii=False)
            )))
        content = self._unwrap_response(normal).strip()
        return ParsedOutput(content=content or None, tool_calls=calls, reasoning=reasoning)


_REGISTRY: dict[str, type[ToolParser]] = {
    "hermes": HermesToolParser,
    "qwen3_xml": Qwen3XMLToolParser,
    "qwen3_coder": Qwen3XMLToolParser,
    "glm": GLMXMLToolParser,
    "glm45": GLMXMLToolParser,
    "glm47": GLMXMLToolParser,
    "glm5": GLMXMLToolParser,
    "kimi_k3": KimiK3ToolParser,
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
        if "<arg_key>" in chat_template or "<arg_value>" in chat_template:
            return "glm"
        if "<function=" in chat_template:
            return "qwen3_xml"
        if "<tool_call>" in chat_template:
            return "hermes"
    ident = f"{model_path} {served_model_name}".lower()
    if "kimi-k3" in ident or "kimi_k3" in ident:
        return "kimi_k3"
    if "glm" in ident:
        return "glm"
    if "qwen3.5" in ident or "qwen3-coder" in ident or "qwen3_coder" in ident:
        return "qwen3_xml"
    return "hermes"
