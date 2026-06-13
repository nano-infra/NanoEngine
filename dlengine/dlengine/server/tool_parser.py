"""Tool-call parsers for OpenAI function-calling compatibility.

Different model families emit tool calls in different surface syntaxes. This
module turns the raw generated text into the OpenAI ``tool_calls`` shape so the
serving layer (:mod:`dlengine.server.openai_server`) can return function calls
that agent scaffolds (SWE-bench, etc.) understand.

Two formats are supported:

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
        return None, text
    open_idx = text.find("<think>")
    if 0 <= open_idx < close:
        reasoning = text[open_idx + len("<think>") : close]
        rest = text[:open_idx] + text[close + len("</think>") :]
    else:
        reasoning = text[:close]
        rest = text[close + len("</think>") :]
    return (reasoning.strip() or None), rest


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


_REGISTRY: dict[str, type[ToolParser]] = {
    "hermes": HermesToolParser,
    "qwen3_xml": Qwen3XMLToolParser,
    "qwen3_coder": Qwen3XMLToolParser,
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
    templates instruct the model to emit ``<function=...>`` XML, while Hermes
    templates use a JSON object inside ``<tool_call>``. Fall back to the model
    path, then to Hermes.
    """
    if chat_template:
        if "<function=" in chat_template:
            return "qwen3_xml"
        if "<tool_call>" in chat_template:
            return "hermes"
    ident = f"{model_path} {served_model_name}".lower()
    if "qwen3.5" in ident or "qwen3-coder" in ident or "qwen3_coder" in ident:
        return "qwen3_xml"
    return "hermes"
