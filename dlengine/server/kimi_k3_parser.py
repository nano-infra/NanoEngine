"""Incremental K3 XTML channel parsing.

Each request owns its channel state and unfinished header/argument. Only
transitions valid in that state consume framing. Response/reasoning text and
completed calls are returned as structured deltas, never filtered HTTP text.

As in other text-based XTML parsers, a bare closing delimiter inside its own
channel body is ambiguous with the real delimiter. This module does not claim
to recover an escaping distinction absent from the decoded stream.
"""

import json

from dlengine.server.tool_parser import Function, ParsedDelta, ToolCall


class KimiK3StreamParser:
    OPEN = "<|open|>"
    CLOSE = "<|close|>"
    SEP = "<|sep|>"
    ARGUMENT_CLOSE = "<|close|>argument<|sep|>"

    def __init__(self, grammar, *, reasoning_open=False):
        self.grammar = grammar
        self.state = "think" if reasoning_open else "start"
        self.pending = ""
        self.reasoning_emitted = False
        self.call_name = ""
        self.arguments = {}
        self.argument_attrs = {}
        self.argument_parts = []

    @staticmethod
    def _overlap(text, markers):
        longest = 0
        for marker in markers:
            for size in range(min(len(text), len(marker) - 1), longest, -1):
                if text.endswith(marker[:size]):
                    longest = size
                    break
        return longest

    def _text(self, text):
        if not text:
            return []
        if self.state == "think":
            self.reasoning_emitted = True
            return [ParsedDelta(reasoning=text)]
        if self.state in ("start", "between"):
            # Whitespace between protocol channels is not response content.
            if not text.strip():
                return []
            self.state = "response"
        return [ParsedDelta(content=text)]

    def _transitions(self):
        g = self.grammar
        transitions = {
            g.MESSAGE_CLOSE: "done",
            g.END_OF_MESSAGE: "done",
            g.TOOLS_OPEN: "tools",
        }
        if self.state == "think":
            transitions.update({g.THINK_CLOSE: "between", g.RESPONSE_OPEN: "response"})
            if not self.reasoning_emitted:
                transitions[g.THINK_OPEN] = "think"
        elif self.state == "response":
            transitions[g.RESPONSE_CLOSE] = "between"
        else:
            transitions.update(
                {
                    g.THINK_OPEN: "think",
                    g.THINK_CLOSE: "between",
                    g.RESPONSE_OPEN: "response",
                    g.RESPONSE_CLOSE: "between",
                }
            )
        return transitions

    def _channel_step(self):
        transitions = self._transitions()
        found = [(self.pending.find(marker), marker) for marker in transitions]
        found = [(position, marker) for position, marker in found if position >= 0]
        if found:
            position, marker = min(found)
            prefix = self.pending[:position]
            if self.state in ("start", "between") and prefix.strip():
                # Plain response text establishes the channel before considering
                # a later marker. Re-evaluate it in that state for chunk invariance.
                self.pending = self.pending[position:]
                return self._text(prefix), True
            events = self._text(prefix)
            self.pending = self.pending[position + len(marker) :]
            self.state = transitions[marker]
            return events, True
        keep = self._overlap(self.pending, transitions)
        size = len(self.pending) - keep
        if not size or (
            self.state in ("start", "between") and self.pending[:size].isspace()
        ):
            return [], False
        text, self.pending = self.pending[:size], self.pending[size:]
        return self._text(text), True

    def _argument_step(self):
        end = self.pending.find(self.ARGUMENT_CLOSE)
        if end < 0:
            keep = self._overlap(self.pending, [self.ARGUMENT_CLOSE])
            size = len(self.pending) - keep
            self.argument_parts.append(self.pending[:size])
            self.pending = self.pending[size:]
            return [], bool(size)
        self.argument_parts.append(self.pending[:end])
        self.pending = self.pending[end + len(self.ARGUMENT_CLOSE) :]
        raw = "".join(self.argument_parts)
        key = self.argument_attrs.get("key")
        if key:
            value = raw
            if self.argument_attrs.get("type", "string") != "string":
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    # Preserve the existing full-parser malformed-JSON behavior.
                    pass
            self.arguments[key] = value
        self.argument_parts = []
        self.state = "call"
        return [], True

    def _container_step(self):
        markers = [self.OPEN, self.CLOSE, self.grammar.END_OF_MESSAGE]
        found = [(self.pending.find(marker), marker) for marker in markers]
        found = [(position, marker) for position, marker in found if position >= 0]
        if not found:
            keep = self._overlap(self.pending, markers)
            self.pending = self.pending[-keep:] if keep else ""
            return [], False
        position, marker = min(found)
        self.pending = self.pending[position:]
        if marker == self.grammar.END_OF_MESSAGE:
            self.state = "done"
            self.pending = ""
            return [], True
        end = self.pending.find(self.SEP, len(marker))
        if end < 0:
            return [], False
        header = self.pending[len(marker) : end].strip().split(None, 1)
        self.pending = self.pending[end + len(self.SEP) :]
        if not header:
            return [], True
        name = header[0]
        attrs = self.grammar._attrs(header[1] if len(header) > 1 else "")
        if marker == self.OPEN and name == "call":
            self.call_name = attrs.get("tool", "")
            self.arguments = {}
            self.state = "call"
        elif self.state == "call" and marker == self.OPEN and name == "argument":
            self.argument_attrs = attrs
            self.argument_parts = []
            self.state = "argument"
        elif self.state == "call" and marker == self.CLOSE and name == "call":
            self.state = "tools"
            if self.call_name:
                call = ToolCall(
                    function=Function(
                        name=self.call_name,
                        arguments=json.dumps(self.arguments, ensure_ascii=False),
                    )
                )
                self.call_name = ""
                self.arguments = {}
                return [ParsedDelta(tool_calls=[call])], True
        elif marker == self.CLOSE and name == "tools":
            self.state = "between"
            self.call_name = ""
            self.arguments = {}
        elif marker == self.CLOSE and name == "message":
            self.state = "done"
        return [], True

    def feed(self, delta):
        self.pending += delta
        events = []
        while self.pending:
            if self.state == "done":
                self.pending = ""
                break
            if self.state == "argument":
                new, progressed = self._argument_step()
            elif self.state in ("tools", "call"):
                new, progressed = self._container_step()
            else:
                new, progressed = self._channel_step()
            events.extend(new)
            if not progressed:
                break
        return events

    def finish(self):
        events = []
        if self.state in ("start", "between", "think", "response"):
            # Keep ambiguous prose such as a lone '<', but do not expose an
            # unfinished protocol header when generation hits its token limit.
            if not self.pending.startswith("<|"):
                events.extend(self._text(self.pending))
        self.pending = ""
        self.argument_parts = []
        self.arguments = {}
        self.state = "done"
        return events
