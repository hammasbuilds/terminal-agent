"""Tool-call data type and the fallback parser for tool calls written as text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCED = re.compile(r"```(?:json|tool_call)?\s*\n(.*?)\n```", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def signature(self) -> str:
        """Canonical string used for repeated-call (loop) detection."""
        return self.name + json.dumps(self.arguments, sort_keys=True, ensure_ascii=False)


def _as_call(obj: Any, idx: int) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        fn = obj.get("function")
        if isinstance(fn, dict):
            return _as_call(fn, idx)
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return ToolCall(name=name, arguments=args, id=f"t{idx}")


def parse_text_tool_calls(content: str) -> list[ToolCall]:
    """Extract tool calls a model wrote as JSON text instead of native tool calls.

    Accepts ``<tool_call>{...}</tool_call>`` blocks, fenced ```json blocks, or a message
    that is itself a single JSON object/list. Only objects with a string ``name`` and
    dict ``arguments`` count; anything else is ignored rather than guessed at.
    """
    chunks = _TAGGED.findall(content) or _FENCED.findall(content)
    if not chunks and content.strip().startswith(("{", "[")):
        chunks = [content.strip()]
    calls: list[ToolCall] = []
    for chunk in chunks:
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            call = _as_call(item, len(calls))
            if call is not None:
                calls.append(call)
    return calls
