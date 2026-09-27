"""JSONL trajectory log and the replay viewer that reads it back.

One JSON object per line: ``{"t": seconds since start, "step": n, "type": ..., ...}``.
Event types: ``run_start``, ``compaction``, ``model``, ``tool_call``, ``approval``,
``tool_result``, ``run_end``. Lines are flushed as they are written, so a crashed or
killed run still leaves a readable log up to the last event.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


class TrajectoryLogger:
    def __init__(self, path: Path | None,
                 listener: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.path = path
        self.listener = listener
        self._fh: TextIO | None = None
        self._t0 = time.monotonic()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8", newline="\n")

    def log(self, type_: str, step: int, **payload: Any) -> None:
        event = {"t": round(time.monotonic() - self._t0, 3), "step": step, "type": type_,
                 **payload}
        if self.listener is not None:
            self.listener(event)
        if self._fh is not None:
            self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def load(path: Path) -> list[dict[str, Any]]:
    """Read a trajectory, skipping (and counting) lines that are not valid JSON."""
    events = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"type": "corrupt_line", "step": -1, "raw": line[:200]})
    return events


@dataclass
class Summary:
    runs: int = 0
    status: list[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: Counter[str] = field(default_factory=Counter)
    tool_errors: Counter[str] = field(default_factory=Counter)
    approvals: Counter[str] = field(default_factory=Counter)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    compactions: int = 0
    seconds: float = 0.0
    corrupt_lines: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs, "status": self.status, "steps": self.steps,
            "tool_calls": dict(self.tool_calls), "tool_errors": dict(self.tool_errors),
            "approvals": dict(self.approvals), "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "compactions": self.compactions,
            "seconds": self.seconds, "corrupt_lines": self.corrupt_lines,
        }


def summarize(events: list[dict[str, Any]]) -> Summary:
    s = Summary()
    for e in events:
        kind = e.get("type")
        if kind == "run_start":
            s.runs += 1
        elif kind == "model":
            s.steps += 1
            s.prompt_tokens += e.get("prompt_tokens") or 0
            s.completion_tokens += e.get("completion_tokens") or 0
        elif kind == "tool_call":
            s.tool_calls[e.get("name", "?")] += 1
        elif kind == "tool_result" and not e.get("ok", True):
            s.tool_errors[e.get("name", "?")] += 1
        elif kind == "approval":
            s.approvals[e.get("decision", "?")] += 1
        elif kind == "compaction":
            s.compactions += 1
        elif kind == "run_end":
            s.status.append(e.get("status", "?"))
            s.seconds = e.get("t", s.seconds)
        elif kind == "corrupt_line":
            s.corrupt_lines += 1
    return s


def _clip(text: str, n: int) -> str:
    text = text.rstrip()
    return text if len(text) <= n else text[:n] + f" ... [{len(text) - n} more chars]"


def render(events: list[dict[str, Any]], full: bool = False, width: int = 600) -> str:
    """Human-readable transcript of a trajectory."""
    limit = 10**9 if full else width
    out: list[str] = []
    for e in events:
        kind, step = e.get("type"), e.get("step")
        if kind == "run_start":
            out.append(f"=== run start  model={e.get('model')}  workspace={e.get('workspace')}")
            out.append(f"task: {_clip(e.get('task', ''), limit)}")
        elif kind == "compaction":
            out.append(f"--- step {step}: context compacted {e.get('tokens_before')} -> "
                       f"{e.get('tokens_after')} est. tokens (stubbed {e.get('stubbed')}, "
                       f"dropped {e.get('dropped')}, squeezed {e.get('squeezed')})")
        elif kind == "model":
            tok = (f", {e.get('prompt_tokens')}+{e.get('completion_tokens')} tokens"
                   if e.get("prompt_tokens") is not None else "")
            cached = " (cached)" if e.get("cached") else ""
            out.append(f"\n[step {step}] model{cached}{tok}")
            if e.get("content"):
                out.append("  says: " + _clip(e["content"], limit).replace("\n", "\n        "))
        elif kind == "tool_call":
            args = json.dumps(e.get("arguments", {}), ensure_ascii=False)
            out.append(f"  -> {e.get('name')}({_clip(args, limit)})")
        elif kind == "approval":
            out.append(f"     approval: {e.get('decision')} [{e.get('risk')}] {e.get('reason')}")
        elif kind == "tool_result":
            mark = "ok" if e.get("ok") else "FAILED"
            body = _clip(e.get("output", ""), limit).replace("\n", "\n       ")
            out.append(f"  <- {mark}: {body}")
        elif kind == "run_end":
            out.append(f"\n=== run end: {e.get('status')} after {e.get('steps')} steps, "
                       f"{e.get('t')}s")
            if e.get("final"):
                out.append(f"final: {_clip(e['final'], limit)}")
        elif kind == "corrupt_line":
            out.append(f"  !! unreadable log line: {e.get('raw')}")
    return "\n".join(out)
