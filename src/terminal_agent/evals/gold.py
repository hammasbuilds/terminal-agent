"""A scripted "model" that applies a gold patch through the agent's own tools.

It behaves like a careful model would: ``read_file`` each target (paging through long
files), then one ``edit`` per hunk using the hunk's own lines as ``old_string``. When the
edit tool answers "occurs N times", it widens the snippet with lines it read until the
snippet is unique, and tries again. New files go through ``write_file``, deletions through
``run_shell``. Every call passes through the policy, the tool layer and the trajectory
log exactly as a model's would; nothing is written behind the agent's back.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from terminal_agent.evals.patches import FilePatch, Hunk, parse_patch
from terminal_agent.llm import ModelTurn
from terminal_agent.protocol import ToolCall

_HEADER = re.compile(r"^\[(?P<path>.+?): lines (?P<a>\d+)-(?P<b>\d+) of (?P<n>\d+)[^\]]*\]\n")
MAX_EXTRA_CONTEXT = 60


@dataclass
class HunkRecord:
    path: str
    index: int
    first_status: str = ""
    final_status: str = ""
    extra_context: int = 0
    attempts: int = 0


@dataclass
class _FileState:
    patch: FilePatch
    lines: list[str] = field(default_factory=list)
    total: int | None = None
    delta: int = 0  # line shift from hunks already applied


class GoldPatchClient:
    model = "gold-patch-script"

    def __init__(self, patch: str) -> None:
        self.files = [_FileState(f) for f in parse_patch(patch) if not f.is_binary]
        self.records: list[HunkRecord] = []
        self._fi = 0
        self._hi = 0
        self._extra = 0
        self._pending: tuple[str, Any] | None = None
        self._n = 0

    # -- ChatClient protocol -------------------------------------------------------
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        last = messages[-1] if messages and messages[-1].get("role") == "tool" else None
        if last is not None and self._pending is not None:
            self._absorb(str(last.get("content", "")))
        call = self._next_call()
        self._n += 1
        if call is None:
            return ModelTurn(
                content="",
                tool_calls=[
                    ToolCall("finish", {"summary": "applied the reference patch"}, f"g{self._n}")
                ],
            )
        call.id = f"g{self._n}"
        return ModelTurn(content="", tool_calls=[call])

    # -- state machine ---------------------------------------------------------------
    def _next_call(self) -> ToolCall | None:
        while self._fi < len(self.files):
            fs = self.files[self._fi]
            fp = fs.patch
            if fp.is_new:
                if self._pending is None or self._pending[0] != "write":
                    text = fp.hunks[0].new_text if fp.hunks else ""
                    self._pending = ("write", fp.path)
                    return ToolCall("write_file", {"path": fp.path, "content": text})
                self._advance_file()
                continue
            if fp.is_deleted:
                if self._pending is None or self._pending[0] != "delete":
                    self._pending = ("delete", fp.path)
                    return ToolCall("run_shell", {"command": f"rm {shlex.quote(fp.path)}"})
                self._advance_file()
                continue
            if fs.total is None or len(fs.lines) < fs.total:
                self._pending = ("read", fp.path)
                return ToolCall("read_file", {"path": fp.path, "offset": len(fs.lines) + 1})
            if self._hi >= len(fp.hunks):
                self._advance_file()
                continue
            hunk = fp.hunks[self._hi]
            old, new = self._snippet(fs, hunk, self._extra)
            if old is None:
                self._record("context_exhausted", final=True, attempted=False)
                self._next_hunk()
                continue
            self._pending = ("edit", (old, new))
            return ToolCall("edit", {"path": fp.path, "old_string": old, "new_string": new})
        self._pending = None
        return None

    def _absorb(self, output: str) -> None:
        assert self._pending is not None
        kind, payload = self._pending
        fs = self.files[self._fi]
        if kind == "read":
            if output.startswith("ERROR:"):
                fs.total = len(fs.lines)  # give up reading; edits will report the failure
                return
            m = _HEADER.match(output)
            body = output[m.end() :] if m else output
            fs.lines.extend(body.split("\n") if body else [])
            fs.total = int(m.group("n")) if m else len(fs.lines)
            return
        if kind == "edit":
            hunk = fs.patch.hunks[self._hi]
            if output.startswith("ERROR:"):
                status = "ambiguous" if "occurs" in output else "not_found"
                if status == "ambiguous" and self._extra < MAX_EXTRA_CONTEXT:
                    self._record(status, final=False)
                    self._extra += 1
                    while self._extra < MAX_EXTRA_CONTEXT and not self._unique(fs, hunk):
                        self._extra += 1
                    return
                self._record(status, final=True)
                self._next_hunk()
                return
            self._record("ok", final=True)
            old, new = payload
            text = "\n".join(fs.lines).replace(old.rstrip("\n"), new.rstrip("\n"), 1)
            fs.lines = text.split("\n")
            fs.total = len(fs.lines)
            fs.delta += hunk.added - hunk.removed
            self._next_hunk()

    # -- helpers ---------------------------------------------------------------------
    def _record(self, status: str, final: bool, attempted: bool = True) -> None:
        path = self.files[self._fi].patch.path
        rec = next((r for r in self.records if r.path == path and r.index == self._hi), None)
        if rec is None:
            rec = HunkRecord(path, self._hi, first_status=status)
            self.records.append(rec)
        rec.attempts += int(attempted)
        rec.extra_context = self._extra
        if final:
            rec.final_status = status

    def _next_hunk(self) -> None:
        self._hi += 1
        self._extra = 0

    def _advance_file(self) -> None:
        self._fi += 1
        self._hi = 0
        self._extra = 0
        self._pending = None

    def _snippet(self, fs: _FileState, hunk: Hunk, extra: int) -> tuple[str | None, str]:
        old, new = hunk.old_text, hunk.new_text
        if extra == 0 and old:
            return old, new
        extra = max(extra, 1)
        n_old = old.count("\n") + (0 if old.endswith("\n") else 1) if old else 0
        start = hunk.old_start - 1 + fs.delta  # 0-based first line of the hunk now
        if hunk.old_len == 0:
            start += 1
        before = fs.lines[max(0, start - extra) : max(0, start)]
        after = fs.lines[start + n_old : start + n_old + extra]
        if len(before) < min(extra, start) or (not before and not after):
            return None, new
        pre = "".join(line + "\n" for line in before)
        post = "".join(line + "\n" for line in after)
        if not old.endswith("\n") and post:
            return None, new
        return pre + old + post, pre + new + post

    def _unique(self, fs: _FileState, hunk: Hunk) -> bool:
        old, _ = self._snippet(fs, hunk, self._extra)
        if old is None:
            return True  # nothing more to add; let the edit report it
        return ("\n".join(fs.lines) + "\n").count(old) == 1
