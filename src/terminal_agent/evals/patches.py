"""Unified-diff parsing, and turning a hunk into the (old_string, new_string) pair of an edit."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF = re.compile(r"^diff --git a/(.*) b/(.*)$")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str] = field(default_factory=list)  # raw lines incl. ' ', '-', '+', '\\'

    def _side(self, keep: str) -> str:
        out: list[str] = []
        no_newline = False
        for i, line in enumerate(self.lines):
            tag, body = line[:1], line[1:]
            if tag == "\\":
                continue
            if tag in (" ", keep):
                nxt = self.lines[i + 1] if i + 1 < len(self.lines) else ""
                no_newline = nxt.startswith("\\")
                out.append(body + ("" if no_newline else "\n"))
        return "".join(out)

    @property
    def old_text(self) -> str:
        return self._side("-")

    @property
    def new_text(self) -> str:
        return self._side("+")

    @property
    def removed(self) -> int:
        return sum(1 for ln in self.lines if ln.startswith("-"))

    @property
    def added(self) -> int:
        return sum(1 for ln in self.lines if ln.startswith("+"))

    @property
    def leading_context(self) -> int:
        n = 0
        for ln in self.lines:
            if not ln.startswith(" "):
                break
            n += 1
        return n

    @property
    def trailing_context(self) -> int:
        n = 0
        for ln in reversed([x for x in self.lines if not x.startswith("\\")]):
            if not ln.startswith(" "):
                break
            n += 1
        return n

    def core(self) -> tuple[str, str]:
        """The hunk with all context stripped: only removed lines -> only added lines."""
        body = self.lines[self.leading_context : len(self.lines) - self.trailing_context]
        trimmed = Hunk(self.old_start, 0, self.new_start, 0, body)
        return trimmed.old_text, trimmed.new_text


@dataclass
class FilePatch:
    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False

    @property
    def path(self) -> str:
        return self.old_path if self.is_deleted else self.new_path


def parse_patch(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    remaining_old = remaining_new = 0
    for line in text.replace("\r\n", "\n").split("\n"):
        m = _DIFF.match(line)
        if m:
            current = FilePatch(m.group(1), m.group(2))
            files.append(current)
            hunk = None
            continue
        if current is None:
            continue
        if hunk is not None and (remaining_old > 0 or remaining_new > 0):
            tag = line[:1]
            if tag in (" ", "-", "+", "\\") or line == "":
                if line == "":  # a blank context line whose leading space was stripped
                    line, tag = " ", " "
                hunk.lines.append(line)
                if tag in (" ", "-"):
                    remaining_old -= 1
                if tag in (" ", "+"):
                    remaining_new -= 1
                continue
        if hunk is not None and line.startswith("\\"):
            hunk.lines.append(line)
            continue
        if line.startswith("new file mode"):
            current.is_new = True
        elif line.startswith("deleted file mode"):
            current.is_deleted = True
        elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
            current.is_binary = True
        else:
            h = _HUNK.match(line)
            if h:
                old_len = int(h.group(2)) if h.group(2) is not None else 1
                new_len = int(h.group(4)) if h.group(4) is not None else 1
                hunk = Hunk(int(h.group(1)), old_len, int(h.group(3)), new_len)
                current.hunks.append(hunk)
                remaining_old, remaining_new = old_len, new_len
    return files


def patch_test_files(test_patch: str) -> list[str]:
    """Files a test patch touches, in order (the tests SWE-bench runs)."""
    return [f.path for f in parse_patch(test_patch)]
