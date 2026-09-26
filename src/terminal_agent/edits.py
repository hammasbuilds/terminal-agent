"""Exact-match string replacement, the mechanism behind the ``edit`` tool.

The contract is gemini-cli's: ``old_string`` must occur exactly ``expected`` times, or
nothing changes and the model is told why. Two relaxations exist and are reported
separately so their effect can be measured:

* ``crlf`` - always on. A CRLF file shown to the model as LF text would otherwise never
  match, which is a harness bug, not a model failure.
* ``rstrip`` / ``indent`` - opt-in (``fuzzy=True``). Line-wise matching that ignores
  trailing whitespace, or all leading/trailing whitespace with the replacement re-indented.
  Still requires a unique match.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Status = Literal["ok", "not_found", "ambiguous", "noop", "empty_old"]


@dataclass
class EditOutcome:
    status: Status
    content: str | None = None
    occurrences: int = 0
    strategy: str = ""
    line: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def detect_eol(content: str) -> str:
    """The file's dominant line ending; LF when there are none."""
    crlf = content.count("\r\n")
    lf = content.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _exact(content: str, old: str, new: str, expected: int, strategy: str) -> EditOutcome:
    count = content.count(old)
    if count == 0:
        return EditOutcome("not_found", occurrences=0, strategy=strategy)
    if count != expected:
        return EditOutcome("ambiguous", occurrences=count, strategy=strategy)
    idx = content.index(old)
    return EditOutcome(
        "ok",
        content=content.replace(old, new),
        occurrences=count,
        strategy=strategy,
        line=content.count("\n", 0, idx) + 1,
    )


def _split_old(text: str) -> tuple[list[str], bool]:
    text = text.replace("\r\n", "\n")
    trailing = text.endswith("\n")
    lines = text.split("\n")
    if trailing:
        lines.pop()
    return lines, trailing


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _line_match(content: str, old: str, new: str,
                norm: Callable[[str], str], reindent: bool, strategy: str) -> EditOutcome:
    old_lines, _ = _split_old(old)
    if not any(line.strip() for line in old_lines):
        return EditOutcome("not_found", strategy=strategy)
    file_lines = content.splitlines(keepends=True)
    bodies = [ln.rstrip("\r\n") for ln in file_lines]
    keys = [norm(b) for b in bodies]
    want = [norm(line) for line in old_lines]
    k = len(want)
    hits = [i for i in range(len(keys) - k + 1) if keys[i : i + k] == want]
    if not hits:
        return EditOutcome("not_found", strategy=strategy)
    if len(hits) > 1:
        return EditOutcome("ambiguous", occurrences=len(hits), strategy=strategy)
    i = hits[0]
    eol = detect_eol(content)
    new_lines, new_trailing = _split_old(new)
    if reindent:
        first_old = next(line for line in old_lines if line.strip())
        first_file = next(bodies[i + j] for j, line in enumerate(old_lines) if line.strip())
        src, dst = _indent(first_old), _indent(first_file)
        if src != dst:
            new_lines = [
                dst + line[len(src):] if line.startswith(src) and line.strip() else line
                for line in new_lines
            ]
    last_eol = file_lines[i + k - 1][len(bodies[i + k - 1]):]
    replacement = eol.join(new_lines)
    if new_lines and (new_trailing or last_eol):
        replacement += last_eol or eol
    before = "".join(file_lines[:i])
    after = "".join(file_lines[i + k :])
    return EditOutcome("ok", content=before + replacement + after, occurrences=1,
                       strategy=strategy, line=i + 1)


def apply_edit(content: str, old: str, new: str, expected: int = 1,
               fuzzy: bool = False) -> EditOutcome:
    """Replace ``old`` with ``new`` in ``content`` if it occurs exactly ``expected`` times."""
    if old == "":
        return EditOutcome("empty_old")
    if old == new:
        return EditOutcome("noop")
    outcome = _exact(content, old, new, expected, "exact")
    if outcome.status != "not_found":
        return outcome
    if "\r\n" in content and "\r" not in old:
        crlf = _exact(content, old.replace("\n", "\r\n"), new.replace("\n", "\r\n"),
                      expected, "crlf")
        if crlf.status != "not_found":
            return crlf
    if not fuzzy or expected != 1:
        return outcome
    for strategy, norm, reindent in (
        ("rstrip", str.rstrip, False),
        ("indent", str.strip, True),
    ):
        result = _line_match(content, old, new, norm, reindent, strategy)
        if result.status != "not_found":
            return result
    return outcome
