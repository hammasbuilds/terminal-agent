"""Exact-match string replacement, the mechanism behind the ``edit`` tool.

The contract is gemini-cli's: ``old_string`` must occur exactly ``expected`` times, or
nothing changes and the model is told why. Two relaxations exist and are reported
separately so their effect can be measured:

* ``crlf`` - always on. A CRLF file shown to the model as LF text would otherwise never
  match, which is a harness bug, not a model failure.
* ``rstrip`` / ``indent`` - opt-in (``fuzzy=True``). ``rstrip`` ignores trailing whitespace;
  ``indent`` tolerates the block being pasted at a different indentation *as a whole* but
  requires the relative indentation between its lines to match exactly, so a snippet whose
  structure differs from the file (a line moved into or out of a block) is refused rather
  than silently applied. Both still require a unique match.
"""

from __future__ import annotations

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


def _shift(lines: list[str], src: str, dst: str) -> list[str]:
    """Re-indent ``lines`` by the difference between ``src`` and ``dst``, uniformly.

    The shift applies to every line, including ones indented *less* than the snippet's
    first line. (Replacing only lines that start with ``src`` - the first version of
    this - left a dedented ``raise`` at column 0 in 7 of 10 real hunks.)
    """
    if src == dst:
        return lines
    if dst.endswith(src):
        pad = dst[: len(dst) - len(src)]
        return [pad + line if line.strip() else line for line in lines]
    if src.endswith(dst):
        cut = len(src) - len(dst)
        prefix = src[:cut]
        return [line[cut:] if line.startswith(prefix) else line for line in lines]
    return [
        dst + line[len(src) :] if line.startswith(src) and line.strip() else line for line in lines
    ]


def _common_lead(lines: list[str]) -> str:
    """Longest whitespace prefix common to every non-blank line."""
    indents = [_indent(ln) for ln in lines if ln.strip()]
    if not indents:
        return ""
    prefix = indents[0]
    for ind in indents[1:]:
        while not ind.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _dedent_key(block: list[str]) -> list[str]:
    """Block with its common leading indent removed, trailing whitespace dropped.

    Preserves *relative* indentation between lines, so two blocks match only when they
    have the same structure - a line moved into or out of a nested block no longer matches.
    """
    lead = _common_lead(block)
    out = []
    for ln in block:
        body = ln[len(lead) :] if ln.startswith(lead) else ln.lstrip(" \t")
        out.append(body.rstrip())
    return out


def _line_match(content: str, old: str, new: str, strategy: str) -> EditOutcome:
    old_lines, _ = _split_old(old)
    if not any(line.strip() for line in old_lines):
        return EditOutcome("not_found", strategy=strategy)
    file_lines = content.splitlines(keepends=True)
    bodies = [ln.rstrip("\r\n") for ln in file_lines]
    k = len(old_lines)
    if strategy == "rstrip":
        want = [ln.rstrip() for ln in old_lines]
        hits = [
            i for i in range(len(bodies) - k + 1) if [b.rstrip() for b in bodies[i : i + k]] == want
        ]
    else:  # indent: compare dedented (structure-preserving) keys
        want = _dedent_key(old_lines)
        hits = [i for i in range(len(bodies) - k + 1) if _dedent_key(bodies[i : i + k]) == want]
    if not hits:
        return EditOutcome("not_found", strategy=strategy)
    if len(hits) > 1:
        return EditOutcome("ambiguous", occurrences=len(hits), strategy=strategy)
    i = hits[0]
    eol = detect_eol(content)
    new_lines, new_trailing = _split_old(new)
    if strategy == "indent":
        new_lines = _shift(new_lines, _common_lead(old_lines), _common_lead(bodies[i : i + k]))
    last_eol = file_lines[i + k - 1][len(bodies[i + k - 1]) :]
    replacement = eol.join(new_lines)
    if new_lines and (new_trailing or last_eol):
        replacement += last_eol or eol
    before = "".join(file_lines[:i])
    after = "".join(file_lines[i + k :])
    return EditOutcome(
        "ok", content=before + replacement + after, occurrences=1, strategy=strategy, line=i + 1
    )


def apply_edit(
    content: str, old: str, new: str, expected: int = 1, fuzzy: bool = False
) -> EditOutcome:
    """Replace ``old`` with ``new`` in ``content`` if it occurs exactly ``expected`` times."""
    if old == "":
        return EditOutcome("empty_old")
    if old == new:
        return EditOutcome("noop")
    outcome = _exact(content, old, new, expected, "exact")
    if outcome.status != "not_found":
        return outcome
    if "\r\n" in content and "\r" not in old:
        crlf = _exact(
            content, old.replace("\n", "\r\n"), new.replace("\n", "\r\n"), expected, "crlf"
        )
        if crlf.status != "not_found":
            return crlf
    if not fuzzy or expected != 1:
        return outcome
    for strategy in ("rstrip", "indent"):
        result = _line_match(content, old, new, strategy)
        if result.status != "not_found":
            return result
    return outcome
