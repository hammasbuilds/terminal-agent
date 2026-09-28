"""The agent's tools. Each takes validated keyword arguments and returns a ToolResult.

File tools act on the local workspace; ``run_shell`` and ``run_tests`` go through a
:class:`~terminal_agent.sandbox.Sandbox`. Files are read and written as bytes, never in
text mode, so a CRLF file stays CRLF and an LF file never becomes CRLF on Windows.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terminal_agent.context import TruncateMode, truncate
from terminal_agent.edits import apply_edit, detect_eol
from terminal_agent.protocol import ToolCall
from terminal_agent.sandbox import Sandbox

# a test target made only of these characters needs no shell quoting (and stays readable
# under cmd.exe, where single quotes are not quotes)
_PLAIN_ARG = re.compile(r"^[\w./:\[\]@+,=-]+$")
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".eggs",
}


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return self.output if self.ok else f"ERROR: {self.output}"


@dataclass
class ToolConfig:
    read_max_lines: int = 1000
    read_max_line_chars: int = 2000
    read_max_chars: int = 6000  # one read window never exceeds this, so it cannot blow the budget
    output_max_chars: int = 8000
    truncate_mode: TruncateMode = "digest"
    shell_timeout: float = 120.0
    test_timeout: float = 900.0
    fuzzy_edit: bool = False
    test_command: str = "python -m pytest -q"
    list_limit: int = 500
    match_limit: int = 200


TRUNCATED_MARK = " [line truncated]"


def window_size(line_lengths: list[int], config: ToolConfig) -> int:
    """How many of these lines (the window's candidates, in order) one read returns.

    Lines longer than ``read_max_line_chars`` are clipped (plus a marker). The window stops
    before the running size would pass ``read_max_chars``, so a dense window cannot blow
    the token budget, but always keeps the first line so a huge single line still returns
    something. The read-window study calls this too, so it measures the tool as it is.
    """
    used = 0
    for n, length in enumerate(line_lengths):
        if length > config.read_max_line_chars:
            length = config.read_max_line_chars + len(TRUNCATED_MARK)
        if n and used + length + 1 > config.read_max_chars:
            return n
        used += length + 1
    return len(line_lengths)


class Toolbox:
    def __init__(self, workspace: Path, sandbox: Sandbox, config: ToolConfig | None = None) -> None:
        self.workspace = workspace.resolve()
        self.sandbox = sandbox
        self.config = config or ToolConfig()
        self._impl: dict[str, Callable[..., ToolResult]] = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit": self.edit,
            "list_dir": self.list_dir,
            "glob": self.glob,
            "grep": self.grep,
            "run_shell": self.run_shell,
            "run_tests": self.run_tests,
        }

    # -- dispatch -----------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return [*self._impl, "finish"]

    def execute(self, call: ToolCall) -> ToolResult:
        impl = self._impl.get(call.name)
        if impl is None:
            return ToolResult(
                False,
                f"unknown tool {call.name!r}; available: {', '.join(self.names)}",
                {"error": "unknown_tool"},
            )
        spec = next(s for s in TOOL_SPECS if s["function"]["name"] == call.name)["function"]
        params = spec["parameters"]
        args = call.arguments
        if "__unparsed__" in args:
            return ToolResult(
                False, "tool arguments were not valid JSON", {"error": "bad_arguments"}
            )
        missing = [p for p in params.get("required", []) if p not in args]
        if missing:
            return ToolResult(
                False,
                f"missing required argument(s): {', '.join(missing)}",
                {"error": "bad_arguments"},
            )
        unknown = [a for a in args if a not in params["properties"]]
        if unknown:
            return ToolResult(
                False,
                f"unknown argument(s): {', '.join(unknown)}; expected "
                f"{', '.join(params['properties'])}",
                {"error": "bad_arguments"},
            )
        for name, value in args.items():
            want = params["properties"][name]["type"]
            if not _type_ok(value, want):
                return ToolResult(
                    False,
                    f"argument {name!r} must be {want}, got {type(value).__name__}",
                    {"error": "bad_arguments"},
                )
        try:
            return impl(**args)
        except OSError as exc:
            return ToolResult(False, f"{type(exc).__name__}: {exc}", {"error": "os_error"})

    def _path(self, raw: str) -> Path:
        return (self.workspace / raw).resolve()

    def _rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(path)

    # -- file tools ---------------------------------------------------------------
    def read_file(self, path: str, offset: int = 1, limit: int | None = None) -> ToolResult:
        target = self._path(path)
        if not target.exists():
            return ToolResult(False, f"{path} does not exist", {"error": "not_found"})
        if target.is_dir():
            return ToolResult(
                False, f"{path} is a directory; use list_dir", {"error": "is_directory"}
            )
        data = target.read_bytes()
        if b"\x00" in data[:8192]:
            return ToolResult(
                False, f"{path} is a binary file ({len(data)} bytes)", {"error": "binary"}
            )
        text = data.decode("utf-8", "replace").replace("\r\n", "\n")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        total = len(lines)
        offset = max(1, offset)
        limit = self.config.read_max_lines if limit is None else max(1, limit)
        candidates = lines[offset - 1 : offset - 1 + limit]
        count = window_size([len(line) for line in candidates], self.config)
        shown, clipped = [], 0
        for line in candidates[:count]:
            if len(line) > self.config.read_max_line_chars:
                clipped += 1
                line = line[: self.config.read_max_line_chars] + TRUNCATED_MARK
            shown.append(line)
        body = "\n".join(shown)
        end = offset + len(shown) - 1
        meta = {"total_lines": total, "first": offset, "last": end, "clipped_lines": clipped}
        if offset > 1 or end < total:
            header = (
                f"[{path}: lines {offset}-{end} of {total}. "
                f"Call read_file with offset={end + 1} to read more.]\n"
                if end < total
                else f"[{path}: lines {offset}-{end} of {total}]\n"
            )
            meta["truncated"] = end < total
            return ToolResult(True, header + body, meta)
        return ToolResult(True, body, meta)

    def write_file(self, path: str, content: str) -> ToolResult:
        target = self._path(path)
        if target.is_dir():
            return ToolResult(False, f"{path} is a directory", {"error": "is_directory"})
        existed = target.exists()
        if existed:
            eol = detect_eol(target.read_bytes().decode("utf-8", "replace"))
            if eol == "\r\n":
                content = content.replace("\r\n", "\n").replace("\n", "\r\n")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
        verb = "overwrote" if existed else "created"
        return ToolResult(
            True, f"{verb} {self._rel(target)} ({len(content)} chars)", {"created": not existed}
        )

    def edit(
        self, path: str, old_string: str, new_string: str, expected_replacements: int = 1
    ) -> ToolResult:
        target = self._path(path)
        if not target.exists():
            if old_string == "":
                return self.write_file(path, new_string)
            return ToolResult(
                False,
                f"{path} does not exist. To create a file pass an "
                f"empty old_string, or use write_file.",
                {"error": "not_found"},
            )
        # surrogateescape round-trips bytes that are not UTF-8 instead of replacing them
        content = target.read_bytes().decode("utf-8", "surrogateescape")
        outcome = apply_edit(
            content, old_string, new_string, expected_replacements, fuzzy=self.config.fuzzy_edit
        )
        meta = {
            "status": outcome.status,
            "strategy": outcome.strategy,
            "occurrences": outcome.occurrences,
        }
        if outcome.status == "not_found":
            return ToolResult(
                False,
                f"old_string not found in {path}. It must match the file "
                f"exactly, including whitespace and indentation; "
                f"read_file the region again and copy it.",
                meta,
            )
        if outcome.status == "ambiguous":
            return ToolResult(
                False,
                f"old_string occurs {outcome.occurrences} times in {path}, "
                f"expected {expected_replacements}. Include more "
                f"surrounding lines so it is unique.",
                meta,
            )
        if outcome.status == "noop":
            return ToolResult(False, "old_string and new_string are identical; nothing to do", meta)
        if outcome.status == "empty_old":
            return ToolResult(
                False,
                f"{path} already exists; an empty old_string only creates "
                f"new files. Use write_file to replace it whole.",
                meta,
            )
        assert outcome.content is not None
        target.write_bytes(outcome.content.encode("utf-8", "surrogateescape"))
        meta["line"] = outcome.line
        return ToolResult(
            True,
            f"edited {self._rel(target)} at line {outcome.line}"
            + (f" ({outcome.strategy} match)" if outcome.strategy != "exact" else ""),
            meta,
        )

    def list_dir(self, path: str = ".") -> ToolResult:
        target = self._path(path)
        if not target.is_dir():
            return ToolResult(False, f"{path} is not a directory", {"error": "not_found"})
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        names = [p.name + ("/" if p.is_dir() else "") for p in entries if p.name != ".git"]
        shown = names[: self.config.list_limit]
        more = len(names) - len(shown)
        return ToolResult(
            True,
            "\n".join(shown) + (f"\n[... {more} more]" if more else ""),
            {"entries": len(names)},
        )

    def _walk(self, root: Path) -> list[Path]:
        out: list[Path] = []
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                children = sorted(d.iterdir())
            except OSError:
                continue
            for c in children:
                if c.is_dir():
                    if c.name not in SKIP_DIRS:
                        stack.append(c)
                else:
                    out.append(c)
        return sorted(out)

    def glob(self, pattern: str, path: str = ".") -> ToolResult:
        root = self._path(path)
        if not root.is_dir():
            return ToolResult(False, f"{path} is not a directory", {"error": "not_found"})
        pat = pattern[2:] if pattern.startswith("./") else pattern
        hits = []
        for f in self._walk(root):
            rel = f.relative_to(root).as_posix()
            if (
                fnmatch.fnmatchcase(rel, pat)
                or (pat.startswith("**/") and fnmatch.fnmatchcase(rel, pat[3:]))
                or ("/" not in pat and fnmatch.fnmatchcase(f.name, pat))
            ):
                hits.append(self._rel(f))
        shown = hits[: self.config.match_limit]
        more = len(hits) - len(shown)
        if not hits:
            return ToolResult(True, f"no files match {pattern!r}", {"matches": 0})
        return ToolResult(
            True,
            "\n".join(shown) + (f"\n[... {more} more]" if more else ""),
            {"matches": len(hits)},
        )

    def grep(self, pattern: str, path: str = ".", include: str | None = None) -> ToolResult:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return ToolResult(
                False, f"invalid regular expression: {exc}", {"error": "bad_arguments"}
            )
        root = self._path(path)
        files = [root] if root.is_file() else self._walk(root) if root.is_dir() else []
        if not files:
            return ToolResult(False, f"{path} does not exist", {"error": "not_found"})
        hits: list[str] = []
        total = 0
        for f in files:
            if include and not fnmatch.fnmatch(f.name, include):
                continue
            try:
                data = f.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            for no, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if rx.search(line):
                    total += 1
                    if len(hits) < self.config.match_limit:
                        hits.append(f"{self._rel(f)}:{no}: {line.strip()[:200]}")
        if not hits:
            return ToolResult(True, f"no matches for {pattern!r}", {"matches": 0})
        more = total - len(hits)
        return ToolResult(
            True,
            "\n".join(hits) + (f"\n[... {more} more matches]" if more else ""),
            {"matches": total},
        )

    # -- execution tools ------------------------------------------------------------
    def _exec(self, command: str, timeout: float) -> ToolResult:
        res = self.sandbox.run(command, timeout)
        text, elided = truncate(res.output, self.config.output_max_chars, self.config.truncate_mode)
        note = f", TIMED OUT after {int(timeout)}s" if res.timed_out else ""
        status = f"[exit code {res.exit_code}{note}]"
        return ToolResult(
            res.exit_code == 0 and not res.timed_out,
            f"{status}\n{text}",
            {
                "exit_code": res.exit_code,
                "timed_out": res.timed_out,
                "chars": len(res.output),
                "elided_chars": elided,
                "seconds": round(res.seconds, 2),
            },
        )

    def run_shell(self, command: str, timeout: int | None = None) -> ToolResult:
        if timeout is not None and timeout <= 0:
            return ToolResult(
                False,
                f"timeout must be a positive number of seconds, got {timeout}",
                {"error": "bad_arguments"},
            )
        limit = float(timeout) if timeout else self.config.shell_timeout
        limit = min(limit, self.config.test_timeout)
        return self._exec(command, limit)

    def run_tests(self, target: str = "") -> ToolResult:
        """Run the test command, with ``target`` passed as ONE argument (never as shell)."""
        target = target.strip()
        if target.startswith("-"):
            return ToolResult(
                False,
                f"target must be a test file, directory or id, not an "
                f"option ({target!r}); use run_shell for options",
                {"error": "bad_arguments"},
            )
        if target and not _PLAIN_ARG.match(target):
            target = shlex.quote(target)  # `x; rm -rf ~` stays one argument
        command = self.config.test_command + (f" {target}" if target else "")
        return self._exec(command, self.config.test_timeout)


def _type_ok(value: Any, want: str) -> bool:
    if want == "string":
        return isinstance(value, str)
    if want == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return True


def _fn(
    name: str, description: str, props: dict[str, dict[str, str]], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


TOOL_SPECS: list[dict[str, Any]] = [
    _fn(
        "read_file",
        "Read a text file. Long files are returned a window at a time; the "
        "header says how to read the next window.",
        {
            "path": {"type": "string", "description": "path relative to the workspace"},
            "offset": {"type": "integer", "description": "1-based first line (default 1)"},
            "limit": {"type": "integer", "description": "max lines to return"},
        },
        ["path"],
    ),
    _fn(
        "write_file",
        "Create a file, or replace a file's whole content.",
        {
            "path": {"type": "string", "description": "path relative to the workspace"},
            "content": {"type": "string", "description": "the full new content"},
        },
        ["path", "content"],
    ),
    _fn(
        "edit",
        "Replace an exact string in a file. old_string must match the file exactly "
        "(whitespace included) and be unique unless expected_replacements says otherwise. "
        "Include a few lines of context around the change.",
        {
            "path": {"type": "string", "description": "path relative to the workspace"},
            "old_string": {"type": "string", "description": "exact text to replace"},
            "new_string": {"type": "string", "description": "replacement text"},
            "expected_replacements": {
                "type": "integer",
                "description": "how many occurrences to replace (default 1)",
            },
        },
        ["path", "old_string", "new_string"],
    ),
    _fn(
        "list_dir",
        "List a directory (directories end with /).",
        {"path": {"type": "string", "description": "directory, default '.'"}},
        [],
    ),
    _fn(
        "glob",
        "Find files by glob pattern, e.g. '**/*.py' or 'src/**/test_*.py'.",
        {
            "pattern": {"type": "string", "description": "glob pattern"},
            "path": {"type": "string", "description": "directory to search, default '.'"},
        },
        ["pattern"],
    ),
    _fn(
        "grep",
        "Search file contents with a Python regular expression. Returns path:line: text.",
        {
            "pattern": {"type": "string", "description": "regular expression"},
            "path": {"type": "string", "description": "file or directory, default '.'"},
            "include": {"type": "string", "description": "only files whose name matches this glob"},
        },
        ["pattern"],
    ),
    _fn(
        "run_shell",
        "Run a shell command in the workspace. Output is truncated; commands that "
        "are destructive or leave the workspace need approval.",
        {
            "command": {"type": "string", "description": "the command"},
            "timeout": {"type": "integer", "description": "seconds (default 120)"},
        },
        ["command"],
    ),
    _fn(
        "run_tests",
        "Run the project's test command, optionally on a target such as a file or test id.",
        {"target": {"type": "string", "description": "test file, directory or test id"}},
        [],
    ),
    _fn(
        "finish",
        "Call when the task is complete, with a short summary of what changed.",
        {"summary": {"type": "string", "description": "what you did"}},
        ["summary"],
    ),
]
