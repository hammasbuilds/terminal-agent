"""Approval policy: which tool calls run unasked, which need a human, which never run.

The shell classifier that rates a command lives in :mod:`terminal_agent.classifier`; this
module maps its rating (and the file tools' targets) to ALLOW / ASK / DENY.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from terminal_agent.classifier import CommandClassifier
from terminal_agent.policy_rules import Risk
from terminal_agent.protocol import ToolCall

__all__ = ["ApprovalPolicy", "CommandClassifier", "Decision", "Verdict"]


class Decision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class Verdict:
    decision: Decision
    risk: Risk
    reason: str


READ_TOOLS = {"read_file", "list_dir", "glob", "grep"}
WRITE_TOOLS = {"write_file", "edit"}


class ApprovalPolicy:
    """Maps a tool call to ALLOW / ASK / DENY.

    ``mode``:
      * ``default`` - read-only commands and tests run; anything else asks.
      * ``auto``    - mutating commands run too; dangerous ones still ask.
    ``allow`` is a list of command prefixes the user trusts (``--allow "make"``); a matching
    *mutating* command runs unasked. A dangerous command is never allowlisted. Headless runs
    have no human, so the agent's approver turns every ASK into a DENY.
    """

    def __init__(
        self,
        workspace: Path,
        shell_root: str | None = None,
        mode: str = "default",
        allow: list[str] | None = None,
    ) -> None:
        if mode not in ("default", "auto"):
            raise ValueError(f"unknown approval mode {mode!r} (use 'default' or 'auto')")
        self.workspace = workspace.resolve()
        self.mode = mode
        self.allow = [a.strip() for a in (allow or []) if a.strip()]
        self.classifier = CommandClassifier(root=shell_root or self.workspace.as_posix())

    def check(self, call: ToolCall) -> Verdict:
        name = call.name
        if name in READ_TOOLS or name in ("run_tests", "finish"):
            return Verdict(Decision.ALLOW, "safe", "read-only tool")
        if name in WRITE_TOOLS:
            return self._check_write(str(call.arguments.get("path", "")))
        if name == "run_shell":
            return self.check_command(str(call.arguments.get("command", "")))
        return Verdict(Decision.ALLOW, "safe", "unknown tools fail at dispatch")

    def _check_write(self, raw: str) -> Verdict:
        try:
            target = (self.workspace / raw).resolve()
        except (OSError, ValueError):
            return Verdict(Decision.DENY, "dangerous", f"unusable path {raw!r}")
        if not target.is_relative_to(self.workspace):
            return Verdict(Decision.ASK, "dangerous", f"writes outside the workspace: {target}")
        if ".git" in target.relative_to(self.workspace).parts:
            return Verdict(Decision.DENY, "dangerous", "writes inside .git")
        return Verdict(Decision.ALLOW, "mutating", "write inside the workspace")

    def check_command(self, command: str) -> Verdict:
        rating = self.classifier.rate(command)
        if rating.risk == "safe":
            return Verdict(Decision.ALLOW, "safe", rating.reason)
        if rating.risk == "dangerous":
            return Verdict(Decision.ASK, "dangerous", rating.reason)
        normalized = " ".join(command.split())
        if self.mode == "auto" or any(
            normalized == p or normalized.startswith(p + " ") for p in self.allow
        ):
            return Verdict(Decision.ALLOW, "mutating", rating.reason)
        return Verdict(Decision.ASK, "mutating", rating.reason)
