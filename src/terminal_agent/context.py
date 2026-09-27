"""Context management: token estimates, output truncation and conversation compaction.

Token counts here are estimates (characters / 3.2 - deliberately pessimistic, since code
tokenises denser than prose and an underestimate means Ollama silently drops the start of
the prompt). Ollama reports the real prompt count after each call; the trajectory logs
both, so the model arm measures the estimator's actual error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

CHARS_PER_TOKEN = 3.2
TruncateMode = Literal["head", "tail", "head_tail", "digest"]
# lines a test runner uses to report a failure, across pytest, unittest/Django and sympy
FAILURE_LINE = re.compile(
    r"^(FAILED|ERROR|FAIL:|ERROR:)\s|\.\.\. (FAIL|ERROR)$|^test_\S+ [FE]$|"
    r"^_{3,} .+ _{3,}$|^E\s{3}|Error: |^\S+Error\b"
)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1 if text else 0


def message_tokens(message: dict[str, Any]) -> int:
    total = estimate_tokens(message.get("content") or "") + 4
    for call in message.get("tool_calls") or []:
        total += estimate_tokens(json.dumps(call, ensure_ascii=False))
    return total


def conversation_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(message_tokens(m) for m in messages)


def truncate(text: str, max_chars: int, mode: TruncateMode = "head_tail") -> tuple[str, int]:
    """Cut ``text`` to about ``max_chars``; return (text, chars elided).

    ``head_tail`` keeps the first 40% and the last 60%: a test runner prints the failing
    test's name at the top of a failure block and its summary at the very end, and
    ``head``-only truncation is exactly what hides the summary.

    ``digest`` first lists the lines that look like failure reports (up to a third of the
    budget), then fills the rest with ``head_tail``. It exists because no single cut works
    for every runner: pytest's ``-rA`` summary is at the end, Django reports each test
    inline as it runs.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text, 0
    if mode == "digest":
        picked: list[str] = []
        used = 0
        for line in text.split("\n"):
            if FAILURE_LINE.search(line) and used + len(line) + 1 <= max_chars // 3:
                picked.append(line)
                used += len(line) + 1
        rest, _ = truncate(text, max_chars - used, "head_tail")
        if not picked:
            return rest, len(text) - (max_chars - used)
        head = "[failure lines]\n" + "\n".join(picked) + "\n[output]\n"
        return head + rest, len(text) - (max_chars - used)
    elided = len(text) - max_chars
    if mode == "head":
        return text[:max_chars] + f"\n[... {elided} chars truncated ...]", elided
    if mode == "tail":
        return f"[... {elided} chars truncated ...]\n" + text[-max_chars:], elided
    head = int(max_chars * 0.4)
    tail = max_chars - head
    return (
        text[:head] + f"\n[... {elided} chars truncated ...]\n" + text[len(text) - tail :],
        elided,
    )


@dataclass
class CompactionReport:
    tokens_before: int
    tokens_after: int
    stubbed: int = 0
    dropped: int = 0
    squeezed: int = 0
    over_budget: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.stubbed or self.dropped or self.squeezed)


class ContextManager:
    """Keeps a conversation under a token budget without a model call.

    Three steps, cheapest information loss first, each stopping once under budget:

    1. **stub** old tool outputs (they can be re-fetched by re-running the tool);
    2. **drop** the oldest assistant/tool exchanges whole, keeping the system prompt,
       the task and the most recent ``keep_recent`` messages;
    3. **squeeze** the largest remaining messages in the middle.
    """

    def __init__(self, budget_tokens: int = 12000, keep_recent: int = 6,
                 stub_min_chars: int = 400) -> None:
        self.budget = budget_tokens
        self.keep_recent = keep_recent
        self.stub_min_chars = stub_min_chars

    def fit(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], CompactionReport]:
        msgs = [dict(m) for m in messages]
        report = CompactionReport(conversation_tokens(msgs), 0)
        if report.tokens_before <= self.budget:
            report.tokens_after = report.tokens_before
            return msgs, report
        head = self._protected_head(msgs)
        recent_start = max(head, len(msgs) - self.keep_recent)

        for i in range(head, recent_start):
            if conversation_tokens(msgs) <= self.budget:
                break
            m = msgs[i]
            content = m.get("content") or ""
            if m.get("role") == "tool" and len(content) >= self.stub_min_chars:
                tool = m.get("tool_name", "tool")
                m["content"] = (
                    f"[compacted: {len(content)} chars of {tool} output removed to fit the "
                    f"context budget; run the tool again if you still need it]"
                )
                report.stubbed += 1

        while conversation_tokens(msgs) > self.budget:
            start = self._protected_head(msgs)
            end = max(start, len(msgs) - self.keep_recent)
            if end <= start:
                break
            # drop one assistant turn together with the tool results that answer it
            j = start + 1
            while j < end and msgs[j].get("role") == "tool":
                j += 1
            del msgs[start:j]
            report.dropped += j - start
        if report.dropped:
            msgs.insert(
                self._protected_head(msgs),
                {"role": "user", "content": f"[{report.dropped} earlier messages were removed "
                                            "to fit the context budget]"},
            )

        while conversation_tokens(msgs) > self.budget:
            idx = max(range(len(msgs)), key=lambda k: len(msgs[k].get("content") or ""))
            content = msgs[idx].get("content") or ""
            if len(content) < 800:
                break
            msgs[idx]["content"], _ = truncate(content, len(content) // 2)
            report.squeezed += 1

        report.tokens_after = conversation_tokens(msgs)
        report.over_budget = report.tokens_after > self.budget
        return msgs, report

    @staticmethod
    def _protected_head(msgs: list[dict[str, Any]]) -> int:
        """System prompt plus the first user message (the task) are never compacted."""
        n = 0
        if msgs and msgs[0].get("role") == "system":
            n = 1
        if len(msgs) > n and msgs[n].get("role") == "user":
            n += 1
        return n
