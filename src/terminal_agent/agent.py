"""The agent loop: ask the model, check each tool call against the policy, run it, repeat."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terminal_agent.context import ContextManager, conversation_tokens
from terminal_agent.llm import ChatClient, ModelError
from terminal_agent.policy import ApprovalPolicy, Decision, Verdict
from terminal_agent.protocol import ToolCall
from terminal_agent.sandbox import LocalSandbox, Sandbox
from terminal_agent.tools import TOOL_SPECS, Toolbox, ToolConfig, ToolResult
from terminal_agent.trajectory import TrajectoryLogger

SYSTEM_PROMPT = """You are a coding agent working in a software repository at {workspace}.
Use the tools to inspect and change files; do not guess at file contents.

Rules:
- Read the relevant code before editing it. Keep changes minimal and focused on the task.
- Use `edit` for changes to existing files. old_string must be copied exactly from the file,
  with enough surrounding lines to be unique.
- Use `run_tests` (or `run_shell`) to check your work when tests exist.
- Commands that delete data, leave the workspace, install packages or use the network need
  approval and may be refused; find another way when that happens.
- When the task is done, call `finish` with a one-paragraph summary."""

Approver = Callable[[ToolCall, Verdict], bool]


def deny_all(call: ToolCall, verdict: Verdict) -> bool:
    """Headless approver: nobody is there to say yes."""
    return False


@dataclass
class AgentConfig:
    max_steps: int = 40
    loop_limit: int = 3  # identical consecutive calls before the run is stopped
    system_prompt: str = SYSTEM_PROMPT


@dataclass
class RunResult:
    status: str  # finished | max_steps | loop | model_error | no_tool_call
    final: str
    steps: int
    tool_calls: Counter[str] = field(default_factory=Counter)
    tool_errors: Counter[str] = field(default_factory=Counter)
    denied: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    compactions: int = 0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "response": self.final, "steps": self.steps,
            "tool_calls": dict(self.tool_calls), "tool_errors": dict(self.tool_errors),
            "denied": self.denied,
            "tokens": {"prompt": self.prompt_tokens, "completion": self.completion_tokens},
            "compactions": self.compactions, "error": self.error,
        }


class Agent:
    def __init__(self, client: ChatClient, toolbox: Toolbox, policy: ApprovalPolicy,
                 approver: Approver = deny_all, logger: TrajectoryLogger | None = None,
                 context: ContextManager | None = None, config: AgentConfig | None = None
                 ) -> None:
        self.client = client
        self.toolbox = toolbox
        self.policy = policy
        self.approver = approver
        self.logger = logger or TrajectoryLogger(None)
        self.context = context or ContextManager()
        self.config = config or AgentConfig()
        self.messages: list[dict[str, Any]] = [{
            "role": "system",
            "content": self.config.system_prompt.format(workspace=toolbox.workspace),
        }]
        self._step = 0

    def reset(self) -> None:
        del self.messages[1:]

    def run(self, task: str) -> RunResult:
        """Run one user request to completion (the conversation so far is kept)."""
        self.logger.log("run_start", self._step, task=task, model=self.client.model,
                        workspace=str(self.toolbox.workspace))
        self.messages.append({"role": "user", "content": task})
        result = RunResult(status="max_steps", final="", steps=0)
        recent: list[str] = []
        for _ in range(self.config.max_steps):
            self._step += 1
            result.steps += 1
            fitted, report = self.context.fit(self.messages)
            if report.changed:
                result.compactions += 1
                self.logger.log("compaction", self._step, tokens_before=report.tokens_before,
                                tokens_after=report.tokens_after, stubbed=report.stubbed,
                                dropped=report.dropped, squeezed=report.squeezed,
                                over_budget=report.over_budget)
            try:
                turn = self.client.chat(fitted, TOOL_SPECS)
            except ModelError as exc:
                result.status, result.error = "model_error", str(exc)
                break
            result.prompt_tokens += turn.prompt_tokens or 0
            result.completion_tokens += turn.completion_tokens or 0
            self.logger.log("model", self._step, content=turn.content,
                            prompt_tokens=turn.prompt_tokens,
                            completion_tokens=turn.completion_tokens,
                            estimated_prompt_tokens=conversation_tokens(fitted),
                            cached=turn.cached, parsed_from_text=turn.parsed_from_text,
                            n_tool_calls=len(turn.tool_calls))
            self.messages.append(_assistant_message(turn.content, turn.tool_calls))
            if not turn.tool_calls:
                result.status, result.final = "no_tool_call", turn.content
                break
            finished = False
            for call in turn.tool_calls:
                result.tool_calls[call.name] += 1
                self.logger.log("tool_call", self._step, name=call.name,
                                arguments=call.arguments)
                if call.name == "finish":
                    finished = True
                    result.final = str(call.arguments.get("summary", turn.content))
                    self.messages.append(_tool_message(call, ToolResult(True, "done")))
                    continue
                outcome = self._run_call(call, result)
                self.messages.append(_tool_message(call, outcome))
                recent.append(call.signature())
            if finished:
                result.status = "finished"
                break
            if len(recent) >= self.config.loop_limit and len(
                    set(recent[-self.config.loop_limit:])) == 1:
                result.status = "loop"
                result.error = f"the same call was made {self.config.loop_limit} times in a row"
                break
        self.logger.log("run_end", self._step, status=result.status, steps=result.steps,
                        final=result.final, error=result.error,
                        tool_calls=dict(result.tool_calls))
        return result

    def _run_call(self, call: ToolCall, result: RunResult) -> ToolResult:
        verdict = self.policy.check(call)
        allowed = verdict.decision is Decision.ALLOW
        if verdict.decision is Decision.ASK:
            allowed = self.approver(call, verdict)
        if verdict.decision is not Decision.ALLOW:
            self.logger.log("approval", self._step, name=call.name,
                            decision="approved" if allowed else "denied",
                            policy=verdict.decision.value, risk=verdict.risk,
                            reason=verdict.reason)
        if not allowed:
            label = call.arguments.get("command") or call.arguments.get("path") or call.name
            result.denied.append(str(label))
            outcome = ToolResult(False, f"not run: the approval policy refused it "
                                        f"({verdict.risk}: {verdict.reason})",
                                 {"error": "denied"})
        else:
            outcome = self.toolbox.execute(call)
        if not outcome.ok:
            result.tool_errors[call.name] += 1
        self.logger.log("tool_result", self._step, name=call.name, ok=outcome.ok,
                        output=outcome.output, meta=outcome.meta)
        return outcome


def _assistant_message(content: str, calls: list[ToolCall]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}}
                             for c in calls]
    return msg


def _tool_message(call: ToolCall, outcome: ToolResult) -> dict[str, Any]:
    return {"role": "tool", "tool_name": call.name, "content": outcome.render()}


def build_agent(workspace: Path, client: ChatClient, *, sandbox: Sandbox | None = None,
                tool_config: ToolConfig | None = None, policy: ApprovalPolicy | None = None,
                approver: Approver = deny_all, trajectory: Path | None = None,
                token_budget: int = 12000, max_steps: int = 40) -> Agent:
    """Assemble an agent with sensible defaults (local sandbox, default policy)."""
    ws = workspace.resolve()
    toolbox = Toolbox(ws, sandbox or LocalSandbox(ws), tool_config or ToolConfig())
    return Agent(
        client=client,
        toolbox=toolbox,
        policy=policy or ApprovalPolicy(ws),
        approver=approver,
        logger=TrajectoryLogger(trajectory),
        context=ContextManager(budget_tokens=token_budget),
        config=AgentConfig(max_steps=max_steps),
    )
