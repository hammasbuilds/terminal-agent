"""Interactive REPL: one conversation, approvals asked at the terminal."""

from __future__ import annotations

import json
from collections.abc import Callable

from terminal_agent.agent import Agent
from terminal_agent.context import conversation_tokens
from terminal_agent.policy import Verdict
from terminal_agent.protocol import ToolCall
from terminal_agent.tools import TOOL_SPECS

HELP = """Type a request and press Enter. Commands:
  /help     this text
  /tools    list the tools the agent can call
  /tokens   estimated size of the conversation so far
  /reset    start a new conversation (the trajectory log continues)
  /exit     quit (also /quit, Ctrl-D)"""


def make_terminal_approver(input_fn: Callable[[str], str] = input,
                           print_fn: Callable[[str], None] = print
                           ) -> Callable[[ToolCall, Verdict], bool]:
    def approve(call: ToolCall, verdict: Verdict) -> bool:
        detail = call.arguments.get("command") or call.arguments.get("path") or json.dumps(
            call.arguments)
        flag = "DANGEROUS" if verdict.risk == "dangerous" else "needs approval"
        print_fn(f"\n[{flag}] {call.name}: {detail}\n  reason: {verdict.reason}")
        try:
            answer = input_fn("  Allow? [y/N] ").strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")

    return approve


def run_repl(agent: Agent, input_fn: Callable[[str], str] = input,
             print_fn: Callable[[str], None] = print) -> int:
    print_fn(f"terminal-agent  model={agent.client.model}  workspace={agent.toolbox.workspace}")
    print_fn("Type /help for commands.")
    while True:
        try:
            line = input_fn("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print_fn("")
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/help":
            print_fn(HELP)
            continue
        if line == "/tools":
            for spec in TOOL_SPECS:
                fn = spec["function"]
                print_fn(f"  {fn['name']:<11} {fn['description']}")
            continue
        if line == "/tokens":
            print_fn(f"  ~{conversation_tokens(agent.messages)} tokens in "
                     f"{len(agent.messages)} messages (budget {agent.context.budget})")
            continue
        if line == "/reset":
            agent.reset()
            print_fn("  conversation cleared")
            continue
        if line.startswith("/"):
            print_fn(f"  unknown command {line.split()[0]}; try /help")
            continue
        try:
            result = agent.run(line)
        except KeyboardInterrupt:
            print_fn("\n  interrupted")
            continue
        for name, n in sorted(result.tool_calls.items()):
            print_fn(f"  [{name} x{n}]")
        if result.error:
            print_fn(f"  ({result.status}) {result.error}")
        print_fn(result.final or f"({result.status})")
