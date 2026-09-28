"""``terminal-agent``: interactive REPL, headless ``-p`` mode, and ``replay``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from terminal_agent.agent import Agent, build_agent, deny_all
from terminal_agent.llm import DEFAULT_HOST, DEFAULT_MODEL, ChatClient, OllamaClient, ScriptedClient
from terminal_agent.policy import ApprovalPolicy
from terminal_agent.repl import make_terminal_approver, run_repl
from terminal_agent.tools import ToolConfig
from terminal_agent.trajectory import load, render, summarize

DEFAULT_TRAJECTORY_DIR = Path.home() / ".terminal-agent" / "trajectories"


def _agent_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="terminal-agent",
        description="A minimal coding agent for the terminal. With no -p it starts an "
        "interactive session; with -p it runs one task headlessly and exits.",
        epilog="Replay a logged run:  terminal-agent replay <trajectory.jsonl> [--summary]",
    )
    p.add_argument("-p", "--prompt", help="run this task headlessly and exit")
    p.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="repository to work in (default: current directory)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama URL (default {DEFAULT_HOST})")
    p.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="headless output format (default text)",
    )
    p.add_argument(
        "--approval-mode",
        choices=["default", "auto"],
        default="default",
        help="default: only read-only commands and tests run unasked; auto: "
        "non-destructive commands run too. Dangerous ones always need a human "
        "(and are refused headless).",
    )
    p.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="PREFIX",
        help="trust commands starting with PREFIX (repeatable), e.g. --allow 'make'",
    )
    p.add_argument("--max-steps", type=int, default=40, help="model calls per task (default 40)")
    p.add_argument(
        "--token-budget",
        type=int,
        default=12000,
        help="estimated prompt tokens before compaction (default 12000)",
    )
    p.add_argument(
        "--test-command",
        default="python -m pytest -q",
        help="what run_tests runs (default 'python -m pytest -q')",
    )
    p.add_argument(
        "--trajectory",
        type=Path,
        help=f"JSONL log path (default {DEFAULT_TRAJECTORY_DIR}/<time>.jsonl)",
    )
    p.add_argument("--cache-dir", type=Path, help="cache model responses here (resumable runs)")
    p.add_argument(
        "--script",
        type=Path,
        help="drive the agent from a JSON list of scripted turns instead of a model "
        "(for testing tools and policy without Ollama)",
    )
    return p


def _replay_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="terminal-agent replay", description="Pretty-print a JSONL trajectory."
    )
    p.add_argument("path", type=Path, help="trajectory .jsonl file")
    p.add_argument("--full", action="store_true", help="do not clip long outputs")
    p.add_argument("--summary", action="store_true", help="print aggregate stats as JSON only")
    return p


def replay_main(argv: list[str]) -> int:
    args = _replay_parser().parse_args(argv)
    if not args.path.is_file():
        print(f"error: no such trajectory file: {args.path}", file=sys.stderr)
        return 2
    events = load(args.path)
    if args.summary:
        print(json.dumps(summarize(events).as_dict(), indent=2))
    else:
        print(render(events, full=args.full))
    return 0


def _client(args: argparse.Namespace) -> ChatClient:
    if args.script:
        return ScriptedClient.from_file(args.script)
    return OllamaClient(model=args.model, host=args.host, cache_dir=args.cache_dir)


def _progress(event: dict[str, Any]) -> None:
    if event["type"] == "tool_call":
        args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
        print(f"  -> {event['name']} {args[:160]}", flush=True)
    elif event["type"] == "tool_result" and not event.get("ok"):
        print(f"     failed: {event.get('output', '')[:200]}", flush=True)
    elif event["type"] == "compaction":
        print(f"  (context compacted to ~{event.get('tokens_after')} tokens)", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["replay"]:
        return replay_main(argv[1:])
    args = _agent_parser().parse_args(argv)
    workspace: Path = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"error: workspace {workspace} is not a directory", file=sys.stderr)
        return 2
    if args.prompt is not None and not args.prompt.strip():
        print("error: -p needs a task; got an empty string", file=sys.stderr)
        return 2
    if args.prompt is None and args.output_format == "json":
        print(
            "error: --output-format json only applies to a headless run; add -p TASK",
            file=sys.stderr,
        )
        return 2
    if args.max_steps < 1 or args.token_budget < 1000:
        print("error: --max-steps must be >= 1 and --token-budget >= 1000", file=sys.stderr)
        return 2
    try:
        client = _client(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y%m%d-%H%M%S")
    trajectory = args.trajectory or DEFAULT_TRAJECTORY_DIR / f"{stamp}.jsonl"
    headless = args.prompt is not None
    agent: Agent = build_agent(
        workspace,
        client,
        tool_config=ToolConfig(test_command=args.test_command),
        policy=ApprovalPolicy(workspace, mode=args.approval_mode, allow=args.allow),
        approver=deny_all if headless else make_terminal_approver(),
        trajectory=trajectory,
        token_budget=args.token_budget,
        max_steps=args.max_steps,
    )
    try:
        if not headless:
            agent.logger.listener = _progress
            return run_repl(agent)
        if args.output_format == "text":
            agent.logger.listener = _progress
        result = agent.run(args.prompt)
    finally:
        agent.logger.close()
    if args.output_format == "json":
        print(json.dumps({**result.as_dict(), "trajectory": str(trajectory)}, indent=2))
    else:
        if result.error:
            print(f"[{result.status}] {result.error}", file=sys.stderr)
        print(result.final or f"({result.status})")
        print(f"[trajectory: {trajectory}]", file=sys.stderr)
    return 0 if result.status in ("finished", "no_tool_call") else 1


if __name__ == "__main__":
    raise SystemExit(main())
