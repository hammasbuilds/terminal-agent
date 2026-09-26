"""The model arm: run the agent with a real model on validated tasks, grade, and classify.

Only tasks that passed harness validation are run, so a failure here is the model's (or the
agent design's), not the harness's. Every generation is cached on disk by the client, and
each task's record is written as soon as it finishes, so an interrupted run resumes.
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from terminal_agent.agent import build_agent, deny_all
from terminal_agent.evals import specs
from terminal_agent.evals.local_tasks import (
    LocalTask,
    git_apply,
    local_test_command,
    run_pytest,
    test_env,
)
from terminal_agent.evals.patches import parse_patch
from terminal_agent.evals.stats import rate
from terminal_agent.evals.tasks import Container, Task, rmtree, test_section
from terminal_agent.llm import ChatClient
from terminal_agent.policy import ApprovalPolicy
from terminal_agent.sandbox import DockerSandbox, LocalSandbox, changed_files, snapshot
from terminal_agent.tools import ToolConfig
from terminal_agent.trajectory import load, summarize

TASK_PROMPT = """Resolve the following issue in the repository in your workspace.

<issue>
{problem}
</issue>

Find the code responsible, make the smallest change that fixes it, and check it with
run_tests if you can. Do not edit or add test files; hidden tests will be run afterwards.
Call finish when you are done."""


def _agent_for(ws: Path, client: ChatClient, sandbox: Any, shell_root: str | None,
               test_command: str, traj: Path, max_steps: int, token_budget: int) -> Any:
    return build_agent(
        ws, client, sandbox=sandbox,
        tool_config=ToolConfig(test_command=test_command, test_timeout=600),
        policy=ApprovalPolicy(ws, shell_root=shell_root, mode="auto"),
        approver=deny_all, trajectory=traj, max_steps=max_steps, token_budget=token_budget)


def run_swebench_task(task: Task, client: ChatClient, run_dir: Path, max_steps: int,
                      token_budget: int) -> dict[str, Any]:
    ws = run_dir / task.instance_id / "workspace"
    traj = run_dir / task.instance_id / "trajectory.jsonl"
    traj.unlink(missing_ok=True)
    t0 = time.monotonic()
    with Container(task) as c:
        c.export(ws)
        before = snapshot(ws)
        agent = _agent_for(ws, client, DockerSandbox(c.name, ws, prelude=specs.PRELUDE),
                           "/testbed", specs.test_command(task.repo, task.version), traj,
                           max_steps, token_budget)
        result = agent.run(TASK_PROMPT.format(problem=task.problem_statement))
        agent.logger.close()
    modified, deleted = changed_files(before, snapshot(ws))
    with Container(task, network=True) as ev:
        ev.push_changes(ws, before)
        patch = ev.diff()
        log, meta = ev.run_tests()
    grade = specs.grade(specs.PARSERS[task.repo](test_section(log)), task.fail_to_pass,
                        task.pass_to_pass)
    rmtree(ws)
    return _record(task.instance_id, task.repo, task.patch, result.as_dict(), modified, deleted,
                   patch, {**grade, **meta}, traj, time.monotonic() - t0)


def run_local_task(task: LocalTask, client: ChatClient, run_dir: Path, max_steps: int,
                   token_budget: int) -> dict[str, Any]:
    root = run_dir / task.instance_id.replace("@", "_")
    ws, ev = root / "workspace", root / "eval"
    traj = root / "trajectory.jsonl"
    traj.unlink(missing_ok=True)
    t0 = time.monotonic()
    task.materialize(ws)
    before = snapshot(ws)
    agent = _agent_for(ws, client, LocalSandbox(ws, env=test_env(ws)), None,
                       local_test_command(), traj, max_steps, token_budget)
    result = agent.run(TASK_PROMPT.format(problem=task.problem_statement))
    agent.logger.close()
    modified, deleted = changed_files(before, snapshot(ws))
    rmtree(ev)
    shutil.copytree(ws, ev)
    pristine = root / "pristine"
    task.materialize(pristine)
    for rel in task.test_files:  # the hidden tests win over anything the agent wrote
        if (pristine / rel).exists():
            (ev / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pristine / rel, ev / rel)
        else:
            (ev / rel).unlink(missing_ok=True)
    applied, _ = git_apply(ev, task.test_patch)
    log, timed_out = run_pytest(ev, task.test_files)
    grade = specs.grade(specs.parse_pytest(log), task.fail_to_pass, task.pass_to_pass)
    rec = _record(task.instance_id, task.repo, task.patch, result.as_dict(), modified, deleted,
                  "", {**grade, "test_patch_applied": applied, "timed_out": timed_out}, traj,
                  time.monotonic() - t0)
    for d in (ws, ev, pristine):
        rmtree(d)
    return rec


def _record(iid: str, repo: str, gold_patch: str, agent: dict[str, Any], modified: list[str],
            deleted: list[str], patch: str, grade: dict[str, Any], traj: Path,
            seconds: float) -> dict[str, Any]:
    events = load(traj) if traj.exists() else []
    edit_status = Counter(
        (e.get("meta") or {}).get("status", "ok" if e.get("ok") else "error")
        for e in events if e.get("type") == "tool_result" and e.get("name") == "edit")
    est = [(e["prompt_tokens"], e["estimated_prompt_tokens"]) for e in events
           if e.get("type") == "model" and e.get("prompt_tokens") and not e.get("cached")
           and e.get("estimated_prompt_tokens")]
    rec = {
        "instance_id": iid, "repo": repo, "agent": agent, "grade": grade,
        "changed": {"modified": modified, "deleted": deleted},
        "gold_files": [f.path for f in parse_patch(gold_patch)],
        "model_patch": patch, "edit_status": dict(edit_status),
        "token_estimate_ratio": [round(a / b, 3) for a, b in est],
        "trajectory_summary": summarize(events).as_dict(), "seconds": round(seconds, 1),
    }
    rec["outcome"] = classify(rec)
    return rec


def classify(rec: dict[str, Any]) -> str:
    """One primary outcome per task, first match wins."""
    status = rec["agent"]["status"]
    if status == "model_error":
        return "model_error"
    if rec["grade"]["resolved"]:
        return "resolved"
    touched = set(rec["changed"]["modified"]) | set(rec["changed"]["deleted"])
    if not touched:
        if rec.get("edit_status") and not rec["edit_status"].get("ok"):
            return "edits_never_applied"
        return {"loop": "loop_without_edit", "max_steps": "ran_out_of_steps",
                "no_tool_call": "gave_up"}.get(status, "no_edit")
    if not touched & set(rec["gold_files"]):
        return "wrong_file"
    if rec["grade"]["f2p_passed"] < rec["grade"]["f2p_total"]:
        return "wrong_fix"
    return "broke_existing_tests"


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {"tasks": 0}
    solved = sum(1 for r in records if r["outcome"] == "resolved")
    steps = [r["agent"]["steps"] for r in records]
    tools: Counter[str] = Counter()
    edits: Counter[str] = Counter()
    ratios: list[float] = []
    for r in records:
        tools.update(r["agent"]["tool_calls"])
        edits.update(r.get("edit_status", {}))
        ratios.extend(r.get("token_estimate_ratio", []))
    total_calls = sum(tools.values()) or 1
    edit_calls = sum(edits.values())
    by_repo: dict[str, list[int]] = {}
    for r in records:
        c = by_repo.setdefault(r["repo"], [0, 0])
        c[0] += r["outcome"] == "resolved"
        c[1] += 1
    return {
        "tasks": n,
        "solve_rate": rate(solved, n),
        "outcomes": dict(Counter(r["outcome"] for r in records).most_common()),
        "steps": {"median": statistics.median(steps), "mean": round(statistics.mean(steps), 2),
                  "max": max(steps)},
        "tokens": {
            "prompt": sum(r["agent"]["tokens"]["prompt"] for r in records),
            "completion": sum(r["agent"]["tokens"]["completion"] for r in records),
            "per_task_median": statistics.median(
                r["agent"]["tokens"]["prompt"] + r["agent"]["tokens"]["completion"]
                for r in records),
        },
        "tool_share": {k: round(v / total_calls, 4) for k, v in tools.most_common()},
        "edit_calls": edit_calls,
        "edit_status": dict(edits),
        "edit_failure_rate": rate(edit_calls - edits.get("ok", 0), edit_calls),
        "token_estimator_ratio_median": statistics.median(ratios) if ratios else None,
        "compactions": sum(r["agent"]["compactions"] for r in records),
        "denied_calls": sum(len(r["agent"]["denied"]) for r in records),
        "by_repo": {k: {"resolved": v[0], "tasks": v[1]} for k, v in sorted(by_repo.items())},
    }


def run_all(jobs: list[Any], runner: Callable[[Any], dict[str, Any]], out_dir: Path,
            log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for job in jobs:
        target = out_dir / f"{job.instance_id.replace('@', '_')}.json"
        if target.exists():
            records.append(json.loads(target.read_text(encoding="utf-8")))
            continue
        rec = runner(job)
        target.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8", newline="\n")
        records.append(rec)
        log(f"{job.instance_id}: {rec['outcome']} in {rec['agent']['steps']} steps")
    return records
