"""The model arm: run the agent with a real model on validated tasks, grade, and classify.

Only tasks that passed harness validation are run, so a failure here is the model's (or the
agent design's), not the harness's. Every generation is cached on disk by the client, and
each task's record is written as soon as it finishes, so an interrupted run resumes.

The agent runs in ``auto`` approval mode, which lets mutating commands (``python x.py``,
``make``) run unasked; the policy is a filter, not a sandbox. So every task's shell runs in
a disposable, network-less container: the SWE-bench image, or ``python:3.11-bookworm`` for
the local suite (:mod:`terminal_agent.evals.local_container`). Running the local suite on
the host needs an explicit ``isolation="host"`` (``ta-eval model-run --local-on-host``).
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypedDict, TypeVar, cast

from terminal_agent.agent import Agent, build_agent, deny_all
from terminal_agent.evals import specs
from terminal_agent.evals.local_container import TEST_COMMAND, LocalTaskContainer
from terminal_agent.evals.local_tasks import (
    LocalTask,
    git_apply,
    git_init_isolated,
    local_test_command,
    run_pytest,
    test_env,
)
from terminal_agent.evals.patches import parse_patch
from terminal_agent.evals.stats import bootstrap_mean_ci, rate
from terminal_agent.evals.tasks import Container, Task, rmtree, test_section
from terminal_agent.llm import DEFAULT_OPTIONS, ChatClient
from terminal_agent.policy import ApprovalPolicy
from terminal_agent.sandbox import (
    DockerSandbox,
    LocalSandbox,
    Sandbox,
    changed_files,
    snapshot,
)
from terminal_agent.tools import ToolConfig
from terminal_agent.trajectory import load, summarize

NUM_CTX = int(DEFAULT_OPTIONS["num_ctx"])
TASK_PROMPT = """Resolve the following issue in the repository in your workspace.

<issue>
{problem}
</issue>

Find the code responsible, make the smallest change that fixes it, and check it with
run_tests if you can. Do not edit or add test files; hidden tests will be run afterwards.
Call finish when you are done."""
# outcomes that are not the model's answer: retried, never persisted, counted as unsolved
ERROR_OUTCOMES = ("model_error", "harness_error")

Isolation = Literal["container", "host"]
Job = TypeVar("Job", Task, LocalTask)


class AgentSummary(TypedDict):
    status: str
    response: str
    steps: int
    tool_calls: dict[str, int]
    tool_errors: dict[str, int]
    denied: list[str]
    tokens: dict[str, int]
    compactions: int
    error: str


class RunRecord(TypedDict):
    instance_id: str
    repo: str
    agent: AgentSummary
    grade: dict[str, object]
    changed: dict[str, list[str]]
    gold_files: list[str]
    model_patch: str
    edit_status: dict[str, int]
    token_estimate_ratio: list[float]
    calls_near_context_limit: int
    trajectory_summary: dict[str, object]
    seconds: float
    outcome: str


def call_bound(tasks: int, max_steps: int, retries: int) -> int:
    """Most model calls a run can make: every attempt of every task uses every step.

    A task is attempted once plus once per retry (a retry follows only a model error, and
    replays earlier steps from the generation cache, so the real count is usually far lower).
    """
    return tasks * max_steps * (1 + retries)


def _agent_for(
    ws: Path,
    client: ChatClient,
    sandbox: Sandbox,
    shell_root: str | None,
    test_command: str,
    traj: Path,
    max_steps: int,
    token_budget: int,
) -> Agent:
    return build_agent(
        ws,
        client,
        sandbox=sandbox,
        tool_config=ToolConfig(test_command=test_command, test_timeout=600),
        policy=ApprovalPolicy(ws, shell_root=shell_root, mode="auto"),
        approver=deny_all,
        trajectory=traj,
        max_steps=max_steps,
        token_budget=token_budget,
    )


def _agent_summary(agent: Agent, prompt: str) -> AgentSummary:
    result = agent.run(prompt)
    agent.logger.close()
    return cast(AgentSummary, result.as_dict())


def run_swebench_task(
    task: Task, client: ChatClient, run_dir: Path, max_steps: int, token_budget: int
) -> RunRecord:
    ws = run_dir / task.instance_id / "workspace"
    traj = run_dir / task.instance_id / "trajectory.jsonl"
    traj.unlink(missing_ok=True)
    t0 = time.monotonic()
    with Container(task) as c:
        c.export(ws)
        before = snapshot(ws)
        agent = _agent_for(
            ws,
            client,
            DockerSandbox(c.name, ws, prelude=specs.PRELUDE),
            "/testbed",
            specs.test_command(task.repo, task.version),
            traj,
            max_steps,
            token_budget,
        )
        summary = _agent_summary(agent, TASK_PROMPT.format(problem=task.problem_statement))
    modified, deleted = changed_files(before, snapshot(ws))
    with Container(task, network=True) as ev:
        ev.push_changes(ws, before)
        patch = ev.diff()
        log, meta = ev.run_tests()
    grade = specs.grade(
        specs.PARSERS[task.repo](test_section(log)), task.fail_to_pass, task.pass_to_pass
    )
    rmtree(ws)
    return _record(
        task.instance_id,
        task.repo,
        task.patch,
        summary,
        modified,
        deleted,
        patch,
        {**grade, **meta},
        traj,
        time.monotonic() - t0,
    )


def _eval_tree(task: LocalTask, ws: Path, root: Path) -> tuple[Path, bool]:
    """A copy of the agent's workspace with the hidden tests restored and the test patch."""
    ev, pristine = root / "eval", root / "pristine"
    rmtree(ev)
    shutil.copytree(ws, ev, ignore=shutil.ignore_patterns(".git"))
    task.materialize(pristine)
    for rel in task.test_files:  # the hidden tests win over anything the agent wrote
        if (pristine / rel).exists():
            (ev / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pristine / rel, ev / rel)
        else:
            (ev / rel).unlink(missing_ok=True)
    rmtree(pristine)
    applied, _ = git_apply(ev, task.test_patch)  # git apply runs nothing from the tree
    return ev, applied


def run_local_task(
    task: LocalTask,
    client: ChatClient,
    run_dir: Path,
    max_steps: int,
    token_budget: int,
    isolation: Isolation = "container",
) -> RunRecord:
    """Run one mined local task. ``isolation="host"`` runs the model's shell on this machine."""
    root = run_dir / task.instance_id.replace("@", "_")
    ws = root / "workspace"
    traj = root / "trajectory.jsonl"
    traj.unlink(missing_ok=True)
    t0 = time.monotonic()
    task.materialize(ws)
    git_init_isolated(ws)  # contain the model's git commands to this throwaway tree
    before = snapshot(ws)
    prompt = TASK_PROMPT.format(problem=task.problem_statement)
    if isolation == "container":
        with LocalTaskContainer() as box:
            box.load(ws)
            agent = _agent_for(
                ws,
                client,
                box.sandbox(ws),
                "/workspace",
                TEST_COMMAND,
                traj,
                max_steps,
                token_budget,
            )
            summary = _agent_summary(agent, prompt)
    else:
        agent = _agent_for(
            ws,
            client,
            LocalSandbox(ws, env=test_env(ws)),
            None,
            local_test_command(),
            traj,
            max_steps,
            token_budget,
        )
        summary = _agent_summary(agent, prompt)
    modified, deleted = changed_files(before, snapshot(ws))
    ev, applied = _eval_tree(task, ws, root)
    if isolation == "container":  # the model's code is imported by the tests: grade inside
        with LocalTaskContainer() as box:
            box.load(ev)
            log, timed_out = box.run_pytest(task.test_files)
    else:
        log, timed_out = run_pytest(ev, task.test_files)
    grade = specs.grade(specs.parse_pytest(log), task.fail_to_pass, task.pass_to_pass)
    rec = _record(
        task.instance_id,
        task.repo,
        task.patch,
        summary,
        modified,
        deleted,
        "",
        {**grade, "test_patch_applied": applied, "timed_out": timed_out, "isolation": isolation},
        traj,
        time.monotonic() - t0,
    )
    for d in (ws, ev):
        rmtree(d)
    return rec


def _record(
    iid: str,
    repo: str,
    gold_patch: str,
    agent: AgentSummary,
    modified: list[str],
    deleted: list[str],
    patch: str,
    grade: dict[str, object],
    traj: Path,
    seconds: float,
) -> RunRecord:
    events = load(traj) if traj.exists() else []
    edit_status = Counter(
        (e.get("meta") or {}).get("status", "ok" if e.get("ok") else "error")
        for e in events
        if e.get("type") == "tool_result" and e.get("name") == "edit"
    )
    est = [
        (e["prompt_tokens"], e["estimated_prompt_tokens"])
        for e in events
        if e.get("type") == "model"
        and e.get("prompt_tokens")
        and not e.get("cached")
        and e.get("estimated_prompt_tokens")
    ]
    rec = RunRecord(
        instance_id=iid,
        repo=repo,
        agent=agent,
        grade=grade,
        changed={"modified": modified, "deleted": deleted},
        gold_files=[f.path for f in parse_patch(gold_patch)],
        model_patch=patch,
        edit_status=dict(edit_status),
        token_estimate_ratio=[round(a / b, 3) for a, b in est],
        calls_near_context_limit=sum(
            1
            for e in events
            if e.get("type") == "model" and (e.get("prompt_tokens") or 0) >= 0.95 * NUM_CTX
        ),
        trajectory_summary=summarize(events).as_dict(),
        seconds=round(seconds, 1),
        outcome="",
    )
    rec["outcome"] = classify(rec)
    return rec


def error_record(iid: str, repo: str, exc: BaseException) -> RunRecord:
    """A task whose run crashed outside the model (docker, the filesystem, a harness bug)."""
    agent = AgentSummary(
        status="harness_error",
        response="",
        steps=0,
        tool_calls={},
        tool_errors={},
        denied=[],
        tokens={"prompt": 0, "completion": 0},
        compactions=0,
        error=f"{type(exc).__name__}: {exc}",
    )
    return RunRecord(
        instance_id=iid,
        repo=repo,
        agent=agent,
        grade={"resolved": False},
        changed={"modified": [], "deleted": []},
        gold_files=[],
        model_patch="",
        edit_status={},
        token_estimate_ratio=[],
        calls_near_context_limit=0,
        trajectory_summary={},
        seconds=0.0,
        outcome="harness_error",
    )


def classify(rec: RunRecord) -> str:
    """One primary outcome per task, first match wins."""
    status = rec["agent"]["status"]
    if status in ERROR_OUTCOMES:
        return status
    if rec["grade"]["resolved"]:
        return "resolved"
    touched = set(rec["changed"]["modified"]) | set(rec["changed"]["deleted"])
    if not touched:
        if rec.get("edit_status") and not rec["edit_status"].get("ok"):
            return "edits_never_applied"
        return {
            "loop": "loop_without_edit",
            "max_steps": "ran_out_of_steps",
            "no_tool_call": "gave_up",
        }.get(status, "no_edit")
    if not touched & set(rec["gold_files"]):
        return "wrong_file"
    if cast(int, rec["grade"]["f2p_passed"]) < cast(int, rec["grade"]["f2p_total"]):
        return "wrong_fix"
    return "broke_existing_tests"


def aggregate(records: Sequence[RunRecord]) -> dict[str, object]:
    # A model_error or harness_error that survived its retries is counted as UNSOLVED in the
    # headline rate: dropping it would flatter the model whenever the server flakes on hard
    # tasks (long prompts time out more). The rate over completed runs is reported beside it.
    total = len(records)
    model_errors = sum(1 for r in records if r["outcome"] == "model_error")
    harness_errors = sum(1 for r in records if r["outcome"] == "harness_error")
    solved = sum(1 for r in records if r["outcome"] == "resolved")
    done = [r for r in records if r["outcome"] not in ERROR_OUTCOMES]
    n = len(done)
    if not n:
        return {
            "tasks": total,
            "model_errors": model_errors,
            "harness_errors": harness_errors,
            "solve_rate": rate(0, total),
            "solve_rate_completed_only": rate(0, 0),
        }
    steps = [r["agent"]["steps"] for r in done]
    tools: Counter[str] = Counter()
    edits: Counter[str] = Counter()
    ratios: list[float] = []
    for r in done:
        tools.update(r["agent"]["tool_calls"])
        edits.update(r.get("edit_status", {}))
        ratios.extend(r.get("token_estimate_ratio", []))
    total_calls = sum(tools.values()) or 1
    edit_calls = sum(edits.values())
    by_repo: dict[str, list[int]] = {}
    for r in done:
        c = by_repo.setdefault(r["repo"], [0, 0])
        c[0] += r["outcome"] == "resolved"
        c[1] += 1
    return {
        "tasks": total,
        "completed": n,
        "model_errors": model_errors,
        "harness_errors": harness_errors,
        "solve_rate": rate(solved, total),
        "solve_rate_completed_only": rate(solved, n),
        "outcomes": dict(Counter(r["outcome"] for r in done).most_common()),
        "steps": {
            "median": statistics.median(steps),
            "mean": round(statistics.mean(steps), 2),
            "mean_ci95": list(bootstrap_mean_ci([float(s) for s in steps])),
            "max": max(steps),
        },
        "tokens": {
            "prompt": sum(r["agent"]["tokens"]["prompt"] for r in done),
            "completion": sum(r["agent"]["tokens"]["completion"] for r in done),
            "per_task_median": statistics.median(
                r["agent"]["tokens"]["prompt"] + r["agent"]["tokens"]["completion"] for r in done
            ),
        },
        "tool_share": {k: round(v / total_calls, 4) for k, v in tools.most_common()},
        "edit_calls": edit_calls,
        "edit_status": dict(edits),
        "edit_failure_rate": rate(edit_calls - edits.get("ok", 0), edit_calls),
        "token_estimator_ratio_median": statistics.median(ratios) if ratios else None,
        "compactions": sum(r["agent"]["compactions"] for r in done),
        "calls_near_context_limit": sum(r.get("calls_near_context_limit", 0) for r in done),
        "denied_calls": sum(len(r["agent"]["denied"]) for r in done),
        "by_repo": {k: {"resolved": v[0], "tasks": v[1]} for k, v in sorted(by_repo.items())},
    }


def run_all(
    jobs: Sequence[Job],
    runner: Callable[[Job], RunRecord],
    out_dir: Path,
    log: Callable[[str], None] = print,
    retries: int = 2,
) -> list[RunRecord]:
    """Run every job, retrying an error outcome rather than recording it.

    A model_error (Ollama unreachable, timeout, a bad response) or a harness_error (the
    runner raised: docker failed, a disk filled) is retried up to ``retries`` times and
    never persisted, so a resumed run tries the task again. One that survives its retries is
    returned (and counted as unsolved by ``aggregate``) but not written to disk. A crash in
    one task never stops the others.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = []

    def attempt(job: Job) -> RunRecord:
        try:
            return runner(job)
        except Exception as exc:
            return error_record(job.instance_id, job.repo, exc)

    for job in jobs:
        target = out_dir / f"{job.instance_id.replace('@', '_')}.json"
        if target.exists():  # a persisted record is always a real (non-error) outcome
            records.append(cast(RunRecord, json.loads(target.read_text(encoding="utf-8"))))
            continue
        rec = attempt(job)
        for n in range(1, retries + 1):
            if rec["outcome"] not in ERROR_OUTCOMES:
                break
            log(
                f"{job.instance_id}: {rec['agent'].get('error') or rec['outcome']}; "
                f"retry {n}/{retries}"
            )
            rec = attempt(job)
        if rec["outcome"] in ERROR_OUTCOMES:
            log(f"{job.instance_id}: {rec['outcome']} after {retries} retries (not recorded)")
            records.append(rec)
            continue
        target.write_text(
            json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        records.append(rec)
        log(f"{job.instance_id}: {rec['outcome']} in {rec['agent']['steps']} steps")
    return records
