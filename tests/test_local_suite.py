import copy
from pathlib import Path

import pytest

from terminal_agent.evals import model_run
from terminal_agent.evals.local_tasks import (
    candidate_commits,
    git_apply,
    load_local_tasks,
    mine_commit,
    save_local_tasks,
)
from terminal_agent.evals.validate import validate_local_task
from terminal_agent.llm import ModelTurn, ScriptedClient
from terminal_agent.protocol import ToolCall


@pytest.fixture(scope="module")
def mined(bugfix_repo, tmp_path_factory):
    repo, sha = bugfix_repo
    assert sha in candidate_commits(repo)
    task = mine_commit(repo, "calcrepo", sha, tmp_path_factory.mktemp("scratch"))
    assert task is not None
    return task


def _task(mined):
    return copy.deepcopy(mined)


def test_mining_finds_fail_to_pass(mined, tmp_path: Path):
    task = _task(mined)
    assert task.fail_to_pass == ["tests/test_calc.py::test_mean"]
    assert task.pass_to_pass == ["tests/test_calc.py::test_clamp"]
    assert "n-1" in task.problem_statement
    assert "len(xs) - 1" in task.patch and "test_mean" in task.test_patch
    save_local_tasks([task], tmp_path / "t.jsonl.gz")
    (back,) = load_local_tasks(tmp_path / "t.jsonl.gz")
    assert back.as_record() == task.as_record()


def test_gold_validation_passes_all_four_checks(mined, tmp_path: Path):
    task = _task(mined)
    rec = validate_local_task(task, tmp_path / "runs", tmp_path / "logs")
    assert rec["verdict"] == "valid", rec["reasons"]
    assert rec["baseline"]["f2p_passed"] == 0 and rec["gold"]["resolved"]
    assert [h["final_status"] for h in rec["hunks"]] == ["ok"]
    assert rec["byte_mismatch"] == []
    assert (tmp_path / "logs" / f"{task.instance_id.replace('@', '_')}.baseline.log.gz").exists()


def test_validation_catches_a_task_whose_tests_do_not_fail(mined, tmp_path: Path):
    task = _task(mined)
    task.fail_to_pass = ["tests/test_calc.py::test_clamp"]  # passes before the fix
    rec = validate_local_task(task, tmp_path / "runs", tmp_path / "logs")
    assert rec["verdict"] == "invalid"
    assert any("already pass before the fix" in r for r in rec["reasons"])


def _fix_script() -> ScriptedClient:
    return ScriptedClient([
        ModelTurn("", [ToolCall("read_file", {"path": "src/calc.py"})]),
        ModelTurn("", [ToolCall("edit", {"path": "src/calc.py",
                                         "old_string": "(len(xs) - 1)",
                                         "new_string": "len(xs)"})]),
        ModelTurn("", [ToolCall("run_tests", {"target": "tests/test_calc.py"})]),
        ModelTurn("", [ToolCall("finish", {"summary": "fixed mean"})]),
    ])


def test_model_run_grades_and_classifies(mined, tmp_path: Path):
    task = _task(mined)
    good = model_run.run_local_task(task, _fix_script(), tmp_path / "m", 10, 12000)
    assert good["outcome"] == "resolved" and good["agent"]["steps"] == 4
    assert good["agent"]["tool_errors"] == {}  # run_tests found and ran the tests
    assert good["edit_status"] == {"ok": 1}
    wrong_place = ScriptedClient([
        ModelTurn("", [ToolCall("write_file", {"path": "src/other.py", "content": "x = 1\n"})]),
        ModelTurn("", [ToolCall("finish", {"summary": "done"})]),
    ])
    rec = model_run.run_local_task(task, wrong_place, tmp_path / "m2", 10, 12000)
    assert rec["outcome"] == "wrong_file"
    gave_up = model_run.run_local_task(task, ScriptedClient([ModelTurn("I cannot.")]),
                                       tmp_path / "m3", 10, 12000)
    assert gave_up["outcome"] == "gave_up"
    cheat = ScriptedClient([
        ModelTurn("", [ToolCall("write_file", {"path": "tests/test_calc.py",
                                               "content": "def test_mean():\n    pass\n"})]),
        ModelTurn("", [ToolCall("finish", {"summary": "tests pass"})]),
    ])
    rec = model_run.run_local_task(task, cheat, tmp_path / "m4", 10, 12000)
    assert rec["outcome"] == "wrong_file" and not rec["grade"]["resolved"]
    summary = model_run.aggregate([good, rec, gave_up])
    assert summary["solve_rate"]["k"] == 1 and summary["tasks"] == 3
    assert summary["outcomes"]["resolved"] == 1


def test_git_apply_works_inside_an_enclosing_repository(mined, bugfix_repo):
    # runs/ lives inside this repo; git used to resolve paths against the outer repo's root,
    # skip every file and still exit 0 (found by harness validation, not by a test)
    repo, _ = bugfix_repo
    nested = repo / "runs" / "ws"
    mined.materialize(nested)
    ok, _ = git_apply(nested, mined.patch)
    assert ok and "sum(xs) / len(xs)\n" in (nested / "src" / "calc.py").read_text()


def test_git_commands_in_a_local_workspace_cannot_touch_the_enclosing_repo(tmp_path):
    # simulate the harness repo: an outer git repo with the workspace nested inside it
    import subprocess

    from terminal_agent.evals.local_tasks import git_init_isolated, test_env
    from terminal_agent.sandbox import LocalSandbox

    def _git(repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                              text=True).stdout

    outer = tmp_path / "harness"
    outer.mkdir()
    _git(outer, "init", "-q")
    (outer / "keep.txt").write_bytes(b"harness data\n")
    _git(outer, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(outer, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "harness")
    outer_head = _git(outer, "rev-parse", "HEAD").strip()

    ws = outer / "runs" / "model" / "ws"
    ws.mkdir(parents=True)
    (ws / "a.py").write_bytes(b"x = 1\n")
    git_init_isolated(ws)
    sb = LocalSandbox(ws, env=test_env(ws))
    (ws / "a.py").write_bytes(b"x = 2\n")
    sb.run("git add -A && git -c user.name=m -c user.email=m@m commit -q -m pwn", 30)
    sb.run("git stash", 30)
    # the enclosing harness repo is untouched: same HEAD, clean tree, file intact
    assert _git(outer, "rev-parse", "HEAD").strip() == outer_head
    # no TRACKED file of the harness was staged, modified or deleted (a nested untracked
    # runs/ dir is fine); the inner repo swallowed the add/commit/stash
    assert _git(outer, "status", "--porcelain", "-uno").strip() == ""
    assert (outer / "keep.txt").read_bytes() == b"harness data\n"


def test_model_errors_are_retried_not_persisted_and_not_scored(tmp_path):
    from types import SimpleNamespace

    calls = {"a@1": 0}

    def runner(job):
        calls[job.instance_id] += 1
        # fail twice with a model_error, then succeed
        if calls[job.instance_id] < 3:
            return {"instance_id": job.instance_id, "repo": "x", "outcome": "model_error",
                    "agent": {"steps": 0, "error": "cannot reach Ollama"}}
        return {"instance_id": job.instance_id, "repo": "x", "outcome": "resolved",
                "agent": {"steps": 4}}

    jobs = [SimpleNamespace(instance_id="a@1")]
    out = tmp_path / "mr"
    recs = model_run.run_all(jobs, runner, out, log=lambda s: None, retries=3)
    assert calls["a@1"] == 3 and recs[0]["outcome"] == "resolved"
    assert (out / "a_1.json").exists()  # only the successful record is persisted


def test_model_error_excluded_from_solve_rate_denominator():
    def rec(outcome, steps):
        return {"outcome": outcome, "repo": "x",
                "agent": {"steps": steps, "tokens": {"prompt": 1, "completion": 1},
                          "tool_calls": {}, "denied": [], "compactions": 0}}

    agg = model_run.aggregate([rec("resolved", 3), rec("wrong_fix", 5), rec("model_error", 0)])
    assert agg["tasks"] == 2 and agg["model_errors"] == 1
    assert agg["solve_rate"]["k"] == 1 and agg["solve_rate"]["n"] == 2  # error not in denominator
