from pathlib import Path

from terminal_agent.evals import model_run
from terminal_agent.evals.local_tasks import (
    candidate_commits,
    load_local_tasks,
    mine_commit,
    save_local_tasks,
)
from terminal_agent.evals.validate import validate_local_task
from terminal_agent.llm import ModelTurn, ScriptedClient
from terminal_agent.protocol import ToolCall


def _task(bugfix_repo, tmp_path):
    repo, sha = bugfix_repo
    assert sha in candidate_commits(repo)
    task = mine_commit(repo, "calcrepo", sha, tmp_path / "scratch")
    assert task is not None
    return task


def test_mining_finds_fail_to_pass(bugfix_repo, tmp_path: Path):
    task = _task(bugfix_repo, tmp_path)
    assert task.fail_to_pass == ["tests/test_calc.py::test_mean"]
    assert task.pass_to_pass == ["tests/test_calc.py::test_clamp"]
    assert "n-1" in task.problem_statement
    assert "len(xs) - 1" in task.patch and "test_mean" in task.test_patch
    save_local_tasks([task], tmp_path / "t.jsonl.gz")
    (back,) = load_local_tasks(tmp_path / "t.jsonl.gz")
    assert back.as_record() == task.as_record()


def test_gold_validation_passes_all_four_checks(bugfix_repo, tmp_path: Path):
    task = _task(bugfix_repo, tmp_path)
    rec = validate_local_task(task, tmp_path / "runs", tmp_path / "logs")
    assert rec["verdict"] == "valid", rec["reasons"]
    assert rec["baseline"]["f2p_passed"] == 0 and rec["gold"]["resolved"]
    assert [h["final_status"] for h in rec["hunks"]] == ["ok"]
    assert rec["byte_mismatch"] == []
    assert (tmp_path / "logs" / f"{task.instance_id.replace('@', '_')}.baseline.log.gz").exists()


def test_validation_catches_a_task_whose_tests_do_not_fail(bugfix_repo, tmp_path: Path):
    task = _task(bugfix_repo, tmp_path)
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
        ModelTurn("", [ToolCall("run_tests", {})]),
        ModelTurn("", [ToolCall("finish", {"summary": "fixed mean"})]),
    ])


def test_model_run_grades_and_classifies(bugfix_repo, tmp_path: Path):
    task = _task(bugfix_repo, tmp_path)
    good = model_run.run_local_task(task, _fix_script(), tmp_path / "m", 10, 12000)
    assert good["outcome"] == "resolved" and good["agent"]["steps"] == 4
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
