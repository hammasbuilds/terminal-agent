"""Runs only with -m docker, against the pulled psf__requests-3362 image."""

from pathlib import Path

import pytest

from terminal_agent.evals import model_run, specs
from terminal_agent.evals.patches import parse_patch
from terminal_agent.evals.tasks import Container, load_tasks, select
from terminal_agent.evals.validate import image_present
from terminal_agent.llm import ModelTurn, ScriptedClient
from terminal_agent.protocol import ToolCall
from terminal_agent.sandbox import DockerSandbox, snapshot
from terminal_agent.tools import Toolbox, ToolConfig

pytestmark = pytest.mark.docker
IID = "psf__requests-3362"


@pytest.fixture(scope="module")
def task():
    (t,) = select(load_tasks(), [IID])
    if not image_present(t.image):
        pytest.skip(f"{t.image} not pulled")
    return t


def test_sandbox_syncs_both_ways(task, tmp_path: Path):
    ws = tmp_path / "ws"
    with Container(task) as c:
        c.export(ws)
        box = Toolbox(ws, DockerSandbox(c.name, ws, prelude=specs.PRELUDE),
                      ToolConfig(test_command=specs.test_command(task.repo, task.version)))
        # a local edit is visible to a command in the container
        assert box.execute(ToolCall("edit", {"path": "requests/__init__.py",
                                             "old_string": "__title__ = 'requests'",
                                             "new_string": "__title__ = 'REQ'"})).ok
        out = box.execute(ToolCall("run_shell", {
            "command": "python -c 'import requests; print(requests.__title__)'"}))
        assert out.ok and "REQ" in out.output
        # a file changed by a command comes back to the workspace
        assert box.execute(ToolCall("run_shell", {
            "command": "sed -i 's/REQ/SED/' requests/__init__.py && echo new > NEW.txt"})).ok
        assert "__title__ = 'SED'" in (ws / "requests" / "__init__.py").read_text()
        assert (ws / "NEW.txt").read_text() == "new\n"
        tests = box.execute(ToolCall("run_tests", {"target": "tests/test_structures.py"}))
        assert tests.ok and "passed" in tests.output
        no_net = box.execute(ToolCall("run_shell", {"command": "curl -sS -m 5 http://example.com"}))
        assert not no_net.ok  # the agent's container is offline
    assert "NEW.txt" in snapshot(ws)


def test_model_run_on_docker_task_with_a_scripted_fix(task, tmp_path: Path):
    (fp,) = parse_patch(task.patch)
    h = fp.hunks[0]
    client = ScriptedClient([
        ModelTurn("", [ToolCall("edit", {"path": fp.path, "old_string": h.old_text,
                                         "new_string": h.new_text})]),
        ModelTurn("", [ToolCall("finish", {"summary": "done"})]),
    ])
    rec = model_run.run_swebench_task(task, client, tmp_path, 5, 12000)
    assert rec["outcome"] == "resolved" and rec["grade"]["resolved"]
    assert rec["model_patch"].startswith("diff --git")


def test_git_status_z_rename_is_parsed_and_synced(task, tmp_path):
    ws = tmp_path / "ws"
    with Container(task) as c:
        c.export(ws)
        sb = DockerSandbox(c.name, ws, prelude=specs.PRELUDE)
        # a rename produces a two-field "R  new\0old" record in `git status -z`
        sb.run("git mv requests/api.py requests/api_renamed.py", 60)
        assert (ws / "requests" / "api_renamed.py").exists()
        assert not (ws / "requests" / "api.py").exists()  # the old path was pulled as gone


def test_pushed_file_keeps_its_executable_bit(task, tmp_path):
    ws = tmp_path / "ws"
    with Container(task) as c:
        c.export(ws)
        c.sh("chmod +x /testbed/setup.py")
        sb = DockerSandbox(c.name, ws, prelude=specs.PRELUDE)
        sb._synced = snapshot(ws)  # pretend nothing synced yet
        (ws / "setup.py").write_bytes((ws / "setup.py").read_bytes() + b"\n# edit\n")
        sb.push()
        _, out = c.sh("test -x /testbed/setup.py && echo EXECUTABLE")
        assert "EXECUTABLE" in out  # overwriting did not strip the +x bit
