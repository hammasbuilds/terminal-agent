import json
from pathlib import Path

import pytest

from terminal_agent.agent import build_agent
from terminal_agent.cli import main
from terminal_agent.evals.cli import main as ta_eval
from terminal_agent.llm import ModelTurn, ScriptedClient
from terminal_agent.protocol import ToolCall
from terminal_agent.repl import make_terminal_approver, run_repl


def _script(tmp_path: Path) -> Path:
    p = tmp_path / "script.json"
    p.write_text(json.dumps([
        {"tool_calls": [{"name": "edit", "arguments": {
            "path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}}]},
        {"tool_calls": [{"name": "run_shell", "arguments": {"command": "git push --force"}}]},
        {"tool_calls": [{"name": "finish", "arguments": {"summary": "x is 2"}}]},
    ]))
    return p


def test_headless_json(tmp_path: Path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_bytes(b"x = 1\n")
    traj = tmp_path / "t.jsonl"
    code = main(["-p", "set x", "-w", str(ws), "--script", str(_script(tmp_path)),
                 "--output-format", "json", "--trajectory", str(traj)])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["status"] == "finished" and out["response"] == "x is 2"
    assert out["denied"] == ["git push --force"] and out["tool_calls"]["edit"] == 1
    assert (ws / "a.py").read_bytes() == b"x = 2\n"
    assert main(["replay", str(traj), "--summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["approvals"] == {"denied": 1} and summary["status"] == ["finished"]
    assert main(["replay", str(traj)]) == 0
    assert "approval: denied [dangerous]" in capsys.readouterr().out


def test_usage_errors(tmp_path: Path, capsys):
    assert main(["-p", "x", "-w", str(tmp_path / "missing")]) == 2
    assert main(["replay", str(tmp_path / "none.jsonl")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert main(["-p", "x", "-w", str(tmp_path), "--script", str(bad)]) == 2
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "headlessly" in capsys.readouterr().out


def test_headless_model_error_exit_code(tmp_path: Path, capsys):
    code = main(["-p", "x", "-w", str(tmp_path), "--host", "http://127.0.0.1:9",
                 "--output-format", "json", "--trajectory", str(tmp_path / "t.jsonl")])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["status"] == "model_error" and "ollama serve" in out["error"]


def test_repl_commands_and_approval(tmp_path: Path):
    client = ScriptedClient([
        ModelTurn("", [ToolCall("run_shell", {"command": "make"})]),
        ModelTurn("", [ToolCall("finish", {"summary": "built"})]),
    ])
    lines = iter(["/help", "/tools", "/tokens", "/bogus", "", "build it", "n", "/reset", "/exit"])
    printed: list[str] = []

    def fake_input(prompt: str) -> str:
        return next(lines)

    agent = build_agent(tmp_path, client,
                        approver=make_terminal_approver(fake_input, printed.append))
    assert run_repl(agent, fake_input, printed.append) == 0
    text = "\n".join(printed)
    assert "/reset" in text and "read_file" in text and "unknown command /bogus" in text
    assert "[needs approval] run_shell: make" in text and "built" in text
    assert len(agent.messages) == 1  # /reset cleared the conversation


def test_ta_eval_reports_unknown_ids_without_a_traceback(capsys):
    assert ta_eval(["validate", "--suite", "swebench", "--ids", "no__such-1"]) == 2
    assert "unknown instance id(s): no__such-1" in capsys.readouterr().err
