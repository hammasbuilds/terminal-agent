import json
from pathlib import Path

from terminal_agent.agent import build_agent
from terminal_agent.context import ContextManager, conversation_tokens, estimate_tokens, truncate
from terminal_agent.llm import ModelError, ModelTurn, ScriptedClient
from terminal_agent.protocol import ToolCall, parse_text_tool_calls
from terminal_agent.trajectory import load, render, summarize


def turn(name: str, **args) -> ModelTurn:
    return ModelTurn("", [ToolCall(name, args)])


def test_agent_runs_tools_logs_and_finishes(tmp_path: Path):
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    client = ScriptedClient([
        turn("read_file", path="a.py"),
        turn("edit", path="a.py", old_string="x = 1", new_string="x = 2"),
        turn("finish", summary="set x to 2"),
    ])
    traj = tmp_path / "t.jsonl"
    agent = build_agent(tmp_path, client, trajectory=traj)
    res = agent.run("change x")
    agent.logger.close()
    assert res.status == "finished" and res.final == "set x to 2" and res.steps == 3
    assert (tmp_path / "a.py").read_bytes() == b"x = 2\n"
    events = load(traj)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("tool_result") == 2
    s = summarize(events)
    assert s.tool_calls == {"read_file": 1, "edit": 1, "finish": 1} and s.status == ["finished"]
    text = render(events)
    assert "-> edit" in text and "run end: finished" in text


def test_headless_denies_what_needs_approval(tmp_path: Path):
    client = ScriptedClient([turn("run_shell", command="rm -rf /"), turn("finish", summary="x")])
    agent = build_agent(tmp_path, client)
    res = agent.run("clean up")
    assert res.denied == ["rm -rf /"] and res.tool_errors["run_shell"] == 1
    tool_msg = next(m for m in agent.messages if m["role"] == "tool")
    assert "approval policy refused" in tool_msg["content"]


def test_interactive_approver_is_consulted(tmp_path: Path):
    asked = []

    def approve(call, verdict):
        asked.append((call.arguments["path"], verdict.risk))
        return True

    client = ScriptedClient([turn("write_file", path="../outside.txt", content="hi"),
                             turn("finish", summary="x")])
    agent = build_agent(tmp_path / "ws", client, approver=approve)
    (tmp_path / "ws").mkdir(exist_ok=True)
    agent.run("write")
    assert asked == [("../outside.txt", "dangerous")]
    assert (tmp_path / "outside.txt").read_text() == "hi"


def test_loop_detection_and_max_steps(tmp_path: Path):
    same = [turn("list_dir") for _ in range(10)]
    res = build_agent(tmp_path, ScriptedClient(same)).run("loop")
    assert res.status == "loop" and res.steps == 3
    varied = [turn("list_dir", path=".") if i % 2 else turn("list_dir") for i in range(10)]
    res = build_agent(tmp_path, ScriptedClient(varied), max_steps=4).run("go")
    assert res.status == "max_steps" and res.steps == 4


def test_text_answer_ends_the_run_and_model_error_is_reported(tmp_path: Path):
    res = build_agent(tmp_path, ScriptedClient([ModelTurn("All done.")])).run("hi")
    assert res.status == "no_tool_call" and res.final == "All done."

    class Broken:
        model = "broken"

        def chat(self, messages, tools):
            raise ModelError("cannot reach Ollama")

    res = build_agent(tmp_path, Broken()).run("hi")
    assert res.status == "model_error" and "Ollama" in res.error


def test_compaction_keeps_the_task_and_recent_turns():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "TASK"}]
    for i in range(20):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "read_file", "arguments": {"i": i}}}]})
        msgs.append({"role": "tool", "tool_name": "read_file", "content": f"{i}" * 2000})
    fitted, report = ContextManager(budget_tokens=3000, keep_recent=4).fit(msgs)
    assert report.changed and report.tokens_after <= 3000 < report.tokens_before
    assert fitted[1]["content"] == "TASK" and fitted[-1] == msgs[-1]
    assert conversation_tokens(fitted) == report.tokens_after
    small, rep = ContextManager(budget_tokens=10**6).fit(msgs)
    assert small == msgs and not rep.changed


def test_truncate_modes():
    text = "HEAD" + "x" * 1000 + "TAIL"
    assert truncate(text, 2000) == (text, 0)
    head, n = truncate(text, 100, "head")
    assert head.startswith("HEAD") and "TAIL" not in head and n == len(text) - 100
    tail, _ = truncate(text, 100, "tail")
    assert tail.endswith("TAIL") and "HEAD" not in tail
    both, _ = truncate(text, 100)
    assert both.startswith("HEAD") and both.endswith("TAIL")
    assert estimate_tokens("") == 0 and estimate_tokens("abcd" * 10) == 13


def test_text_tool_call_fallback_parser():
    tagged = 'ok <tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>'
    assert parse_text_tool_calls(tagged)[0].arguments == {"path": "a"}
    fenced = '```json\n{"name": "grep", "arguments": "{\\"pattern\\": \\"x\\"}"}\n```'
    assert parse_text_tool_calls(fenced)[0].name == "grep"
    listed = json.dumps([{"function": {"name": "list_dir", "arguments": {}}}])
    assert parse_text_tool_calls(listed)[0].name == "list_dir"
    assert parse_text_tool_calls("I will read the file now.") == []
    assert parse_text_tool_calls('{"name": 3}') == []


def test_digest_truncation_lists_failures_first():
    lines = [f"test_{i} (app.tests.T) ... ok" for i in range(300)]
    lines[5] = "test_bad (app.tests.T) ... FAIL"
    text = "\n".join([*lines, "", "Ran 300 tests", "OK"])
    cut, elided = truncate(text, 2000, "digest")
    assert cut.startswith("[failure lines]\ntest_bad (app.tests.T) ... FAIL")
    assert elided > 0 and len(cut) < 2100
    assert truncate("short", 2000, "digest") == ("short", 0)
    plain, _ = truncate("x" * 5000, 1000, "digest")  # nothing looks like a failure
    assert "[failure lines]" not in plain and "chars truncated" in plain


def test_compaction_protects_the_latest_task_not_just_the_first():
    # a second REPL task sits in the middle; compaction must keep it (reviewer issue 7)
    cm = ContextManager(budget_tokens=1000, keep_recent=4)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "TASK ONE"}]
    for i in range(6):
        msgs += [{"role": "assistant", "content": "",
                  "tool_calls": [{"function": {"name": "read_file", "arguments": {"i": i}}}]},
                 {"role": "tool", "tool_name": "read_file", "content": "x" * 400}]
    msgs.append({"role": "user", "content": "TASK TWO: rename foo to bar"})
    for i in range(6):
        msgs += [{"role": "assistant", "content": "",
                  "tool_calls": [{"function": {"name": "grep", "arguments": {"i": i}}}]},
                 {"role": "tool", "tool_name": "grep", "content": "y" * 400}]
    fitted, _ = cm.fit(msgs)
    kept = [m["content"] for m in fitted if m.get("role") == "user"]
    assert "TASK TWO: rename foo to bar" in kept and "TASK ONE" in kept


def test_compaction_never_squeezes_the_protected_task():
    # a single huge task must be returned intact, not truncated 8043 -> 2058 (reviewer issue 6)
    cm = ContextManager(budget_tokens=1000, keep_recent=4)
    task = "ISSUE " + "z" * 7000 + " KEY DETAIL " + "z" * 1000 + " END"
    fitted, rep = cm.fit([{"role": "system", "content": "s"}, {"role": "user", "content": task}])
    assert fitted[1]["content"] == task and rep.squeezed == 0
    assert "KEY DETAIL" in fitted[1]["content"]
