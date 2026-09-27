import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from terminal_agent.llm import ModelError, OllamaClient, ScriptedClient, turn_from_response


class _Stub(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.requests.append(body)
        if _Stub.status != 200:
            self.send_response(_Stub.status)
            self.end_headers()
            self.wfile.write(b'{"error": "model not found"}')
            return
        reply = {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]},
            "prompt_eval_count": 120, "eval_count": 9}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub():
    _Stub.requests, _Stub.status = [], 200
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_client_sends_options_parses_tool_calls_and_caches(stub, tmp_path: Path):
    client = OllamaClient(model="m", host=stub, cache_dir=tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    first = client.chat(msgs, [])
    assert first.tool_calls[0].name == "read_file" and first.prompt_tokens == 120
    assert not first.cached
    sent = _Stub.requests[0]
    assert sent["options"]["num_ctx"] == 16384 and sent["stream"] is False
    again = client.chat(msgs, [])
    assert again.cached and len(_Stub.requests) == 1
    client.chat([{"role": "user", "content": "other"}], [])
    assert len(_Stub.requests) == 2


def test_missing_model_and_unreachable_server_give_actionable_errors(stub):
    _Stub.status = 404
    with pytest.raises(ModelError, match="ollama pull m"):
        OllamaClient(model="m", host=stub).chat([], [])
    with pytest.raises(ModelError, match="ollama serve"):
        OllamaClient(model="m", host="http://127.0.0.1:9", timeout=2).chat([], [])


def test_text_fallback_and_string_arguments():
    t = turn_from_response({"message": {"content": '<tool_call>{"name": "glob", "arguments": '
                                                   '{"pattern": "*.py"}}</tool_call>'}})
    assert t.parsed_from_text and t.tool_calls[0].arguments == {"pattern": "*.py"}
    t = turn_from_response({"message": {"content": "", "tool_calls": [
        {"function": {"name": "grep", "arguments": '{"pattern": "x"}'}}]}})
    assert t.tool_calls[0].arguments == {"pattern": "x"} and not t.parsed_from_text
    t = turn_from_response({"message": {"tool_calls": [
        {"function": {"name": "grep", "arguments": "{broken"}}]}})
    assert "__unparsed__" in t.tool_calls[0].arguments


def test_scripted_client_from_file(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"tool_calls": [{"name": "list_dir"}]}, {"content": "bye"}]))
    c = ScriptedClient.from_file(p)
    assert c.chat([], []).tool_calls[0].name == "list_dir"
    assert c.chat([], []).content == "bye" and c.chat([], []).content == "(script finished)"
    p.write_text("{}")
    with pytest.raises(ValueError):
        ScriptedClient.from_file(p)


def test_malformed_script_raises_value_error_not_traceback(tmp_path):
    for bad in ("[1]", '[{"tool_calls":[{"arguments":{}}]}]', "{}", '[{"tool_calls":[5]}]'):
        p = tmp_path / "s.json"
        p.write_text(bad)
        with pytest.raises(ValueError):
            ScriptedClient.from_file(p)


def test_non_json_ollama_body_becomes_a_model_error(stub, monkeypatch):
    import terminal_agent.llm as llm

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>502 Bad Gateway</html>"

    monkeypatch.setattr(llm._OPENER, "open", lambda *a, **k: R())
    with pytest.raises(ModelError, match="non-JSON"):
        OllamaClient(model="m", host=stub).chat([], [])


def test_non_dict_tool_arguments_do_not_crash_the_tool_layer():
    # a JSON string that decodes to a list/scalar is not valid tool arguments (reviewer probe)
    from pathlib import Path

    from terminal_agent.sandbox import LocalSandbox
    from terminal_agent.tools import Toolbox

    t = turn_from_response({"message": {"tool_calls": [
        {"function": {"name": "read_file", "arguments": "[1, 2]"}}]}})
    assert isinstance(t.tool_calls[0].arguments, dict)
    res = Toolbox(Path("."), LocalSandbox(Path("."))).execute(t.tool_calls[0])
    assert not res.ok and res.meta["error"] == "bad_arguments"


def test_client_ignores_http_proxy_for_the_local_server(stub, monkeypatch):
    # a proxy on the discard port would swallow the request if urllib honoured it
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    turn = OllamaClient(model="m", host=stub, timeout=5).chat([{"role": "user", "content": "x"}], [])
    assert turn.tool_calls[0].name == "read_file"
