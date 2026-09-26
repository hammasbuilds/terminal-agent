import sys
from pathlib import Path

import pytest

from terminal_agent.protocol import ToolCall
from terminal_agent.sandbox import LocalSandbox
from terminal_agent.tools import Toolbox, ToolConfig


@pytest.fixture
def box(tmp_path: Path) -> Toolbox:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_bytes(b"def f():\n    return 1\n")
    (tmp_path / "pkg" / "crlf.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (tmp_path / "big.txt").write_bytes("".join(f"line {i}\n" for i in range(1, 2501)).encode())
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_bytes(b"ref")
    return Toolbox(tmp_path, LocalSandbox(tmp_path), ToolConfig(output_max_chars=300))


def call(box: Toolbox, name: str, **args):
    return box.execute(ToolCall(name, args))


def test_read_file_windows_long_files_with_a_paging_header(box):
    first = call(box, "read_file", path="big.txt")
    assert first.ok and first.output.startswith("[big.txt: lines 1-1000 of 2500.")
    assert "offset=1001" in first.output and first.meta["truncated"]
    last = call(box, "read_file", path="big.txt", offset=2001)
    assert last.output.startswith("[big.txt: lines 2001-2500 of 2500]")
    assert "line 2500" in last.output
    small = call(box, "read_file", path="pkg/a.py")
    assert small.output == "def f():\n    return 1"


def test_read_file_errors(box):
    assert "does not exist" in call(box, "read_file", path="nope.py").output
    assert call(box, "read_file", path="bin.dat").meta["error"] == "binary"
    assert call(box, "read_file", path="pkg").meta["error"] == "is_directory"


def test_edit_keeps_crlf_and_bytes(box, tmp_path):
    res = call(box, "edit", path="pkg/crlf.py", old_string="y = 2\n", new_string="y = 3\n")
    assert res.ok and res.meta["strategy"] == "crlf"
    assert (tmp_path / "pkg" / "crlf.py").read_bytes() == b"x = 1\r\ny = 3\r\n"


def test_edit_error_messages_guide_the_model(box):
    amb = call(box, "edit", path="big.txt", old_string="line 2", new_string="x")
    assert not amb.ok and amb.meta["status"] == "ambiguous" and "Include more" in amb.output
    missing = call(box, "edit", path="pkg/a.py", old_string="return 2", new_string="x")
    assert not missing.ok and "not found" in missing.output
    assert missing.meta["status"] == "not_found"
    create = call(box, "edit", path="pkg/new.py", old_string="", new_string="z = 1\n")
    assert create.ok and create.meta["created"]
    exists = call(box, "edit", path="pkg/a.py", old_string="", new_string="z")
    assert not exists.ok and "already exists" in exists.output


def test_non_utf8_bytes_survive_an_edit(box, tmp_path):
    p = tmp_path / "latin.py"
    p.write_bytes(b"name = '\xe9t\xe9'\nx = 1\n")
    assert call(box, "edit", path="latin.py", old_string="x = 1", new_string="x = 2").ok
    assert p.read_bytes() == b"name = '\xe9t\xe9'\nx = 2\n"


def test_write_file_preserves_an_existing_crlf_convention(box, tmp_path):
    assert call(box, "write_file", path="pkg/crlf.py", content="a\nb\n").ok
    assert (tmp_path / "pkg" / "crlf.py").read_bytes() == b"a\r\nb\r\n"
    assert call(box, "write_file", path="deep/new/f.txt", content="a\n").meta["created"]
    assert (tmp_path / "deep" / "new" / "f.txt").read_bytes() == b"a\n"


def test_list_glob_grep_skip_git(box):
    listing = call(box, "list_dir").output
    assert "pkg/" in listing and ".git" not in listing
    assert call(box, "glob", pattern="**/*.py").output.splitlines() == ["pkg/a.py", "pkg/crlf.py"]
    assert call(box, "glob", pattern="*.md").output.startswith("no files match")
    hits = call(box, "grep", pattern=r"return \d")
    assert hits.output == "pkg/a.py:2: return 1"
    assert call(box, "grep", pattern="ref").output.startswith("no matches")
    assert call(box, "grep", pattern="(").meta["error"] == "bad_arguments"


def test_argument_validation(box):
    assert "unknown tool" in call(box, "delete_everything").output
    assert "missing required" in call(box, "read_file").output
    assert "unknown argument" in call(box, "read_file", path="a", bogus=1).output
    assert "must be integer" in call(box, "read_file", path="a", offset="3").output
    bad = box.execute(ToolCall("read_file", {"__unparsed__": "{oops"}))
    assert bad.meta["error"] == "bad_arguments"


def test_run_shell_truncates_head_and_tail(box):
    py = Path(sys.executable).as_posix()
    res = call(box, "run_shell", command=f'"{py}" -c "print(\'A\' * 2000); print(\'END\')"')
    assert res.ok and res.meta["elided_chars"] > 1500
    assert "chars truncated" in res.output and res.output.rstrip().endswith("END")


def test_run_shell_timeout_kills_the_process(box):
    py = Path(sys.executable).as_posix()
    res = call(box, "run_shell", command=f'"{py}" -c "import time; time.sleep(30)"', timeout=2)
    assert not res.ok and res.meta["timed_out"] and res.meta["seconds"] < 20
    assert "TIMED OUT" in res.output


def test_run_shell_reports_exit_code(box):
    res = call(box, "run_shell", command="exit 3")
    assert not res.ok and res.meta["exit_code"] == 3 and res.output.startswith("[exit code 3]")
