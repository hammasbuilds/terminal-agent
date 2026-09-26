import difflib
from pathlib import Path

from terminal_agent.agent import build_agent
from terminal_agent.evals import specs
from terminal_agent.evals.gold import GoldPatchClient
from terminal_agent.evals.patches import parse_patch, patch_test_files
from terminal_agent.policy import ApprovalPolicy


def unified(path: str, before: str, after: str, context: int = 3) -> str:
    diff = difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                f"a/{path}", f"b/{path}", n=context)
    return f"diff --git a/{path} b/{path}\n" + "".join(diff)


REPEATED = "".join(f"def f{i}():\n    return None\n\n\n" for i in range(3))


def test_parse_patch_sides_and_core():
    before = "a\nb\nc\nd\ne\nf\ng\n"
    after = "a\nb\nc\nD\ne\nf\ng\n"
    (fp,) = parse_patch(unified("x.txt", before, after))
    (h,) = fp.hunks
    assert h.old_text == "a\nb\nc\nd\ne\nf\ng\n" and h.new_text == after
    assert (h.leading_context, h.trailing_context, h.removed, h.added) == (3, 3, 1, 1)
    assert h.core() == ("d\n", "D\n")


def test_parse_patch_no_newline_marker_and_new_file():
    patch = ("diff --git a/n.py b/n.py\nnew file mode 100644\n--- /dev/null\n+++ b/n.py\n"
             "@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n\\ No newline at end of file\n"
             "diff --git a/tests/test_n.py b/tests/test_n.py\n--- a/tests/test_n.py\n"
             "+++ b/tests/test_n.py\n@@ -1 +1 @@\n-old\n+new\n")
    files = parse_patch(patch)
    assert files[0].is_new and files[0].hunks[0].new_text == "x = 1\ny = 2"
    assert patch_test_files(patch) == ["n.py", "tests/test_n.py"]


def _drive(tmp_path: Path, files: dict[str, str], patch: str):
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(text.encode())
    client = GoldPatchClient(patch)
    agent = build_agent(tmp_path, client, policy=ApprovalPolicy(tmp_path, mode="auto"),
                        max_steps=200)
    res = agent.run("apply")
    return res, client


def test_gold_client_widens_context_when_the_hunk_is_ambiguous(tmp_path: Path):
    after = REPEATED.replace("def f1():\n    return None", "def f1():\n    return 1")
    patch = unified("m.py", REPEATED, after, context=0)
    res, client = _drive(tmp_path, {"m.py": REPEATED}, patch)
    assert res.status == "finished"
    assert (tmp_path / "m.py").read_text() == after
    (rec,) = client.records
    assert rec.first_status == "ambiguous" and rec.final_status == "ok"
    assert rec.attempts == 2 and rec.extra_context >= 1


def test_gold_client_pages_long_files_creates_and_deletes(tmp_path: Path):
    long_before = "".join(f"v{i} = {i}\n" for i in range(2400))
    long_after = long_before.replace("v2300 = 2300\n", "v2300 = -1\n").replace(
        "v5 = 5\n", "v5 = -5\n")
    patch = (unified("long.py", long_before, long_after)
             + "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n"
               "@@ -0,0 +1 @@\n+created = True\n"
             + "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n"
               "+++ /dev/null\n@@ -1 +0,0 @@\n-gone = True\n")
    res, client = _drive(tmp_path, {"long.py": long_before, "old.py": "gone = True\n"}, patch)
    assert res.status == "finished"
    assert (tmp_path / "long.py").read_text() == long_after
    assert (tmp_path / "new.py").read_text() == "created = True\n"
    assert not (tmp_path / "old.py").exists()
    assert [r.final_status for r in client.records] == ["ok", "ok"]
    assert res.tool_calls["read_file"] == 3  # 2400 lines in 1000-line windows


def test_pytest_and_sympy_and_django_parsers():
    log = ("PASSED tests/t.py::test_a\nFAILED tests/t.py::test_b - AssertionError: x\n"
           "SKIPPED [1] tests/t.py:3: why\nERROR tests/t.py::test_c\n")
    assert specs.parse_pytest(log) == {"tests/t.py::test_a": "PASSED",
                                       "tests/t.py::test_b": "FAILED",
                                       "tests/t.py::test_c": "ERROR"}
    opts = specs.parse_pytest_options("PASSED t.py::test_x[/abs/path/data.txt]\n")
    assert opts == {"t.py::test_x[/data.txt]": "PASSED"}
    sym = specs.parse_sympy("test_one ok\ntest_two F\ntest_three E\n")
    assert sym == {"test_one": "PASSED", "test_two": "FAILED", "test_three": "ERROR"}
    dj = specs.parse_django("test_x (app.tests.T) ... ok\ntest_y (app.tests.T) ... FAIL\n")
    assert dj == {"test_x (app.tests.T)": "PASSED", "test_y (app.tests.T)": "FAILED"}


def test_grade_requires_every_listed_test():
    statuses = {"a": "PASSED", "b": "XFAIL", "c": "FAILED"}
    g = specs.grade(statuses, ["a", "b"], ["c"])
    assert not g["resolved"] and g["f2p_passed"] == 2 and g["p2p_failing"] == ["c"]
    assert specs.grade(statuses, ["a"], ["missing"])["p2p_passed"] == 0
    assert specs.grade(statuses, ["a", "b"], [])["resolved"]


def test_repo_specific_commands():
    assert specs.test_directives("django/django", ["tests/admin_views/tests.py"]) == [
        "admin_views.tests"]
    assert specs.test_directives("psf/requests", ["test_requests.py", "x.json"]) == [
        "test_requests.py"]
    assert "bin/test" in specs.test_command("sympy/sympy", "1.1")
    assert specs.image_name("psf__requests-3362") == (
        "swebench/sweb.eval.x86_64.psf_1776_requests-3362:latest")
