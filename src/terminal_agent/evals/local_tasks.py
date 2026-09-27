"""A second, fully local task suite mined from real bug-fix commits.

A commit qualifies when it changes both code and tests, and the tests it adds or changes
fail on the parent commit and pass on the commit itself - the SWE-bench construction,
run on this machine with nothing but ``python -m pytest``. The parent tree (code files
only, size-capped) is stored with the task so the suite runs from a clean clone, without
Docker or network, and without the original repositories.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terminal_agent.evals import specs
from terminal_agent.evals.patches import patch_test_files
from terminal_agent.evals.tasks import rmtree
from terminal_agent.sandbox import run_argv

LOCAL_DATA = Path(__file__).resolve().parents[3] / "data" / "local_tasks.jsonl.gz"
TEST_RE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")
KEEP_SUFFIXES = {".py", ".toml", ".cfg", ".ini", ".txt", ".json", ".yaml", ".yml", ".jsonl"}
SKIP_PARTS = {"results", "docs", "data", ".github", "images", "notebooks", "assets"}
MAX_FILE = 200_000
# a tree mentioning any of these could reach a model server or a GPU when its tests run
FORBIDDEN = (b"11434", b"ollama", b"import torch", b"from torch", b"transformers", b"cuda")
TEST_ARGS = ["-m", "pytest", "-rA", "--tb=no", "-p", "no:cacheprovider", "-q"]


@dataclass
class LocalTask:
    instance_id: str
    repo: str
    base_commit: str
    commit: str
    problem_statement: str
    patch: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    tree_b64: str

    @property
    def test_files(self) -> list[str]:
        return [f for f in patch_test_files(self.test_patch) if not f.endswith("conftest.py")]

    def materialize(self, dest: Path) -> None:
        rmtree(dest)
        dest.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(self.tree_b64)), mode="r:gz") as t:
            for m in t.getmembers():
                if m.isfile():
                    target = dest / m.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fh = t.extractfile(m)
                    target.write_bytes(fh.read() if fh else b"")

    def as_record(self) -> dict[str, Any]:
        return {"instance_id": self.instance_id, "repo": self.repo,
                "base_commit": self.base_commit, "commit": self.commit,
                "problem_statement": self.problem_statement, "patch": self.patch,
                "test_patch": self.test_patch, "FAIL_TO_PASS": self.fail_to_pass,
                "PASS_TO_PASS": self.pass_to_pass, "tree_b64": self.tree_b64}


def load_local_tasks(path: Path = LOCAL_DATA) -> list[LocalTask]:
    if not path.exists():
        return []
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.append(LocalTask(r["instance_id"], r["repo"], r["base_commit"], r["commit"],
                                 r["problem_statement"], r["patch"], r["test_patch"],
                                 r["FAIL_TO_PASS"], r["PASS_TO_PASS"], r["tree_b64"]))
    return out


def save_local_tasks(tasks: list[LocalTask], path: Path = LOCAL_DATA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        for t in sorted(tasks, key=lambda t: t.instance_id):
            fh.write(json.dumps(t.as_record()) + "\n")


# -- running tests locally ----------------------------------------------------------------

def test_env(workspace: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env["PYTHONPATH"] = os.pathsep.join([str(workspace / "src"), str(workspace)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OLLAMA_HOST"] = "http://127.0.0.1:9"  # discard port: fail fast, never reach a model
    # a local-task workspace lives under runs/, inside the harness repo; without a ceiling a
    # model's `git add -A`/`git commit` would resolve to the harness repo and could corrupt it
    env["GIT_CEILING_DIRECTORIES"] = str(workspace.resolve().parent)
    return env


def git_init_isolated(workspace: Path) -> None:
    """Make the workspace its own git repo so git commands are contained, not the harness's."""
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(workspace.resolve().parent),
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    for args in (["init", "-q"], ["-c", "user.name=ta", "-c", "user.email=ta@localhost",
                                  "-c", "commit.gpgsign=false", "add", "-A"],
                 ["-c", "user.name=ta", "-c", "user.email=ta@localhost",
                  "-c", "commit.gpgsign=false", "commit", "-q", "-m", "task base"]):
        subprocess.run(["git", *args], cwd=workspace, env=env, capture_output=True, check=False)


def run_pytest(workspace: Path, files: list[str], timeout: float = 300) -> tuple[str, bool]:
    """Run pytest on ``files`` with this interpreter; return (log, timed_out)."""
    res = run_argv([sys.executable, *TEST_ARGS, *files], workspace, test_env(workspace), timeout)
    return res.output, res.timed_out


def git_apply(workspace: Path, patch: str) -> tuple[bool, str]:
    """``git apply`` a patch to a plain directory.

    If the directory sits inside some other git work tree, git applies paths relative to
    *that* repository's root and silently skips files outside the current directory - exit
    code 0, nothing changed. The ceiling stops git from discovering an enclosing repo.
    """
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(workspace.resolve().parent)}
    proc = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=workspace, env=env,
                          input=patch.encode("utf-8"), capture_output=True, check=False)
    return proc.returncode == 0, (proc.stdout + proc.stderr).decode("utf-8", "replace")


def local_test_command() -> str:
    """The run_tests command an agent gets on a local task (runs in the workspace)."""
    # PYTHONPATH comes from test_env(), which the sandbox runs with; spelling it here as
    # "src:." would be wrong on Windows, where the separator is ";"
    py = Path(sys.executable).as_posix()
    return f'"{py}" -m pytest -rA -p no:cacheprovider --tb=short'


# -- mining --------------------------------------------------------------------------------

def _git(repo: Path, *args: str, binary: bool = False) -> Any:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-300:])
    return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")


def _keep(path: str, size: int) -> bool:
    parts = path.split("/")
    if any(p in SKIP_PARTS for p in parts[:-1]) or size > MAX_FILE:
        return False
    return Path(path).suffix in KEEP_SUFFIXES or Path(path).name in ("Makefile",)


def _filtered_tree(repo: Path, commit: str) -> bytes:
    raw = _git(repo, "archive", "--format=tar", commit, binary=True)
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw)) as src, \
            tarfile.open(fileobj=out, mode="w:gz") as dst:
        for m in src.getmembers():
            if m.isfile() and _keep(m.name, m.size):
                fh = src.extractfile(m)
                data = fh.read() if fh else b""
                info = tarfile.TarInfo(m.name)
                info.size = len(data)
                info.mtime = 0
                dst.addfile(info, io.BytesIO(data))
    return out.getvalue()


def _mentions_forbidden(tree_gz: bytes) -> bool:
    with tarfile.open(fileobj=io.BytesIO(tree_gz), mode="r:gz") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".py"):
                fh = t.extractfile(m)
                low = (fh.read() if fh else b"").lower()
                if any(word in low for word in FORBIDDEN):
                    return True
    return False


def candidate_commits(repo: Path, limit: int = 400) -> list[str]:
    log = _git(repo, "log", "--no-merges", f"-n{limit}", "--format=@%H", "--name-only")
    out = []
    for block in log.split("@")[1:]:
        lines = [ln for ln in block.strip().split("\n") if ln]
        sha, files = lines[0], lines[1:]
        py = [f for f in files if f.endswith(".py")]
        tests = [f for f in py if TEST_RE.search(f)]
        code = [f for f in py if not TEST_RE.search(f)]
        if tests and code and len(py) <= 6:
            out.append(sha)
    return out


def mine_commit(repo: Path, repo_name: str, sha: str, scratch: Path,
                max_patch_lines: int = 120) -> LocalTask | None:
    try:
        parent = _git(repo, "rev-parse", f"{sha}^").strip()
    except RuntimeError:
        return None
    names = _git(repo, "diff", "--name-only", parent, sha).split()
    tests = [f for f in names if TEST_RE.search(f) and f.endswith(".py")]
    code = [f for f in names if not TEST_RE.search(f) and f.endswith(".py")
            and not any(p in SKIP_PARTS for p in f.split("/")[:-1])]
    if not tests or not code:
        return None
    patch = _git(repo, "diff", "--no-color", parent, sha, "--", *code)
    test_patch = _git(repo, "diff", "--no-color", parent, sha, "--", *tests)
    if sum(1 for ln in patch.split("\n") if ln[:1] in "+-") > max_patch_lines:
        return None
    tree = _filtered_tree(repo, parent)
    if _mentions_forbidden(tree):
        return None
    task = LocalTask(f"{repo.name}@{sha[:10]}", repo_name, parent, sha,
                     _git(repo, "log", "-1", "--format=%B", sha).strip(), patch, test_patch,
                     [], [], base64.b64encode(tree).decode())
    runnable = [f for f in task.test_files if f.endswith(".py")]
    if not runnable:
        return None
    ws = scratch / "ws"
    task.materialize(ws)
    ok, _ = git_apply(ws, test_patch)
    if not ok:
        return None
    before, t1 = run_pytest(ws, runnable)
    ok, _ = git_apply(ws, patch)
    if not ok:
        return None
    after, t2 = run_pytest(ws, runnable)
    rmtree(ws)
    if t1 or t2:
        return None
    s0, s1 = specs.parse_pytest(before), specs.parse_pytest(after)
    f2p = sorted(t for t, s in s1.items() if specs.passed(s) and not specs.passed(s0.get(t)))
    p2p = sorted(t for t, s in s1.items() if specs.passed(s) and specs.passed(s0.get(t)))
    broke = [t for t, s in s0.items() if specs.passed(s) and not specs.passed(s1.get(t))]
    if not f2p or broke:
        return None
    task.fail_to_pass, task.pass_to_pass = f2p, p2p
    return task


def mine(repos: list[tuple[Path, str]], limit: int = 400, log: Any = print) -> list[LocalTask]:
    tasks: list[LocalTask] = []
    with tempfile.TemporaryDirectory(prefix="ta-mine-") as tmp:
        for repo, name in repos:
            for sha in candidate_commits(repo, limit):
                task = mine_commit(repo, name, sha, Path(tmp))
                log(f"{repo.name}@{sha[:10]}: {'TASK' if task else '-'}"
                    + (f" f2p={len(task.fail_to_pass)} p2p={len(task.pass_to_pass)}"
                       if task else ""))
                if task:
                    tasks.append(task)
    return tasks
