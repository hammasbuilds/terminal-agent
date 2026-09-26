"""SWE-bench Lite tasks and the Docker containers they are evaluated in.

Each task uses the official prebuilt image ``swebench/sweb.eval.x86_64.<id>``: the repo at
``base_commit`` in ``/testbed`` with its conda environment already installed. The agent's
workspace is a local, byte-exact export of that tree (``git archive``), so file tools run
on the host while tests run in the image.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import stat
import sys
import tarfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terminal_agent.evals import specs
from terminal_agent.evals.patches import patch_test_files
from terminal_agent.sandbox import changed_files, docker, snapshot, tar_files

DATA = Path(__file__).resolve().parents[3] / "data" / "swebench_lite.jsonl.gz"
WORKDIR = "/testbed"
HEREDOC = "EOF_TERMINAL_AGENT_5f3c1"


@dataclass
class Task:
    instance_id: str
    repo: str
    base_commit: str
    version: str
    problem_statement: str
    patch: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]

    @property
    def image(self) -> str:
        return specs.image_name(self.instance_id)

    @property
    def test_files(self) -> list[str]:
        return patch_test_files(self.test_patch)


def load_tasks(path: Path = DATA) -> list[Task]:
    tasks = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            tasks.append(Task(r["instance_id"], r["repo"], r["base_commit"], r["version"],
                              r["problem_statement"], r["patch"], r["test_patch"],
                              r["FAIL_TO_PASS"], r["PASS_TO_PASS"]))
    return tasks


def select(tasks: list[Task], ids: list[str] | None) -> list[Task]:
    if not ids:
        return tasks
    by_id = {t.instance_id: t for t in tasks}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise KeyError(f"unknown instance id(s): {', '.join(unknown)}")
    return [by_id[i] for i in ids]


def _force_remove(func: Callable[..., Any], path: str, _exc: object) -> None:
    """rmtree error handler: clear the read-only bit Windows leaves on some files, retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def rmtree(path: Path) -> None:
    """Delete a tree, including read-only files (``ignore_errors`` would leave them behind)."""
    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_force_remove)
    else:
        shutil.rmtree(path, onerror=_force_remove)


@dataclass
class ExportReport:
    files: int = 0
    skipped_symlinks: list[str] = field(default_factory=list)
    skipped_invalid: list[str] = field(default_factory=list)
    case_collisions: list[str] = field(default_factory=list)


class Container:
    """A throwaway container from the task image. Only ever removes itself.

    The agent's container has no network (``network=False``); grading containers keep
    Docker's default network, as the official harness does, because some PASS_TO_PASS
    tests (requests' connect-timeout tests) need a routable network to time out rather
    than fail fast.
    """

    def __init__(self, task: Task, memory: str = "3g", network: bool = False) -> None:
        self.task = task
        self.name = f"ta-{task.instance_id.lower().replace('__', '-')}-{uuid.uuid4().hex[:6]}"
        self.memory = memory
        self.network = network
        self.started = False

    def __enter__(self) -> Container:
        net = [] if self.network else ["--network", "none"]
        docker(["run", "-d", "--name", self.name, "--memory", self.memory, *net,
                self.task.image, "tail", "-f", "/dev/null"], timeout=300)
        self.started = True
        return self

    def __exit__(self, *exc: object) -> None:
        if self.started:
            docker(["rm", "-f", self.name], timeout=120, check=False)
            self.started = False

    def sh(self, script: str, timeout: float = 600, stdin: bytes | None = None
           ) -> tuple[int, str]:
        args = ["exec", "-i", self.name, "bash", "-c", script] if stdin is not None else [
            "exec", self.name, "bash", "-c", script]
        proc = docker(args, input_bytes=stdin, timeout=timeout, check=False)
        text = (proc.stdout + proc.stderr).decode("utf-8", "replace")
        return proc.returncode, text

    def write(self, path: str, data: bytes) -> None:
        code, out = self.sh(f"cat > {path}", stdin=data)
        if code != 0:
            raise RuntimeError(f"could not write {path} in {self.name}: {out[-300:]}")

    def tree_matches(self, commit: str) -> bool:
        """Is HEAD's content that of ``commit``?

        The official images add a "SWE-bench" commit on top of base_commit that changes
        only file modes, so comparing HEAD's hash with base_commit would reject every
        image. Compare (path, blob) pairs instead, which ignores modes.
        """
        def tree(rev: str) -> set[tuple[str, str]] | None:
            code, out = self.sh(f"git -C {WORKDIR} ls-tree -r {rev}")
            if code != 0:
                return None
            rows = (line.split(None, 3) for line in out.splitlines() if line.strip())
            return {(r[3], r[2]) for r in rows if len(r) == 4}

        head, base = tree("HEAD"), tree(commit)
        return head is not None and head == base

    def export(self, dest: Path) -> ExportReport:
        """Extract the tracked tree at HEAD to ``dest`` (bytes preserved, no .git)."""
        rmtree(dest)
        dest.mkdir(parents=True)
        proc = docker(["exec", self.name, "git", "-C", WORKDIR, "archive", "--format=tar",
                       "HEAD"], timeout=600)
        report = ExportReport()
        seen: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
            for member in tar.getmembers():
                if member.issym() or member.islnk():
                    report.skipped_symlinks.append(member.name)
                    continue
                if not member.isfile():
                    continue
                low = member.name.lower()
                if low in seen:
                    report.case_collisions.append(f"{seen[low]} vs {member.name}")
                    continue
                seen[low] = member.name
                target = dest / member.name
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fh = tar.extractfile(member)
                    target.write_bytes(fh.read() if fh else b"")
                except (OSError, ValueError):
                    report.skipped_invalid.append(member.name)
                    continue
                report.files += 1
        return report

    def push_changes(self, workspace: Path, before: dict[str, str]) -> tuple[list[str], list[str]]:
        """Copy files the agent changed into /testbed; return (modified, deleted)."""
        modified, deleted = changed_files(before, snapshot(workspace))
        if modified:
            docker(["exec", "-i", self.name, "tar", "-x", "-C", WORKDIR],
                   input_bytes=tar_files(workspace, modified), timeout=300)
        if deleted:
            docker(["exec", self.name, "rm", "-f", "--", *[f"{WORKDIR}/{d}" for d in deleted]])
        return modified, deleted

    def diff(self) -> str:
        """The working-tree change as a patch, the form SWE-bench predictions take."""
        _, out = self.sh(f"cd {WORKDIR} && git -c core.fileMode=false diff --text")
        return out

    def run_tests(self, timeout: float = 1800) -> tuple[str, dict[str, Any]]:
        """Apply the task's test patch and run its tests; return (log, meta)."""
        t = self.task
        files = t.test_files
        self.write("/tmp/test.patch", t.test_patch.encode("utf-8"))
        lines = [specs.PRELUDE, f"cd {WORKDIR}", *specs.pre_test_commands(t.repo, t.version),
                 f"git checkout {t.base_commit} -- {' '.join(files)} 2>/dev/null || true",
                 "git apply -v /tmp/test.patch || { echo '>>>>> TEST PATCH FAILED'; exit 97; }",
                 "echo '>>>>> Start Test Output'",
                 f"{specs.test_command(t.repo, t.version)} "
                 f"{' '.join(specs.test_directives(t.repo, files))}",
                 "echo \">>>>> Test Exit Code $?\"",
                 "echo '>>>>> End Test Output'"]
        script = _quote("\n".join(lines))
        code, log = self.sh(f"timeout -s KILL {int(timeout)} bash -c '{script}'",
                            timeout=timeout + 60)
        meta = {"exit_code": code, "test_patch_applied": ">>>>> TEST PATCH FAILED" not in log,
                "timed_out": code == 137}
        return log, meta


def _quote(script: str) -> str:
    return script.replace("'", "'\"'\"'")


def test_section(log: str) -> str:
    """The part of an eval log between the start/end markers."""
    start = log.find(">>>>> Start Test Output")
    end = log.find(">>>>> End Test Output")
    if start < 0:
        return log
    return log[start : end if end > start else len(log)]
