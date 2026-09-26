"""Where shell commands run: the local machine, or a Docker container mirroring the workspace.

File tools always operate on the local workspace. A :class:`DockerSandbox` pushes local
changes into the container before each command and pulls back files the command changed,
so ``run_shell("sed -i ...")`` and ``edit`` stay consistent with each other.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import signal
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool = False
    seconds: float = 0.0


class Sandbox(Protocol):
    def run(self, command: str, timeout: float) -> ExecResult:
        ...


def _decode(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", "replace").replace("\r\n", "\n")


def local_shell() -> list[str]:
    """argv prefix for running a command string locally."""
    if os.name == "nt":
        bash = shutil.which("bash")
        return [bash, "-c"] if bash else ["cmd", "/d", "/s", "/c"]
    return ["/bin/bash", "-c"] if Path("/bin/bash").exists() else ["/bin/sh", "-c"]


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill the process we started and its children - never anything else."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True, check=False)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    proc.kill()


class LocalSandbox:
    def __init__(self, workspace: Path, env: dict[str, str] | None = None) -> None:
        self.workspace = workspace
        self.env = env

    def run(self, command: str, timeout: float) -> ExecResult:
        start = time.monotonic()
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [*local_shell(), command],
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=self.env,
            **kwargs,  # type: ignore[arg-type]
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
            return ExecResult(proc.returncode, _decode(out), False, time.monotonic() - start)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            out, _ = proc.communicate()
            return ExecResult(124, _decode(out), True, time.monotonic() - start)


def file_digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict[str, str]:
    """Relative POSIX path -> sha1 of every file under ``root``."""
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_file() and ".git" not in p.relative_to(root).parts:
            out[p.relative_to(root).as_posix()] = file_digest(p)
    return out


def changed_files(before: dict[str, str], after: dict[str, str]) -> tuple[list[str], list[str]]:
    """(modified or added, deleted) between two snapshots."""
    modified = sorted(k for k, v in after.items() if before.get(k) != v)
    deleted = sorted(k for k in before if k not in after)
    return modified, deleted


def tar_files(root: Path, rel_paths: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel in rel_paths:
            data = (root / rel).read_bytes()
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def docker(args: list[str], *, input_bytes: bytes | None = None, timeout: float = 600,
           check: bool = True) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run(["docker", *args], input=input_bytes, capture_output=True,
                          timeout=timeout, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])} failed ({proc.returncode}): "
                           f"{_decode(proc.stderr)[-800:]}")
    return proc


class DockerSandbox:
    """Runs commands inside ``container`` at ``workdir``, keeping it in sync with ``workspace``."""

    def __init__(self, container: str, workspace: Path, workdir: str = "/testbed",
                 prelude: str = "") -> None:
        self.container = container
        self.workspace = workspace
        self.workdir = workdir
        self.prelude = prelude
        self._synced = snapshot(workspace)

    def push(self) -> None:
        """Copy local edits made since the last sync into the container."""
        now = snapshot(self.workspace)
        modified, deleted = changed_files(self._synced, now)
        if modified:
            docker(["exec", "-i", self.container, "tar", "-x", "-C", self.workdir],
                   input_bytes=tar_files(self.workspace, modified))
        if deleted:
            docker(["exec", self.container, "rm", "-f", "--",
                    *[f"{self.workdir}/{d}" for d in deleted]])
        self._synced = now

    def pull(self) -> None:
        """Copy files a command changed inside the container back to the workspace."""
        proc = docker(["exec", self.container, "git", "-C", self.workdir, "status",
                       "--porcelain", "-z", "--untracked-files=all"], check=False)
        if proc.returncode != 0:
            return
        entries = [e for e in proc.stdout.decode("utf-8", "replace").split("\0") if e]
        present, gone = [], []
        for entry in entries:
            code, rel = entry[:2], entry[3:]
            if not rel or rel.endswith("/"):
                continue
            (gone if "D" in code else present).append(rel)
        wanted = [r for r in present if self._container_differs(r)]
        if wanted:
            data = docker(["exec", self.container, "tar", "-c", "-C", self.workdir, "--",
                           *wanted]).stdout
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    target = self.workspace / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fh = tar.extractfile(member)
                    if fh is not None:
                        target.write_bytes(fh.read())
        for rel in gone:
            (self.workspace / rel).unlink(missing_ok=True)
        self._synced = snapshot(self.workspace)

    def _container_differs(self, rel: str) -> bool:
        local = self.workspace / rel
        if not local.exists():
            return True
        proc = docker(["exec", self.container, "sha1sum", f"{self.workdir}/{rel}"], check=False)
        return proc.returncode != 0 or proc.stdout.split(b" ")[0].decode() != file_digest(local)

    def run(self, command: str, timeout: float) -> ExecResult:
        self.push()
        script = f"exec 2>&1\n{self.prelude}\ncd {self.workdir}\n{command}"
        start = time.monotonic()
        try:
            proc = docker(["exec", self.container, "timeout", "-s", "KILL", str(int(timeout)),
                           "bash", "-c", script], timeout=timeout + 30, check=False)
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", True, time.monotonic() - start)
        timed_out = proc.returncode == 137 and time.monotonic() - start >= timeout - 1
        self.pull()
        return ExecResult(proc.returncode, _decode(proc.stdout) + _decode(proc.stderr),
                          timed_out, time.monotonic() - start)
