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
import tempfile
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


def run_argv(argv: list[str], cwd: Path, env: dict[str, str] | None, timeout: float
             ) -> ExecResult:
    """Run a process in its own group; on timeout kill the tree and return what it wrote.

    Output goes to a real temp file, not a pipe. A pipe stays open as long as *any*
    descendant holds its write end, so a backgrounded grandchild makes ``communicate()``
    block long past the timeout and lose everything; a file never blocks the parent.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    start = time.monotonic()
    deadline = start + timeout
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    timed_out = False
    with tempfile.TemporaryFile() as out:
        proc = subprocess.Popen(argv, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, **kwargs)  # type: ignore[call-overload]
        while proc.poll() is None:
            if time.monotonic() >= deadline:
                _kill_tree(proc)
                timed_out = True
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                break
            time.sleep(0.05)
        out.seek(0)
        data = out.read()
    code = 124 if timed_out else (proc.returncode if proc.returncode is not None else 124)
    return ExecResult(code, _decode(data), timed_out, time.monotonic() - start)


class LocalSandbox:
    def __init__(self, workspace: Path, env: dict[str, str] | None = None) -> None:
        self.workspace = workspace
        self.env = env

    def run(self, command: str, timeout: float) -> ExecResult:
        return run_argv([*local_shell(), command], self.workspace, self.env, timeout)


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


def tar_files(root: Path, rel_paths: list[str], modes: dict[str, int] | None = None) -> bytes:
    """Tar the given files. ``modes`` gives a per-path octal mode to preserve (e.g. +x)."""
    buf = io.BytesIO()
    modes = modes or {}
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel in rel_paths:
            data = (root / rel).read_bytes()
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = modes.get(rel, 0o644)
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
            # Windows loses the executable bit; keep whatever the container already had so
            # overwriting a file (e.g. sympy's bin/test) does not make it non-executable.
            modes = {f: 0o755 for f in self._container_executables(modified)}
            docker(["exec", "-i", self.container, "tar", "-x", "-C", self.workdir],
                   input_bytes=tar_files(self.workspace, modified, modes))
        if deleted:
            docker(["exec", self.container, "rm", "-f", "--",
                    *[f"{self.workdir}/{d}" for d in deleted]])
        self._synced = now

    def _container_executables(self, rel_paths: list[str]) -> list[str]:
        script = "".join(f'test -x "{self.workdir}/{r}" && printf "%s\\0" "{r}"; '
                         for r in rel_paths)
        proc = docker(["exec", self.container, "sh", "-c", script], check=False)
        return [r for r in proc.stdout.decode("utf-8", "replace").split("\0") if r]

    def pull(self) -> None:
        """Copy files a command changed inside the container back to the workspace."""
        proc = docker(["exec", self.container, "git", "-C", self.workdir, "status",
                       "--porcelain", "-z", "--untracked-files=all"], check=False)
        if proc.returncode != 0:
            return
        # porcelain -z: a rename/copy is two NUL fields ("R  <new>\0<old>\0"), so the old
        # path must be consumed explicitly, not read as its own status entry.
        fields = proc.stdout.decode("utf-8", "replace").split("\0")
        present, gone = [], []
        i = 0
        while i < len(fields):
            entry = fields[i]
            i += 1
            if not entry:
                continue
            code, rel = entry[:2], entry[3:]
            if code[:1] in ("R", "C") or code[1:2] in ("R", "C"):
                orig = fields[i] if i < len(fields) else ""
                i += 1
                if orig and not orig.endswith("/"):
                    gone.append(orig)
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
