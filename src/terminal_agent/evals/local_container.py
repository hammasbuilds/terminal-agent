"""An offline container for a mined local task, so a model's commands never run on the host.

The model arm runs in ``auto`` approval mode, where the policy lets mutating commands run
unasked (``python evil.py``, ``make``): the policy is a filter, not a sandbox. SWE-bench
tasks already run in a disposable container; this gives the local suite the same. The
agent's file tools still edit the host copy of the workspace (a throwaway directory under
``runs/``), and :class:`~terminal_agent.sandbox.DockerSandbox` syncs it with the container
before and after every shell command. Grading runs in a second fresh container.

The image is the official ``python:3.11-bookworm`` (Python 3.11 like the harness, and it
ships git, which the sandbox's sync and the model's own ``git diff`` need). pytest is not
installed from the network: the pure-Python packages of this interpreter's pytest are
copied in, so the container never needs a network at all.
"""

from __future__ import annotations

import importlib.util
import io
import tarfile
import uuid
from pathlib import Path

from terminal_agent.evals.local_tasks import TEST_ARGS
from terminal_agent.sandbox import DockerSandbox, docker

LOCAL_IMAGE = "python:3.11-bookworm"
WORKDIR = "/workspace"
PYTEST_HOME = "/opt/ta-pytest"
PYTEST_PACKAGES = ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "py")
PRELUDE = (
    f"export PYTHONPATH={WORKDIR}/src:{WORKDIR}:{PYTEST_HOME} PYTHONDONTWRITEBYTECODE=1 "
    "PYTHONIOENCODING=utf-8 GIT_CEILING_DIRECTORIES=/"
)
TEST_COMMAND = "python -m pytest -rA -p no:cacheprovider --tb=short"


def image_present(image: str = LOCAL_IMAGE) -> bool:
    try:
        return docker(["image", "inspect", image], check=False, timeout=60).returncode == 0
    except (OSError, RuntimeError):
        return False


def _pytest_tar() -> bytes:
    """The host interpreter's pytest and its pure-Python dependencies, as a tar."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in PYTEST_PACKAGES:
            spec = importlib.util.find_spec(name)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"{name} is not importable here; run under `uv run`")
            origin = Path(spec.origin)
            src = origin.parent if origin.name == "__init__.py" else origin
            tar.add(src, arcname=src.name, filter=_skip_bytecode)
    return buf.getvalue()


def _skip_bytecode(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if "__pycache__" in info.name or info.name.endswith(".pyc") else info


def _tree_tar(root: Path) -> bytes:
    """Every file under ``root`` (including ``.git``), for loading into the container."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                info = tarfile.TarInfo(p.relative_to(root).as_posix())
                data = p.read_bytes()
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class LocalTaskContainer:
    """A throwaway, network-less container holding one local task's workspace."""

    def __init__(self, image: str = LOCAL_IMAGE, memory: str = "2g") -> None:
        self.image = image
        self.memory = memory
        self.name = f"ta-local-{uuid.uuid4().hex[:10]}"
        self.started = False

    def __enter__(self) -> LocalTaskContainer:
        if not image_present(self.image):
            raise RuntimeError(
                f"docker image {self.image} is not pulled; run: docker pull {self.image}"
            )
        docker(
            [
                "run",
                "-d",
                "--name",
                self.name,
                "--memory",
                self.memory,
                "--network",
                "none",
                self.image,
                "tail",
                "-f",
                "/dev/null",
            ],
            timeout=300,
        )
        self.started = True
        docker(["exec", self.name, "mkdir", "-p", WORKDIR, PYTEST_HOME])
        docker(["exec", "-i", self.name, "tar", "-x", "-C", PYTEST_HOME], input_bytes=_pytest_tar())
        return self

    def __exit__(self, *exc: object) -> None:
        if self.started:  # only ever removes the container it created
            docker(["rm", "-f", self.name], timeout=120, check=False)
            self.started = False

    def load(self, workspace: Path) -> None:
        """Copy the workspace (with its throwaway ``.git``) into the container."""
        docker(
            ["exec", "-i", self.name, "tar", "-x", "-C", WORKDIR], input_bytes=_tree_tar(workspace)
        )

    def sandbox(self, workspace: Path) -> DockerSandbox:
        return DockerSandbox(self.name, workspace, workdir=WORKDIR, prelude=PRELUDE)

    def run_pytest(self, files: list[str], timeout: float = 300) -> tuple[str, bool]:
        """Run the task's tests inside the container; return (log, timed_out)."""
        args = " ".join(TEST_ARGS[2:] + files)  # TEST_ARGS starts with "-m pytest"
        script = f"exec 2>&1\n{PRELUDE}\ncd {WORKDIR}\npython -m pytest {args}"
        proc = docker(
            ["exec", self.name, "timeout", "-s", "KILL", str(int(timeout)), "bash", "-c", script],
            timeout=timeout + 60,
            check=False,
        )
        log = (proc.stdout + proc.stderr).decode("utf-8", "replace")
        return log, proc.returncode == 137
