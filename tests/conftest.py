import subprocess
from pathlib import Path

import pytest

BUGGY = '''def mean(xs):
    """Arithmetic mean."""
    return sum(xs) / (len(xs) - 1)


def clamp(x, lo, hi):
    return max(lo, min(x, hi))
'''
FIXED = BUGGY.replace("(len(xs) - 1)", "len(xs)")
TEST_OLD = """from calc import clamp


def test_clamp():
    assert clamp(5, 0, 3) == 3
"""
TEST_NEW = (
    TEST_OLD.replace("from calc import clamp", "from calc import clamp, mean")
    + """

def test_mean():
    assert mean([1, 2, 3]) == 2
"""
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture(scope="module")
def bugfix_repo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """A two-commit repo: the second commit fixes ``mean`` and adds its test."""
    repo = tmp_path_factory.mktemp("repo") / "calcrepo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "calc.py").write_bytes(BUGGY.encode())
    (repo / "tests" / "test_calc.py").write_bytes(TEST_OLD.encode())
    (repo / "pyproject.toml").write_bytes(b'[project]\nname = "calc"\nversion = "0"\n')
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "calc")
    (repo / "src" / "calc.py").write_bytes(FIXED.encode())
    (repo / "tests" / "test_calc.py").write_bytes(TEST_NEW.encode())
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-qam",
        "mean divided by n-1\n\nmean([1, 2, 3]) returned 3.0 instead of 2.",
    )
    return repo, _git(repo, "rev-parse", "HEAD").strip()
