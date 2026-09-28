"""Per-repository test commands and log parsers for SWE-bench Lite.

The commands and parsing rules reimplement the SWE-bench harness's per-repo conventions
(``pytest -rA`` summaries, sympy's ``bin/test`` lines, Django's ``runtests.py`` output) so a
test id here is spelled exactly as in the dataset's FAIL_TO_PASS / PASS_TO_PASS lists.
"""

from __future__ import annotations

import re
from collections.abc import Callable

PASSED, FAILED, SKIPPED, ERROR, XFAIL = "PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL"
STATUSES = (PASSED, FAILED, SKIPPED, ERROR, XFAIL)
NON_TEST_EXTS = (
    ".json",
    ".png",
    "csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
)

PYTEST = "pytest --no-header -rA --tb=no -p no:cacheprovider"
PYTEST_ASTROPY = "pytest -rA -vv -o console_output_style=classic --tb=no"
SEABORN = "pytest --no-header -rA"
SPHINX = "tox --current-env -epy39 -v --"
SYMPY = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"
DJANGO = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"

PRELUDE = "source /opt/miniconda3/bin/activate && conda activate testbed"


def test_command(repo: str, version: str) -> str:
    if repo == "django/django":
        return "./tests/runtests.py --verbosity 2" if version == "1.9" else DJANGO
    return {
        "astropy/astropy": PYTEST_ASTROPY,
        "mwaskom/seaborn": SEABORN,
        "sphinx-doc/sphinx": SPHINX,
        "sympy/sympy": SYMPY,
    }.get(repo, PYTEST)


def pre_test_commands(repo: str, version: str) -> list[str]:
    """Environment fixes SWE-bench applies before the tests run."""
    if repo == "django/django":
        return ["export LANG=en_US.UTF-8", "export LANGUAGE=en_US:en", "export LC_ALL=en_US.UTF-8"]
    if repo == "sphinx-doc/sphinx":
        return ["sed -i 's/pytest/pytest -rA/' tox.ini"]
    return []


def test_directives(repo: str, test_files: list[str]) -> list[str]:
    files = [f for f in test_files if not f.endswith(NON_TEST_EXTS)]
    if repo == "django/django":
        out = []
        for f in files:
            mod = f[: -len(".py")] if f.endswith(".py") else f
            mod = mod[len("tests/") :] if mod.startswith("tests/") else mod
            out.append(mod.replace("/", "."))
        return out
    return files


def image_name(instance_id: str) -> str:
    return "swebench/sweb.eval.x86_64." + instance_id.replace("__", "_1776_").lower() + ":latest"


# -- log parsers --------------------------------------------------------------------------


def parse_pytest(log: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in log.split("\n"):
        if not line.startswith(STATUSES):
            continue
        if line.startswith(FAILED):
            line = line.replace(" - ", " ")
        parts = line.split()
        if len(parts) <= 1:
            continue
        if parts[0] == SKIPPED and re.match(r"^\[\d+\]$", parts[1]):
            continue
        out[parts[1]] = parts[0]
    return out


def parse_pytest_options(log: str) -> dict[str, str]:
    """pytest, but ``test[/abs/path/x]`` parameters are shortened to their basename."""
    out: dict[str, str] = {}
    for name, status in parse_pytest(log).items():
        m = re.match(r"(.*?)\[(.*)\]", name)
        if m:
            main, option = m.groups()
            if option.startswith("/") and not option.startswith("//") and "*" not in option:
                option = "/" + option.split("/")[-1]
            name = f"{main}[{option}]"
        out[name] = status
    return out


def parse_pytest_v2(log: str) -> dict[str, str]:
    out: dict[str, str] = {}
    escapes = "".join(chr(c) for c in range(1, 32))
    table = str.maketrans("", "", escapes)
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line).translate(table)
        if line.startswith(STATUSES):
            if line.startswith(FAILED):
                line = line.split(" - ", 1)[0]
            parts = line.split()
            if len(parts) >= 2 and not (parts[0] == SKIPPED and re.match(r"^\[\d+\]$", parts[1])):
                out[" ".join(parts[1:])] = parts[0]
        elif line.endswith(STATUSES):
            parts = line.split()
            if len(parts) >= 2:
                out[" ".join(parts[:-1])] = parts[-1]
    return out


def parse_seaborn(log: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in log.split("\n"):
        parts = line.split()
        if len(parts) < 2:
            continue
        if line.startswith(FAILED):
            out[parts[1]] = FAILED
        elif f" {PASSED} " in line and parts[1] == PASSED:
            out[parts[0]] = PASSED
        elif line.startswith(PASSED):
            out[parts[1]] = PASSED
    return out


def parse_matplotlib(log: str) -> dict[str, str]:
    return parse_pytest(log.replace("MouseButton.LEFT", "1").replace("MouseButton.RIGHT", "3"))


def parse_sympy(log: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        out[f"{m[1]}.py:{m[2]}"] = FAILED
    for line in log.split("\n"):
        line = line.strip()
        if not line.startswith("test_"):
            continue
        name = line.split()[0]
        if line.endswith(" E"):
            out[name] = ERROR
        elif line.endswith(" F"):
            out[name] = FAILED
        elif line.endswith(" ok"):
            out[name] = PASSED
    return out


def parse_django(log: str) -> dict[str, str]:
    out: dict[str, str] = {}
    prev = None
    for raw in log.split("\n"):
        line = raw.strip()
        if "--version is equivalent to version" in line:
            out["--version is equivalent to version"] = PASSED
        if " ... " in line:
            prev = line.split(" ... ")[0]
        for suffix in (" ... ok", " ... OK", " ...  OK"):
            if line.endswith(suffix):
                if line.startswith("Applying sites.0002_alter_domain_unique...test_no_migrations"):
                    line = line.split("...", 1)[-1].strip()
                out[line.rsplit(suffix, 1)[0]] = PASSED
                break
        if " ... skipped" in line:
            out[line.split(" ... skipped")[0]] = SKIPPED
        if line.endswith(" ... FAIL"):
            out[line.split(" ... FAIL")[0]] = FAILED
        if line.startswith("FAIL:"):
            out[line.split()[1].strip()] = FAILED
        if line.endswith(" ... ERROR"):
            out[line.split(" ... ERROR")[0]] = ERROR
        if line.startswith("ERROR:"):
            out[line.split()[1].strip()] = ERROR
        if line.lstrip().startswith("ok") and prev is not None:
            out[prev] = PASSED
    for pattern in (
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
    ):
        for m in re.finditer(pattern, log, re.MULTILINE):
            out[m.group(1)] = PASSED
    return out


PARSERS: dict[str, Callable[[str], dict[str, str]]] = {
    "astropy/astropy": parse_pytest_v2,
    "django/django": parse_django,
    "matplotlib/matplotlib": parse_matplotlib,
    "mwaskom/seaborn": parse_seaborn,
    "pallets/flask": parse_pytest,
    "psf/requests": parse_pytest_options,
    "pydata/xarray": parse_pytest,
    "pylint-dev/pylint": parse_pytest_options,
    "pytest-dev/pytest": parse_pytest,
    "scikit-learn/scikit-learn": parse_pytest_v2,
    "sphinx-doc/sphinx": parse_pytest_v2,
    "sympy/sympy": parse_sympy,
}


def passed(status: str | None) -> bool:
    return status in (PASSED, XFAIL)


def grade(
    statuses: dict[str, str], fail_to_pass: list[str], pass_to_pass: list[str]
) -> dict[str, object]:
    """SWE-bench grading: resolved iff every F2P and every P2P test passes."""
    f2p_ok = [t for t in fail_to_pass if passed(statuses.get(t))]
    p2p_ok = [t for t in pass_to_pass if passed(statuses.get(t))]
    return {
        "resolved": len(f2p_ok) == len(fail_to_pass) and len(p2p_ok) == len(pass_to_pass),
        "f2p_passed": len(f2p_ok),
        "f2p_total": len(fail_to_pass),
        "p2p_passed": len(p2p_ok),
        "p2p_total": len(pass_to_pass),
        "f2p_failing": sorted(set(fail_to_pass) - set(f2p_ok)),
        "p2p_failing": sorted(set(pass_to_pass) - set(p2p_ok)),
    }
