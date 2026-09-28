"""How the classifier rates commands that run code: shells, interpreters, test runners,
package managers, make and editors."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from terminal_agent.policy_rules import (
    CODE_RED_FLAGS,
    INSTALL_VERBS,
    PLACEHOLDER,
    RUFF_WRITE,
    SAFE,
    Rating,
    dangerous,
    mutating,
    opt,
    option_values,
    writes_into_git,
)
from terminal_agent.rules_files import paths_rating, write_target

if TYPE_CHECKING:
    from terminal_agent.classifier import CommandClassifier


# options of test/lint verbs whose value is a file or directory the tool writes
TEST_VERB_WRITE_OPTS = {
    "go": {
        "-o",
        "-coverprofile",
        "-cpuprofile",
        "-memprofile",
        "-blockprofile",
        "-mutexprofile",
        "-trace",
        "-outputdir",
    },
    "cargo": {"--target-dir", "--artifact-dir"},
    "ruff": {"-o", "--output-file", "--cache-dir"},
}


def rate_test_verb(c: CommandClassifier, name: str, verb: str, args: list[str], cwd: str) -> Rating:
    for flag, val in option_values(args, TEST_VERB_WRITE_OPTS.get(name, set())):
        return write_target(c, f"{name} {verb} {flag}", val, cwd)
    if name == "ruff" and any(a in RUFF_WRITE for a in args):
        return mutating("ruff --fix rewrites files")
    if (
        name == "go"
        and verb == "test"
        and any(a == "-exec" or a.startswith("-exec=") for a in args)
    ):
        return dangerous("go test -exec runs an arbitrary program")
    if name in ("npm", "pnpm", "yarn") and "-g" in args:
        return dangerous(f"{name} -g changes global packages")
    return SAFE


def rate_pkg(name: str, args: list[str]) -> Rating:
    verb = next((a for a in args if not a.startswith("-")), "")
    if name in ("conda", "mamba") and any(a in INSTALL_VERBS | {"create"} for a in args):
        return dangerous(f"{name} changes environments or installed packages")
    if (
        verb in INSTALL_VERBS
        or (name in ("npm", "pnpm", "yarn") and "-g" in args)
        or (name == "pipx" and verb == "run")
    ):
        return dangerous(f"{name} {verb} changes installed packages")
    if name in ("pip", "pip3") and verb in ("list", "show", "freeze", "check"):
        return SAFE
    if name == "go" and verb in ("install", "get"):
        return dangerous("go install changes installed packages")
    return mutating(f"runs {name}")


def rate_uv(args: list[str]) -> Rating:
    sub = args[:2]
    if sub[:1] == ["pip"] and len(sub) > 1 and sub[1] in INSTALL_VERBS | {"sync"}:
        return dangerous("uv pip changes installed packages")
    if sub[:1] in (["add"], ["remove"], ["sync"]):
        return dangerous("uv changes installed packages")
    return mutating("runs uv")


def rate_make(c: CommandClassifier, args: list[str], cwd: str) -> Rating:
    dry = any(a in ("-n", "--dry-run", "--just-print", "--recon", "-q", "--question") for a in args)
    for f in ("-f", "--file", "--makefile"):
        if f in args:
            val = args[args.index(f) + 1] if args.index(f) + 1 < len(args) else ""
            if c.outside(val, cwd) and not dry:
                return dangerous(f"make -f runs a makefile outside the workspace ({val})")
    targets = [a for a in args if not a.startswith("-") and "=" not in a]
    if not dry and any(t in ("install", "uninstall") for t in targets):
        return dangerous("make install writes outside the workspace")
    return SAFE if dry else mutating("runs make")


def rate_editor(name: str, args: list[str]) -> Rating:
    if any(
        a in ("-c", "--command", "+", "-e", "--eval", "--batch")
        or a.startswith("+")
        or (a.startswith("-c") and len(a) > 2)
        for a in args
    ):
        return dangerous(f"{name} runs editor commands that can execute shell code")
    return mutating(f"opens {name}")


def rate_shell(c: CommandClassifier, name: str, args: list[str], piped: bool, depth: int) -> Rating:
    lowered = [a.lower() for a in args]
    if name in ("pwsh", "powershell") and any(
        a in ("-enc", "-encodedcommand", "-e", "-ec") for a in lowered
    ):
        return dangerous("runs an encoded PowerShell command")
    for flag_idx, a in enumerate(lowered):
        if a in ("-c", "-lc", "-ic", "/c", "/k", "-command", "-cmd") or (
            a.startswith("-") and a.endswith("c") and len(a) <= 4 and name != "cmd"
        ):
            script = (
                " ".join(args[flag_idx + 1 :])
                if name == "cmd"
                else (args[flag_idx + 1] if flag_idx + 1 < len(args) else "")
            )
            return c.rate(script, depth + 1)
    script_files = [a for a in args if not a.startswith("-")]
    if piped or not script_files:
        return dangerous(f"pipes input into {name}, which runs it as code")
    return mutating(f"runs the script {script_files[0]}")


def rate_interpreter(
    c: CommandClassifier, name: str, args: list[str], piped: bool, cwd: str
) -> Rating:
    if name in ("perl", "ruby") and any(
        a.startswith("-") and not a.startswith("--") and "i" in a for a in args
    ):
        return paths_rating(
            c,
            [a for a in args if not a.startswith("-")][1:],
            f"{name} -i edits files in place",
            cwd,
        )
    if not args or args == ["-"]:
        if piped:
            return dangerous(f"pipes input into {name}, which runs it as code")
        return mutating(f"starts {name}")
    if args[0] in ("--version", "-V", "--help", "-h"):
        return SAFE
    if args[0] == "-m" and len(args) > 1:
        mod = args[1]
        if mod in ("pytest", "unittest", "doctest", "py_compile", "compileall", "tabnanny"):
            if mod == "pytest":
                return rate_pytest(c, args[2:], cwd)
            return SAFE
        if mod == "pip":
            return rate_pkg("pip", args[2:])
        return mutating(f"runs the module {mod}")
    for flag in ("-c", "-e", "-E", "--eval", "-p", "-r", "--run"):
        if flag in args:
            code = args[args.index(flag) + 1] if args.index(flag) + 1 < len(args) else ""
            low = code.lower()
            if PLACEHOLDER in code or "$(" in code or "`" in code:
                return dangerous(f"inline {name} code is computed at run time")
            hit = next((f for f in CODE_RED_FLAGS if f in low), None)
            if hit:
                return dangerous(f"inline {name} code calls {hit.rstrip('(.')}")
            return _inline_code_paths(c, name, code, cwd)
    return mutating(f"runs {name} {args[0]}")


_STRING_LITERAL = re.compile(r"""(['"])(.*?)(?<!\\)\1""")
_WRITE_VERBS = (
    "open",
    "write",
    "rename",
    "replace",
    "copy",
    "move",
    "mkdir",
    "makedirs",
    "delete",
    "remove",
    "unlink",
    "symlink",
    "link",
    "chmod",
    "touch",
    "rm",
)


def _inline_code_paths(c: CommandClassifier, name: str, code: str, cwd: str) -> Rating:
    """Inline code that opens or writes a path literal outside the workspace is dangerous.

    ``python -c "open('../x','w')"`` names no dangerous function, but the path is readable
    in the string. Reading and writing cannot be told apart statically, so any file verb
    together with an outside path literal is treated as a write (fails closed).
    """
    low = code.lower()
    if any(v in low for v in _WRITE_VERBS):
        for _, literal in _STRING_LITERAL.findall(code):
            if not literal or not (
                literal.startswith(("/", "~", "..", "$"))
                or "/../" in literal
                or re.match(r"^[A-Za-z]:[\\/]", literal)
                or literal.startswith(".git")
                or "/.git" in literal
            ):
                continue
            if c.outside(literal, cwd):
                return dangerous(
                    f"inline {name} code opens a path outside the workspace ({literal})"
                )
            if writes_into_git(literal):
                return dangerous(f"inline {name} code opens a path inside .git ({literal})")
    return mutating(f"runs inline {name} code")


# pytest options whose value is a file or directory pytest writes (``--cov-report`` takes
# ``TYPE:PATH``; ``-o``/``--override-ini`` takes ``key=value``, checked for path keys)
PYTEST_WRITE_OPTS = {
    "--basetemp",
    "--rootdir",  # the cache (.pytest_cache) is written under the rootdir
    "--junitxml",
    "--junit-xml",
    "--result-log",
    "--report-log",
    "--log-file",
    "--debug",
    "--cov-report",
    "--cache-clear-dir",
}
PYTEST_PATH_INI_KEYS = {"cache_dir", "log_file"}
# pytest options that take a separate value (so the value is not a test path)
PYTEST_VALUE_OPTS = {
    "-k",
    "-m",
    "-p",
    "-o",
    "-c",
    "-W",
    "-r",
    "-n",
    "--tb",
    "--deselect",
    "--ignore",
    "--ignore-glob",
    "--maxfail",
    "--durations",
    "--confcutdir",
    "--import-mode",
    "--log-level",
    "--override-ini",
    "--capture",
    "--color",
    "--cov",
    "--cov-config",
    "--config-file",
    "--rootdir",
    "--basetemp",
    "--junitxml",
    "--junit-xml",
    "--log-file",
    "--cov-report",
    "--report-log",
    "--result-log",
    "--dist",
    "--timeout",
}


def rate_pytest(c: CommandClassifier, args: list[str], cwd: str) -> Rating:
    """pytest reads and runs tests, but some options write files or upload the results."""
    if any(opt(a) == "--pastebin" for a in args):
        return dangerous("pytest --pastebin uploads the test session to bpaste.net")
    rating = SAFE
    for flag, val in option_values(args, PYTEST_WRITE_OPTS):
        if flag == "--cov-report" and ":" in val:
            val = val.split(":", 1)[1]
        elif flag == "--cov-report" or (flag == "--debug" and not val):
            rating = rating.worse(mutating(f"pytest {flag} writes a report"))
            continue
        rating = rating.worse(write_target(c, f"pytest {flag}", val, cwd))
    for _, setting in option_values(args, {"-o", "--override-ini"}):
        key, _, val = setting.partition("=")
        if key.strip() in PYTEST_PATH_INI_KEYS:
            rating = rating.worse(write_target(c, f"pytest -o {key.strip()}", val, cwd))
    for path in _pytest_paths(args):
        # tests outside move the rootdir (and its .pytest_cache) outside too, and run
        # whatever conftest.py lives there
        if c.outside(path.split("::", 1)[0], cwd):
            rating = rating.worse(dangerous(f"pytest runs tests outside the workspace ({path})"))
    return rating


def _pytest_paths(args: list[str]) -> list[str]:
    out = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in PYTEST_VALUE_OPTS:
            i += 2
            continue
        if not a.startswith("-"):
            out.append(a)
        i += 1
    return out
