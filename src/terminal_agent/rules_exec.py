"""How the classifier rates commands that run code: shells, interpreters, test runners,
package managers, make and editors."""

from __future__ import annotations

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
)
from terminal_agent.rules_files import paths_rating

if TYPE_CHECKING:
    from terminal_agent.classifier import CommandClassifier


def rate_test_verb(name: str, verb: str, args: list[str]) -> Rating:
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
            return mutating(f"runs inline {name} code")
    return mutating(f"runs {name} {args[0]}")


def rate_pytest(c: CommandClassifier, args: list[str], cwd: str) -> Rating:
    """pytest reads and runs tests, but --basetemp / -p writes and can load plugins."""
    for i, a in enumerate(args):
        base = opt(a)
        if base in (
            "--basetemp",
            "--rootdir",
            "--junitxml",
            "--result-log",
            "--report-log",
            "--cache-clear-dir",
        ):
            val = a.split("=", 1)[1] if "=" in a else (args[i + 1] if i + 1 < len(args) else "")
            if c.outside(val, cwd):
                return dangerous(f"pytest writes outside the workspace ({val})")
            return mutating(f"pytest writes {val or base}")
    return SAFE
