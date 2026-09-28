"""How the classifier rates a ``git`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from terminal_agent.policy_rules import (
    GIT_EXEC_OPT,
    GIT_READ,
    GIT_WRITE_OPT,
    SAFE,
    Rating,
    dangerous,
    mutating,
    opt,
)

if TYPE_CHECKING:
    from terminal_agent.classifier import CommandClassifier


def rate_git(c: CommandClassifier, args: list[str], cwd: str) -> Rating:
    rest = list(args)
    while rest and rest[0].startswith("-"):
        tok = rest.pop(0)
        base = opt(tok)
        if base == "-c":  # `git -c key=value` can inject an executed hook
            return dangerous("git -c overrides config, which can run a program")
        if base in ("--config-env",):
            return dangerous("git --config-env overrides config from the environment")
        if base in ("-C", "--git-dir", "--work-tree", "--namespace") and "=" not in tok:
            val = rest.pop(0) if rest else ""
            if base == "-C" and c.outside(val, cwd):
                return dangerous(f"git -C runs in {val}, outside the workspace")
    if not rest:
        return SAFE
    sub, sargs = rest[0], rest[1:]
    if any(opt(a) in GIT_EXEC_OPT for a in sargs):
        return dangerous(f"git {sub} opens results in a pager / external diff")
    for a in sargs:
        if opt(a) in GIT_WRITE_OPT:
            val = (
                a.split("=", 1)[1]
                if "=" in a
                else (sargs[sargs.index(a) + 1] if sargs.index(a) + 1 < len(sargs) else "")
            )
            if c.outside(val, cwd):
                return dangerous(f"git {sub} writes outside the workspace ({val})")
    if sub in GIT_READ:
        if sub == "reflog" and sargs[:1] in (["expire"], ["delete"]):
            return dangerous("git reflog expire destroys history")
        return SAFE
    if sub == "push":
        return dangerous("git push publishes (or overwrites) remote history")
    if sub in ("filter-branch", "filter-repo", "rebase", "update-ref", "replace"):
        return dangerous(f"git {sub} rewrites history")
    if sub == "reset" and ("--hard" in sargs or "--merge" in sargs or "--keep" in sargs):
        return dangerous("git reset --hard discards work")
    if sub == "clean":
        return dangerous("git clean deletes untracked files")
    if sub == "gc" and any(a.startswith("--prune") for a in sargs):
        return dangerous("git gc --prune destroys unreachable objects")
    if sub in ("checkout", "restore"):
        pos = [a for a in sargs if not a.startswith("-")]
        if (
            "--" in sargs
            or "." in sargs
            or "-f" in sargs
            or (sub == "restore" and sargs)
            or (sub == "checkout" and len(pos) >= 2)
        ):
            return dangerous(f"git {sub} discards working-tree changes")
        return mutating(f"git {sub}")
    if sub == "switch":
        if any(a in ("--discard-changes", "-f", "--force") for a in sargs):
            return dangerous("git switch --discard-changes throws away changes")
        return mutating("git switch")
    if sub == "commit":
        if any(a in ("--amend",) for a in sargs):
            return dangerous("git commit --amend rewrites the last commit")
        return mutating("git commit")
    if sub == "branch":
        if any(a in ("-D", "-d", "--delete", "-M", "-m", "--move", "-f") for a in sargs):
            return dangerous("git branch deletes or moves a branch")
        return SAFE
    if sub == "stash":
        if sargs[:1] in (["drop"], ["clear"], ["pop"]):
            return dangerous("git stash drop/clear/pop discards or replaces work")
        if sargs[:1] in (["list"], ["show"]):
            return SAFE
        return mutating("git stash")
    if sub == "remote":
        if sargs[:1] in (["add"], ["remove"], ["rm"], ["set-url"], ["rename"]):
            return dangerous("git remote changes where code is pushed")
        return SAFE
    if sub == "config":
        if any(a in ("--global", "--system") for a in sargs):
            return dangerous("git config writes global settings")
        exec_keys = (
            "core.hookspath",
            "core.fsmonitor",
            "core.pager",
            "core.sshcommand",
            "diff.external",
            "sequence.editor",
        )
        if any(s.lower().startswith(exec_keys) for s in sargs):
            return dangerous("git config sets a key that runs a program")
        if any(a in ("--get", "--list", "-l", "--get-all") for a in sargs):
            return SAFE
        return mutating("git config")
    if sub == "tag":
        return SAFE if not sargs or sargs[0] in ("-l", "--list") else mutating("git tag")
    return mutating(f"git {sub}")
