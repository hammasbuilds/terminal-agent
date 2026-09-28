"""How the classifier rates a ``git`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from terminal_agent.policy_rules import (
    GIT_CONFIG_READ_FLAGS,
    GIT_CONFIG_SAFE_KEYS,
    GIT_EXEC_OPT,
    GIT_READ,
    GIT_REDIRECT_OPT,
    GIT_WRITE_OPT,
    SAFE,
    Rating,
    dangerous,
    mutating,
    opt,
    opt_value,
    writes_into_git,
)

if TYPE_CHECKING:
    from terminal_agent.classifier import CommandClassifier


def rate_git(c: CommandClassifier, args: list[str], cwd: str, depth: int = 0) -> Rating:
    rest = list(args)
    while rest and rest[0].startswith("-"):
        tok = rest.pop(0)
        base = opt(tok)
        if base == "-c":  # `git -c key=value` can inject an executed hook
            return dangerous("git -c overrides config, which can run a program")
        if base == "--config-env":
            return dangerous("git --config-env overrides config from the environment")
        if base in GIT_REDIRECT_OPT:
            if base == "--exec-path" and "=" not in tok:
                continue  # bare --exec-path only prints the directory
            # another git dir's config (core.fsmonitor, hooks) runs on the next command
            return dangerous(
                f"git {base} points git at a repository or program directory "
                "whose config and hooks can run a program"
            )
        if base in ("-C", "--namespace") and "=" not in tok:
            val = rest.pop(0) if rest else ""
            if base == "-C" and c.outside(val, cwd):
                return dangerous(f"git -C runs in {val}, outside the workspace")
    if not rest:
        return SAFE
    sub, sargs = rest[0], rest[1:]
    if any(opt(a) in GIT_EXEC_OPT for a in sargs):
        return dangerous(f"git {sub} opens results in a pager / external diff or tool")
    wrote = _output_option(c, sub, sargs, cwd)
    if wrote is not None:
        return wrote
    if sub in GIT_READ:
        if sub == "reflog" and sargs[:1] in (["expire"], ["delete"]):
            return dangerous("git reflog expire destroys history")
        return SAFE
    if sub == "config":
        return _rate_config(sargs)
    if sub in ("submodule", "bisect"):
        return _rate_runs_command(c, sub, sargs, cwd, depth)
    return _rate_writing_sub(sub, sargs)


def _output_option(c: CommandClassifier, sub: str, sargs: list[str], cwd: str) -> Rating | None:
    """``git diff --output=f`` / ``git log -o f``: a read-only subcommand that writes."""
    for i, a in enumerate(sargs):
        if opt(a) in GIT_WRITE_OPT and not (sub == "grep" and opt(a) == "-o"):  # --only-matching
            val = opt_value(sargs, i)[0]
            if c.outside(val, cwd):
                return dangerous(f"git {sub} writes outside the workspace ({val})")
            if writes_into_git(val):
                return dangerous(f"git {sub} writes inside .git or a git-dir-shaped path")
            return mutating(f"git {sub} writes {val}")
    return None


def _rate_config(sargs: list[str]) -> Rating:
    """Reading config is safe; setting a key is allowed only for keys that run nothing."""
    if any(a in ("--global", "--system") for a in sargs):
        return dangerous("git config writes global settings")
    if any(opt(a) in ("-f", "--file", "--blob") for a in sargs):
        return dangerous("git config -f reads or writes an arbitrary config file")
    positional = [a for a in sargs if not a.startswith("-")]
    verb = (
        positional[0]
        if positional
        and positional[0]
        in ("get", "list", "set", "unset", "rename-section", "remove-section", "edit")
        else ""
    )
    if verb:
        positional = positional[1:]
    if verb in ("get", "list") or any(a in GIT_CONFIG_READ_FLAGS for a in sargs):
        return SAFE
    if verb == "edit" or "-e" in sargs or "--edit" in sargs:
        return dangerous("git config --edit opens an editor")
    if not positional:
        return SAFE
    key = positional[0].lower()
    setting = (
        len(positional) > 1
        or verb in ("set", "unset", "rename-section", "remove-section")
        or any(
            a
            in (
                "--unset",
                "--unset-all",
                "--add",
                "--replace-all",
                "--remove-section",
                "--rename-section",
            )
            for a in sargs
        )
    )
    if not setting:
        return SAFE  # `git config KEY` prints the value
    if key.startswith(GIT_CONFIG_SAFE_KEYS):
        return mutating(f"git config sets {key}")
    return dangerous(
        f"git config sets {key}, a key that is not known to be inert "
        "(aliases, filters, hooks, pagers and helpers run programs)"
    )


def _rate_runs_command(
    c: CommandClassifier, sub: str, sargs: list[str], cwd: str, depth: int
) -> Rating:
    """``git submodule foreach CMD`` and ``git bisect run CMD...`` run a command."""
    if sub == "submodule" and "foreach" in sargs:
        after = [a for a in sargs[sargs.index("foreach") + 1 :] if not a.startswith("-")]
        return c.rate(" ".join(after), depth + 1).worse(mutating("git submodule foreach"))
    if sub == "bisect" and sargs[:1] == ["run"]:
        inner = sargs[1:]
        return (c.rate_words(inner, False, depth + 1, cwd) if inner else SAFE).worse(
            mutating("git bisect run")
        )
    return mutating(f"git {sub}")


def _rate_writing_sub(sub: str, sargs: list[str]) -> Rating:
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
        if "--amend" in sargs:
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
    if sub == "tag":
        return SAFE if not sargs or sargs[0] in ("-l", "--list") else mutating("git tag")
    return mutating(f"git {sub}")
