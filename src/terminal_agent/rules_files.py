"""How the classifier rates commands that read, write or delete files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from terminal_agent.policy_rules import (
    DELETERS,
    PLACEHOLDER,
    SAFE,
    TEXTUTIL_WRITE_OPT,
    Rating,
    command_name,
    dangerous,
    mutating,
    opt,
)

if TYPE_CHECKING:
    from terminal_agent.classifier import CommandClassifier


def paths_rating(c: CommandClassifier, args: list[str], what: str, cwd: str) -> Rating:
    outside = [a for a in args if c.outside(a, cwd)]
    if outside:
        return dangerous(f"{what} outside the workspace ({outside[0]})")
    return mutating(what)


def rate_textutil(c: CommandClassifier, name: str, args: list[str], cwd: str) -> Rating:
    write_opts = TEXTUTIL_WRITE_OPT.get(name, set())
    positionals = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            positionals.extend(x for x in args[i + 1 :])
            break
        base = opt(a)
        if base in write_opts:
            val = a.split("=", 1)[1] if "=" in a else (args[i + 1] if i + 1 < len(args) else "")
            return (
                dangerous(f"{name} writes outside the workspace ({val})")
                if c.outside(val, cwd)
                else mutating(f"{name} writes {val}")
            )
        if a.startswith("-"):
            i += 1
            continue
        positionals.append(a)
        i += 1
    # uniq/split take an OUTPUT positional (the last one, when there are two)
    if name in ("uniq", "split") and len(positionals) >= 2:
        out = positionals[-1]
        return (
            dangerous(f"{name} writes outside the workspace ({out})")
            if c.outside(out, cwd)
            else mutating(f"{name} writes {out}")
        )
    return SAFE


def has_exec_search_opt(args: list[str]) -> bool:
    return any(opt(a) in ("--pre", "--pre-glob", "--hostname-bin") for a in args)


def rate_sed(c: CommandClassifier, args: list[str], cwd: str) -> Rating:
    inplace = any(a == "-i" or a.startswith(("-i", "--in-place")) for a in args)
    scripts = [a for a in args if not a.startswith("-")]
    # -e/-f take the next token as the script/file; treat the joined non-options as scripts
    joined = " ".join(scripts)
    if (
        re.search(r"(^|[;{}\s])[wW]\s+\S", joined)
        or re.search(r"s#?/[^/]*/[^/]*/[a-z0-9]*[wW]", joined)
        or "s/.*/.*/w" in joined
    ):
        return dangerous("sed writes a file with its w command")
    if re.search(r"(^|[;{}\s])e($|[;\s])", joined) or re.search(r"/[a-z]*e[a-z]*(;|$| )", joined):
        return dangerous("sed executes a command with its e command")
    if not inplace:
        return SAFE
    targets = list(scripts[1:]) if scripts else []
    return paths_rating(c, targets, "sed edits files in place", cwd)


def rate_awk(args: list[str]) -> Rating:
    program = " ".join(args)
    if "system(" in program or re.search(r'\|\s*(getline|"|\w)', program):
        return dangerous("awk runs shell commands")
    if re.search(r"(print|printf)[^;{}]*>>?", program):
        return dangerous("awk writes a file with a print redirect")
    return SAFE


def rate_tar(c: CommandClassifier, args: list[str], cwd: str) -> Rating:
    if any(a in ("-P", "--absolute-names") for a in args):
        return dangerous("tar --absolute-names can write anywhere")
    if any(
        a.startswith("--checkpoint-action")
        or a.startswith("--to-command")
        or a == "--use-compress-program"
        for a in args
    ):
        return dangerous("tar runs a program via --checkpoint-action / --to-command")
    return paths_rating(c, args, "tar writes files", cwd)


def rate_rm(c: CommandClassifier, args: list[str], cwd: str) -> Rating:
    opts = []
    for a in args:
        if a == "--":
            break
        if a.startswith("-"):
            opts.append(a)
    flags = "".join(o[1:] for o in opts if not o.startswith("--"))
    long = set(opts)
    if "f" in flags or "--force" in long:
        return dangerous("forced delete")
    if "r" in flags or "R" in flags or "--recursive" in long:
        return dangerous("recursive delete")
    if any("*" in a for a in args):
        return dangerous("wildcard delete")
    return paths_rating(c, [a for a in args if a != "--"], "deletes", cwd)


def rate_find(c: CommandClassifier, args: list[str], depth: int, cwd: str) -> Rating:
    if "-delete" in args:
        return dangerous("find -delete removes files")
    for w in ("-fprint", "-fprint0", "-fprintf", "-fls"):
        if w in args:
            target = args[args.index(w) + 1] if args.index(w) + 1 < len(args) else ""
            return (
                dangerous(f"find {w} writes outside the workspace")
                if c.outside(target, cwd)
                else mutating(f"find {w} writes {target}")
            )
    for flag in ("-exec", "-execdir", "-ok", "-okdir"):
        if flag in args:
            start = args.index(flag) + 1
            inner = []
            for a in args[start:]:
                if a in (";", "\\;", "+"):
                    break
                inner.append(a)
            if inner and command_name(inner[0]) in DELETERS:
                return dangerous("find -exec deletes every match")
            inner_rating = c.rate_words(inner, False, depth + 1, cwd) if inner else SAFE
            return inner_rating.worse(mutating("find runs a command per file"))
    return SAFE


def rate_download(c: CommandClassifier, name: str, args: list[str], cwd: str) -> Rating:
    upload = {
        "-d",
        "--data",
        "--data-binary",
        "--data-raw",
        "--data-urlencode",
        "-F",
        "--form",
        "-T",
        "--upload-file",
        "--post-file",
        "--post-data",
        "--body-file",
    }
    if any(PLACEHOLDER in a or "$(" in a or "`" in a for a in args):
        return dangerous(f"{name} sends data computed by another command")
    out_flags = {"-o", "--output", "-O", "--output-document", "-P", "--directory-prefix"}
    for i, a in enumerate(args):
        flag, _, val = a.partition("=")
        target = val if val else (args[i + 1] if i + 1 < len(args) else "")
        if opt(flag) in out_flags and c.outside(target, cwd):
            return dangerous(f"{name} writes a download outside the workspace ({target})")
    for a in args:
        flag = a.split("=", 1)[0]
        if flag in upload or (a.startswith("-d") and len(a) > 2 and name == "curl"):
            return dangerous(f"{name} uploads data")
    if "-X" in args:
        method = args[args.index("-X") + 1] if args.index("-X") + 1 < len(args) else ""
        if method.upper() in ("POST", "PUT", "PATCH", "DELETE"):
            return dangerous(f"{name} sends a {method.upper()} request")
    return mutating(f"{name} downloads from the network")


def rate_chmod(c: CommandClassifier, args: list[str], cwd: str) -> Rating:
    if any(a.startswith("-") and "R" in a for a in args) or "--recursive" in args:
        return dangerous("recursive permission change")
    modes = [a for a in args if not a.startswith("-")]
    if any(re.search(r"[ugoa]*[+=][rwxXt]*s", m) or re.match(r"[2467]\d{3}$", m) for m in modes):
        return dangerous("sets a setuid/setgid bit")
    return paths_rating(c, args, "changes permissions", cwd)
