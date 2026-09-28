"""Rate a shell command string ``safe``, ``mutating`` or ``dangerous``.

Shell commands are tokenised (``shlex`` with shell punctuation), split into simple commands
at ``; && || | & ( )`` and newlines, unwrapped (``sudo``, ``env``, ``timeout``, ``xargs``,
``nice``, ``bash -c '...'``, ``trap '...'``, ``$( ... )`` ...) and each simple command is
rated ``safe`` (read-only or a test run), ``mutating`` or ``dangerous``. The command's rating
is the worst of its parts. The per-family rules live in ``rules_git``, ``rules_files`` and
``rules_exec``; this module holds the parsing, path resolution and dispatch.

Two design rules keep it from being defeated by a flag:

* **A command name is not a promise.** A read-only tool with an option that writes a file
  (``sort -o``, ``git diff --output=``, ``find -fprint``) or runs a helper program
  (``rg --pre``, ``git grep -O``, ``git -c core.fsmonitor=...``) is not safe.
* **A write target must resolve, statically, to inside the workspace.** ``cd`` is tracked
  across ``&&``/``;``; a path built from ``$VAR``, ``~``, ``$( )`` or an unknown ``cd`` is
  treated as outside. Failing closed is the only honest answer to a path it cannot read.
"""

from __future__ import annotations

import posixpath
import re

from terminal_agent import rules_exec, rules_files, rules_git
from terminal_agent.policy_rules import (
    ALWAYS_DANGEROUS,
    CODE_RED_FLAGS,
    DELETERS,
    EDITORS,
    ENV_EXEC,
    ENV_HIJACK,
    EXEC_PROCESS_SUBST,
    HEREDOC_TO_INTERPRETER,
    INFRA,
    INFRA_DESTRUCTIVE,
    INTERPRETERS,
    KEYWORDS,
    MAX_DEPTH,
    PACKAGE_MANAGERS,
    PLACEHOLDER,
    READ_ONLY,
    SAFE,
    SHELLS,
    TEST_RUNNERS,
    TEST_VERBS,
    TEXTUTIL_READONLY,
    TEXTUTIL_WRITE_OPT,
    UNKNOWN_CWD,
    WRAPPERS,
    WRITE_WRAPPER_OPT,
    Rating,
    command_name,
    dangerous,
    mutating,
    opt,
    skip_options,
    writes_into_git,
)
from terminal_agent.shell_parse import split_units, substitutions, tokenize

OUTSIDE = "\x00outside"  # _resolve's answer for a path that is certainly outside (C:\, \\host)


class CommandClassifier:
    """Rates a shell command string. ``root`` is the workspace as the shell sees it."""

    def __init__(self, root: str = "/workspace") -> None:
        self.root = root.rstrip("/") or "/"

    # -- paths ---------------------------------------------------------------------
    def _resolve(self, word: str, cwd: str) -> str | None:
        """Absolute posix path a word points at, or None if not statically resolvable."""
        if not word or word.startswith("-") or "://" in word or PLACEHOLDER in word:
            return None
        if word == "/dev/null" or word.startswith(("/dev/std", "/dev/fd/")):
            return self.root  # a harmless pseudo-file; treat as "inside"
        if "$" in word or "`" in word or word.startswith("~"):
            return None
        if re.match(r"^[A-Za-z]:[\\/]", word) or word.startswith("\\\\"):
            return OUTSIDE
        w = word.replace("\\", "/")
        if w.startswith("/"):
            return posixpath.normpath(w)
        if cwd == UNKNOWN_CWD:
            return None
        return posixpath.normpath(posixpath.join(cwd, w))

    def outside(self, word: str, cwd: str | None = None) -> bool:
        """True when ``word`` as a *write* target is outside the workspace or unresolvable."""
        base = self.root if cwd is None else cwd
        if not word or word.startswith("-"):
            return False
        resolved = self._resolve(word, base)
        if resolved == self.root:
            return False
        if resolved is None or resolved == OUTSIDE:
            return True
        return not (resolved == self.root or resolved.startswith(self.root + "/"))

    def _cd_target(self, words: list[str], cwd: str) -> str:
        """cwd after a ``cd`` simple command (UNKNOWN_CWD if it cannot be resolved)."""
        rest = list(words)
        while rest and (rest[0] in KEYWORDS or re.match(r"^[A-Za-z_]\w*=", rest[0])):
            rest.pop(0)
        if not rest or command_name(rest[0]) not in ("cd", "pushd"):
            return cwd
        args = [a for a in rest[1:] if not a.startswith("-")]
        if not args:  # `cd` with no argument goes home
            return UNKNOWN_CWD
        resolved = self._resolve(args[0], cwd)
        if resolved is None or resolved == OUTSIDE:
            return UNKNOWN_CWD
        return resolved

    # -- entry point ---------------------------------------------------------------
    def rate(self, command: str, depth: int = 0) -> Rating:
        if depth > MAX_DEPTH:
            return dangerous("nested too deeply to inspect")
        if not command.strip():
            return SAFE
        rating = SAFE
        if EXEC_PROCESS_SUBST.search(command):
            rating = dangerous("runs a downloaded or generated script via <( )")
        if HEREDOC_TO_INTERPRETER.search(command):
            hit = next((f for f in CODE_RED_FLAGS if f in command.lower()), None)
            if hit:
                rating = dangerous(f"heredoc code calls {hit.rstrip('(.')}")
        bodies, stripped = substitutions(command)
        for inner in bodies:
            rating = rating.worse(self.rate(inner, depth + 1))
        try:
            tokens = tokenize(stripped)
        except ValueError as exc:
            return dangerous(f"could not parse the command ({exc})")
        return rating.worse(self._rate_tokens(tokens, command, depth))

    def _rate_tokens(self, tokens: list[str], raw: str, depth: int) -> Rating:
        rating = SAFE
        if re.search(r"\w*\s*\(\s*\)\s*\{", raw):
            rating = rating.worse(dangerous("defines a shell function (fork-bomb shape)"))
        if "&" in tokens:  # a lone & backgrounds a process that outlives the timeout
            rating = rating.worse(dangerous("backgrounds a process that outlives the timeout"))
        cwd = self.root
        for unit in split_units(tokens):
            for op, target in unit.redirects:
                rating = rating.worse(self._rate_redirect(op, target, cwd))
            if unit.words:
                rating = rating.worse(self.rate_words(unit.words, unit.piped, depth, cwd))
            cwd = self._cd_target(unit.words, cwd)
        return rating

    def _rate_redirect(self, op: str, target: str, cwd: str) -> Rating:
        if op in ("<", "<<", "<<<", "<&") or (op == ">&" and target.isdigit()):
            return SAFE
        if target.startswith(("/dev/tcp", "/dev/udp")):
            return dangerous("redirects to a network socket")
        if target.startswith(("/dev/sd", "/dev/nvme")):
            return dangerous("writes to a raw disk")
        if self.outside(target, cwd):
            return dangerous(f"writes outside the workspace ({target})")
        if writes_into_git(target):
            return dangerous("writes inside .git (a git hook or config runs on the next command)")
        return mutating(f"writes {target}")

    # -- one simple command --------------------------------------------------------
    def rate_words(self, words: list[str], piped: bool, depth: int, cwd: str) -> Rating:
        """Rate one simple command given as words (assignments, name, arguments)."""
        words = list(words)
        while words and words[0] in KEYWORDS:
            words.pop(0)
        if not words or words[0] in ("for", "in", "case"):
            return SAFE
        rating = SAFE
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            rating = rating.worse(self._rate_assignment(words[0]))
            words.pop(0)
        if not words:
            return rating
        name_raw = words[0]
        if "$" in name_raw or "`" in name_raw or PLACEHOLDER in name_raw:
            return dangerous("the command name is computed at run time")
        name = command_name(name_raw)
        return rating.worse(self._rate_named(name, words[1:], piped, depth, cwd))

    def _rate_assignment(self, token: str) -> Rating:
        name = token.split("=", 1)[0]
        if ENV_HIJACK.match(token):
            return dangerous(f"overrides {name}")
        if name in ENV_EXEC:
            return dangerous(f"{name} names a program that gets executed")
        return SAFE

    def _rate_named(self, name: str, args: list[str], piped: bool, depth: int, cwd: str) -> Rating:
        wrapped = self._rate_wrapping(name, args, piped, depth, cwd)
        if wrapped is not None:
            return wrapped
        if name in ALWAYS_DANGEROUS:
            return dangerous(ALWAYS_DANGEROUS[name])
        if name.startswith("mkfs"):
            return dangerous("formats a filesystem")
        if name in SHELLS:
            return rules_exec.rate_shell(self, name, args, piped, depth)
        if name in INTERPRETERS:
            return rules_exec.rate_interpreter(self, name, args, piped, cwd)
        verb0 = next((a for a in args if not a.startswith("-")), "")
        if (name, verb0) in TEST_VERBS:
            return rules_exec.rate_test_verb(name, verb0, args)
        if name in INFRA:
            if any(a in INFRA_DESTRUCTIVE for a in args):
                return dangerous(f"{name} changes or destroys remote infrastructure")
            return mutating(f"runs {name}")
        if name in PACKAGE_MANAGERS:
            return rules_exec.rate_pkg(name, args)
        return self._rate_tool(name, args, cwd, depth)

    def _rate_wrapping(
        self, name: str, args: list[str], piped: bool, depth: int, cwd: str
    ) -> Rating | None:
        """Commands that run another command or string; None if ``name`` is not one."""
        if name in ("sudo", "doas", "su", "runas"):
            return dangerous(ALWAYS_DANGEROUS[name])
        if name == "env":
            return self._rate_env(args, piped, depth, cwd)
        if name == "export":
            rating = SAFE
            for a in args:
                if "=" in a:
                    rating = rating.worse(self._rate_assignment(a))
            return rating
        if name == "flock":
            return self._rate_flock(args, piped, depth, cwd)
        if name in WRAPPERS:
            return self._rate_wrapper(name, args, piped, depth, cwd)
        if name == "watch":
            rest = skip_options(args, {"-n", "--interval", "-d", "--differences"})
            return self.rate(" ".join(rest), depth + 1) if rest else SAFE  # watch runs a string
        if name in ("timeout", "xargs"):
            rest = skip_options(
                args, {"-n", "-s", "-k", "-a", "-I", "-d", "-L", "-P", "--signal", "--interval"}
            )
            if name == "timeout" and rest:
                rest = rest[1:]  # the duration
            if not rest:
                return SAFE
            if name == "xargs" and command_name(rest[0]) in DELETERS:
                return dangerous("xargs deletes every listed path")
            return self.rate_words(rest, piped, depth + 1, cwd)
        if name in EDITORS:
            return rules_exec.rate_editor(name, args)
        if name == "make":
            return rules_exec.rate_make(self, args, cwd)
        if name == "trap":
            action = next((a for a in args if a != "--"), "")
            return self.rate(action, depth + 1) if action and not action.startswith("-") else SAFE
        return None

    def _rate_tool(self, name: str, args: list[str], cwd: str, depth: int) -> Rating:
        """Everything that is not a wrapper, shell, interpreter or package manager."""
        if name == "uv":
            return rules_exec.rate_uv(args)
        if name == "git":
            return rules_git.rate_git(self, args, cwd)
        if name == "date":
            return (
                dangerous("date -s sets the system clock")
                if any(a in ("-s", "--set") or a.startswith("--set=") for a in args)
                else SAFE
            )
        if name == "mypy":
            return (
                dangerous("mypy --install-types installs packages")
                if any(a.startswith("--install-types") for a in args)
                else SAFE
            )
        if name == "rm":
            return rules_files.rate_rm(self, args, cwd)
        if name in ("rmdir", "del", "erase"):
            if any(a.lower() in ("/s", "/q", "-p", "--parents") for a in args) or any(
                "*" in a for a in args
            ):
                return dangerous(f"{name} removes trees or wildcards")
            return rules_files.paths_rating(self, args, f"{name} deletes", cwd)
        if name == "find":
            return rules_files.rate_find(self, args, depth, cwd)
        if name == "docker":
            sub = next((a for a in args if not a.startswith("-")), "")
            if sub in ("ps", "images", "logs", "inspect", "version", "info"):
                return SAFE
            return dangerous(f"docker {sub} changes containers or the host")
        if name in ("curl", "wget"):
            return rules_files.rate_download(self, name, args, cwd)
        if name == "tar":
            return rules_files.rate_tar(self, args, cwd)
        if name == "chmod":
            return rules_files.rate_chmod(self, args, cwd)
        if name == "sed":
            return rules_files.rate_sed(self, args, cwd)
        if name in ("awk", "gawk"):
            return rules_files.rate_awk(args)
        if name in ("rg", "ag") and rules_files.has_exec_search_opt(args):
            return dangerous(f"{name} runs a helper program per file")
        if name in TEXTUTIL_READONLY or name in TEXTUTIL_WRITE_OPT:
            return rules_files.rate_textutil(self, name, args, cwd)
        if name in (
            "cp",
            "mv",
            "ln",
            "tee",
            "touch",
            "mkdir",
            "install",
            "unzip",
            "zip",
            "gunzip",
            "patch",
        ):
            return rules_files.paths_rating(self, args, f"{name} writes files", cwd)
        if name in TEST_RUNNERS:
            return rules_exec.rate_pytest(self, args, cwd)
        if name in READ_ONLY:
            return SAFE
        return mutating(f"'{name}' is not a known read-only command")

    # -- unwrappers ----------------------------------------------------------------
    def _rate_env(self, args: list[str], piped: bool, depth: int, cwd: str) -> Rating:
        rating = SAFE
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--":
                i += 1
                break
            if a in ("-S", "--split-string") and i + 1 < len(args):
                return self.rate(args[i + 1], depth + 1)  # env -S re-splits into a command
            if a.startswith(("-S", "--split-string=")):
                return self.rate(a.split("=", 1)[-1] if "=" in a else a[2:], depth + 1)
            if a in ("-i", "--ignore-environment", "-0", "--null"):
                i += 1
                continue
            if a in ("-u", "--unset", "-C", "--chdir") and i + 1 < len(args):
                i += 2
                continue
            if "=" in a and not a.startswith("-"):
                rating = rating.worse(self._rate_assignment(a))
                i += 1
                continue
            break
        rest = args[i:]
        return rating.worse(self.rate_words(rest, piped, depth + 1, cwd)) if rest else rating

    def _rate_flock(self, args: list[str], piped: bool, depth: int, cwd: str) -> Rating:
        """flock [opts] LOCKFILE command...  or  flock [opts] -c 'command string'."""
        i = 0
        while i < len(args) and args[i].startswith("-"):
            tok = args[i]
            if opt(tok) == "-c":  # flock -c 'string'
                return self.rate(args[i + 1], depth + 1) if i + 1 < len(args) else SAFE
            if opt(tok) in WRAPPERS["flock"] and "=" not in tok and len(tok) <= 2:
                i += 2
            else:
                i += 1
        rest = args[i:]
        if not rest:
            return SAFE
        return self.rate_words(rest[1:], piped, depth + 1, cwd) if len(rest) > 1 else SAFE

    def _rate_wrapper(
        self, name: str, args: list[str], piped: bool, depth: int, cwd: str
    ) -> Rating:
        value_opts = WRAPPERS[name]
        rating = SAFE
        i = 0
        while i < len(args) and args[i].startswith("-") and args[i] != "--":
            tok = args[i]
            base = opt(tok)
            if name == "time" and base in WRITE_WRAPPER_OPT:  # time -o FILE writes
                val = (
                    tok.split("=", 1)[1]
                    if "=" in tok
                    else (args[i + 1] if i + 1 < len(args) else "")
                )
                rating = rating.worse(
                    mutating(f"{name} writes {val}")
                    if not self.outside(val, cwd)
                    else dangerous(f"{name} writes outside the workspace ({val})")
                )
                i += 1 if "=" in tok else 2
                continue
            if base in value_opts and "=" not in tok and len(tok) <= len(base):
                i += 2  # option plus its separate value
            else:
                i += 1  # a flag, or a short option with its value attached
        if i < len(args) and args[i] == "--":
            i += 1
        rest = args[i:]
        if name == "exec" and not rest:
            return rating
        return rating.worse(self.rate_words(rest, piped, depth + 1, cwd)) if rest else rating
