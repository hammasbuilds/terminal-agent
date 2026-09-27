"""Approval policy: which tool calls run unasked, which need a human, which never run.

Shell commands are tokenised (``shlex`` with shell punctuation), split into simple commands
at ``; && || | & ( )`` and newlines, unwrapped (``sudo``, ``env``, ``timeout``, ``xargs``,
``nice``, ``bash -c '...'``, ``trap '...'``, ``$( ... )`` ...) and each simple command is
rated ``safe`` (read-only or a test run), ``mutating`` or ``dangerous``. The command's rating
is the worst of its parts.

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
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from terminal_agent.policy_rules import (
    _EXEC_PROCESS_SUBST,
    _HEREDOC_TO_INTERPRETER,
    _SUBST,
    ALWAYS_DANGEROUS,
    CODE_RED_FLAGS,
    DELETERS,
    EDITORS,
    ENV_EXEC,
    ENV_HIJACK,
    GIT_EXEC_OPT,
    GIT_READ,
    GIT_WRITE_OPT,
    INFRA,
    INFRA_DESTRUCTIVE,
    INSTALL_VERBS,
    INTERPRETERS,
    KEYWORDS,
    MAX_DEPTH,
    OPERATORS,
    PACKAGE_MANAGERS,
    PLACEHOLDER,
    READ_ONLY,
    REDIRECTS,
    RUFF_WRITE,
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
    Risk,
    _command_name,
    _dangerous,
    _mutating,
    _opt,
    _skip_options,
    _writes_into_git,
)
from terminal_agent.protocol import ToolCall


class Decision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class Verdict:
    decision: Decision
    risk: Risk
    reason: str

















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
            return "\x00outside"
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
        if resolved is None or resolved == "\x00outside":
            return True
        return not (resolved == self.root or resolved.startswith(self.root + "/"))

    def _cd_target(self, words: list[str], cwd: str) -> str:
        """cwd after a ``cd`` simple command (UNKNOWN_CWD if it cannot be resolved)."""
        rest = list(words)
        while rest and (rest[0] in KEYWORDS or re.match(r"^[A-Za-z_]\w*=", rest[0])):
            rest.pop(0)
        if not rest or _command_name(rest[0]) not in ("cd", "pushd"):
            return cwd
        args = [a for a in rest[1:] if not a.startswith("-")]
        if not args:  # `cd` with no argument goes home
            return UNKNOWN_CWD
        resolved = self._resolve(args[0], cwd)
        if resolved is None or resolved == "\x00outside":
            return UNKNOWN_CWD
        return resolved

    # -- entry point ---------------------------------------------------------------
    def rate(self, command: str, depth: int = 0) -> Rating:
        if depth > MAX_DEPTH:
            return _dangerous("nested too deeply to inspect")
        if not command.strip():
            return SAFE
        rating = SAFE
        if _EXEC_PROCESS_SUBST.search(command):
            rating = _dangerous("runs a downloaded or generated script via <( )")
        if _HEREDOC_TO_INTERPRETER.search(command):
            hit = next((f for f in CODE_RED_FLAGS if f in command.lower()), None)
            if hit:
                rating = _dangerous(f"heredoc code calls {hit.rstrip('(.')}")
        bodies, stripped = self._substitutions(command)
        for inner in bodies:
            rating = rating.worse(self.rate(inner, depth + 1))
        try:
            flat = stripped.replace("\r", "").replace("\n", " ; ")
            lexer = shlex.shlex(flat, posix=True, punctuation_chars=";&|()<>")
            lexer.whitespace = " \t"
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError as exc:
            return _dangerous(f"could not parse the command ({exc})")
        return rating.worse(self._rate_tokens(tokens, command, depth))

    @staticmethod
    def _substitutions(command: str) -> tuple[list[str], str]:
        """Split out ``$( )``, backtick and ``<( )``/``>( )`` bodies (outermost only)."""
        bodies: list[str] = []
        out: list[str] = []
        i = 0
        while i < len(command):
            m = _SUBST.match(command, i)
            if not m:
                out.append(command[i])
                i += 1
                continue
            start = m.end()
            if m.group() == "`":
                end = command.find("`", start)
                end = len(command) if end < 0 else end
                bodies.append(command[start:end])
                i = end + 1
            else:
                depth, j = 1, start
                while j < len(command) and depth:
                    depth += {"(": 1, ")": -1}.get(command[j], 0)
                    j += 1
                bodies.append(command[start : j - 1] if depth == 0 else command[start:])
                i = j
            out.append(PLACEHOLDER)
        return bodies, "".join(out)

    def _split_units(
        self, tokens: list[str]
    ) -> list[tuple[list[str], list[tuple[str, str]], bool]]:
        """Ordered simple commands: (words, redirects, is-piped-into)."""
        units: list[tuple[list[str], list[tuple[str, str]], bool]] = []
        words: list[str] = []
        redirects: list[tuple[str, str]] = []
        piped = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in OPERATORS or "\n" in tok:
                units.append((words, redirects, piped))
                words, redirects = [], []
                piped = tok in ("|", "|&")
                i += 1
                continue
            if tok in REDIRECTS or (tok.isdigit() and i + 1 < len(tokens)
                                    and tokens[i + 1] in REDIRECTS):
                if tok.isdigit():
                    i += 1
                    tok = tokens[i]
                target = tokens[i + 1] if i + 1 < len(tokens) else ""
                redirects.append((tok, target))
                i += 2
                continue
            words.append(tok)
            i += 1
        units.append((words, redirects, piped))
        return units

    def _rate_tokens(self, tokens: list[str], raw: str, depth: int) -> Rating:
        rating = SAFE
        if re.search(r"\w*\s*\(\s*\)\s*\{", raw):
            rating = rating.worse(_dangerous("defines a shell function (fork-bomb shape)"))
        if "&" in tokens:  # a lone & backgrounds a process that outlives the timeout
            rating = rating.worse(_dangerous("backgrounds a process that outlives the timeout"))
        cwd = self.root
        for words, redirects, piped in self._split_units(tokens):
            for op, target in redirects:
                rating = rating.worse(self._rate_redirect(op, target, cwd))
            if words:
                rating = rating.worse(self._rate_simple(words, piped, depth, cwd))
            cwd = self._cd_target(words, cwd)
        return rating

    def _rate_redirect(self, op: str, target: str, cwd: str) -> Rating:
        if op in ("<", "<<", "<<<", "<&") or (op == ">&" and target.isdigit()):
            return SAFE
        if target.startswith(("/dev/tcp", "/dev/udp")):
            return _dangerous("redirects to a network socket")
        if target.startswith(("/dev/sd", "/dev/nvme")):
            return _dangerous("writes to a raw disk")
        if self.outside(target, cwd):
            return _dangerous(f"writes outside the workspace ({target})")
        if _writes_into_git(target):
            return _dangerous("writes inside .git (a git hook or config runs on the next command)")
        return _mutating(f"writes {target}")

    # -- one simple command --------------------------------------------------------
    def _rate_simple(self, words: list[str], piped: bool, depth: int, cwd: str) -> Rating:
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
            return _dangerous("the command name is computed at run time")
        name = _command_name(name_raw)
        return rating.worse(self._rate_named(name, words[1:], piped, depth, cwd))

    def _rate_assignment(self, token: str) -> Rating:
        name = token.split("=", 1)[0]
        if ENV_HIJACK.match(token):
            return _dangerous(f"overrides {name}")
        if name in ENV_EXEC:
            return _dangerous(f"{name} names a program that gets executed")
        return SAFE

    def _rate_named(self, name: str, args: list[str], piped: bool, depth: int,
                    cwd: str) -> Rating:
        if name in ("sudo", "doas", "su", "runas"):
            return _dangerous(ALWAYS_DANGEROUS[name])
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
            rest = _skip_options(args, {"-n", "--interval", "-d", "--differences"})
            return self.rate(" ".join(rest), depth + 1) if rest else SAFE  # watch runs a string
        if name in ("timeout", "xargs"):
            rest = _skip_options(args, {"-n", "-s", "-k", "-a", "-I", "-d", "-L", "-P",
                                        "--signal", "--interval"})
            if name == "timeout" and rest:
                rest = rest[1:]  # the duration
            if not rest:
                return SAFE
            if name == "xargs" and _command_name(rest[0]) in DELETERS:
                return _dangerous("xargs deletes every listed path")
            return self._rate_simple(rest, piped, depth + 1, cwd)
        if name in EDITORS:
            if any(a in ("-c", "--command", "+", "-e", "--eval", "--batch") or a.startswith("+")
                   or (a.startswith("-c") and len(a) > 2) for a in args):
                return _dangerous(f"{name} runs editor commands that can execute shell code")
            return _mutating(f"opens {name}")
        if name == "make":
            dry = any(a in ("-n", "--dry-run", "--just-print", "--recon", "-q", "--question")
                      for a in args)
            for f in ("-f", "--file", "--makefile"):
                if f in args:
                    val = args[args.index(f) + 1] if args.index(f) + 1 < len(args) else ""
                    if self.outside(val, cwd) and not dry:
                        return _dangerous(f"make -f runs a makefile outside the workspace ({val})")
            targets = [a for a in args if not a.startswith("-") and "=" not in a]
            if not dry and any(t in ("install", "uninstall") for t in targets):
                return _dangerous("make install writes outside the workspace")
            return SAFE if dry else _mutating("runs make")
        if name == "trap":
            action = next((a for a in args if a != "--"), "")
            return self.rate(action, depth + 1) if action and not action.startswith("-") else SAFE
        if name in ALWAYS_DANGEROUS:
            return _dangerous(ALWAYS_DANGEROUS[name])
        if name.startswith("mkfs"):
            return _dangerous("formats a filesystem")
        if name in SHELLS:
            return self._rate_shell(name, args, piped, depth)
        if name in INTERPRETERS:
            return self._rate_interpreter(name, args, piped, depth, cwd)
        verb0 = next((a for a in args if not a.startswith("-")), "")
        if (name, verb0) in TEST_VERBS:
            return self._rate_test_verb(name, verb0, args)
        if name in INFRA:
            if any(a in INFRA_DESTRUCTIVE for a in args):
                return _dangerous(f"{name} changes or destroys remote infrastructure")
            return _mutating(f"runs {name}")
        if name in PACKAGE_MANAGERS:
            return self._rate_pkg(name, args)
        if name == "uv":
            sub = args[:2]
            if sub[:1] == ["pip"] and len(sub) > 1 and sub[1] in INSTALL_VERBS | {"sync"}:
                return _dangerous("uv pip changes installed packages")
            if sub[:1] in (["add"], ["remove"], ["sync"]):
                return _dangerous("uv changes installed packages")
            return _mutating("runs uv")
        if name == "git":
            return self._rate_git(args, cwd)
        if name == "date":
            return _dangerous("date -s sets the system clock") if any(
                a in ("-s", "--set") or a.startswith("--set=") for a in args) else SAFE
        if name == "mypy":
            return _dangerous("mypy --install-types installs packages") if any(
                a.startswith("--install-types") for a in args) else SAFE
        if name == "rm":
            return self._rate_rm(args, cwd)
        if name in ("rmdir", "del", "erase"):
            if any(a.lower() in ("/s", "/q", "-p", "--parents") for a in args) or any(
                    "*" in a for a in args):
                return _dangerous(f"{name} removes trees or wildcards")
            return self._paths_rating(args, f"{name} deletes", cwd)
        if name == "find":
            return self._rate_find(args, depth, cwd)
        if name == "docker":
            sub = next((a for a in args if not a.startswith("-")), "")
            if sub in ("ps", "images", "logs", "inspect", "version", "info"):
                return SAFE
            return _dangerous(f"docker {sub} changes containers or the host")
        if name in ("curl", "wget"):
            return self._rate_download(name, args, cwd)
        if name == "tar":
            return self._rate_tar(args, cwd)
        if name == "chmod":
            if any(a.startswith("-") and "R" in a for a in args) or "--recursive" in args:
                return _dangerous("recursive permission change")
            modes = [a for a in args if not a.startswith("-")]
            if any(re.search(r"[ugoa]*[+=][rwxXt]*s", m) or re.match(r"[2467]\d{3}$", m)
                   for m in modes):
                return _dangerous("sets a setuid/setgid bit")
            return self._paths_rating(args, "changes permissions", cwd)
        if name == "sed":
            return self._rate_sed(args, cwd)
        if name == "awk" or name == "gawk":
            return self._rate_awk(args)
        if name in ("rg", "ag") and self._has_exec_search_opt(args):
            return _dangerous(f"{name} runs a helper program per file")
        if name in TEXTUTIL_READONLY or name in TEXTUTIL_WRITE_OPT:
            return self._rate_textutil(name, args, cwd)
        if name in ("cp", "mv", "ln", "tee", "touch", "mkdir", "install", "unzip", "zip",
                    "gunzip", "patch"):
            return self._paths_rating(args, f"{name} writes files", cwd)
        if name in TEST_RUNNERS:
            return self._rate_pytest_module(args, cwd)
        if name in READ_ONLY:
            return SAFE
        return _mutating(f"'{name}' is not a known read-only command")

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
        return rating.worse(self._rate_simple(rest, piped, depth + 1, cwd)) if rest else rating

    def _rate_flock(self, args: list[str], piped: bool, depth: int, cwd: str) -> Rating:
        """flock [opts] LOCKFILE command...  or  flock [opts] -c 'command string'."""
        i = 0
        while i < len(args) and args[i].startswith("-"):
            opt = args[i]
            if _opt(opt) == "-c":  # flock -c 'string'
                return self.rate(args[i + 1], depth + 1) if i + 1 < len(args) else SAFE
            if _opt(opt) in WRAPPERS["flock"] and "=" not in opt and len(opt) <= 2:
                i += 2
            else:
                i += 1
        rest = args[i:]
        if not rest:
            return SAFE
        return self._rate_simple(rest[1:], piped, depth + 1, cwd) if len(rest) > 1 else SAFE

    def _rate_wrapper(self, name: str, args: list[str], piped: bool, depth: int,
                      cwd: str) -> Rating:
        value_opts = WRAPPERS[name]
        rating = SAFE
        i = 0
        while i < len(args) and args[i].startswith("-") and args[i] != "--":
            opt = args[i]
            base = _opt(opt)
            if name == "time" and base in WRITE_WRAPPER_OPT:  # time -o FILE writes
                val = opt.split("=", 1)[1] if "=" in opt else (
                    args[i + 1] if i + 1 < len(args) else "")
                rating = rating.worse(_mutating(f"{name} writes {val}") if not self.outside(
                    val, cwd) else _dangerous(f"{name} writes outside the workspace ({val})"))
                i += 1 if "=" in opt else 2
                continue
            if base in value_opts and "=" not in opt and len(opt) <= len(base):
                i += 2  # option plus its separate value
            else:
                i += 1  # a flag, or a short option with its value attached
        if i < len(args) and args[i] == "--":
            i += 1
        rest = args[i:]
        if name == "exec" and not rest:
            return rating
        return rating.worse(self._rate_simple(rest, piped, depth + 1, cwd)) if rest else rating

    # -- families ------------------------------------------------------------------
    def _rate_test_verb(self, name: str, verb: str, args: list[str]) -> Rating:
        if name == "ruff" and any(a in RUFF_WRITE for a in args):
            return _mutating("ruff --fix rewrites files")
        if name == "go" and verb == "test" and any(
                a == "-exec" or a.startswith("-exec=") for a in args):
            return _dangerous("go test -exec runs an arbitrary program")
        if name in ("npm", "pnpm", "yarn") and "-g" in args:
            return _dangerous(f"{name} -g changes global packages")
        return SAFE

    def _rate_pkg(self, name: str, args: list[str]) -> Rating:
        verb = next((a for a in args if not a.startswith("-")), "")
        if verb in INSTALL_VERBS or (name in ("npm", "pnpm", "yarn") and "-g" in args) or (
                name == "pipx" and verb == "run"):
            return _dangerous(f"{name} {verb} changes installed packages")
        if name in ("pip", "pip3") and verb in ("list", "show", "freeze", "check"):
            return SAFE
        if name == "go" and verb in ("install", "get"):
            return _dangerous("go install changes installed packages")
        return _mutating(f"runs {name}")

    def _rate_textutil(self, name: str, args: list[str], cwd: str) -> Rating:
        write_opts = TEXTUTIL_WRITE_OPT.get(name, set())
        positionals = []
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--":
                positionals.extend(x for x in args[i + 1:])
                break
            base = _opt(a)
            if base in write_opts:
                val = a.split("=", 1)[1] if "=" in a else (args[i + 1] if i + 1 < len(args) else "")
                return _dangerous(f"{name} writes outside the workspace ({val})") \
                    if self.outside(val, cwd) else _mutating(f"{name} writes {val}")
            if a.startswith("-"):
                i += 1
                continue
            positionals.append(a)
            i += 1
        # uniq/split take an OUTPUT positional (the last one, when there are two)
        if name in ("uniq", "split") and len(positionals) >= 2:
            out = positionals[-1]
            return _dangerous(f"{name} writes outside the workspace ({out})") \
                if self.outside(out, cwd) else _mutating(f"{name} writes {out}")
        return SAFE

    def _has_exec_search_opt(self, args: list[str]) -> bool:
        for a in args:
            base = _opt(a)
            if base in ("--pre", "--pre-glob", "--hostname-bin"):
                return True
        return False

    def _rate_sed(self, args: list[str], cwd: str) -> Rating:
        inplace = any(a == "-i" or a.startswith(("-i", "--in-place")) for a in args)
        scripts = [a for a in args if not a.startswith("-")]
        # -e/-f take the next token as the script/file; treat the joined non-options as scripts
        joined = " ".join(scripts)
        if re.search(r"(^|[;{}\s])[wW]\s+\S", joined) or re.search(r"s#?/[^/]*/[^/]*/[a-z0-9]*[wW]",
                                                                   joined) or "s/.*/.*/w" in joined:
            return _dangerous("sed writes a file with its w command")
        if re.search(r"(^|[;{}\s])e($|[;\s])", joined) or re.search(r"/[a-z]*e[a-z]*(;|$| )",
                                                                    joined):
            return _dangerous("sed executes a command with its e command")
        if not inplace:
            return SAFE
        targets = [a for a in scripts[1:]] if scripts else []
        return self._paths_rating(targets, "sed edits files in place", cwd)

    def _rate_awk(self, args: list[str]) -> Rating:
        program = " ".join(args)
        if "system(" in program or re.search(r'\|\s*(getline|"|\w)', program):
            return _dangerous("awk runs shell commands")
        if re.search(r'(print|printf)[^;{}]*>>?', program):
            return _dangerous("awk writes a file with a print redirect")
        return SAFE

    def _rate_tar(self, args: list[str], cwd: str) -> Rating:
        if any(a in ("-P", "--absolute-names") for a in args):
            return _dangerous("tar --absolute-names can write anywhere")
        if any(a.startswith("--checkpoint-action") or a.startswith("--to-command")
               or a == "--use-compress-program" for a in args):
            return _dangerous("tar runs a program via --checkpoint-action / --to-command")
        return self._paths_rating(args, "tar writes files", cwd)

    def _paths_rating(self, args: list[str], what: str, cwd: str) -> Rating:
        outside = [a for a in args if self.outside(a, cwd)]
        if outside:
            return _dangerous(f"{what} outside the workspace ({outside[0]})")
        return _mutating(what)

    def _rate_rm(self, args: list[str], cwd: str) -> Rating:
        opts = []
        for a in args:
            if a == "--":
                break
            if a.startswith("-"):
                opts.append(a)
        flags = "".join(o[1:] for o in opts if not o.startswith("--"))
        long = set(opts)
        if "f" in flags or "--force" in long:
            return _dangerous("forced delete")
        if "r" in flags or "R" in flags or "--recursive" in long:
            return _dangerous("recursive delete")
        if any("*" in a for a in args):
            return _dangerous("wildcard delete")
        return self._paths_rating([a for a in args if a != "--"], "deletes", cwd)

    def _rate_find(self, args: list[str], depth: int, cwd: str) -> Rating:
        if "-delete" in args:
            return _dangerous("find -delete removes files")
        for w in ("-fprint", "-fprint0", "-fprintf", "-fls"):
            if w in args:
                target = args[args.index(w) + 1] if args.index(w) + 1 < len(args) else ""
                return _dangerous(f"find {w} writes outside the workspace") \
                    if self.outside(target, cwd) else _mutating(f"find {w} writes {target}")
        for flag in ("-exec", "-execdir", "-ok", "-okdir"):
            if flag in args:
                start = args.index(flag) + 1
                inner = []
                for a in args[start:]:
                    if a in (";", "\\;", "+"):
                        break
                    inner.append(a)
                if inner and _command_name(inner[0]) in DELETERS:
                    return _dangerous("find -exec deletes every match")
                inner_rating = self._rate_simple(inner, False, depth + 1, cwd) if inner else SAFE
                return inner_rating.worse(_mutating("find runs a command per file"))
        return SAFE

    def _rate_download(self, name: str, args: list[str], cwd: str) -> Rating:
        upload = {"-d", "--data", "--data-binary", "--data-raw", "--data-urlencode", "-F",
                  "--form", "-T", "--upload-file", "--post-file", "--post-data", "--body-file"}
        if any(PLACEHOLDER in a or "$(" in a or "`" in a for a in args):
            return _dangerous(f"{name} sends data computed by another command")
        out_flags = {"-o", "--output", "-O", "--output-document", "-P", "--directory-prefix"}
        for i, a in enumerate(args):
            flag, _, val = a.partition("=")
            target = val if val else (args[i + 1] if i + 1 < len(args) else "")
            if _opt(flag) in out_flags and self.outside(target, cwd):
                return _dangerous(f"{name} writes a download outside the workspace ({target})")
        for a in args:
            flag = a.split("=", 1)[0]
            if flag in upload or (a.startswith("-d") and len(a) > 2 and name == "curl"):
                return _dangerous(f"{name} uploads data")
        if "-X" in args:
            method = args[args.index("-X") + 1] if args.index("-X") + 1 < len(args) else ""
            if method.upper() in ("POST", "PUT", "PATCH", "DELETE"):
                return _dangerous(f"{name} sends a {method.upper()} request")
        return _mutating(f"{name} downloads from the network")

    def _rate_git(self, args: list[str], cwd: str) -> Rating:
        rest = list(args)
        while rest and rest[0].startswith("-"):
            opt = rest.pop(0)
            base = _opt(opt)
            if base == "-c":  # `git -c key=value` can inject an executed hook
                return _dangerous("git -c overrides config, which can run a program")
            if base in ("--config-env",):
                return _dangerous("git --config-env overrides config from the environment")
            if base in ("-C", "--git-dir", "--work-tree", "--namespace") and "=" not in opt:
                val = rest.pop(0) if rest else ""
                if base == "-C" and self.outside(val, cwd):
                    return _dangerous(f"git -C runs in {val}, outside the workspace")
        if not rest:
            return SAFE
        sub, sargs = rest[0], rest[1:]
        if any(_opt(a) in GIT_EXEC_OPT for a in sargs):
            return _dangerous(f"git {sub} opens results in a pager / external diff")
        for a in sargs:
            if _opt(a) in GIT_WRITE_OPT:
                val = a.split("=", 1)[1] if "=" in a else (
                    sargs[sargs.index(a) + 1] if sargs.index(a) + 1 < len(sargs) else "")
                if self.outside(val, cwd):
                    return _dangerous(f"git {sub} writes outside the workspace ({val})")
        if sub in GIT_READ:
            if sub == "reflog" and sargs[:1] in (["expire"], ["delete"]):
                return _dangerous("git reflog expire destroys history")
            return SAFE
        if sub == "push":
            return _dangerous("git push publishes (or overwrites) remote history")
        if sub in ("filter-branch", "filter-repo", "rebase", "update-ref", "replace"):
            return _dangerous(f"git {sub} rewrites history")
        if sub == "reset" and ("--hard" in sargs or "--merge" in sargs or "--keep" in sargs):
            return _dangerous("git reset --hard discards work")
        if sub == "clean":
            return _dangerous("git clean deletes untracked files")
        if sub == "gc" and any(a.startswith("--prune") for a in sargs):
            return _dangerous("git gc --prune destroys unreachable objects")
        if sub in ("checkout", "restore"):
            pos = [a for a in sargs if not a.startswith("-")]
            if "--" in sargs or "." in sargs or "-f" in sargs or (sub == "restore" and sargs) \
                    or (sub == "checkout" and len(pos) >= 2):
                return _dangerous(f"git {sub} discards working-tree changes")
            return _mutating(f"git {sub}")
        if sub == "switch":
            if any(a in ("--discard-changes", "-f", "--force") for a in sargs):
                return _dangerous("git switch --discard-changes throws away changes")
            return _mutating("git switch")
        if sub == "commit":
            if any(a in ("--amend",) for a in sargs):
                return _dangerous("git commit --amend rewrites the last commit")
            return _mutating("git commit")
        if sub == "branch":
            if any(a in ("-D", "-d", "--delete", "-M", "-m", "--move", "-f") for a in sargs):
                return _dangerous("git branch deletes or moves a branch")
            return SAFE
        if sub == "stash":
            if sargs[:1] in (["drop"], ["clear"], ["pop"]):
                return _dangerous("git stash drop/clear/pop discards or replaces work")
            if sargs[:1] in (["list"], ["show"]):
                return SAFE
            return _mutating("git stash")
        if sub == "remote":
            if sargs[:1] in (["add"], ["remove"], ["rm"], ["set-url"], ["rename"]):
                return _dangerous("git remote changes where code is pushed")
            return SAFE
        if sub == "config":
            if any(a in ("--global", "--system") for a in sargs):
                return _dangerous("git config writes global settings")
            exec_keys = ("core.hookspath", "core.fsmonitor", "core.pager", "core.sshcommand",
                         "diff.external", "sequence.editor")
            if any(s.lower().startswith(exec_keys) for s in sargs):
                return _dangerous("git config sets a key that runs a program")
            if any(a in ("--get", "--list", "-l", "--get-all") for a in sargs):
                return SAFE
            return _mutating("git config")
        if sub == "tag":
            return SAFE if not sargs or sargs[0] in ("-l", "--list") else _mutating("git tag")
        return _mutating(f"git {sub}")

    def _rate_shell(self, name: str, args: list[str], piped: bool, depth: int) -> Rating:
        lowered = [a.lower() for a in args]
        if name in ("pwsh", "powershell") and any(
                a in ("-enc", "-encodedcommand", "-e", "-ec") for a in lowered):
            return _dangerous("runs an encoded PowerShell command")
        for flag_idx, a in enumerate(lowered):
            if a in ("-c", "-lc", "-ic", "/c", "/k", "-command", "-cmd") or (
                    a.startswith("-") and a.endswith("c") and len(a) <= 4 and name != "cmd"):
                script = " ".join(args[flag_idx + 1 :]) if name == "cmd" else (
                    args[flag_idx + 1] if flag_idx + 1 < len(args) else "")
                return self.rate(script, depth + 1)
        script_files = [a for a in args if not a.startswith("-")]
        if piped or not script_files:
            return _dangerous(f"pipes input into {name}, which runs it as code")
        return _mutating(f"runs the script {script_files[0]}")

    def _rate_interpreter(self, name: str, args: list[str], piped: bool, depth: int,
                          cwd: str) -> Rating:
        if name in ("perl", "ruby") and any(
                a.startswith("-") and not a.startswith("--") and "i" in a for a in args):
            return self._paths_rating([a for a in args if not a.startswith("-")][1:],
                                      f"{name} -i edits files in place", cwd)
        if not args or args == ["-"]:
            if piped:
                return _dangerous(f"pipes input into {name}, which runs it as code")
            return _mutating(f"starts {name}")
        if args[0] in ("--version", "-V", "--help", "-h"):
            return SAFE
        if args[0] == "-m" and len(args) > 1:
            mod = args[1]
            if mod in ("pytest", "unittest", "doctest", "py_compile", "compileall", "tabnanny"):
                if mod == "pytest":
                    return self._rate_pytest_module(args[2:], cwd)
                return SAFE
            if mod == "pip":
                return self._rate_pkg("pip", args[2:])
            return _mutating(f"runs the module {mod}")
        for flag in ("-c", "-e", "-E", "--eval", "-p", "-r", "--run"):
            if flag in args:
                code = args[args.index(flag) + 1] if args.index(flag) + 1 < len(args) else ""
                low = code.lower()
                if PLACEHOLDER in code or "$(" in code or "`" in code:
                    return _dangerous(f"inline {name} code is computed at run time")
                hit = next((f for f in CODE_RED_FLAGS if f in low), None)
                if hit:
                    return _dangerous(f"inline {name} code calls {hit.rstrip('(.')}")
                return _mutating(f"runs inline {name} code")
        return _mutating(f"runs {name} {args[0]}")

    def _rate_pytest_module(self, args: list[str], cwd: str) -> Rating:
        """pytest reads and runs tests, but --basetemp / -p writes and can load plugins."""
        for i, a in enumerate(args):
            base = _opt(a)
            if base in ("--basetemp", "--rootdir", "--junitxml", "--result-log",
                        "--report-log", "--cache-clear-dir"):
                val = a.split("=", 1)[1] if "=" in a else (args[i + 1] if i + 1 < len(args) else "")
                if self.outside(val, cwd):
                    return _dangerous(f"pytest writes outside the workspace ({val})")
                return _mutating(f"pytest writes {val or base}")
        return SAFE






READ_TOOLS = {"read_file", "list_dir", "glob", "grep"}
WRITE_TOOLS = {"write_file", "edit"}


class ApprovalPolicy:
    """Maps a tool call to ALLOW / ASK / DENY.

    ``mode``:
      * ``default`` - read-only commands and tests run; anything else asks.
      * ``auto``    - mutating commands run too; dangerous ones still ask.
    ``allow`` is a list of command prefixes the user trusts (``--allow "make"``); a matching
    *mutating* command runs unasked. A dangerous command is never allowlisted. Headless runs
    have no human, so the agent's approver turns every ASK into a DENY.
    """

    def __init__(self, workspace: Path, shell_root: str | None = None, mode: str = "default",
                 allow: list[str] | None = None) -> None:
        if mode not in ("default", "auto"):
            raise ValueError(f"unknown approval mode {mode!r} (use 'default' or 'auto')")
        self.workspace = workspace.resolve()
        self.mode = mode
        self.allow = [a.strip() for a in (allow or []) if a.strip()]
        self.classifier = CommandClassifier(root=shell_root or self.workspace.as_posix())

    def check(self, call: ToolCall) -> Verdict:
        name = call.name
        if name in READ_TOOLS or name in ("run_tests", "finish"):
            return Verdict(Decision.ALLOW, "safe", "read-only tool")
        if name in WRITE_TOOLS:
            return self._check_write(str(call.arguments.get("path", "")))
        if name == "run_shell":
            return self.check_command(str(call.arguments.get("command", "")))
        return Verdict(Decision.ALLOW, "safe", "unknown tools fail at dispatch")

    def _check_write(self, raw: str) -> Verdict:
        try:
            target = (self.workspace / raw).resolve()
        except (OSError, ValueError):
            return Verdict(Decision.DENY, "dangerous", f"unusable path {raw!r}")
        if not target.is_relative_to(self.workspace):
            return Verdict(Decision.ASK, "dangerous", f"writes outside the workspace: {target}")
        if ".git" in target.relative_to(self.workspace).parts:
            return Verdict(Decision.DENY, "dangerous", "writes inside .git")
        return Verdict(Decision.ALLOW, "mutating", "write inside the workspace")

    def check_command(self, command: str) -> Verdict:
        rating = self.classifier.rate(command)
        if rating.risk == "safe":
            return Verdict(Decision.ALLOW, "safe", rating.reason)
        if rating.risk == "dangerous":
            return Verdict(Decision.ASK, "dangerous", rating.reason)
        normalized = " ".join(command.split())
        if self.mode == "auto" or any(
                normalized == p or normalized.startswith(p + " ") for p in self.allow):
            return Verdict(Decision.ALLOW, "mutating", rating.reason)
        return Verdict(Decision.ASK, "mutating", rating.reason)
