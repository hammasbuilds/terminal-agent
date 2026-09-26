"""Approval policy: which tool calls run unasked, which need a human, which never run.

Shell commands are tokenised (``shlex`` with shell punctuation), split into simple
commands at ``; && || | & ( )`` and newlines, unwrapped (``sudo``, ``env``, ``timeout``,
``xargs``, ``bash -c '...'``, ``trap '...'``, ``$( ... )`` ...) and each simple command is
rated ``safe`` (read-only or a test run), ``mutating`` or ``dangerous``. The command's
rating is the worst of its parts. Anything the classifier cannot parse, or whose command
name is computed at run time (``$cmd``, ``$(echo rm)``), is ``dangerous``: failing closed is
the only honest answer to a command it cannot read.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from terminal_agent.protocol import ToolCall

Risk = Literal["safe", "mutating", "dangerous"]
_ORDER = {"safe": 0, "mutating": 1, "dangerous": 2}
MAX_DEPTH = 6


class Decision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class Verdict:
    decision: Decision
    risk: Risk
    reason: str


@dataclass
class Rating:
    risk: Risk
    reason: str

    def worse(self, other: Rating) -> Rating:
        return other if _ORDER[other.risk] > _ORDER[self.risk] else self


SAFE = Rating("safe", "read-only")

READ_ONLY = {
    "ls", "dir", "pwd", "cat", "head", "tail", "less", "more", "wc", "grep", "egrep", "fgrep",
    "rg", "ag", "echo", "printf", "which", "where", "type", "file", "stat", "du", "df", "tree",
    "sort", "uniq", "cut", "tr", "diff", "cmp", "comm", "basename", "dirname", "realpath",
    "readlink", "date", "whoami", "uname", "true", "false", "test", "[", "jq", "nl", "column",
    "md5sum", "sha1sum", "sha256sum", "cd", "sleep", "seq", "yes", "ps", "printenv", "mypy",
}
TEST_RUNNERS = {"pytest", "py.test", "runtests.py", "test"}
# (command, first argument) pairs that only run tests or linters
TEST_VERBS = {("npm", "test"), ("cargo", "test"), ("go", "test"), ("go", "vet"),
              ("ruff", "check"), ("tox", "-l")}
ALWAYS_DANGEROUS = {
    "dd": "raw disk/file writes", "fdisk": "partitions disks", "wipefs": "wipes disks",
    "shred": "destroys files", "format": "formats a disk", "mount": "remounts filesystems",
    "umount": "unmounts filesystems", "kill": "kills processes", "pkill": "kills processes",
    "killall": "kills processes", "taskkill": "kills processes", "shutdown": "stops the machine",
    "reboot": "reboots the machine", "halt": "stops the machine", "poweroff": "stops the machine",
    "systemctl": "controls system services", "service": "controls system services",
    "su": "switches user", "sudo": "runs as root", "doas": "runs as root", "runas": "runs as root",
    "crontab": "edits scheduled jobs", "chown": "changes ownership", "reg": "edits the registry",
    "setx": "persists environment variables", "wsl": "controls WSL", "nc": "raw network socket",
    "ncat": "raw network socket", "netcat": "raw network socket", "telnet": "remote shell",
    "scp": "copies files to another host", "rsync": "copies files to another host",
    "ssh": "remote shell", "sftp": "remote file transfer", "ftp": "remote file transfer",
    "unlink": "deletes a file", "truncate": "empties files", "alias": "redefines commands",
    "eval": "runs a string as code", "source": "runs a file as shell code",
    ".": "runs a file as shell code", "iex": "runs a string as code",
    "invoke-expression": "runs a string as code", "irm": "downloads for execution",
    "remove-item": "deletes files", "rd": "deletes directories",
    "blkdiscard": "discards a disk's blocks", "vssadmin": "deletes shadow copies",
    "diskpart": "partitions disks", "bcdedit": "edits boot configuration",
    "schtasks": "edits scheduled tasks", "at": "schedules jobs", "launchctl": "edits services",
    "sc": "controls Windows services", "ssh-keygen": "creates or overwrites keys",
    "history": "edits shell history", "net": "manages Windows users and shares", "npx": "downloads and runs a package",
    "uvx": "downloads and runs a package", "bunx": "downloads and runs a package",
}
# infrastructure CLIs: these verbs destroy remote resources
INFRA = {"kubectl", "terraform", "aws", "gcloud", "az", "helm", "pulumi", "doctl", "gsutil"}
INFRA_DESTRUCTIVE = {"delete", "destroy", "rb", "rm", "uninstall", "terminate", "drain",
                     "apply", "scale", "remove", "purge", "down", "replace", "patch"}
PACKAGE_MANAGERS = {"pip", "pip3", "apt", "apt-get", "yum", "dnf", "brew", "conda", "choco",
                    "winget", "npm", "pnpm", "yarn", "gem", "cargo", "mamba", "pipx", "go"}
INSTALL_VERBS = {"install", "uninstall", "remove", "add", "i", "update", "upgrade", "reinstall",
                 "get", "exec", "dlx"}
SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "fish", "cmd", "pwsh", "powershell"}
INTERPRETERS = {"python", "python3", "python2", "perl", "ruby", "node", "php", "deno", "bun"}
WRAPPERS = {"command", "builtin", "exec", "nohup", "nice", "time", "stdbuf", "busybox",
            "ionice", "chronic"}
KEYWORDS = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "case", "esac",
            "{", "}", "!", "[[", "]]", "function", "select"}
GIT_READ = {"status", "diff", "log", "show", "blame", "rev-parse", "ls-files", "grep",
            "describe", "shortlog", "reflog", "cat-file", "ls-tree", "whatchanged", "version"}
CODE_RED_FLAGS = ("rmtree", "remove(", "unlink", "rmsync", "rmdir", "system(", "subprocess",
                  "popen", "exec(", "spawn", "child_process", "os.kill", "chmod", "truncate",
                  "urlopen", "requests.", "socket")
ENV_HIJACK = re.compile(r"^(PATH|LD_PRELOAD|LD_LIBRARY_PATH|PYTHONPATH|PYTHONSTARTUP|"
                        r"BASH_ENV|ENV|PROMPT_COMMAND|NODE_OPTIONS)=")
OPERATORS = {";", "&&", "||", "|", "&", "(", ")", ";;", "|&", "\n"}
REDIRECTS = {">", ">>", ">|", "&>", "&>>", ">&", "<", "<<", "<<<", "<>", "<&"}
_SUBST = re.compile(r"\$\(|`|<\(|>\(")
_EXEC_PROCESS_SUBST = re.compile(
    r"(^|[;&|\s])(bash|sh|zsh|dash|ksh|python3?|perl|ruby|node|source|\.)\s+<\(")
_HEREDOC_TO_INTERPRETER = re.compile(r"(^|[;&|\s])(python3?|perl|ruby|node|php)\s+(-\s+)?<<")
DELETERS = {"rm", "unlink", "shred", "rmdir", "del", "erase", "remove-item"}


def _dangerous(reason: str) -> Rating:
    return Rating("dangerous", reason)


def _mutating(reason: str) -> Rating:
    return Rating("mutating", reason)


class CommandClassifier:
    """Rates a shell command string. ``root`` is the workspace as the shell sees it."""

    def __init__(self, root: str = "/workspace") -> None:
        self.root = root.rstrip("/") or "/"

    # -- paths ------------------------------------------------------------------
    def outside(self, word: str) -> bool:
        """Does ``word``, read as a path, point outside the workspace?"""
        if not word or word.startswith("-") or "://" in word:
            return False
        if re.match(r"^[A-Za-z]:[\\/]", word) or word.startswith("\\\\"):
            return True
        if word.startswith("~"):
            return True
        if word.startswith("$"):
            return bool(re.match(r"^\$\{?(HOME|USERPROFILE|APPDATA)\b", word))
        if word == "/dev/null" or word.startswith("/dev/std") or word.startswith("/dev/fd/"):
            return False
        norm = posixpath.normpath(posixpath.join(self.root, word.replace("\\", "/")))
        return not (norm == self.root or norm.startswith(self.root + "/"))

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
        for inner in self._substitutions(command):
            rating = rating.worse(self.rate(inner, depth + 1))
        try:
            flat = command.replace("\r", "").replace("\n", " ; ")
            lexer = shlex.shlex(flat, posix=True,
                                punctuation_chars=";&|()<>")
            lexer.whitespace = " \t"
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError as exc:
            return _dangerous(f"could not parse the command ({exc})")
        return rating.worse(self._rate_tokens(tokens, command, depth))

    @staticmethod
    def _substitutions(command: str) -> list[str]:
        """Bodies of ``$( )``, backticks and process substitutions, outermost first."""
        found: list[str] = []
        for m in _SUBST.finditer(command):
            start = m.end()
            if m.group() == "`":
                end = command.find("`", start)
                if end > 0:
                    found.append(command[start:end])
                continue
            depth, i = 1, start
            while i < len(command) and depth:
                depth += {"(": 1, ")": -1}.get(command[i], 0)
                i += 1
            found.append(command[start : i - 1] if depth == 0 else command[start:])
        return found

    def _rate_tokens(self, tokens: list[str], raw: str, depth: int) -> Rating:
        rating = SAFE
        segments: list[list[str]] = [[]]
        after_pipe: list[bool] = [False]
        redirect_targets: list[tuple[str, str]] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in OPERATORS or "\n" in tok:
                if tok == "&":
                    rating = rating.worse(_dangerous("backgrounds a process that outlives "
                                                     "the timeout"))
                segments.append([])
                after_pipe.append(tok in ("|", "|&"))
                i += 1
                continue
            if tok in REDIRECTS or (tok.isdigit() and i + 1 < len(tokens)
                                    and tokens[i + 1] in REDIRECTS):
                if tok.isdigit():
                    i += 1
                    tok = tokens[i]
                target = tokens[i + 1] if i + 1 < len(tokens) else ""
                redirect_targets.append((tok, target))
                i += 2
                continue
            segments[-1].append(tok)
            i += 1

        if re.search(r"\w*\s*\(\s*\)\s*\{", raw):
            rating = rating.worse(_dangerous("defines a shell function (fork-bomb shape)"))
        for op, target in redirect_targets:
            if op in ("<", "<<", "<<<", "<&"):
                continue
            if op == ">&" and target.isdigit():
                continue
            if target.startswith("/dev/tcp") or target.startswith("/dev/udp"):
                rating = rating.worse(_dangerous("redirects to a network socket"))
            elif target.startswith("/dev/sd") or target.startswith("/dev/nvme"):
                rating = rating.worse(_dangerous("writes to a raw disk"))
            elif self.outside(target):
                rating = rating.worse(_dangerous(f"writes outside the workspace ({target})"))
            else:
                rating = rating.worse(_mutating(f"writes {target}"))
        for words, piped in zip(segments, after_pipe, strict=True):
            if words:
                rating = rating.worse(self._rate_simple(words, piped, depth))
        return rating

    # -- one simple command ---------------------------------------------------------
    def _rate_simple(self, words: list[str], piped: bool, depth: int) -> Rating:
        words = list(words)
        while words and words[0] in KEYWORDS:
            words.pop(0)
        if not words:
            return SAFE
        if words[0] in ("for", "in", "case"):
            return SAFE
        rating = SAFE
        # leading VAR=value assignments
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            if ENV_HIJACK.match(words[0]):
                rating = rating.worse(_dangerous(f"overrides {words[0].split('=')[0]}"))
            words.pop(0)
        if not words:
            return rating
        name_raw = words[0]
        if "$" in name_raw or "`" in name_raw:
            return _dangerous("the command name is computed at run time")
        name = _command_name(name_raw)
        args = words[1:]
        return rating.worse(self._rate_named(name, args, piped, depth))

    def _rate_named(self, name: str, args: list[str], piped: bool, depth: int) -> Rating:
        if name in ("sudo", "doas", "su", "runas"):
            return _dangerous(ALWAYS_DANGEROUS[name])
        if name == "env":
            rest = [a for a in args if not (a.startswith("-") or "=" in a)]
            hijack = [a for a in args if ENV_HIJACK.match(a)]
            if hijack:
                return _dangerous(f"overrides {hijack[0].split('=')[0]}")
            return self._rate_simple(rest, piped, depth + 1) if rest else SAFE
        if name == "export":
            hijack = [a for a in args if ENV_HIJACK.match(a)]
            return _dangerous(f"overrides {hijack[0].split('=')[0]}") if hijack else SAFE
        if name in WRAPPERS:
            return self._rate_simple(args, piped, depth + 1) if args else SAFE
        if name in ("timeout", "watch", "xargs"):
            rest = _skip_options(args, takes_value={"-n", "-s", "-k", "-a", "-I", "-d", "-L",
                                                    "-P", "--signal", "--interval"})
            if name == "timeout" and rest:
                rest = rest[1:]  # the duration
            if name == "xargs" and not rest:
                return SAFE
            if name == "xargs" and _command_name(rest[0]) in DELETERS:
                return _dangerous("xargs deletes every listed path")
            return self._rate_simple(rest, piped, depth + 1) if rest else SAFE
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
            return self._rate_interpreter(name, args, piped, depth)
        verb0 = next((a for a in args if not a.startswith("-")), "")
        if (name, verb0) in TEST_VERBS and "--fix" not in args:
            return SAFE
        if name in INFRA:
            if any(a in INFRA_DESTRUCTIVE for a in args):
                return _dangerous(f"{name} changes or destroys remote infrastructure")
            return _mutating(f"runs {name}")
        if name in PACKAGE_MANAGERS:
            verb = next((a for a in args if not a.startswith("-")), "")
            if verb in INSTALL_VERBS or (name in ("npm", "pnpm", "yarn") and "-g" in args) or (
                    name == "pipx" and verb == "run"):
                return _dangerous(f"{name} {verb} changes installed packages")
            if name in ("pip", "pip3") and verb in ("list", "show", "freeze", "check"):
                return SAFE
            return _mutating(f"runs {name}")
        if name == "uv":
            sub = args[:2]
            if sub[:1] == ["pip"] and len(sub) > 1 and sub[1] in INSTALL_VERBS | {"sync"}:
                return _dangerous("uv pip changes installed packages")
            if sub[:1] in (["add"], ["remove"], ["sync"]):
                return _dangerous("uv changes installed packages")
            return _mutating("runs uv")
        if name == "git":
            return self._rate_git(args)
        if name == "rm":
            return self._rate_rm(args)
        if name in ("rmdir", "del", "erase"):
            if any(a.lower() in ("/s", "/q", "-p", "--parents") for a in args) or any(
                    "*" in a for a in args):
                return _dangerous(f"{name} removes trees or wildcards")
            return self._paths_rating(args, f"{name} deletes", destructive=True)
        if name == "find":
            return self._rate_find(args, depth)
        if name == "docker":
            sub = next((a for a in args if not a.startswith("-")), "")
            if sub in ("ps", "images", "logs", "inspect", "version", "info"):
                return SAFE
            return _dangerous(f"docker {sub} changes containers or the host")
        if name in ("curl", "wget"):
            return self._rate_download(name, args)
        if name == "tar" and any(a in ("-P", "--absolute-names") for a in args):
            return _dangerous("tar --absolute-names can write anywhere")

        if name == "chmod":
            if any(a.startswith("-") and "R" in a for a in args) or "--recursive" in args:
                return _dangerous("recursive permission change")
            return self._paths_rating(args, "changes permissions", destructive=False)
        if name == "sed":
            inplace = any(a == "-i" or a.startswith("-i") or a.startswith("--in-place")
                          for a in args)
            if not inplace:
                return SAFE
            return self._paths_rating(args[1:], "edits files in place", destructive=False)
        if name == "awk":
            program = " ".join(args)
            if "system(" in program or ("|" in program and "getline" in program):
                return _dangerous("awk runs shell commands")
            return SAFE
        if name in ("cp", "mv", "ln", "tee", "touch", "mkdir", "install", "patch", "tar",
                    "unzip", "zip", "gzip", "gunzip"):
            return self._paths_rating(args, f"{name} writes files", destructive=False)
        if name in READ_ONLY:
            return SAFE
        if name in TEST_RUNNERS:
            return SAFE
        return _mutating(f"'{name}' is not a known read-only command")

    def _paths_rating(self, args: list[str], what: str, destructive: bool) -> Rating:
        outside = [a for a in args if self.outside(a)]
        if outside:
            return _dangerous(f"{what} outside the workspace ({outside[0]})")
        return _mutating(what)

    def _rate_rm(self, args: list[str]) -> Rating:
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
        return self._paths_rating([a for a in args if a != "--"], "deletes", destructive=True)

    def _rate_find(self, args: list[str], depth: int) -> Rating:
        if "-delete" in args:
            return _dangerous("find -delete removes files")
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
                inner_rating = self._rate_simple(inner, False, depth + 1) if inner else SAFE
                return inner_rating.worse(_mutating("find runs a command per file"))
        return SAFE

    def _rate_download(self, name: str, args: list[str]) -> Rating:
        upload = {"-d", "--data", "--data-binary", "--data-raw", "--data-urlencode", "-F",
                  "--form", "-T", "--upload-file", "--post-file", "--post-data", "--body-file"}
        if any("$(" in a or "`" in a for a in args):
            return _dangerous(f"{name} sends data computed by another command")
        out_flags = {"-o", "--output", "-O", "--output-document", "-P", "--directory-prefix"}
        for i, a in enumerate(args):
            flag, _, val = a.partition("=")
            target = val if val else (args[i + 1] if i + 1 < len(args) else "")
            if flag in out_flags and self.outside(target):
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

    def _rate_git(self, args: list[str]) -> Rating:
        # skip global options such as -C <dir> / -c k=v
        rest = list(args)
        while rest and rest[0].startswith("-"):
            opt = rest.pop(0)
            if opt in ("-C", "-c", "--git-dir", "--work-tree") and rest:
                rest.pop(0)
        if not rest:
            return SAFE
        sub, sargs = rest[0], rest[1:]
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
            if "--" in sargs or "." in sargs or (sub == "restore" and sargs) or "-f" in sargs:
                return _dangerous(f"git {sub} discards working-tree changes")
            return _mutating(f"git {sub}")
        if sub == "branch":
            if any(a in ("-D", "-d", "--delete", "-M", "-m", "--move", "-f") for a in sargs):
                return _dangerous("git branch deletes or moves a branch")
            return SAFE
        if sub == "stash":
            if sargs[:1] in (["drop"], ["clear"]):
                return _dangerous("git stash drop/clear discards work")
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

    def _rate_interpreter(self, name: str, args: list[str], piped: bool, depth: int) -> Rating:
        if name in ("perl", "ruby") and any(
                a.startswith("-") and not a.startswith("--") and "i" in a for a in args):
            return self._paths_rating([a for a in args if not a.startswith("-")][1:],
                                      f"{name} -i edits files in place", destructive=False)
        if not args or args == ["-"]:
            if piped:
                return _dangerous(f"pipes input into {name}, which runs it as code")
            return _mutating(f"starts {name}")
        if args[0] in ("--version", "-V", "--help", "-h"):
            return SAFE
        if args[0] == "-m" and len(args) > 1:
            mod = args[1]
            if mod in ("pytest", "unittest", "doctest", "py_compile", "compileall", "tabnanny"):
                return SAFE
            if mod == "pip":
                return self._rate_named("pip", args[2:], piped, depth)
            return _mutating(f"runs the module {mod}")
        for flag in ("-c", "-e", "-E", "--eval", "-p"):
            if flag in args:
                code = args[args.index(flag) + 1] if args.index(flag) + 1 < len(args) else ""
                low = code.lower()
                if "$(" in code or "`" in code:
                    return _dangerous(f"inline {name} code is computed at run time")
                hit = next((f for f in CODE_RED_FLAGS if f in low), None)
                if hit:
                    return _dangerous(f"inline {name} code calls {hit.rstrip('(.')}")
                return _mutating(f"runs inline {name} code")
        return _mutating(f"runs {name} {args[0]}")


def _command_name(word: str) -> str:
    """``/bin/rm`` -> ``rm``; ``C:\\x\\RM.EXE`` -> ``rm``; ``\\rm`` -> ``rm``."""
    name = word.lstrip("\\")
    name = re.split(r"[\\/]", name)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _skip_options(args: list[str], takes_value: set[str]) -> list[str]:
    rest = list(args)
    while rest and rest[0].startswith("-"):
        opt = rest.pop(0)
        if opt in takes_value and rest:
            rest.pop(0)
    return rest


READ_TOOLS = {"read_file", "list_dir", "glob", "grep"}
WRITE_TOOLS = {"write_file", "edit"}


class ApprovalPolicy:
    """Maps a tool call to ALLOW / ASK / DENY.

    ``mode``:
      * ``default`` - read-only commands and tests run; anything else asks.
      * ``auto``    - mutating commands run too; dangerous ones still ask.
    ``allow`` is a list of command prefixes the user trusts (``--allow "npm test"``); a
    matching *mutating* command runs unasked. A dangerous command is never allowlisted.
    Headless runs have no human, so the agent's approver turns every ASK into a DENY.
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
