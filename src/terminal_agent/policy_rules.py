"""Shared vocabulary of the command classifier: the ``Rating`` type, the rule tables, and a
few pure token helpers.

The tables are data (which commands are read-only, which environment variables name a
program that gets executed, which git subcommands only read, ...). The helpers
(``opt``, ``opt_value``, ``command_name``, ``skip_options``, ``writes_into_git``) are small
pure functions over tokens with no knowledge of the workspace. The decisions that use them
live in ``classifier.py`` and the ``rules_*`` modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Risk = Literal["safe", "mutating", "dangerous"]
_ORDER = {"safe": 0, "mutating": 1, "dangerous": 2}
MAX_DEPTH = 6
# cwd sentinel: the shell's working directory is not statically known
UNKNOWN_CWD = "\x00unknown"


@dataclass
class Rating:
    risk: Risk
    reason: str

    def worse(self, other: Rating) -> Rating:
        return other if _ORDER[other.risk] > _ORDER[self.risk] else self


SAFE = Rating("safe", "read-only")

# ── read-only commands and text utilities ────────────────────────────────────────────
READ_ONLY = {
    "ls",
    "dir",
    "pwd",
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ag",
    "echo",
    "printf",
    "which",
    "where",
    "type",
    "file",
    "stat",
    "du",
    "df",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "whoami",
    "uname",
    "true",
    "false",
    "test",
    "[",
    "jq",
    "cmp",
    "diff",
    "hexdump",
    "od",
    "strings",
    "printenv",
    "env",
    "sha1sum",
    "sha256sum",
    "md5sum",
    "cksum",
    "yes",
    "seq",
    "sleep",
    "ps",
    "id",
    "groups",
    "tty",
    "arch",
    "nproc",
    "getconf",
    "locale",
    "cal",
    "cd",
    "pushd",
    "popd",
    "dirs",
}
# text utilities that read stdin/files but can also *write* a named output file
TEXTUTIL_WRITE_OPT = {
    "sort": {"-o", "--output"},
    "tree": {"-o", "--output"},
    "gzip": {"-o"},
    "shuf": {"-o", "--output"},
    "csplit": {"-f", "--prefix", "-b", "--suffix-format"},
}
# uniq / split write their *positional* output; comm/join/cut/paste/nl/fold/fmt do not
TEXTUTIL_READONLY = {
    "cut",
    "paste",
    "nl",
    "fold",
    "fmt",
    "expand",
    "unexpand",
    "column",
    "comm",
    "join",
    "rev",
    "tac",
    "tr",
    "sort",
    "uniq",
    "tree",
    "shuf",
    "look",
}
TEST_RUNNERS = {"pytest", "py.test", "runtests.py"}
# (command, first positional) pairs that only run tests or linters (read-only) -
# unless a fix/exec option is present (checked below)
TEST_VERBS = {
    ("npm", "test"),
    ("cargo", "test"),
    ("cargo", "check"),
    ("cargo", "clippy"),
    ("go", "test"),
    ("go", "vet"),
    ("ruff", "check"),
    ("tox", "-l"),
    ("pnpm", "test"),
    ("yarn", "test"),
}
RUFF_WRITE = {"--fix", "--fix-only", "--unsafe-fixes"}
# ── always-dangerous commands, infrastructure and package managers ───────────────────
ALWAYS_DANGEROUS = {
    "dd": "raw disk/file writes",
    "fdisk": "partitions disks",
    "wipefs": "wipes disks",
    "shred": "destroys files",
    "format": "formats a disk",
    "mount": "remounts filesystems",
    "umount": "unmounts filesystems",
    "kill": "kills processes",
    "pkill": "kills processes",
    "killall": "kills processes",
    "taskkill": "kills processes",
    "shutdown": "stops the machine",
    "reboot": "reboots the machine",
    "halt": "stops the machine",
    "poweroff": "stops the machine",
    "systemctl": "controls system services",
    "service": "controls system services",
    "su": "switches user",
    "sudo": "runs as root",
    "doas": "runs as root",
    "runas": "runs as root",
    "crontab": "edits scheduled jobs",
    "chown": "changes ownership",
    "reg": "edits the registry",
    "setx": "persists environment variables",
    "wsl": "controls WSL",
    "nc": "raw network socket",
    "ncat": "raw network socket",
    "netcat": "raw network socket",
    "telnet": "remote shell",
    "scp": "copies files to another host",
    "rsync": "copies files to another host",
    "ssh": "remote shell",
    "sftp": "remote file transfer",
    "ftp": "remote file transfer",
    "unlink": "deletes a file",
    "truncate": "empties files",
    "alias": "redefines commands",
    "eval": "runs a string as code",
    "source": "runs a file as shell code",
    ".": "runs a file as shell code",
    "iex": "runs a string as code",
    "invoke-expression": "runs a string as code",
    "irm": "downloads for execution",
    "remove-item": "deletes files",
    "rd": "deletes directories",
    "blkdiscard": "discards a disk's blocks",
    "vssadmin": "deletes shadow copies",
    "diskpart": "partitions disks",
    "bcdedit": "edits boot configuration",
    "schtasks": "edits scheduled tasks",
    "at": "schedules jobs",
    "launchctl": "edits services",
    "sc": "controls Windows services",
    "ssh-keygen": "creates or overwrites keys",
    "history": "edits shell history",
    "net": "manages Windows users and shares",
    "npx": "downloads and runs a package",
    "uvx": "downloads and runs a package",
    "bunx": "downloads and runs a package",
    # Windows living-off-the-land binaries and PowerShell cmdlets that download, run or
    # change permissions (the review found them rated merely 'unknown')
    "certutil": "downloads or decodes files",
    "bitsadmin": "downloads files",
    "mshta": "runs a remote script",
    "rundll32": "runs code from a DLL",
    "regsvr32": "runs code from a DLL",
    "wmic": "starts processes / changes the system",
    "icacls": "changes file permissions",
    "takeown": "takes ownership of files",
    "cipher": "wipes free space",
    "invoke-webrequest": "downloads files",
    "iwr": "downloads files",
    "start-process": "starts a process",
    "set-content": "writes files",
    "out-file": "writes files",
    # database clients act on a server, not on the workspace
    "psql": "runs SQL against a database server",
    "mysql": "runs SQL against a database server",
    "redis-cli": "runs commands against a Redis server",
    "mongosh": "runs commands against a database server",
    "mongo": "runs commands against a database server",
}
# container engines: only their read-only subcommands are safe
CONTAINER_ENGINES = {"docker", "podman", "nerdctl"}
INFRA = {"kubectl", "terraform", "aws", "gcloud", "az", "helm", "pulumi", "doctl", "gsutil"}
INFRA_DESTRUCTIVE = {
    "delete",
    "destroy",
    "rb",
    "rm",
    "uninstall",
    "terminate",
    "drain",
    "apply",
    "scale",
    "remove",
    "purge",
    "down",
    "replace",
    "patch",
}
PACKAGE_MANAGERS = {
    "pip",
    "pip3",
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "brew",
    "conda",
    "choco",
    "winget",
    "npm",
    "pnpm",
    "yarn",
    "gem",
    "cargo",
    "mamba",
    "pipx",
    "go",
    "poetry",
    "pdm",
    "hatch",
    "rye",
    "nix-env",
}
INSTALL_VERBS = {
    "install",
    "uninstall",
    "remove",
    "add",
    "i",
    "update",
    "upgrade",
    "reinstall",
    "get",
    "exec",
    "dlx",
    "ci",
    "sync",
    "lock",
    "rm",
}
# ── shells, interpreters, wrappers and editors ───────────────────────────────────────
SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "fish", "cmd", "pwsh", "powershell"}
INTERPRETERS = {"python", "python3", "python2", "perl", "ruby", "node", "php", "deno", "bun"}
# wrapper -> options that take a value (so the wrapped command starts after them)
WRAPPERS = {
    "command": set(),
    "builtin": set(),
    "exec": {"-a"},
    "nohup": set(),
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "--class", "-n", "--classdata", "-p"},
    "time": {"-o", "--output", "-f", "--format"},
    "stdbuf": {"-i", "-o", "-e"},
    "chrt": {"-p"},
    "setsid": set(),
    "chronic": set(),
    "setarch": set(),
    "busybox": set(),
    "flock": {"-w", "--timeout", "-E", "--conflict-exit-code"},
}
EDITORS = {
    "vim",
    "vi",
    "nvim",
    "view",
    "emacs",
    "emacsclient",
    "ex",
    "nano",
    "pico",
    "code",
    "gedit",
    "kak",
    "hx",
    "micro",
}
WRITE_WRAPPER_OPT = {"-o", "--output"}  # time -o FILE writes; check the value
KEYWORDS = {
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "do",
    "done",
    "while",
    "until",
    "case",
    "esac",
    "{",
    "}",
    "!",
    "[[",
    "]]",
    "function",
    "select",
}
# ── git ─────────────────────────────────────────────────────────────────────────────
GIT_READ = {
    "status",
    "diff",
    "log",
    "show",
    "blame",
    "rev-parse",
    "ls-files",
    "grep",
    "describe",
    "shortlog",
    "reflog",
    "cat-file",
    "ls-tree",
    "whatchanged",
    "version",
}
GIT_EXEC_OPT = {  # run a pager, an external diff or a tool
    "-O",
    "--open-files-in-pager",
    "--ext-diff",
    "--extcmd",
    "--tool",
}
# global options that point git at another repository, work tree or program directory
GIT_REDIRECT_OPT = {"--git-dir", "--work-tree", "--exec-path"}
# git config keys an agent may set in auto mode; every other key could name a program
# (core.fsmonitor, alias.x = !cmd, filter.*.clean, diff.*.textconv, credential.helper, ...)
GIT_CONFIG_SAFE_KEYS = (
    "user.name",
    "user.email",
    "core.autocrlf",
    "core.filemode",
    "core.ignorecase",
    "core.eol",
    "core.safecrlf",
    "init.defaultbranch",
    "pull.rebase",
    "pull.ff",
    "merge.ff",
    "color.",
    "advice.",
    "commit.gpgsign",
)
GIT_CONFIG_READ_FLAGS = {
    "--get",
    "--get-all",
    "--get-regexp",
    "--get-urlmatch",
    "--list",
    "-l",
    "--show-origin",
    "--show-scope",
}
GIT_WRITE_OPT = {"-o", "--output"}
# ── inline-code red flags and executed environment variables ─────────────────────────
CODE_RED_FLAGS = (
    "rmtree",
    "remove(",
    "unlink",
    "rmsync",
    "rmdir",
    "system",
    "from os",
    "import os as",
    "from subprocess",
    "from shutil",
    "from pty",
    "pty.",
    "__builtins__",
    "builtins",
    "globals(",
    "setattr(",
    "write_text",
    "write_bytes",
    "file.delete",
    "fileutils",
    "fs.rm",
    "fs.write",
    "writefile",
    "subprocess",
    "popen",
    "exec(",
    "spawn",
    "child_process",
    "os.kill",
    "chmod",
    "truncate",
    "urlopen",
    "requests.",
    "socket",
    "shutil",
    "pathlib",
    "os.remove",
    "getattr(",
    "__import__",
    "importlib",
    "eval(",
    "compile(",
    "os.environ",
    "ctypes",
    "marshal",
    "pickle",
)
# environment variables whose value NAMES a program that gets executed
ENV_EXEC = {
    "PAGER",
    "GIT_PAGER",
    "MANPAGER",
    "GIT_EXTERNAL_DIFF",
    "GIT_DIFF_OPTS",
    "LESSOPEN",
    "LESSCLOSE",
    "GIT_EDITOR",
    "EDITOR",
    "VISUAL",
    "GIT_SEQUENCE_EDITOR",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_ASKPASS",
    "SSH_ASKPASS",
    "BROWSER",
    "GIT_PROXY_COMMAND",
    "SHELL",
    "BASH_ENV",
    "ENV",
    "PROMPT_COMMAND",
    "PS1",
    "FIGNORE",
    "GIT_CONFIG",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_TEMPLATE_DIR",
    "GIT_ATTR_SOURCE",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_INDEX_FILE",
    # point git at another repository (whose config can run a program) or program dir
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_EXEC_PATH",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
}
# GIT_CONFIG_KEY_<n>=core.fsmonitor / GIT_CONFIG_VALUE_<n>=cmd inject config the same way
ENV_EXEC_PATTERN = re.compile(r"^GIT_CONFIG_(KEY|VALUE)_\d+$")
# shell builtins that assign (and may export) variables, like `export`
DECLARERS = {"export", "declare", "typeset", "readonly", "local"}
ENV_HIJACK = re.compile(
    r"^(PATH|LD_PRELOAD|LD_LIBRARY_PATH|PYTHONPATH|PYTHONSTARTUP|"
    r"BASH_ENV|ENV|PROMPT_COMMAND|NODE_OPTIONS|DYLD_INSERT_LIBRARIES)="
)
# ── shell operators, redirects, substitutions ────────────────────────────────────────
OPERATORS = {";", "&&", "||", "|", "&", "(", ")", ";;", "|&", "\n"}
REDIRECTS = {">", ">>", ">|", "&>", "&>>", ">&", "<", "<<", "<<<", "<>", "<&"}
SUBST = re.compile(r"\$\(|`|<\(|>\(")
PLACEHOLDER = "__SUBSTITUTION__"
EXEC_PROCESS_SUBST = re.compile(
    r"(^|[;&|\s])(bash|sh|zsh|dash|ksh|python3?|perl|ruby|node|source|\.)\s+<\("
)
HEREDOC_TO_INTERPRETER = re.compile(r"(^|[;&|\s])(python3?|perl|ruby|node|php)\s+(-\s+)?<<")
DELETERS = {"rm", "unlink", "shred", "rmdir", "del", "erase", "remove-item"}


# ── pure helpers ────────────────────────────────────────────────────────────────────
def dangerous(reason: str) -> Rating:
    return Rating("dangerous", reason)


def mutating(reason: str) -> Rating:
    return Rating("mutating", reason)


def writes_into_git(target: str) -> bool:
    """True for a write inside ``.git`` or one that makes a directory look like a git dir.

    git treats any directory holding ``HEAD``, ``objects/`` and ``refs/`` as a repository
    (``git --git-dir=d``, ``cd d && git status``), and runs the programs its ``config``
    names. Writing a file called ``HEAD`` is the one step no ordinary edit needs.
    """
    parts = target.replace("\\", "/").rstrip("/").split("/")
    return ".git" in parts or parts[-1].lower() == "head"


def opt_value(args: list[str], i: int) -> tuple[str, int]:
    """The value of the option at ``args[i]`` and how many tokens option+value take.

    ``--out=x`` and ``-o=x`` -> (x, 1); ``-ox`` (short option, value attached) -> (x, 1);
    ``--out x`` / ``-o x`` -> (x, 2). A missing value is "".
    """
    tok = args[i]
    nxt = args[i + 1] if i + 1 < len(args) else ""
    if tok.startswith("--"):
        return (tok.split("=", 1)[1], 1) if "=" in tok else (nxt, 2)
    if len(tok) > 2:
        attached = tok[2:]
        return (attached[1:] if attached.startswith("=") else attached), 1
    return nxt, 2


def option_values(args: list[str], names: set[str]) -> list[tuple[str, str]]:
    """(option, value) for every occurrence of an option in ``names``, spelled exactly.

    Unlike :func:`opt` this does not shorten ``-coverprofile`` to ``-c``: single-dash long
    options (go's ``-coverprofile=f``) are matched whole, and a one-letter name also
    matches its attached form (``-o../x``).
    """
    out = []
    for i, tok in enumerate(args):
        head = tok.split("=", 1)[0]
        if head in names:
            if "=" in tok:
                out.append((head, tok.split("=", 1)[1]))
            else:
                out.append((head, args[i + 1] if i + 1 < len(args) else ""))
        elif len(tok) > 2 and tok[0] == "-" and tok[1] != "-" and tok[:2] in names:
            out.append((tok[:2], opt_value(args, i)[0]))
    return out


def opt(token: str) -> str:
    """The option name of ``--out=x`` / ``-oL`` / ``-o`` (best effort)."""
    base = token.split("=", 1)[0]
    if base.startswith("--"):
        return base
    if len(base) > 2 and base[1] != "-":  # attached short value, e.g. -n5 / -oL
        return base[:2]
    return base


def command_name(word: str) -> str:
    """``/bin/rm`` -> ``rm``; ``C:\\x\\RM.EXE`` -> ``rm``; ``\\rm`` -> ``rm``."""
    name = word.lstrip("\\")
    name = re.split(r"[\\/]", name)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def skip_options(args: list[str], takes_value: set[str]) -> list[str]:
    rest = list(args)
    while rest and rest[0].startswith("-"):
        tok = rest.pop(0)
        if tok in takes_value and rest:
            rest.pop(0)
    return rest
