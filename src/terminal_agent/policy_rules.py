"""Rule tables and pure helpers for the approval policy, grouped by domain.

Split out of ``policy.py`` (which keeps the ``CommandClassifier`` and ``ApprovalPolicy``
logic) so the git / package-manager / path / wrapper rules live in one named place. Every
name here is data or a small pure helper; none of it has behaviour, and the classifier's
output is byte-for-byte identical to before the split (verified over 843 commands).
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
    "ls", "dir", "pwd", "cat", "head", "tail", "less", "more", "wc", "grep", "egrep", "fgrep",
    "rg", "ag", "echo", "printf", "which", "where", "type", "file", "stat", "du", "df",
    "basename", "dirname", "realpath", "readlink", "whoami", "uname", "true", "false", "test",
    "[", "jq", "cmp", "diff", "hexdump", "xxd", "od", "strings", "printenv", "env",
    "sha1sum", "sha256sum", "md5sum", "cksum", "yes", "seq", "sleep", "ps", "id", "groups",
    "tty", "hostname", "arch", "nproc", "getconf", "locale", "cal",
    "cd", "pushd", "popd", "dirs", "date",
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
TEXTUTIL_READONLY = {"cut", "paste", "nl", "fold", "fmt", "expand", "unexpand", "column",
                     "comm", "join", "rev", "tac", "tr", "sort", "uniq", "tree", "shuf", "look"}
TEST_RUNNERS = {"pytest", "py.test", "runtests.py"}
# (command, first positional) pairs that only run tests or linters (read-only) -
# unless a fix/exec option is present (checked below)
TEST_VERBS = {("npm", "test"), ("cargo", "test"), ("cargo", "check"), ("cargo", "clippy"),
              ("go", "test"), ("go", "vet"), ("ruff", "check"), ("tox", "-l"), ("pnpm", "test"),
              ("yarn", "test")}
RUFF_WRITE = {"--fix", "--fix-only", "--unsafe-fixes"}
# ── always-dangerous commands, infrastructure and package managers ───────────────────
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
    "history": "edits shell history", "net": "manages Windows users and shares",
    "npx": "downloads and runs a package", "uvx": "downloads and runs a package",
    "bunx": "downloads and runs a package",
}
INFRA = {"kubectl", "terraform", "aws", "gcloud", "az", "helm", "pulumi", "doctl", "gsutil"}
INFRA_DESTRUCTIVE = {"delete", "destroy", "rb", "rm", "uninstall", "terminate", "drain",
                     "apply", "scale", "remove", "purge", "down", "replace", "patch"}
PACKAGE_MANAGERS = {"pip", "pip3", "apt", "apt-get", "yum", "dnf", "brew", "conda", "choco",
                    "winget", "npm", "pnpm", "yarn", "gem", "cargo", "mamba", "pipx", "go",
                    "poetry", "pdm", "hatch", "rye", "nix-env"}
INSTALL_VERBS = {"install", "uninstall", "remove", "add", "i", "update", "upgrade", "reinstall",
                 "get", "exec", "dlx", "ci", "sync", "lock", "rm"}
# ── shells, interpreters, wrappers and editors ───────────────────────────────────────
SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "fish", "cmd", "pwsh", "powershell"}
INTERPRETERS = {"python", "python3", "python2", "perl", "ruby", "node", "php", "deno", "bun"}
# wrapper -> options that take a value (so the wrapped command starts after them)
WRAPPERS = {
    "command": set(), "builtin": set(), "exec": {"-a"}, "nohup": set(),
    "nice": {"-n", "--adjustment"}, "ionice": {"-c", "--class", "-n", "--classdata", "-p"},
    "time": {"-o", "--output", "-f", "--format"}, "stdbuf": {"-i", "-o", "-e"},
    "chrt": {"-p"}, "setsid": set(), "chronic": set(), "setarch": set(), "busybox": set(),
    "flock": {"-w", "--timeout", "-E", "--conflict-exit-code"},
}
EDITORS = {"vim", "vi", "nvim", "view", "emacs", "emacsclient", "ex", "nano", "pico", "code",
           "gedit", "kak", "hx", "micro"}
WRITE_WRAPPER_OPT = {"-o", "--output"}  # time -o FILE writes; check the value
KEYWORDS = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "case", "esac",
            "{", "}", "!", "[[", "]]", "function", "select"}
# ── git ─────────────────────────────────────────────────────────────────────────────
GIT_READ = {"status", "diff", "log", "show", "blame", "rev-parse", "ls-files", "grep",
            "describe", "shortlog", "reflog", "cat-file", "ls-tree", "whatchanged", "version"}
GIT_EXEC_OPT = {"-O", "--open-files-in-pager", "--ext-diff"}  # run a pager / external diff
GIT_WRITE_OPT = {"-o", "--output"}
# ── inline-code red flags and executed environment variables ─────────────────────────
CODE_RED_FLAGS = ("rmtree", "remove(", "unlink", "rmsync", "rmdir", "system(", "subprocess",
                  "popen", "exec(", "spawn", "child_process", "os.kill", "chmod", "truncate",
                  "urlopen", "requests.", "socket", "shutil", "pathlib", "os.remove",
                  "getattr(", "__import__", "importlib", "eval(", "compile(", "os.environ",
                  "ctypes", "marshal", "pickle")
# environment variables whose value NAMES a program that gets executed
ENV_EXEC = {"PAGER", "GIT_PAGER", "MANPAGER", "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS", "LESSOPEN",
            "LESSCLOSE", "GIT_EDITOR", "EDITOR", "VISUAL", "GIT_SEQUENCE_EDITOR", "GIT_SSH",
            "GIT_SSH_COMMAND", "GIT_ASKPASS", "SSH_ASKPASS", "BROWSER", "GIT_PROXY_COMMAND",
            "SHELL", "BASH_ENV", "ENV", "PROMPT_COMMAND", "PS1", "FIGNORE", "GIT_CONFIG",
            "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_TEMPLATE_DIR", "GIT_ATTR_SOURCE",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_INDEX_FILE"}
ENV_HIJACK = re.compile(r"^(PATH|LD_PRELOAD|LD_LIBRARY_PATH|PYTHONPATH|PYTHONSTARTUP|"
                        r"BASH_ENV|ENV|PROMPT_COMMAND|NODE_OPTIONS|DYLD_INSERT_LIBRARIES)=")
# ── shell operators, redirects, substitutions ────────────────────────────────────────
OPERATORS = {";", "&&", "||", "|", "&", "(", ")", ";;", "|&", "\n"}
REDIRECTS = {">", ">>", ">|", "&>", "&>>", ">&", "<", "<<", "<<<", "<>", "<&"}
_SUBST = re.compile(r"\$\(|`|<\(|>\(")
PLACEHOLDER = "__SUBSTITUTION__"
_EXEC_PROCESS_SUBST = re.compile(
    r"(^|[;&|\s])(bash|sh|zsh|dash|ksh|python3?|perl|ruby|node|source|\.)\s+<\(")
_HEREDOC_TO_INTERPRETER = re.compile(r"(^|[;&|\s])(python3?|perl|ruby|node|php)\s+(-\s+)?<<")
DELETERS = {"rm", "unlink", "shred", "rmdir", "del", "erase", "remove-item"}
# ── pure helpers ────────────────────────────────────────────────────────────────────
def _dangerous(reason: str) -> Rating:
    return Rating("dangerous", reason)
def _mutating(reason: str) -> Rating:
    return Rating("mutating", reason)
def _writes_into_git(target: str) -> bool:
    parts = target.replace("\\", "/").split("/")
    return ".git" in parts
def _opt(token: str) -> str:
    """The option name of ``--out=x`` / ``-oL`` / ``-o`` (best effort)."""
    base = token.split("=", 1)[0]
    if base.startswith("--"):
        return base
    if len(base) > 2 and base[1] != "-":  # attached short value, e.g. -n5 / -oL
        return base[:2]
    return base
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
