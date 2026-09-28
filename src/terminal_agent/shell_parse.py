"""Turn a shell command string into simple commands the classifier can rate.

Pure functions, no policy: pull out ``$( )`` / backtick / ``<( )`` bodies, tokenise with
``shlex`` (shell punctuation kept as separate tokens), and split the tokens into simple
commands at ``; && || | & ( )`` and newlines, separating redirections from words.
"""

from __future__ import annotations

import shlex
from itertools import pairwise
from typing import NamedTuple

from terminal_agent.policy_rules import OPERATORS, PLACEHOLDER, REDIRECTS, SUBST


class Unit(NamedTuple):
    """One simple command: its words, its redirections, and whether it is piped into."""

    words: list[str]
    redirects: list[tuple[str, str]]
    piped: bool


def substitutions(command: str) -> tuple[list[str], str]:
    """Split out ``$( )``, backtick and ``<( )``/``>( )`` bodies (outermost only).

    Returns the bodies and the command with each one replaced by ``PLACEHOLDER``.
    """
    bodies: list[str] = []
    out: list[str] = []
    i = 0
    while i < len(command):
        m = SUBST.match(command, i)
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


def tokenize(stripped: str) -> list[str]:
    """shlex tokens of a substitution-free command; raises ValueError on bad quoting."""
    flat = stripped.replace("\r", "").replace("\n", " ; ")
    lexer = shlex.shlex(flat, posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace = " \t"
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


MAX_BRACE_WORDS = 256


def _brace_alternatives(word: str) -> tuple[str, list[str], str] | None:
    """(prefix, alternatives, suffix) of the first comma brace group in ``word``, if any.

    ``${VAR}`` and ``{a..z}`` sequences are left alone: a sequence only yields letters or
    numbers, and a parameter expansion is not a brace expansion.
    """
    i = 0
    while i < len(word):
        if word[i] == "{" and (i == 0 or word[i - 1] != "$"):
            depth, j, cuts = 1, i + 1, []
            while j < len(word) and depth:
                ch = word[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                elif ch == "," and depth == 1:
                    cuts.append(j)
                j += 1
            if depth == 0 and cuts:
                bounds = [i, *cuts, j - 1]
                alts = [word[a + 1 : b] for a, b in pairwise(bounds)]
                return word[:i], alts, word[j:]
        i += 1
    return None


def expand_braces(words: list[str]) -> list[str] | None:
    """Bash brace expansion of each word (``{rm,-rf,..}`` -> ``rm -rf ..``).

    ``shlex`` has already removed quotes, so a quoted ``'{a,b}'`` is expanded too; callers
    rate both the original and the expanded words and keep the worse, which can only make a
    rating stricter. Returns None when the expansion exceeds ``MAX_BRACE_WORDS``.
    """
    out: list[str] = []
    for word in words:
        pending = [word]
        while pending:
            w = pending.pop(0)
            found = _brace_alternatives(w)
            if found is None:
                out.append(w)
            else:
                prefix, alts, suffix = found
                pending[:0] = [prefix + a + suffix for a in alts]
            if len(out) + len(pending) > MAX_BRACE_WORDS:
                return None
    return out


def split_units(tokens: list[str]) -> list[Unit]:
    """Ordered simple commands, each with its redirections pulled out of its words."""
    units: list[Unit] = []
    words: list[str] = []
    redirects: list[tuple[str, str]] = []
    piped = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in OPERATORS or "\n" in tok:
            units.append(Unit(words, redirects, piped))
            words, redirects = [], []
            piped = tok in ("|", "|&")
            i += 1
            continue
        if tok in REDIRECTS or (
            tok.isdigit() and i + 1 < len(tokens) and tokens[i + 1] in REDIRECTS
        ):
            if tok.isdigit():
                i += 1
                tok = tokens[i]
            target = tokens[i + 1] if i + 1 < len(tokens) else ""
            redirects.append((tok, target))
            i += 2
            continue
        words.append(tok)
        i += 1
    units.append(Unit(words, redirects, piped))
    return units
