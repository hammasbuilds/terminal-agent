"""Turn a shell command string into simple commands the classifier can rate.

Pure functions, no policy: pull out ``$( )`` / backtick / ``<( )`` bodies, tokenise with
``shlex`` (shell punctuation kept as separate tokens), and split the tokens into simple
commands at ``; && || | & ( )`` and newlines, separating redirections from words.
"""

from __future__ import annotations

import shlex
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
