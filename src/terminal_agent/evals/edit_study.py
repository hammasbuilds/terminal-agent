"""How often does an exact-match edit fail on real patches, and what would fuzzy matching buy?

Every gold-patch hunk is replayed against the real pre-image file (hunks of one file in
order, each against the file as earlier hunks left it). For each hunk:

* ``git3`` - old_string = the hunk as git wrote it (3 lines of context): unique / ambiguous;
* ``core`` - old_string = only the removed lines, the minimal edit a model might emit;
* ``k_min`` - the fewest lines of context each side that make the change unique;
* ``perturbations`` - the git3 snippet with a copying slip a model plausibly makes
  (trailing whitespace dropped, tabs expanded, first-line indent lost, snippet dedented),
  applied with the exact tool and with the opt-in fuzzy tool. A fuzzy "success" is only
  counted as correct if the file equals the gold result up to that same whitespace slip.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from terminal_agent.edits import apply_edit
from terminal_agent.evals.patches import Hunk, parse_patch
from terminal_agent.evals.stats import cluster_rate
from terminal_agent.evals.tasks import Task

MAX_K = 60


def _map_lines(text: str, fn: Callable[[str], str]) -> str:
    return "".join(fn(line[:-1]) + "\n" if line.endswith("\n") else fn(line)
                   for line in text.splitlines(keepends=True))


def _common_indent(text: str) -> str:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    indents = [ln[: len(ln) - len(ln.lstrip(" \t"))] for ln in lines]
    prefix = indents[0]
    for ind in indents[1:]:
        while not ind.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _dedent_pair(old: str, new: str) -> tuple[str, str]:
    ind = _common_indent(old)
    if not ind:
        return old, new
    strip = lambda s: _map_lines(s, lambda ln: ln[len(ind):] if ln.startswith(ind) else ln)  # noqa: E731
    return strip(old), strip(new)


def _first_line_lstrip(old: str, new: str) -> tuple[str, str]:
    head, sep, rest = old.partition("\n")
    nhead, nsep, nrest = new.partition("\n")
    lost = head[: len(head) - len(head.lstrip(" \t"))]
    new2 = (nhead[len(lost):] if nhead.startswith(lost) else nhead) + nsep + nrest
    return head.lstrip(" \t") + sep + rest, new2


PERTURBATIONS: dict[str, Callable[[str, str], tuple[str, str]]] = {
    "trailing_ws_dropped": lambda o, n: (_map_lines(o, str.rstrip), _map_lines(n, str.rstrip)),
    "tabs_expanded": lambda o, n: (o.expandtabs(4), n.expandtabs(4)),
    "first_line_unindented": _first_line_lstrip,
    "snippet_dedented": _dedent_pair,
}


def _normalize(text: str) -> list[str]:
    return [ln.expandtabs(4).rstrip() for ln in text.replace("\r\n", "\n").split("\n")]


def _apply_at(lines: list[str], hunk: Hunk, delta: int) -> tuple[list[str], bool]:
    """Apply a hunk positionally (what git does); report whether its old lines matched."""
    old = [ln[1:] for ln in hunk.lines if ln[:1] in (" ", "-")]
    new = [ln[1:] for ln in hunk.lines if ln[:1] in (" ", "+")]
    start = hunk.old_start - 1 + delta if hunk.old_len else hunk.old_start + delta
    matched = lines[start : start + len(old)] == old
    return lines[:start] + new + lines[start + len(old) :], matched


def _k_min(content: str, lines: list[str], hunk: Hunk, delta: int) -> int | None:
    lead = hunk.leading_context
    core_old, _ = hunk.core()
    n_core = core_old.count("\n")
    start = hunk.old_start - 1 + delta + lead if hunk.old_len else hunk.old_start + delta
    for k in range(0 if n_core else 1, MAX_K + 1):
        a, b = max(0, start - k), min(len(lines), start + n_core + k)
        snippet = "\n".join(lines[a:b])
        if snippet and content.count(snippet) == 1:
            return k
    return None


def study_task(task: Task, preimages: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for fp in parse_patch(task.patch):
        if fp.is_new or fp.is_deleted or fp.is_binary or fp.path not in preimages:
            continue
        raw = preimages[fp.path]
        crlf = "\r\n" in raw
        content = raw.replace("\r\n", "\n")
        lines = content.split("\n")
        delta = 0
        for idx, hunk in enumerate(fp.hunks):
            row: dict[str, Any] = {"instance_id": task.instance_id, "repo": task.repo,
                                   "path": fp.path, "hunk": idx, "crlf_file": crlf,
                                   "removed": hunk.removed, "added": hunk.added}
            old, new = hunk.old_text, hunk.new_text
            row["git3"] = apply_edit(content, old, new).status
            core_old, core_new = hunk.core()
            row["core"] = apply_edit(content, core_old, core_new).status if core_old \
                else "no_anchor"
            row["k_min"] = _k_min(content, lines, hunk, delta)
            new_lines, matched = _apply_at(lines, hunk, delta)
            row["positional_match"] = matched
            expected = "\n".join(new_lines)
            perturbed: dict[str, Any] = {}
            for name, fn in PERTURBATIONS.items():
                p_old, p_new = fn(old, new)
                if p_old == old:
                    continue
                exact = apply_edit(content, p_old, p_new)
                fuzzy = apply_edit(content, p_old, p_new, fuzzy=True)
                if fuzzy.ok:
                    assert fuzzy.content is not None
                    fz = ("correct" if _normalize(fuzzy.content) == _normalize(expected)
                          else "wrong_result")
                else:
                    fz = fuzzy.status
                perturbed[name] = {"exact": exact.status, "fuzzy": fz,
                                   "fuzzy_strategy": fuzzy.strategy}
            row["perturbations"] = perturbed
            rows.append(row)
            lines = new_lines
            content = expected
            delta += hunk.added - hunk.removed
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["instance_id"], []).append(r)

    def crate(pred: Callable[[dict[str, Any]], bool],
              where: Callable[[dict[str, Any]], bool] = lambda r: True) -> dict[str, object]:
        groups = [(sum(1 for r in rs if where(r) and pred(r)), sum(1 for r in rs if where(r)))
                  for rs in by_task.values()]
        return cluster_rate([g for g in groups if g[1]])

    with_removal = lambda r: r["removed"] > 0  # noqa: E731
    k_values = [r["k_min"] for r in rows if r["k_min"] is not None]
    k_hist: dict[str, int] = {}
    for k in k_values:
        key = str(k) if k <= 5 else "6+"
        k_hist[key] = k_hist.get(key, 0) + 1
    pert: dict[str, Any] = {}
    for name in PERTURBATIONS:
        affected = [r for r in rows if name in r["perturbations"]]
        cells = [r["perturbations"][name] for r in affected]
        pert[name] = {
            "hunks_affected": crate(lambda r, n=name: n in r["perturbations"]),
            "exact_not_found": sum(1 for c in cells if c["exact"] == "not_found"),
            "exact_ok": sum(1 for c in cells if c["exact"] == "ok"),
            "fuzzy_correct": sum(1 for c in cells if c["fuzzy"] == "correct"),
            "fuzzy_wrong_result": sum(1 for c in cells if c["fuzzy"] == "wrong_result"),
            "fuzzy_ambiguous": sum(1 for c in cells if c["fuzzy"] == "ambiguous"),
            "fuzzy_not_found": sum(1 for c in cells if c["fuzzy"] == "not_found"),
            "n": len(cells),
        }
    return {
        "tasks": len(by_task),
        "hunks": len(rows),
        "files_crlf": len({(r["instance_id"], r["path"]) for r in rows if r["crlf_file"]}),
        "positional_mismatch": sum(1 for r in rows if not r["positional_match"]),
        "git3_unique": crate(lambda r: r["git3"] == "ok"),
        "git3_ambiguous": crate(lambda r: r["git3"] == "ambiguous"),
        "git3_not_found": crate(lambda r: r["git3"] == "not_found"),
        "core_ambiguous_given_removal": crate(lambda r: r["core"] == "ambiguous", with_removal),
        "core_unique_given_removal": crate(lambda r: r["core"] == "ok", with_removal),
        "pure_insertions": sum(1 for r in rows if not r["removed"]),
        "k_min_histogram": dict(sorted(k_hist.items())),
        "k_min_over_3": crate(lambda r: r["k_min"] is not None and r["k_min"] > 3),
        "k_min_unresolved": sum(1 for r in rows if r["k_min"] is None),
        "perturbations": pert,
    }


def load_preimages(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for line in fh:
            r = json.loads(line)
            out.setdefault(r["instance_id"], {})[r["path"]] = r["content"]
    return out


def save_preimages(path: Path, data: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        for iid in sorted(data):
            for p in sorted(data[iid]):
                # ensure_ascii keeps surrogate-escaped (non-UTF-8) bytes representable
                fh.write(json.dumps({"instance_id": iid, "path": p,
                                     "content": data[iid][p]}) + "\n")
