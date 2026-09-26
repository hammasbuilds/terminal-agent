"""Does the agent still see which test failed after its tool output is truncated?

Input: real failing test logs (the baseline runs from harness validation, or the local
suite). Each log is cut the way ``run_shell``/``run_tests`` would cut it - ``head``,
``tail`` or ``head_tail`` at several character budgets - and then parsed with the repo's
own log parser. A FAIL_TO_PASS test counts as *visible* if the truncated text still
reports it as failed or errored.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from terminal_agent.context import TruncateMode, truncate
from terminal_agent.evals.stats import cluster_rate

BUDGETS = (1000, 2000, 4000, 8000, 16000, 32000)
MODES: tuple[TruncateMode, ...] = ("head", "tail", "head_tail")
FAILING = ("FAILED", "ERROR")


def visible_failures(text: str, parser: Callable[[str], dict[str, str]],
                     tests: list[str]) -> int:
    statuses = parser(text)
    return sum(1 for t in tests if statuses.get(t) in FAILING)


def study_log(log: str, parser: Callable[[str], dict[str, str]], f2p: list[str]
              ) -> dict[str, Any] | None:
    """Per budget and mode, how many of the F2P tests the full log shows failing stay visible."""
    shown = [t for t in f2p if parser(log).get(t) in FAILING]
    if not shown:
        return None
    cells: dict[str, dict[str, int]] = {}
    for mode in MODES:
        for budget in BUDGETS:
            cut, _ = truncate(log, budget, mode)
            cells[f"{mode}@{budget}"] = {"visible": visible_failures(cut, parser, shown),
                                         "total": len(shown)}
    return {"chars": len(log), "f2p_failing_in_full_log": len(shown), "cells": cells}


def aggregate(per_log: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"logs": len(per_log)}
    lengths = sorted(r["chars"] for r in per_log.values())
    if lengths:
        out["log_chars"] = {"median": lengths[len(lengths) // 2], "max": lengths[-1],
                            "over_8000": sum(1 for n in lengths if n > 8000)}
    table: dict[str, Any] = {}
    for mode in MODES:
        for budget in BUDGETS:
            key = f"{mode}@{budget}"
            groups = [(r["cells"][key]["visible"], r["cells"][key]["total"])
                      for r in per_log.values()]
            table[key] = cluster_rate(groups)
    out["visible_rate"] = table
    return out
