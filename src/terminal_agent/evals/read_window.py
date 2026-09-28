"""How many ``read_file`` calls it takes to reach SWE-bench Lite's edit sites.

``read_file`` returns a window capped by characters (``ToolConfig.read_max_chars``), not a
fixed number of lines, so whether an edit site is in the first window depends on how long
every line before it is. This study replays the tool's own window rule
(:func:`terminal_agent.tools.window_size`) over the real line lengths of every pre-image
file the gold patches edit (``data/lite_line_lengths.json.gz``), paging from line 1 the way
the tool's header tells the model to.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import statistics
from pathlib import Path
from typing import Any

from terminal_agent.evals.patches import parse_patch
from terminal_agent.evals.stats import rate
from terminal_agent.evals.tasks import Task
from terminal_agent.tools import ToolConfig, window_size

CAPS = (3000, 6000, 12000, 24000)  # chars per read; 6000 is the tool's default


def load_line_lengths(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return json.loads(gzip.decompress(path.read_bytes()))


def reads_to_reach(lengths: list[int], line: int, config: ToolConfig) -> int:
    """Read calls, paging from line 1, until ``line`` (1-based) has been shown."""
    offset, reads = 1, 0
    while offset <= min(line, len(lengths)):
        reads += 1
        candidates = lengths[offset - 1 : offset - 1 + config.read_max_lines]
        offset += max(1, window_size(candidates, config))
    return max(reads, 1)


def task_reads(task: Task, files: dict[str, dict[str, Any]], config: ToolConfig) -> int | None:
    """Worst file of the task: reads needed to reach its last edited line (None if unknown)."""
    worst = 0
    for f in parse_patch(task.patch):
        if f.is_new:
            continue
        entry = files.get(f.path)
        if not entry or entry.get("missing"):
            return None
        last = max(h.old_start + max(h.old_len, 1) - 1 for h in f.hunks)
        worst = max(worst, reads_to_reach(entry["lengths"], last, config))
    return worst or None


def study(tasks: list[Task], lengths: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    base = ToolConfig()
    by_cap: dict[str, Any] = {}
    measured: list[str] = []
    for cap in CAPS:
        cfg = dataclasses.replace(base, read_max_chars=cap)
        reads = {t.instance_id: task_reads(t, lengths.get(t.instance_id, {}), cfg) for t in tasks}
        known = [r for r in reads.values() if r is not None]
        measured = [i for i, r in reads.items() if r is not None]
        n = len(known)
        by_cap[str(cap)] = {
            "edit_beyond_first_read": rate(sum(1 for r in known if r > 1), n),
            "needs_more_than_3_reads": rate(sum(1 for r in known if r > 3), n),
            "reads_to_reach_edit": {
                "median": statistics.median(known) if known else None,
                "mean": round(statistics.mean(known), 2) if known else None,
                "p90": sorted(known)[int(0.9 * (n - 1))] if known else None,
                "max": max(known) if known else None,
            },
        }
    # the line-count view the README used before reads were capped by characters
    last_lines = []
    for t in tasks:
        ends = [h.old_start + max(h.old_len, 1) - 1 for f in parse_patch(t.patch) for h in f.hunks]
        last_lines.append(max(ends) if ends else 0)
    return {
        "tasks": len(tasks),
        "tasks_measured": len(measured),
        "unmeasured": sorted({t.instance_id for t in tasks} - set(measured)),
        "default_read_max_chars": base.read_max_chars,
        "read_max_lines": base.read_max_lines,
        "by_char_cap": by_cap,
        "legacy_line_windows": {
            str(w): rate(sum(1 for n in last_lines if n > w), len(tasks))
            for w in (250, 500, 1000, 2000)
        },
        "median_last_edited_line": sorted(last_lines)[len(last_lines) // 2],
        "max_last_edited_line": max(last_lines),
    }
