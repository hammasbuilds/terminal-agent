"""Harness validation: drive the agent with each task's gold patch and check fail -> pass.

For every task, four things must hold before the harness is trusted with model time:

1. **tool layer** - every hunk of the gold patch lands through ``read_file``/``edit``, and
   the resulting files are byte-identical to ``git apply`` of the same patch;
2. **baseline** - with only the test patch applied, every FAIL_TO_PASS test fails and every
   PASS_TO_PASS test passes;
3. **gold** - with the agent's edits pushed in, every FAIL_TO_PASS and PASS_TO_PASS passes;
4. **export** - the local workspace is a faithful copy (HEAD == base_commit, nothing
   skipped that the patch touches).

A task failing any of these is a harness or task bug, and is reported with the reason.
"""

from __future__ import annotations

import gzip
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from terminal_agent.agent import build_agent
from terminal_agent.evals import specs
from terminal_agent.evals.gold import GoldPatchClient
from terminal_agent.evals.local_tasks import LocalTask, git_apply, run_pytest, test_env
from terminal_agent.evals.patches import parse_patch
from terminal_agent.evals.tasks import Container, Task, rmtree, test_section
from terminal_agent.policy import ApprovalPolicy
from terminal_agent.sandbox import (
    DockerSandbox,
    LocalSandbox,
    changed_files,
    docker,
    file_digest,
    snapshot,
)


def image_present(image: str) -> bool:
    return docker(["image", "inspect", image], check=False, timeout=60).returncode == 0


def _save_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _grade(task: Task, log: str) -> tuple[dict[str, Any], dict[str, str]]:
    statuses = specs.PARSERS[task.repo](test_section(log))
    return specs.grade(statuses, task.fail_to_pass, task.pass_to_pass), statuses


def validate_task(task: Task, run_dir: Path, log_dir: Path, test_timeout: float = 1800
                  ) -> dict[str, Any]:
    rec: dict[str, Any] = {"instance_id": task.instance_id, "repo": task.repo,
                           "version": task.version}
    t0 = time.monotonic()
    if not image_present(task.image):
        rec["verdict"] = "image_missing"
        rec["reasons"] = [f"docker image {task.image} is not pulled"]
        return rec
    ws = run_dir / task.instance_id / "workspace"
    traj = run_dir / task.instance_id / "trajectory.jsonl"
    traj.unlink(missing_ok=True)
    gold_files = [f.path for f in parse_patch(task.patch)]

    with Container(task) as c:
        rec["head_matches_base"] = c.head() == task.base_commit
        export = c.export(ws)
        rec["export"] = asdict(export)
        before = snapshot(ws)
        rec["preimages"] = {p: (ws / p).read_bytes().decode("utf-8", "surrogateescape")
                            for p in gold_files if (ws / p).is_file()}
        client = GoldPatchClient(task.patch)
        agent = build_agent(
            ws, client,
            sandbox=DockerSandbox(c.name, ws, prelude=specs.PRELUDE),
            policy=ApprovalPolicy(ws, shell_root="/testbed", mode="auto"),
            trajectory=traj, max_steps=400, token_budget=10**9,
        )
        result = agent.run(task.problem_statement)
        agent.logger.close()
        rec["agent"] = result.as_dict()
        rec["hunks"] = [asdict(r) for r in client.records]

    modified, deleted = changed_files(before, snapshot(ws))
    with Container(task) as ref:
        ref.write("/tmp/gold.patch", task.patch.encode("utf-8"))
        code, out = ref.sh("cd /testbed && git apply -v /tmp/gold.patch")
        rec["git_apply_ok"] = code == 0
        mismatched = []
        for rel in sorted(set(modified) | set(gold_files)):
            code, out = ref.sh(f"sha1sum /testbed/{rel} 2>/dev/null")
            theirs = out.split(" ")[0] if code == 0 else None
            ours = file_digest(ws / rel) if (ws / rel).is_file() else None
            if theirs != ours:
                mismatched.append(rel)
        rec["byte_mismatch"] = mismatched
        rec["changed_files"] = {"modified": modified, "deleted": deleted}

    with Container(task) as base:
        log, meta = base.run_tests(test_timeout)
        _save_log(log_dir / f"{task.instance_id}.baseline.log.gz", log)
        grade, _ = _grade(task, log)
        rec["baseline"] = {**grade, **meta}

    with Container(task) as fin:
        fin.push_changes(ws, before)
        rec["model_patch"] = fin.diff()
        log, meta = fin.run_tests(test_timeout)
        _save_log(log_dir / f"{task.instance_id}.gold.log.gz", log)
        grade, _ = _grade(task, log)
        rec["gold"] = {**grade, **meta}

    rec["reasons"] = diagnose(rec, gold_files)
    rec["verdict"] = "valid" if not rec["reasons"] else "invalid"
    rec["seconds"] = round(time.monotonic() - t0, 1)
    rmtree(ws)
    return rec


def diagnose(rec: dict[str, Any], gold_files: list[str]) -> list[str]:
    reasons = []
    if not rec.get("head_matches_base"):
        reasons.append("export: image HEAD is not the task's base_commit")
    skipped = set(rec["export"]["skipped_symlinks"]) | set(rec["export"]["skipped_invalid"])
    if skipped & set(gold_files):
        reasons.append("export: a file the patch touches could not be exported")
    bad_hunks = [h for h in rec.get("hunks", []) if h["final_status"] != "ok"]
    if bad_hunks:
        reasons.append(f"tool layer: {len(bad_hunks)} hunk(s) did not apply "
                       f"({', '.join(sorted({h['final_status'] for h in bad_hunks}))})")
    if rec["agent"]["status"] != "finished":
        reasons.append(f"agent loop ended with {rec['agent']['status']}")
    if not rec.get("git_apply_ok"):
        reasons.append("reference: git apply of the gold patch failed")
    if rec.get("byte_mismatch"):
        reasons.append(f"tool layer: result differs from git apply in "
                       f"{', '.join(rec['byte_mismatch'])}")
    b, g = rec["baseline"], rec["gold"]
    for label, r in (("baseline", b), ("gold", g)):
        if not r.get("test_patch_applied", True):
            reasons.append(f"{label}: the test patch did not apply")
        if r.get("timed_out"):
            reasons.append(f"{label}: tests timed out")
    if b["f2p_passed"]:
        reasons.append(f"task: {b['f2p_passed']}/{b['f2p_total']} FAIL_TO_PASS already pass "
                       f"before the fix")
    if b["p2p_passed"] < b["p2p_total"]:
        reasons.append(f"task: {b['p2p_total'] - b['p2p_passed']} PASS_TO_PASS fail before "
                       f"the fix")
    if g["f2p_passed"] < g["f2p_total"]:
        reasons.append(f"gold: {g['f2p_total'] - g['f2p_passed']} FAIL_TO_PASS still fail")
    if g["p2p_passed"] < g["p2p_total"]:
        reasons.append(f"gold: {g['p2p_total'] - g['p2p_passed']} PASS_TO_PASS fail")
    return reasons


def load_records(out_dir: Path) -> dict[str, dict[str, Any]]:
    recs = {}
    for p in sorted(out_dir.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        recs[rec["instance_id"]] = rec
    return recs


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ran = [r for r in records if r.get("verdict") in ("valid", "invalid")]
    valid = [r for r in ran if r["verdict"] == "valid"]
    hunks = [h for r in ran for h in r.get("hunks", [])]
    by_repo: dict[str, dict[str, int]] = {}
    for r in ran:
        d = by_repo.setdefault(r["repo"], {"tasks": 0, "valid": 0})
        d["tasks"] += 1
        d["valid"] += r["verdict"] == "valid"
    reason_counts: dict[str, int] = {}
    for r in ran:
        for reason in r.get("reasons", []):
            key = reason.split(":")[0] + ": " + reason.split(":")[1].split("(")[0].strip()
            reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "tasks_attempted": len(ran),
        "tasks_valid": len(valid),
        "valid_rate": round(len(valid) / len(ran), 4) if ran else None,
        "image_missing": sum(1 for r in records if r.get("verdict") == "image_missing"),
        "by_repo": by_repo,
        "reason_counts": reason_counts,
        "hunks": len(hunks),
        "hunks_first_try_ok": sum(1 for h in hunks if h["first_status"] == "ok"),
        "hunks_first_try_ambiguous": sum(1 for h in hunks if h["first_status"] == "ambiguous"),
        "hunks_final_ok": sum(1 for h in hunks if h["final_status"] == "ok"),
        "byte_identical_tasks": sum(1 for r in ran if not r.get("byte_mismatch")),
        "invalid": {r["instance_id"]: r["reasons"] for r in ran if r["verdict"] != "valid"},
    }


def validate_local_task(task: LocalTask, run_dir: Path, log_dir: Path) -> dict[str, Any]:
    """The same four checks for a mined local task, with pytest on this machine."""
    rec: dict[str, Any] = {"instance_id": task.instance_id, "repo": task.repo,
                           "version": task.base_commit[:10], "head_matches_base": True}
    t0 = time.monotonic()
    root = run_dir / task.instance_id.replace("@", "_")
    ws, ref, base, fin = (root / n for n in ("workspace", "ref", "baseline", "gold"))
    task.materialize(ws)
    rec["export"] = {"files": sum(1 for _ in ws.rglob("*")), "skipped_symlinks": [],
                     "skipped_invalid": [], "case_collisions": []}
    before = snapshot(ws)
    gold_files = [f.path for f in parse_patch(task.patch)]
    rec["preimages"] = {p: (ws / p).read_bytes().decode("utf-8", "surrogateescape")
                        for p in gold_files if (ws / p).is_file()}
    client = GoldPatchClient(task.patch)
    agent = build_agent(ws, client, sandbox=LocalSandbox(ws, env=test_env(ws)),
                        policy=ApprovalPolicy(ws, mode="auto"),
                        trajectory=root / "trajectory.jsonl", max_steps=400,
                        token_budget=10**9)
    result = agent.run(task.problem_statement)
    agent.logger.close()
    rec["agent"] = result.as_dict()
    rec["hunks"] = [asdict(r) for r in client.records]
    modified, deleted = changed_files(before, snapshot(ws))
    rec["changed_files"] = {"modified": modified, "deleted": deleted}

    task.materialize(ref)
    rec["git_apply_ok"], _ = git_apply(ref, task.patch)
    rec["byte_mismatch"] = [
        rel for rel in sorted(set(modified) | set(gold_files))
        if ((ws / rel).is_file() and file_digest(ws / rel)) != (
            (ref / rel).is_file() and file_digest(ref / rel))
    ]
    runnable = task.test_files
    for label, tree in (("baseline", base), ("gold", fin)):
        if label == "baseline":
            task.materialize(tree)
        else:
            rmtree(tree)
            shutil.copytree(ws, tree)
        applied, _ = git_apply(tree, task.test_patch)
        log, timed_out = run_pytest(tree, runnable)
        _save_log(log_dir / f"{task.instance_id.replace('@', '_')}.{label}.log.gz", log)
        grade = specs.grade(specs.parse_pytest(log), task.fail_to_pass, task.pass_to_pass)
        rec[label] = {**grade, "test_patch_applied": applied, "timed_out": timed_out}
    rec["reasons"] = diagnose(rec, gold_files)
    rec["verdict"] = "valid" if not rec["reasons"] else "invalid"
    rec["seconds"] = round(time.monotonic() - t0, 1)
    rmtree(root / "ref")
    rmtree(root / "baseline")
    rmtree(root / "gold")
    rmtree(ws)
    return rec
