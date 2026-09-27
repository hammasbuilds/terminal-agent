"""``ta-eval``: build and validate task suites, run the tool-layer studies, run the model arm."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from terminal_agent.evals import (
    edit_study,
    model_run,
    safety_study,
    specs,
    truncation_study,
    validate,
)
from terminal_agent.evals.local_tasks import LOCAL_DATA, load_local_tasks, mine, save_local_tasks
from terminal_agent.evals.patches import parse_patch
from terminal_agent.evals.stats import rate
from terminal_agent.evals.tasks import load_tasks, select, test_section
from terminal_agent.llm import DEFAULT_HOST, DEFAULT_MODEL, OllamaClient

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
PREIMAGES = ROOT / "data" / "gold_preimages.jsonl.gz"


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")
    print(f"wrote {path.relative_to(ROOT)}")


# -- mine-local ------------------------------------------------------------------------------

def cmd_mine_local(args: argparse.Namespace) -> int:
    repos: list[tuple[Path, str]] = []
    public = set(args.public or [])
    for d in sorted(Path(args.root).iterdir()):
        if not (d / ".git").exists() or d.resolve() == ROOT:
            continue
        url = subprocess.run(["git", "-C", str(d), "remote", "get-url", "origin"],
                             capture_output=True, text=True, check=False).stdout.strip()
        name = url.rstrip("/").removesuffix(".git").split("/")[-1] if url else ""
        if public and name not in public:
            continue
        repos.append((d, name))
    print(f"mining {len(repos)} repositories")
    tasks = mine(repos, limit=args.limit)
    save_local_tasks(tasks, Path(args.out))
    print(f"{len(tasks)} tasks -> {args.out}")
    return 0


# -- validate --------------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    out_dir = RESULTS / "validation" / args.suite
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    preimages = edit_study.load_preimages(PREIMAGES)
    if args.suite == "swebench":
        tasks = select(load_tasks(), args.ids)
        if args.repos:
            tasks = [t for t in tasks if t.repo in args.repos]
        if args.pulled_only:
            pulled = validate.pulled_images()
            tasks = [t for t in tasks if t.image in pulled]
        jobs: list[Any] = tasks
    else:
        local = load_local_tasks()
        jobs = [t for t in local if not args.ids or t.instance_id in args.ids]
    print(f"validating {len(jobs)} {args.suite} task(s)")
    for task in jobs:
        target = out_dir / f"{task.instance_id.replace('@', '_')}.json"
        if target.exists() and not args.force:
            continue
        t0 = time.monotonic()
        if args.suite == "swebench":
            rec = validate.validate_task(task, RUNS / "validate", log_dir, args.test_timeout,
                                         args.repeat)
        else:
            rec = validate.validate_local_task(task, RUNS / "validate-local", log_dir,
                                               args.repeat)
        pre = rec.pop("preimages", None)
        if pre:
            preimages[task.instance_id] = pre
            edit_study.save_preimages(PREIMAGES, preimages)
        if rec.get("verdict") != "image_missing":
            target.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8", newline="\n")
        print(f"{task.instance_id}: {rec['verdict']} ({time.monotonic() - t0:.0f}s) "
              + "; ".join(rec.get("reasons", [])), flush=True)
    records = list(validate.load_records(out_dir).values())
    _write(RESULTS / f"harness_validation_{args.suite}.json", validate.summarize(records))
    return 0


# -- studies ---------------------------------------------------------------------------------

def _all_tasks() -> dict[str, Any]:
    tasks: dict[str, Any] = {t.instance_id: t for t in load_tasks()}
    tasks.update({t.instance_id: t for t in load_local_tasks()})
    return tasks


def cmd_edit_study(args: argparse.Namespace) -> int:
    preimages = edit_study.load_preimages(PREIMAGES)
    tasks = _all_tasks()
    rows = []
    for iid, files in sorted(preimages.items()):
        if iid in tasks:
            rows.extend(edit_study.study_task(tasks[iid], files))
    swe = [r for r in rows if "@" not in r["instance_id"]]
    loc = [r for r in rows if "@" in r["instance_id"]]
    _write(RESULTS / "edit_study.json", {
        "all": edit_study.aggregate(rows),
        "swebench": edit_study.aggregate(swe),
        "local": edit_study.aggregate(loc),
        "per_hunk": rows,
    })
    return 0


def cmd_read_window(args: argparse.Namespace) -> int:
    """Where in the file SWE-bench Lite's gold edits sit, from the patch headers alone."""
    tasks = load_tasks()
    windows = (250, 500, 1000, 2000)
    last_lines = []
    for t in tasks:
        ends = [h.old_start + max(h.old_len, 1) - 1 for f in parse_patch(t.patch)
                for h in f.hunks]
        last_lines.append(max(ends) if ends else 0)
    preimages = edit_study.load_preimages(PREIMAGES)
    long_line_files = [
        f"{iid}:{p}" for iid, files in preimages.items() for p, c in files.items()
        if any(len(line) > 2000 for line in c.split("\n"))
    ]
    _write(RESULTS / "read_window.json", {
        "tasks": len(tasks),
        "edit_beyond_first_window": {str(w): rate(sum(1 for n in last_lines if n > w),
                                                  len(tasks)) for w in windows},
        "median_last_edited_line": sorted(last_lines)[len(last_lines) // 2],
        "max_last_edited_line": max(last_lines),
        "preimage_files_with_lines_over_2000_chars": long_line_files,
        "preimage_files": sum(len(v) for v in preimages.values()),
    })
    return 0


def cmd_truncation_study(args: argparse.Namespace) -> int:
    tasks = _all_tasks()
    per_log: dict[str, Any] = {}
    for suite in ("swebench", "local"):
        log_dir = RESULTS / "validation" / suite / "logs"
        for p in sorted(log_dir.glob("*.baseline.log.gz")):
            key = p.name.removesuffix(".baseline.log.gz")
            iid = key if suite == "swebench" else key.replace("_", "@", 1)
            task = tasks.get(iid) or next(
                (t for k, t in tasks.items() if k.replace("@", "_") == key), None)
            if task is None:
                continue
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                log = fh.read()
            if suite == "swebench":
                log = test_section(log)
                parser = specs.PARSERS[task.repo]
            else:
                parser = specs.parse_pytest
            row = truncation_study.study_log(log, parser, task.fail_to_pass)
            if row is not None:
                per_log[task.instance_id] = row
    _write(RESULTS / "truncation_study.json",
           {"summary": truncation_study.aggregate(per_log), "per_log": per_log})
    return 0


def cmd_safety(args: argparse.Namespace) -> int:
    out = {}
    for name in ("risky_commands", "risky_commands_heldout", "risky_commands_heldout2"):
        corpus = safety_study.load_corpus(ROOT / "data" / f"{name}.jsonl")
        out[name] = safety_study.score(corpus)
    _write(RESULTS / "safety.json", out)
    for name, res in out.items():
        p = res["policy"]
        print(f"{name}: dangerous caught {p['dangerous_caught']['rate']}, safe friction "
              f"{p['safe_friction']['rate']}; forced_rm caught "
              f"{res['forced_rm']['dangerous_caught']['rate']}, blocklist caught "
              f"{res['blocklist']['dangerous_caught']['rate']}")
    return 0


def valid_ids(suite: str) -> list[str]:
    out_dir = RESULTS / "validation" / suite
    return sorted(r["instance_id"] for r in validate.load_records(out_dir).values()
                  if r.get("verdict") == "valid")


def cmd_model_run(args: argparse.Namespace) -> int:
    ids = valid_ids(args.suite)
    if args.ids:
        ids = [i for i in ids if i in set(args.ids)]
    if args.limit:
        ids = ids[: args.limit]
    upper = len(ids) * args.max_steps
    print(f"model arm: {args.model} on {len(ids)} validated {args.suite} task(s); "
          f"at most {upper} model calls ({args.max_steps} steps x {len(ids)} tasks)")
    if args.dry_run:
        for i in ids:
            print(f"  {i}")
        return 0
    if not ids:
        print("nothing to run: validate the suite first (ta-eval validate)", file=sys.stderr)
        return 1
    client = OllamaClient(model=args.model, host=args.host,
                          cache_dir=ROOT / "cache" / "ollama" / args.model.replace(":", "_"))
    out_dir = RESULTS / "model_run" / args.model.replace(":", "_") / args.suite
    runs = RUNS / "model" / args.suite
    if args.suite == "swebench":
        by_id = {t.instance_id: t for t in load_tasks()}
        jobs: list[Any] = [by_id[i] for i in ids]

        def runner(job: Any) -> dict[str, Any]:
            return model_run.run_swebench_task(job, client, runs, args.max_steps,
                                               args.token_budget)
    else:
        by_local = {t.instance_id: t for t in load_local_tasks()}
        jobs = [by_local[i] for i in ids]

        def runner(job: Any) -> dict[str, Any]:
            return model_run.run_local_task(job, client, runs, args.max_steps,
                                            args.token_budget)
    records = model_run.run_all(jobs, runner, out_dir)
    _write(RESULTS / f"model_run_{args.model.replace(':', '_')}_{args.suite}.json",
           {"model": args.model, "max_steps": args.max_steps, "token_budget": args.token_budget,
            "summary": model_run.aggregate(records)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ta-eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine-local", help="mine bug-fix commits from local git repos into tasks")
    m.add_argument("--root", default=str(ROOT.parent), help="directory holding the repos")
    m.add_argument("--public", nargs="*", help="only repos whose origin name is in this list")
    m.add_argument("--limit", type=int, default=400, help="commits per repo to consider")
    m.add_argument("--out", default=str(LOCAL_DATA))
    m.set_defaults(fn=cmd_mine_local)

    v = sub.add_parser("validate", help="apply gold patches through the agent; check fail->pass")
    v.add_argument("--suite", choices=["swebench", "local"], default="local")
    v.add_argument("--ids", nargs="*", help="only these instance ids")
    v.add_argument("--repos", nargs="*", help="only these repos (swebench), e.g. psf/requests")
    v.add_argument("--pulled-only", action="store_true",
                   help="skip SWE-bench tasks whose Docker image is not pulled")
    v.add_argument("--force", action="store_true", help="re-run tasks that already have a record")
    v.add_argument("--test-timeout", type=float, default=1800)
    v.add_argument("--repeat", type=int, default=0,
                   help="extra runs of every test stage, to catch flaky tasks (default 0)")
    v.set_defaults(fn=cmd_validate)

    sub.add_parser("edit-study", help="exact-match edit failure rates on real hunks"
                   ).set_defaults(fn=cmd_edit_study)
    sub.add_parser("read-window", help="where gold edits sit relative to read_file's window"
                   ).set_defaults(fn=cmd_read_window)
    sub.add_parser("truncation-study", help="does truncation hide the failing test?"
                   ).set_defaults(fn=cmd_truncation_study)
    sub.add_parser("safety", help="score the approval policy on the command corpora"
                   ).set_defaults(fn=cmd_safety)

    r = sub.add_parser("model-run", help="run the agent with a real model on validated tasks")
    r.add_argument("--suite", choices=["swebench", "local"], default="local")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--host", default=DEFAULT_HOST)
    r.add_argument("--max-steps", type=int, default=30)
    r.add_argument("--token-budget", type=int, default=12000)
    r.add_argument("--ids", nargs="*")
    r.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    r.add_argument("--dry-run", action="store_true", help="list the jobs and exit")
    r.set_defaults(fn=cmd_model_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
