"""Fetch the pre-image of every file SWE-bench Lite's gold patches edit; keep line lengths.

The read-window study needs to know how many lines of a file one ``read_file`` call shows,
and since reads are capped by characters that depends on every line's length. Only the
lengths are stored (plus a sha1 of the file, so a later fetch can be checked), which keeps
``data/lite_line_lengths.json.gz`` small. Files come from raw.githubusercontent.com at the
task's ``base_commit``; the run resumes from the output file if interrupted.

    uv run python scripts/fetch_lite_line_lengths.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from terminal_agent.evals.patches import parse_patch
from terminal_agent.evals.tasks import load_tasks

OUT = Path(__file__).resolve().parents[1] / "data" / "lite_line_lengths.json.gz"
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def fetch(url: str, attempts: int = 6) -> bytes | None:
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"could not fetch {url}")


def line_lengths(data: bytes) -> list[int]:
    text = data.decode("utf-8", "replace").replace("\r\n", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [len(line) for line in lines]


def main() -> int:
    done: dict[str, dict[str, dict[str, object]]] = {}
    if OUT.exists():
        done = json.loads(gzip.decompress(OUT.read_bytes()))
    jobs = []
    for t in load_tasks():
        for f in parse_patch(t.patch):
            if f.is_new or f.path in done.get(t.instance_id, {}):
                continue
            jobs.append(
                (t.instance_id, f.path, RAW.format(repo=t.repo, commit=t.base_commit, path=f.path))
            )
    print(f"{len(jobs)} files to fetch", flush=True)

    def one(job: tuple[str, str, str]) -> tuple[str, str, bytes | None]:
        iid, path, url = job
        return iid, path, fetch(url)

    with ThreadPoolExecutor(4) as pool:
        for n, (iid, path, data) in enumerate(pool.map(one, jobs), 1):
            entry = (
                {"missing": True}
                if data is None
                else {"sha1": hashlib.sha1(data).hexdigest(), "lengths": line_lengths(data)}
            )
            done.setdefault(iid, {})[path] = entry
            if n % 25 == 0 or n == len(jobs):
                OUT.write_bytes(
                    gzip.compress(json.dumps(done, sort_keys=True).encode(), 9, mtime=0)
                )
                print(f"{n}/{len(jobs)}", flush=True)
    OUT.write_bytes(gzip.compress(json.dumps(done, sort_keys=True).encode(), 9, mtime=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
