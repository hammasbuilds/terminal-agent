"""Export SWE-bench Lite (test split) from the local Hugging Face cache to data/.

One-off, needs ``pyarrow`` (not a dependency of the package). Usage:

    python scripts/export_swebench_lite.py [path/to/swe-bench_lite-test.arrow]

The default path is where ``datasets.load_dataset("princeton-nlp/SWE-bench_Lite")`` caches
the split. The output is a gzipped JSONL of the fields the harness uses; hints are left out
because the agent is not given them.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pyarrow as pa

DEFAULT = (
    Path.home() / ".cache/huggingface/datasets/princeton-nlp___swe-bench_lite/default/"
    "0.0.0/6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2/swe-bench_lite-test.arrow"
)
FIELDS = [
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "created_at",
    "problem_statement",
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
]
OUT = Path(__file__).resolve().parents[1] / "data" / "swebench_lite.jsonl.gz"


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    with pa.memory_map(str(src)) as fh:
        table = pa.ipc.open_stream(fh).read_all()
    rows = table.to_pylist()
    with gzip.open(OUT, "wt", encoding="utf-8", newline="\n") as out:
        for row in sorted(rows, key=lambda r: r["instance_id"]):
            rec = {k: row[k] for k in FIELDS}
            for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                rec[k] = json.loads(rec[k])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} instances to {OUT}")


if __name__ == "__main__":
    main()
