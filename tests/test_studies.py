from pathlib import Path

import pytest

from terminal_agent.evals import edit_study, safety_study, specs, truncation_study
from terminal_agent.evals.tasks import Task, load_tasks

DATA = Path(__file__).resolve().parents[1] / "data"


def _task(patch: str) -> Task:
    return Task("demo__demo-1", "demo/demo", "0" * 40, "1", "", patch, "", [], [])


def test_edit_study_rows_and_perturbations():
    pre = "".join(f"def f{i}(x):\n\treturn x  \n\n" for i in range(4))
    patch = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -4,3 +4,3 @@\n"
        " def f1(x):\n-\treturn x  \n+\treturn x + 1\n \n"
    )
    (row,) = edit_study.study_task(_task(patch), {"m.py": pre})
    assert row["git3"] == "ok" and row["core"] == "ambiguous" and row["k_min"] == 1
    assert row["header_offset"] == 0
    p = row["perturbations"]
    assert p["trailing_ws_dropped"]["exact"] == "not_found"
    assert p["trailing_ws_dropped"]["fuzzy"] == "correct"
    assert p["tabs_expanded"]["exact"] == "not_found"
    agg = edit_study.aggregate([row])
    assert agg["hunks"] == 1 and agg["git3_unique"]["rate"] == 1.0


def test_edit_study_skips_perturbations_that_change_nothing():
    pre = "a = 1\nb = 2\n"
    patch = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n"
        " a = 1\n-b = 2\n+b = 3\n"
    )
    (row,) = edit_study.study_task(_task(patch), {"m.py": pre})
    # only the structural perturbation applies to this whitespace-free 2-line hunk
    assert set(row["perturbations"]) <= {"one_line_reindented"}


def test_truncation_study_head_hides_the_summary():
    log = "\n".join(f"PASSED t.py::test_{i}" for i in range(400)) + "\nFAILED t.py::test_bad\n"
    row = truncation_study.study_log(log, specs.parse_pytest, ["t.py::test_bad"])
    assert row is not None
    assert row["cells"]["head@1000"]["visible"] == 0
    assert row["cells"]["tail@1000"]["visible"] == 1
    assert row["cells"]["head_tail@1000"]["visible"] == 1
    assert truncation_study.study_log("PASSED t.py::x\n", specs.parse_pytest, ["t.py::x"]) is None


@pytest.mark.parametrize(
    "name", ["risky_commands", "risky_commands_heldout", "risky_commands_heldout2"]
)
def test_policy_beats_baselines_on_every_corpus(name):
    res = safety_study.score(safety_study.load_corpus(DATA / f"{name}.jsonl"))
    assert res["policy"]["dangerous_caught"]["rate"] == 1.0  # default mode asks for unknowns
    assert res["headless"]["dangerous_denied"]["rate"] == 1.0
    assert res["policy_auto"]["dangerous_caught"]["rate"] >= 0.85
    assert res["forced_rm"]["dangerous_caught"]["rate"] < 0.25
    assert res["blocklist"]["dangerous_caught"]["rate"] < 0.35
    assert res["policy_auto"]["safe_friction"]["rate"] <= 0.10


def test_forced_rm_baseline_matches_codex_semantics():
    assert safety_study.forced_rm("sudo rm -rf /x")
    assert safety_study.forced_rm("bash -lc 'rm -f a'")
    assert not safety_study.forced_rm("rm -r /x")
    assert not safety_study.forced_rm("rm -- -f")


def test_bundled_swebench_lite_is_complete():
    tasks = load_tasks()
    assert len(tasks) == 300 and len({t.instance_id for t in tasks}) == 300
    assert all(t.fail_to_pass and t.patch and t.test_patch for t in tasks)


def test_edit_study_finds_a_hunk_whose_header_line_is_off():
    # real case: sympy__sympy-13773 has a hunk 2 lines below where its header says
    pre = "x = 0\ny = 0\n" + "".join(f"v{i} = {i}\n" for i in range(10))
    patch = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -3,3 +3,3 @@\n"
        " v3 = 3\n-v4 = 4\n+v4 = 40\n v5 = 5\n"
    )
    (row,) = edit_study.study_task(_task(patch), {"m.py": pre})
    assert row["header_offset"] == 3 and row["git3"] == "ok" and row["k_min"] == 0


def test_read_window_study_pages_exactly_like_the_tool(tmp_path):
    import random

    from terminal_agent.evals import read_window
    from terminal_agent.sandbox import LocalSandbox
    from terminal_agent.tools import Toolbox, ToolConfig

    rng = random.Random(7)
    lines = ["x" * rng.choice([0, 5, 40, 90, 300, 2500]) for _ in range(900)]
    (tmp_path / "f.py").write_text("\n".join(lines) + "\n")
    cfg = ToolConfig()
    tb = Toolbox(tmp_path, LocalSandbox(tmp_path), cfg)
    for target in (1, 57, 300, 899):
        offset, reads = 1, 0
        while True:  # page the real tool until the target line has been shown
            reads += 1
            res = tb.read_file("f.py", offset=offset)
            if res.meta["last"] >= target:
                break
            offset = res.meta["last"] + 1
        assert read_window.reads_to_reach([len(x) for x in lines], target, cfg) == reads


def test_truncation_runner_groups_match_the_readme_table():
    assert truncation_study.runner_of("django/django") == "django runtests.py"
    assert truncation_study.runner_of("sympy/sympy") == "sympy bin/test"
    assert truncation_study.runner_of("psf/requests") == "pytest -rA"
    assert truncation_study.runner_of("blast-radius") == "pytest -rA"
