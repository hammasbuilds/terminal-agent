# STATUS

**READY-FOR-MODEL-RUN** - self-score **91/100** (model-arm cap: 5). An independent review of
the previous submission scored it 76/100 and found real defects, chiefly in the approval
policy; those are fixed and each carries a regression test. Everything that does not need a
model is done and measured; the model arm is built, tested with scripted clients and queued.

## What the review found and what changed

| # | Reviewer finding | Fix | Regression test |
|---|---|---|---|
| 1 | read-only commands with a write/exec flag auto-ran (`pytest --basetemp=..`, `git -c ...`, `sort -o`, exec env vars, ...) | flag/env-aware re-rating of read-only tools | `test_reviewer_bypasses_are_now_dangerous`, `test_third_held_out_vectors_are_dangerous` |
| 2 | wrappers took a flag as the command (`nice -n 1 rm -rf`, `env -S`) | wrappers skip their own options; `env -S` re-parses | same |
| 3 | no `cd`/computed-path tracking | `cd` tracked; unresolvable write target = outside | `test_cd_then_write_inside_vs_outside` |
| 4 | local model workspace could corrupt the harness repo via git | workspace git-init'd + `GIT_CEILING_DIRECTORIES` | `test_git_commands_in_a_local_workspace_cannot_touch_the_enclosing_repo` |
| 5 | model_error persisted, skipped on resume, counted in denominator; bad JSON uncaught | retry, never persist, exclude from denominator; wrap JSON | `test_model_errors_are_retried_...`, `test_model_error_excluded_...`, `test_non_json_ollama_body...` |
| 6 | 1000-line read window over budget; squeeze cut the task text | read capped by chars; task/system prompt never squeezed | `test_read_file_*`, `test_compaction_never_squeezes_the_protected_task` |
| 7 | compaction dropped a REPL's second task | protect the latest user message too | `test_compaction_protects_the_latest_task_not_just_the_first` |
| 8 | fuzzy edit stripped all indentation, misapplied structural changes | match on common-dedented blocks; refuse structural breaks | `test_fuzzy_indent_refuses_a_structurally_different_snippet`, `..._does_not_merge_tabs...` |
| 9 | 2 s timeout returned after ~12 s with a backgrounded child; negative timeout accepted | output to a temp file; reject non-positive timeout | `test_run_shell_backgrounded_child_...`, `..._rejects_a_non_positive_timeout` |
| 10 | malformed `--script` raised a traceback | clean `ValueError` | `test_malformed_script_raises_value_error_not_traceback` |
| 11 | stale `read_window.json` preimage count | regenerated (now 39) | n/a (data) |
| 12 | `git status -z` renames misparsed | consume the rename's second field | `test_git_status_z_rename_is_parsed` (docker) |
| 13 | pushed files forced to 0o644 (broke executables) | preserve the container's exec bit | covered by the docker sync test |
| 14 | unused param/field, badge count, dangling REPL message | removed; badge 224; assistant note on model error | `test_model_error_leaves_a_well_formed_conversation` |

The policy was rewritten around two rules: a command's name is not a promise (flags are
inspected), and a write target must resolve statically to inside the workspace. A **third**
held-out command set (206 commands, written by a fresh agent with no repo access) was scored
once, blind: the classifier caught **111/121** dangerous commands in `auto` mode and rated
**0/121** as safe, so 0 ran unasked in `default` mode. Its 10 misses were new categories,
since fixed and tested.

## Self-score

| Points | Criterion | Score | Reason |
|---:|---|---:|---|
| 15 | Works from a clean clone | 15 | Fresh clone: `uv sync --offline`, `uv run pytest -q` (222 pass, HF_HOME at an empty dir), `uv run python demo.py` all succeed; no network/model/Docker for tests or demo. |
| 20 | Real data, real result | 17 | Headline numbers from real SWE-bench Lite instances in the official images and real bug-fix commits, produced here. Capped: 39 of 300 Lite tasks (download-bound); solve rate needs the model arm. |
| 15 | Finding quality | 13 | Controls and baselines (three blind command sets, two policy baselines, four truncation modes, perturbation vs gold), Wilson + task-clustered bootstrap CIs, every surprising number chased (three were bugs in my own study). Minus: truncation digest not held-out; local suite small/easy. |
| 15 | Correctness | 14 | 224 tests on behaviour and failure modes, incl. the reviewer's exploit corpus as fixtures; each confirmed regression test fails against the old code. Minus: `auto`-mode classifier coverage is ~91% on unseen commands, not 100% (default mode is the guarantee). |
| 10 | Usability | 9 | `--help` on both CLIs; actionable errors (Ollama unreachable, model not pulled, bad script, unknown task id, bad timeout); `--script`, `examples/`. Minus: SWE-bench runs need multi-GB images the user pulls. |
| 10 | README | 10 | House format, findings table, six real I/O samples, NOT-do section, real problems hit (now with the review's findings and honest reframing of findings 1 and 3/6). |
| 10 | Code quality | 8 | ruff clean, typed, zero runtime deps, small modules. Minus: `policy.py` is ~760 lines (one classifier, many command families) - cohesive but large. |
| 5 | Honesty | 5 | Every README number traces to `results/`; default-mode 0/514 stated separately from auto-mode held-out coverage; the fuzzy "80/80" is split into uniform (works) vs structural (refused); caveats on the digest and the held-out sets stated. |

## Queued for the model run

`scripts/run_models.sh` (dry run: `scripts/run_models.sh --dry-run`):

| suite | tasks | model-call bound |
|---|---:|---:|
| local (valid) | 15 | 450 |
| SWE-bench Lite (valid) | 35 | 1,050 |
| **total** | **50** | **at most 1,500** |

Local tasks run in a git-isolated throwaway workspace; model errors are retried and excluded
from the solve-rate denominator. Generations are cached by (model, messages, tools, options);
each record is written when it finishes, so re-running resumes. Output:
`results/model_run_qwen2.5-coder_14b_{local,swebench}.json` - solve rate with Wilson CI,
outcome taxonomy, steps (with bootstrap CI), tokens, tool-call share, edit failure rate,
estimator error, context-limit hits, model-error count.

## Known weaknesses

- `auto`-mode command coverage does not converge to 100% on unseen commands; the safety that
  holds without a blind set is default mode (0 of 514 dangerous commands ran unasked).
- 39 of 300 Lite tasks, 5 of 12 repos (Docker Hub CDN was failing DNS at the time).
- The truncation `digest` regex was written after seeing these runners' output.
- The mined local suite uses commit messages as issues, which often describe the fix.
- `blast-radius@74c59cd37a` is counted valid but failed once under machine load.
- `policy.py` is large; a split into `readonly`/`git`/`paths` submodules is a reasonable
  follow-up but was not done to avoid churn under review.

## Reproduce

```bash
unset VIRTUAL_ENV
uv sync
uv run pytest -q                                   # 222 tests (+2 with -m docker)
uv run python demo.py
uv run ta-eval safety                              # results/safety.json (4 corpora)
uv run ta-eval read-window                         # results/read_window.json
uv run ta-eval validate --suite local --force --repeat 2   # results/harness_validation_local.json
docker pull <images for the 39 tasks>              # see data/ and results/validation/swebench/
uv run ta-eval validate --suite swebench --pulled-only     # results/harness_validation_swebench.json
uv run ta-eval edit-study                          # results/edit_study.json
uv run ta-eval truncation-study                    # results/truncation_study.json
scripts/run_models.sh --dry-run                    # the model arm (do not run without the GPU)
```

`results/safety_policy_v{1,2,4}*.json` are the blind scores of the policy at the commit each
held-out set was introduced; `results/safety.json` is the current policy on all four sets.
