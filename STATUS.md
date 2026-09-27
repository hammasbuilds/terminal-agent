# STATUS

**READY-FOR-MODEL-RUN** - self-score **90/100**. Everything that does not need a model is
done and measured; the model arm (qwen2.5-coder:14b solve rate, steps, tokens, tool usage,
failure taxonomy) is built, tested with scripted clients and queued in `scripts/run_models.sh`.

## Self-score

| Points | Criterion | Score | Reason |
|---:|---|---:|---|
| 15 | Works from a clean clone | 15 | Fresh `git clone` into a temp dir: `uv sync --offline`, `uv run pytest -q` (all pass, `HF_HOME` pointed at an empty dir), `uv run python demo.py` all succeed. No network, model or Docker needed for tests or demo. |
| 20 | Real data, real result | 17 | Headline numbers come from real SWE-bench Lite instances in the official images and from real bug-fix commits, all produced here. Capped: 39 of 300 Lite tasks (download-bound), and the solve rate needs the model arm. |
| 15 | Finding quality | 13 | Controls and baselines (two held-out command sets written blind, codex's rule and a blocklist as baselines; four truncation modes; perturbation vs gold), Wilson and task-clustered bootstrap CIs, every surprising number investigated (three turned out to be bugs in my own study). Minus: the truncation digest is not held-out; local suite is small and easy. |
| 15 | Correctness | 13 | 128 tests (+2 Docker) on behaviour and failure modes; each regression test verified to fail before its fix. Minus: policy is a classifier, not a sandbox, and its coverage on unseen commands is ~85%; one local task is valid but was seen to fail once under load. |
| 10 | Usability | 9 | `--help` on both CLIs, actionable errors (Ollama unreachable, model not pulled, bad script, unknown task id), `--script` to try the CLI without a model, `examples/`. Minus: SWE-bench runs need multi-GB images the user must pull. |
| 10 | README | 10 | House format, findings table near the top, six real Input/Output samples, NOT-do section, real problems hit. |
| 10 | Code quality | 9 | ruff clean, typed, zero runtime dependencies, small modules. Minus: `policy.py` is long (one classifier with many command families). |
| 5 | Honesty | 4 | Every README number is in `results/`; limitations and the non-held-out caveats stated. Minus one for scope: the SWE-bench subset was chosen by image layers (stated), not at random from all 300. |

## Done

- Agent: REPL, headless `-p` with JSON output, 8 tools + `finish`, loop detection, step limit,
  approval policy (default / auto / allowlist; headless denies), compaction under a token
  budget, JSONL trajectories and `terminal-agent replay`, Ollama client with a disk cache and
  a text-tool-call fallback parser.
- Harness: SWE-bench Lite in the official Docker images (two-way workspace sync, offline agent
  container, SWE-bench log parsers), a mined local suite (17 tasks, no Docker), gold-patch
  replay through the agent's tools with four checks and flaky re-runs.
- Results: `results/harness_validation_{swebench,local}.json`, `results/validation/**`,
  `results/edit_study.json`, `results/truncation_study.json`, `results/read_window.json`,
  `results/safety.json` (+ `safety_policy_v1.json`, `safety_policy_v2.json` as scored at the time).

## Queued for the model run

`scripts/run_models.sh` (dry run: `scripts/run_models.sh --dry-run`):

| suite | tasks | model-call bound |
|---|---:|---:|
| local (valid) | 15 | 450 (30 steps x 15) |
| SWE-bench Lite (valid) | 35 | 1,050 |
| **total** | **50** | **at most 1,500** |

It checks free RAM, free VRAM, that Ollama answers and has the model, pulls any missing image,
then runs `ta-eval model-run` per suite. Generations are cached by (model, messages, tools,
options) and each task's record is written when it finishes, so re-running resumes. Output:
`results/model_run_qwen2.5-coder_14b_{local,swebench}.json` - solve rate with Wilson CI,
outcome taxonomy (resolved / wrong_fix / wrong_file / broke_existing_tests / gave_up /
ran_out_of_steps / loop_without_edit / edits_never_applied / model_error), steps, tokens,
tool-call share, edit failure rate in the wild, estimator error, context-limit hits.

## Known weaknesses

- 39 of 300 Lite tasks, 5 of 12 repos; more images are queued in the scratch pull script but
  Docker Hub's CDN was failing DNS at the time of writing.
- The truncation `digest` regex was written after seeing these runners' output.
- The mined local suite uses commit messages as issues, which often describe the fix.
- Token estimates are chars/3.2 until the model arm measures the real ratio.
- `blast-radius@74c59cd37a` is counted valid but failed once under machine load.

## Reproduce

```bash
unset VIRTUAL_ENV
uv sync
uv run pytest -q                                   # 128 tests
uv run python demo.py
uv run ta-eval safety                              # results/safety.json
uv run ta-eval read-window                         # results/read_window.json
uv run ta-eval validate --suite local --force --repeat 2   # results/harness_validation_local.json
docker pull <images for the 39 tasks>              # see data/ and results/validation/swebench/
uv run ta-eval validate --suite swebench --pulled-only     # results/harness_validation_swebench.json
uv run ta-eval edit-study                          # results/edit_study.json
uv run ta-eval truncation-study                    # results/truncation_study.json
scripts/run_models.sh --dry-run                    # the model arm (do not run without the GPU)
```

`results/safety_policy_v1.json` and `v2.json` were produced by the policy as of commits
`cf7b9eb` (v1) and `0b0b065` (v2); check those out and run `ta-eval safety` to reproduce.
