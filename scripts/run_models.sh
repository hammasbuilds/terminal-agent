#!/usr/bin/env bash
# Run the model arm end to end: qwen2.5-coder:14b on every validated task, both suites.
#
#   scripts/run_models.sh --dry-run    list the jobs and the model-call bound, run nothing
#   scripts/run_models.sh              check resources, then run (resumable: re-run to continue)
#
# Environment: MODEL (default qwen2.5-coder:14b), OLLAMA_HOST (default http://127.0.0.1:11434),
# MAX_STEPS (default 30), RETRIES (default 2), MIN_FREE_GB (default 5),
# MIN_FREE_VRAM_MB (default 11000), LOCAL_ON_HOST (default 0).
#
# RISK: the agent runs in `auto` approval mode, where the policy lets mutating commands run
# unasked - including `python x.py`, `make`, and tests whose conftest.py the model just
# wrote. The policy is a filter, not a sandbox. Every task's shell therefore runs in a
# network-less container (the SWE-bench image, or python:3.11-bookworm for the local
# suite); this script refuses to run the local suite without Docker unless you set
# LOCAL_ON_HOST=1, which runs the model's commands on THIS machine.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen2.5-coder:14b}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MAX_STEPS="${MAX_STEPS:-30}"
RETRIES="${RETRIES:-2}"
LOCAL_ON_HOST="${LOCAL_ON_HOST:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
MIN_FREE_VRAM_MB="${MIN_FREE_VRAM_MB:-11000}"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
unset VIRTUAL_ENV

run() { uv run --offline ta-eval "$@"; }
HOST_FLAG=()
[[ "$LOCAL_ON_HOST" == "1" ]] && HOST_FLAG=(--local-on-host)

echo "== job list"
run model-run --suite local --model "$MODEL" --max-steps "$MAX_STEPS" --retries "$RETRIES"   "${HOST_FLAG[@]}" --dry-run
run model-run --suite swebench --model "$MODEL" --max-steps "$MAX_STEPS" --retries "$RETRIES"   --dry-run
if [[ $DRY -eq 1 ]]; then
  echo "(dry run: nothing executed)"
  exit 0
fi

echo "== resource checks"
free_gb=$(powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" 2>/dev/null \
  || awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "free RAM: ${free_gb} GB (need ${MIN_FREE_GB})"
if (( free_gb < MIN_FREE_GB )); then echo "not enough free RAM; aborting" >&2; exit 1; fi

if command -v nvidia-smi >/dev/null; then
  vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "free VRAM: ${vram} MB (need ${MIN_FREE_VRAM_MB})"
  if (( vram < MIN_FREE_VRAM_MB )); then echo "GPU is busy; aborting" >&2; exit 1; fi
fi

if ! curl -fsS "$HOST/api/tags" -o /tmp/ta_tags.json; then
  echo "cannot reach Ollama at $HOST; start it with 'ollama serve'" >&2; exit 1
fi
if ! grep -q "\"$MODEL\"" /tmp/ta_tags.json; then
  echo "model $MODEL is not pulled; run: ollama pull $MODEL" >&2; exit 1
fi

if docker info >/dev/null 2>&1; then
  echo "== SWE-bench images for validated tasks"
  uv run --offline python - <<'PY'
import json, pathlib, subprocess
from terminal_agent.evals.tasks import load_tasks
valid = {json.loads(p.read_text())["instance_id"] for p in pathlib.Path("results/validation/swebench").glob("*.json")
         if json.loads(p.read_text()).get("verdict") == "valid"}
for t in load_tasks():
    if t.instance_id in valid and subprocess.run(["docker", "image", "inspect", t.image], capture_output=True).returncode:
        print("pulling", t.image, flush=True)
        subprocess.run(["docker", "pull", "-q", t.image], check=True)
PY
  if ! docker image inspect python:3.11-bookworm >/dev/null 2>&1; then
    echo "pulling python:3.11-bookworm (the local suite's offline container)"
    docker pull -q python:3.11-bookworm
  fi
  SUITES="local swebench"
elif [[ "$LOCAL_ON_HOST" == "1" ]]; then
  echo "docker is not running: SWE-bench suite skipped; LOCAL_ON_HOST=1, so the local suite"
  echo "runs the model's commands ON THIS MACHINE"
  SUITES="local"
else
  echo "docker is not running: nothing can run isolated. Start Docker, or set LOCAL_ON_HOST=1" >&2
  echo "to run the local suite on this machine (auto mode runs the model's commands unasked)." >&2
  exit 1
fi

for suite in $SUITES; do
  echo "== model arm: $suite"
  extra=()
  [[ "$suite" == "local" ]] && extra=("${HOST_FLAG[@]}")
  OLLAMA_HOST="$HOST" run model-run --suite "$suite" --model "$MODEL" --host "$HOST"     --max-steps "$MAX_STEPS" --retries "$RETRIES" "${extra[@]}"
done
echo "done: results/model_run_*.json"
