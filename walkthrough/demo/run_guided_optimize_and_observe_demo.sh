#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PHASE_RUNNER="$SCRIPT_DIR/guided_optimize_and_observe.py"

MODE="mock"
SCALE="small"
OBSERVABILITY="backend"
FRONTEND_URL="${TRAIGENT_DEMO_FRONTEND_URL:-http://localhost:3002}"
ARTIFACTS_DIR="${TRAIGENT_GUIDED_DEMO_ARTIFACTS_DIR:-$SCRIPT_DIR/artifacts/guided_optimize_observe}"
ENV_FILE=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASELINE_RUNS="3"
POST_RUNS="3"
RUN_ID="guided-$(date -u +%Y%m%dT%H%M%SZ)"

usage() {
  cat <<'EOF'
Usage:
  bash walkthrough/demo/run_guided_optimize_and_observe_demo.sh [options]

Options:
  --mode <mock|real|auto>         Execution mode. Default: mock
  --scale <tiny|small|medium|large>
                                  Optimization scale. Default: small
  --observability <backend|memory|auto>
                                  Observability sink. Default: backend
  --frontend-url <url>            Frontend URL shown in prompts. Default: http://localhost:3002
  --artifacts-dir <path>          Directory for saved run artifacts
  --env-file <path>               Source this env file before running
  --python-bin <path>             Python interpreter. Default: python3
  --baseline-runs <n>             Baseline observed runs. Default: 3
  --post-runs <n>                 Post-best-config runs. Default: 3
  --run-id <value>                Stable run id for FE filtering
  --help                          Show this help

Required for FE-visible traces:
  TRAIGENT_BACKEND_URL or TRAIGENT_API_URL
  TRAIGENT_API_KEY

Optional for real mode:
  OPENAI_API_KEY and/or ANTHROPIC_API_KEY and/or GOOGLE_API_KEY
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --scale)
      SCALE="$2"
      shift 2
      ;;
    --frontend-url)
      FRONTEND_URL="$2"
      shift 2
      ;;
    --observability)
      OBSERVABILITY="$2"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --baseline-runs)
      BASELINE_RUNS="$2"
      shift 2
      ;;
    --post-runs)
      POST_RUNS="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Env file not found: $ENV_FILE" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

BACKEND_URL="${TRAIGENT_BACKEND_URL:-${TRAIGENT_API_URL:-}}"
if [[ "$OBSERVABILITY" == "backend" && ( -z "${BACKEND_URL}" || -z "${TRAIGENT_API_KEY:-}" ) ]]; then
  echo "This guided demo needs backend observability enabled." >&2
  echo "Set TRAIGENT_BACKEND_URL or TRAIGENT_API_URL, and TRAIGENT_API_KEY." >&2
  exit 1
fi

RUN_DIR="$ARTIFACTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

pause_for_enter() {
  local prompt="$1"
  read -r -p "$prompt" _
}

print_phase_summary() {
  local summary_path="$1"
  "$PYTHON_BIN" - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
print(f"Summary file: {path}")
print(f"Trace name: {data['trace_name']}")
print(f"Environment: {data['environment']}")
print("Tags: " + ", ".join(data["tags"]))
if data.get("active_config"):
    print("Active config: " + json.dumps(data["active_config"], sort_keys=True))
if data.get("best_config"):
    print("Best config: " + json.dumps(data["best_config"], sort_keys=True))
if data.get("best_metrics"):
    print("Best metrics: " + json.dumps(data["best_metrics"], sort_keys=True))
if data.get("frontend_experiment_url"):
    print(f"Experiment URL: {data['frontend_experiment_url']}")
print(f"Observability URL: {data['frontend_observability_url']}")
PY
}

run_phase() {
  local phase="$1"
  shift
  (
    cd "$SDK_ROOT"
    "$PYTHON_BIN" "$PHASE_RUNNER" \
      --phase "$phase" \
      --run-id "$RUN_ID" \
      --mode "$MODE" \
      --scale "$SCALE" \
      --observability "$OBSERVABILITY" \
      --artifacts-dir "$ARTIFACTS_DIR" \
      --frontend-url "$FRONTEND_URL" \
      "$@"
  )
}

echo "Guided Traigent observe + optimize demo"
echo "SDK root: $SDK_ROOT"
echo "Run id: $RUN_ID"
echo "Mode: $MODE"
echo "Scale: $SCALE"
echo "Observability: $OBSERVABILITY"
echo "Frontend: $FRONTEND_URL"
echo "Backend: ${BACKEND_URL:-not-configured}"
echo "Artifacts: $RUN_DIR"
echo
echo "What this does:"
echo "1. Emit baseline observed traces with a known default config."
echo "2. Run optimization and save the best config."
echo "3. Load and apply the best config, then emit new observed traces."
echo
echo "FE filters to use throughout:"
echo "  Tag: run:$RUN_ID"
echo "  Environment: walkthrough-guided-$MODE"
echo

pause_for_enter "Press Enter to run step 1: baseline observability... "
run_phase baseline --baseline-runs "$BASELINE_RUNS"
print_phase_summary "$RUN_DIR/baseline_summary.json"
echo
echo "Inspect now in the FE:"
echo "  Open $FRONTEND_URL/observability"
echo "  Filter by tag run:$RUN_ID"
echo "  Open a baseline trace and inspect record-execution-context"
echo "  You should see the default config: model=gpt-4o-mini, temperature=0.7, response_style=bullet"
echo

pause_for_enter "Press Enter after you inspect baseline traces to run step 2: optimization... "
run_phase optimize
print_phase_summary "$RUN_DIR/optimize_summary.json"
echo
echo "Inspect now in the FE:"
echo "  Open $FRONTEND_URL/experiments or the experiment URL above"
echo "  Confirm the best config differs from the baseline default config"
echo "  Keep the same run tag run:$RUN_ID for the observability side"
echo

pause_for_enter "Press Enter after you inspect optimization results to run step 3: post-optimization observability... "
run_phase post --post-runs "$POST_RUNS"
print_phase_summary "$RUN_DIR/post_summary.json"
echo
echo "Final FE check:"
echo "  Open $FRONTEND_URL/observability"
echo "  Filter by tag run:$RUN_ID"
echo "  Compare baseline traces vs post-best traces"
echo "  In record-execution-context, baseline should show the default config"
echo "  Post traces should show the applied best config from optimization"
echo
echo "Artifacts saved under: $RUN_DIR"
