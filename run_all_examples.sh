#!/bin/bash
# =============================================================================
# Run ALL Traigent Examples — Sequential or Parallel
# =============================================================================
#
# USAGE:
#   ./run_all_examples.sh                    # Sequential, mock mode (default)
#   ./run_all_examples.sh --parallel         # Parallel, mock mode
#   ./run_all_examples.sh --real             # Sequential, real mode
#   ./run_all_examples.sh --parallel --real  # Parallel, real mode
#   ./run_all_examples.sh --jobs 4           # Parallel with 4 workers
#
# FLAGS:
#   --sequential   Run examples one at a time (default)
#   --parallel     Run examples concurrently (default: 4 workers)
#   --jobs N       Number of parallel workers (implies --parallel)
#   --mock         Use simulated LLM responses (default)
#   --real         Use real API calls (requires API keys)
#   --timeout N    Per-example timeout in seconds (default: 180 mock, 300 real)
#   --report FILE  Write JSON report to FILE (default: stdout summary only)
#   --verify       Snapshot backend before/after and diff to verify results logged
#                  NOTE: Only meaningful in --real mode (mock+offline skips backend)
#
# EXAMPLES:
#   ./run_all_examples.sh --mock                          # Quick smoke test
#   ./run_all_examples.sh --parallel --jobs 8 --mock      # Fast parallel mock
#   ./run_all_examples.sh --real --timeout 600             # Real with longer timeout
#   ./run_all_examples.sh --parallel --report results.json # Parallel with JSON report
#
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# --- Find Python ---
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "ERROR: No Python interpreter found"
    exit 1
fi

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# --- Parse arguments ---
MODE="mock"
EXECUTION="sequential"
JOBS=4
TIMEOUT=""
REPORT_FILE=""
VERIFY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --sequential)  EXECUTION="sequential"; shift ;;
        --parallel)    EXECUTION="parallel"; shift ;;
        --jobs)        EXECUTION="parallel"; JOBS="$2"; shift 2 ;;
        --mock)        MODE="mock"; shift ;;
        --real)        MODE="real"; shift ;;
        --timeout)     TIMEOUT="$2"; shift 2 ;;
        --report)      REPORT_FILE="$2"; shift 2 ;;
        --verify)      VERIFY=true; shift ;;
        -h|--help)
            sed -n '2,/^# =====/p' "$0" | head -n -1 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help)"
            exit 1
            ;;
    esac
done

# --- Set defaults ---
if [ -z "$TIMEOUT" ]; then
    if [ "$MODE" = "mock" ]; then TIMEOUT=180; else TIMEOUT=300; fi
fi

# --- All examples (paths relative to repo root) ---
declare -a ALL_EXAMPLES=(
    # === hello.py (repo root) ===
    "hello.py"

    # === Core (12) ===
    "examples/core/rag-optimization/run.py"
    "examples/core/simple-prompt/run.py"
    "examples/core/few-shot-classification/run.py"
    "examples/core/prompt-style-optimization/run.py"
    "examples/core/structured-output-json/run.py"
    "examples/core/safety-guardrails/run.py"
    "examples/core/token-budget-summarization/run.py"
    "examples/core/prompt-ab-test/run.py"
    "examples/core/chunking-long-context/run.py"
    "examples/core/tool-use-calculator/run.py"
    "examples/core/production-deployment/run.py"
    "examples/core/error-handling/run.py"

    # === Multi-Objective (5) ===
    "examples/core/multi-objective-tradeoff/run_anthropic.py"
    "examples/core/multi-objective-tradeoff/run_openai.py"
    "examples/core/multi-objective-tradeoff/run_openai_optuna.py"
    "examples/core/multi-objective-tradeoff/run_openai_optuna_concurrency.py"
    "examples/core/multi-objective-tradeoff/run_many_providers.py"

    # === Quickstart (3) ===
    "examples/quickstart/01_simple_qa.py"
    "examples/quickstart/02_customer_support_rag.py"
    "examples/quickstart/03_custom_objectives.py"

    # === TVL Tutorials (5) ===
    "examples/tvl/tutorials/01_getting_started/run_optimization.py"
    "examples/tvl/tutorials/02_typed_tvars/explore_tvars.py"
    "examples/tvl/tutorials/03_multi_objective/analyze_tradeoffs.py"
    "examples/tvl/tutorials/04_promotion_policy/test_promotion.py"
    "examples/tvl/tutorials/05_statistical_testing/tost_demo.py"

    # === RAGAS (3) ===
    "examples/advanced/ragas/basics/run.py"
    "examples/advanced/ragas/column_map/run.py"
    "examples/advanced/ragas/with_llm/run.py"

    # === Docs / Gallery (2) ===
    "examples/gallery/page-inline/core-concepts/ex01-configuration-spaces/run.py"
    "examples/gallery/page-inline/core-concepts/ex02-objectives-metrics/run.py"

    # === Walkthrough Mock (8) ===
    "walkthrough/mock/01_tuning_qa.py"
    "walkthrough/mock/02_zero_code_change.py"
    "walkthrough/mock/03_parameter_mode.py"
    "walkthrough/mock/04_multi_objective.py"
    "walkthrough/mock/05_rag_parallel.py"
    "walkthrough/mock/06_custom_evaluator.py"
    "walkthrough/mock/07_multi_provider.py"
    "walkthrough/mock/08_privacy_modes.py"

    # === Advanced Walkthrough Mock (5) ===
    "walkthrough/mock/advanced/01_tuned_variables.py"
    "walkthrough/mock/advanced/02_prompt_optimization.py"
    "walkthrough/mock/advanced/03_multi_agent.py"
    "walkthrough/mock/advanced/04_workflow_traces_demo.py"
    "walkthrough/mock/advanced/05_langgraph_multiagent_demo.py"
)

TOTAL=${#ALL_EXAMPLES[@]}

# --- Environment setup ---
export TRAIGENT_DATASET_ROOT="$REPO_ROOT"
export TRAIGENT_BATCH_MODE=true
export TRAIGENT_COST_APPROVED=true
export TRAIGENT_PAUSE_ON_ERROR=false

if [ "$MODE" = "mock" ]; then
    export TRAIGENT_MOCK_LLM=true
    export TRAIGENT_OFFLINE_MODE=true
    RESULTS_DIR="$REPO_ROOT/.traigent_validation_results"
else
    if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
        echo -e "${RED}ERROR: No API key found for real mode${NC}"
        echo "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY"
        exit 1
    fi
    export TRAIGENT_MOCK_LLM=false
    export TRAIGENT_RUN_COST_LIMIT=${TRAIGENT_RUN_COST_LIMIT:-10}
    RESULTS_DIR="$REPO_ROOT/.traigent_validation_results"
fi
: "${TRAIGENT_RESULTS_FOLDER:=$RESULTS_DIR}"
export TRAIGENT_RESULTS_FOLDER
mkdir -p "$TRAIGENT_RESULTS_FOLDER"
export JOBLIB_TEMP_FOLDER="$TRAIGENT_RESULTS_FOLDER/joblib"
mkdir -p "$JOBLIB_TEMP_FOLDER"

# --- Temp directory for per-example logs ---
LOG_DIR=$(mktemp -d)
trap "rm -rf $LOG_DIR" EXIT

# --- Run a single example ---
# Writes result to $LOG_DIR/<index>.result (PASS|FAIL|SKIP|TIMEOUT)
# and output to $LOG_DIR/<index>.log
run_one() {
    local idx=$1
    local example=$2
    local example_path="$REPO_ROOT/$example"

    if [ ! -f "$example_path" ]; then
        echo "SKIP" > "$LOG_DIR/$idx.result"
        echo "File not found: $example_path" > "$LOG_DIR/$idx.log"
        return
    fi

    local example_dir
    example_dir="$(dirname "$example_path")"
    local example_name
    example_name="$(basename "$example_path")"

    local start_ts
    start_ts=$(date +%s)

    # Run from the example's directory so relative paths work
    local output
    output=$(cd "$example_dir" && timeout "$TIMEOUT" "$PYTHON" "$example_name" 2>&1)
    local rc=$?

    local end_ts
    end_ts=$(date +%s)
    local duration=$(( end_ts - start_ts ))

    echo "$duration" > "$LOG_DIR/$idx.duration"
    echo "$output" > "$LOG_DIR/$idx.log"

    if [ $rc -eq 0 ]; then
        echo "PASS" > "$LOG_DIR/$idx.result"
    elif [ $rc -eq 77 ]; then
        echo "SKIP" > "$LOG_DIR/$idx.result"
    elif [ $rc -eq 124 ]; then
        echo "TIMEOUT" > "$LOG_DIR/$idx.result"
    else
        echo "FAIL" > "$LOG_DIR/$idx.result"
    fi
}

# --- Print header ---
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Traigent Example Runner — ${BOLD}${TOTAL} examples${NC}${CYAN} (${MODE} mode, ${EXECUTION})${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Python:    $($PYTHON --version 2>&1)"
echo -e "  Mode:      ${MODE}"
echo -e "  Execution: ${EXECUTION}$([ "$EXECUTION" = "parallel" ] && echo " (${JOBS} workers)")"
echo -e "  Timeout:   ${TIMEOUT}s per example"
echo ""

# --- Pre-run verification snapshot ---
if [ "$VERIFY" = true ]; then
    echo -e "${CYAN}  Taking backend snapshot (before)...${NC}"
    "$PYTHON" "$REPO_ROOT/scripts/verify_example_results.py" --snapshot before 2>/dev/null || echo -e "  ${YELLOW}Backend snapshot skipped (not reachable)${NC}"
    echo ""
fi

START_TIME=$(date +%s)

# --- Execute ---
if [ "$EXECUTION" = "parallel" ]; then
    echo -e "${BLUE}  Launching ${TOTAL} examples across ${JOBS} workers...${NC}"
    echo ""

    # Export function and vars for GNU parallel / xargs
    export -f run_one
    export REPO_ROOT TIMEOUT PYTHON LOG_DIR

    # Use background jobs with a semaphore
    active=0
    for i in "${!ALL_EXAMPLES[@]}"; do
        run_one "$i" "${ALL_EXAMPLES[$i]}" &
        active=$((active + 1))
        if [ "$active" -ge "$JOBS" ]; then
            wait -n 2>/dev/null || true
            active=$((active - 1))
        fi
    done
    wait
else
    for i in "${!ALL_EXAMPLES[@]}"; do
        example="${ALL_EXAMPLES[$i]}"
        printf "  [%2d/%d] %-65s " "$((i+1))" "$TOTAL" "$example"
        run_one "$i" "$example"

        result=$(cat "$LOG_DIR/$i.result" 2>/dev/null || echo "FAIL")
        duration=$(cat "$LOG_DIR/$i.duration" 2>/dev/null || echo "?")

        case "$result" in
            PASS)    echo -e "${GREEN}PASS${NC} (${duration}s)" ;;
            SKIP)    echo -e "${YELLOW}SKIP${NC}" ;;
            TIMEOUT) echo -e "${RED}TIMEOUT${NC} (>${TIMEOUT}s)" ;;
            FAIL)    echo -e "${RED}FAIL${NC} (${duration}s)" ;;
        esac
    done
fi

END_TIME=$(date +%s)
WALL_TIME=$(( END_TIME - START_TIME ))

# --- Collect results ---
passed=0
failed=0
skipped=0
timed_out=0
declare -a failed_list=()
declare -a timeout_list=()
declare -a skip_list=()

for i in "${!ALL_EXAMPLES[@]}"; do
    result=$(cat "$LOG_DIR/$i.result" 2>/dev/null || echo "FAIL")
    case "$result" in
        PASS)    passed=$((passed + 1)) ;;
        SKIP)    skipped=$((skipped + 1)); skip_list+=("${ALL_EXAMPLES[$i]}") ;;
        TIMEOUT) timed_out=$((timed_out + 1)); timeout_list+=("${ALL_EXAMPLES[$i]}") ;;
        FAIL)    failed=$((failed + 1)); failed_list+=("${ALL_EXAMPLES[$i]}") ;;
    esac
done

# --- Print per-example results for parallel mode ---
if [ "$EXECUTION" = "parallel" ]; then
    for i in "${!ALL_EXAMPLES[@]}"; do
        example="${ALL_EXAMPLES[$i]}"
        result=$(cat "$LOG_DIR/$i.result" 2>/dev/null || echo "FAIL")
        duration=$(cat "$LOG_DIR/$i.duration" 2>/dev/null || echo "?")

        case "$result" in
            PASS)    echo -e "  ${GREEN}PASS${NC}    ${example} (${duration}s)" ;;
            SKIP)    echo -e "  ${YELLOW}SKIP${NC}    ${example}" ;;
            TIMEOUT) echo -e "  ${RED}TIMEOUT${NC} ${example} (>${TIMEOUT}s)" ;;
            FAIL)    echo -e "  ${RED}FAIL${NC}    ${example} (${duration}s)" ;;
        esac
    done
fi

# --- Summary ---
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Summary${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}   $passed"
echo -e "  ${RED}Failed:${NC}   $failed"
echo -e "  ${RED}Timeout:${NC}  $timed_out"
echo -e "  ${YELLOW}Skipped:${NC}  $skipped"
echo -e "  Total:    $TOTAL"
echo -e "  Wall time: ${WALL_TIME}s"
echo ""

if [ ${#failed_list[@]} -gt 0 ]; then
    echo -e "${RED}Failed examples:${NC}"
    for ex in "${failed_list[@]}"; do
        echo "  - $ex"
        tail -5 "$LOG_DIR/$(for i in "${!ALL_EXAMPLES[@]}"; do [ "${ALL_EXAMPLES[$i]}" = "$ex" ] && echo "$i"; done).log" 2>/dev/null | sed 's/^/      /'
    done
    echo ""
fi

if [ ${#timeout_list[@]} -gt 0 ]; then
    echo -e "${RED}Timed-out examples:${NC}"
    for ex in "${timeout_list[@]}"; do echo "  - $ex"; done
    echo ""
fi

# --- JSON report ---
if [ -n "$REPORT_FILE" ]; then
    echo "{" > "$REPORT_FILE"
    echo "  \"date\": \"$(date -Iseconds)\"," >> "$REPORT_FILE"
    echo "  \"mode\": \"$MODE\"," >> "$REPORT_FILE"
    echo "  \"execution\": \"$EXECUTION\"," >> "$REPORT_FILE"
    echo "  \"total\": $TOTAL," >> "$REPORT_FILE"
    echo "  \"passed\": $passed," >> "$REPORT_FILE"
    echo "  \"failed\": $failed," >> "$REPORT_FILE"
    echo "  \"timed_out\": $timed_out," >> "$REPORT_FILE"
    echo "  \"skipped\": $skipped," >> "$REPORT_FILE"
    echo "  \"wall_time_seconds\": $WALL_TIME," >> "$REPORT_FILE"
    echo "  \"results\": [" >> "$REPORT_FILE"
    for i in "${!ALL_EXAMPLES[@]}"; do
        result=$(cat "$LOG_DIR/$i.result" 2>/dev/null || echo "FAIL")
        duration=$(cat "$LOG_DIR/$i.duration" 2>/dev/null || echo "0")
        comma=","
        [ "$i" -eq $((TOTAL - 1)) ] && comma=""
        echo "    {\"example\": \"${ALL_EXAMPLES[$i]}\", \"status\": \"$result\", \"duration_seconds\": $duration}$comma" >> "$REPORT_FILE"
    done
    echo "  ]" >> "$REPORT_FILE"
    echo "}" >> "$REPORT_FILE"
    echo -e "  Report written to: ${BOLD}$REPORT_FILE${NC}"
    echo ""
fi

# --- Post-run verification ---
if [ "$VERIFY" = true ]; then
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Backend Verification${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}  Taking backend snapshot (after)...${NC}"
    "$PYTHON" "$REPO_ROOT/scripts/verify_example_results.py" --snapshot after 2>/dev/null || echo -e "  ${YELLOW}Backend snapshot skipped (not reachable)${NC}"
    "$PYTHON" "$REPO_ROOT/scripts/verify_example_results.py" --diff before after 2>/dev/null || echo -e "  ${YELLOW}Backend diff skipped${NC}"

    # Also store per-example local result snapshots
    SNAPSHOT_DIR="$REPO_ROOT/.validation_results/examples"
    mkdir -p "$SNAPSHOT_DIR"
    for i in "${!ALL_EXAMPLES[@]}"; do
        example="${ALL_EXAMPLES[$i]}"
        result=$(cat "$LOG_DIR/$i.result" 2>/dev/null || echo "UNKNOWN")
        duration=$(cat "$LOG_DIR/$i.duration" 2>/dev/null || echo "0")
        safe_name=$(echo "$example" | tr '/' '_' | tr ' ' '_')

        # Extract best config/score from output
        best_config=""
        best_score=""
        if [ -f "$LOG_DIR/$i.log" ]; then
            best_config=$(grep -i "best config" "$LOG_DIR/$i.log" 2>/dev/null | head -1 | sed 's/.*Best config[^:]*: *//' || true)
            best_score=$(grep -i "best score" "$LOG_DIR/$i.log" 2>/dev/null | head -1 | sed 's/.*Best score[^:]*: *//' || true)
        fi

        cat > "$SNAPSHOT_DIR/${safe_name}.json" <<SNAP
{
  "example": "$example",
  "status": "$result",
  "duration_seconds": $duration,
  "mode": "$MODE",
  "best_config": "$best_config",
  "best_score": "$best_score",
  "timestamp": "$(date -Iseconds)"
}
SNAP
    done
    echo -e "  Local result snapshots saved to: ${BOLD}.validation_results/examples/${NC}"
    echo ""
fi

# --- Exit code ---
if [ $failed -eq 0 ] && [ $timed_out -eq 0 ]; then
    echo -e "${GREEN}All examples completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}Some examples failed. Review output above.${NC}"
    exit 1
fi
