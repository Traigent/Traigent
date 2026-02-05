#!/bin/bash
# =============================================================================
# Test Traigent Examples by Category
# =============================================================================
#
# This script runs Traigent examples to verify they work correctly.
# It supports running examples by specific categories/subfolders.
#
# USAGE:
#   ./test_all_examples.sh                         # Show help
#   ./test_all_examples.sh core                    # Run core examples
#   ./test_all_examples.sh advanced                # Run advanced examples
#   ./test_all_examples.sh docs                    # Run docs examples
#   ./test_all_examples.sh ragas                   # Run RAGAS examples only
#   ./test_all_examples.sh walkthrough             # Run walkthrough examples
#   ./test_all_examples.sh all                     # Run all examples
#   ./test_all_examples.sh --real core             # Real mode (needs API keys)
#
# CATEGORIES:
#   core       - Main examples demonstrating Traigent features (10 examples)
#   advanced   - Execution modes and specialized features
#   ragas      - RAGAS evaluation integration examples (3 examples)
#   docs       - Documentation inline examples (2 examples)
#   walkthrough - Walkthrough tutorial examples (8 examples)
#   all        - Run all categories
#
# MODE FLAGS:
#   --mock     - Use simulated LLM responses (default)
#   --real     - Make actual API calls (requires API keys)
#
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Find Python - prefer venv, fallback to python3
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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
MODE="--mock"
CATEGORY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --mock|--real)
            MODE="$1"
            shift
            ;;
        core|advanced|ragas|docs|walkthrough|all)
            CATEGORY="$1"
            shift
            ;;
        -h|--help|"")
            echo "Usage: $0 [--mock|--real] <category>"
            echo ""
            echo "Categories:"
            echo "  core        - Main Traigent feature examples (10 examples)"
            echo "  advanced    - Execution modes and specialized features"
            echo "  ragas       - RAGAS evaluation integration (3 examples)"
            echo "  docs        - Documentation inline examples (2 examples)"
            echo "  walkthrough - Tutorial walkthrough examples (8 examples)"
            echo "  all         - Run all categories"
            echo ""
            echo "Mode flags:"
            echo "  --mock      - Simulated LLM responses (default, no API keys)"
            echo "  --real      - Real API calls (requires API keys)"
            echo ""
            echo "Examples:"
            echo "  $0 core                    # Run core examples in mock mode"
            echo "  $0 ragas                   # Run RAGAS examples"
            echo "  $0 --real advanced         # Run advanced with real APIs"
            echo "  $0 all                     # Run everything"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

if [ -z "$CATEGORY" ]; then
    echo "ERROR: No category specified"
    echo "Run with --help for usage"
    exit 1
fi

# Core examples - main Traigent feature demonstrations
declare -a CORE_EXAMPLES=(
    "core/hello-world/run.py"
    "core/simple-prompt/run.py"
    "core/few-shot-classification/run.py"
    "core/prompt-style-optimization/run.py"
    "core/structured-output-json/run.py"
    "core/safety-guardrails/run.py"
    "core/token-budget-summarization/run.py"
    "core/prompt-ab-test/run.py"
    "core/chunking-long-context/run.py"
    "core/tool-use-calculator/run.py"
)

# RAGAS examples - evaluation integration (work in mock mode)
declare -a RAGAS_EXAMPLES=(
    "advanced/ragas/basics/run.py"
    "advanced/ragas/column_map/run.py"
    "advanced/ragas/with_llm/run.py"
)

# Advanced examples - execution modes (require API keys for most)
declare -a ADVANCED_EXAMPLES=(
    # Note: execution-modes require API keys (mock fallbacks are import-time only)
    # Uncomment if you have OPENAI_API_KEY set
    # "advanced/execution-modes/ex01-local-basic/run.py"
    # "advanced/execution-modes/ex02-local-privacy/run.py"
    # ...
)

# Docs examples - documentation inline (only those with runtime mock checks)
declare -a DOCS_EXAMPLES=(
    "docs/page-inline/core-concepts/ex01-configuration-spaces/run.py"
    "docs/page-inline/core-concepts/ex02-objectives-metrics/run.py"
)

# Walkthrough examples - tutorial series
declare -a WALKTHROUGH_EXAMPLES=(
    "../walkthrough/examples/mock/01_tuning_qa.py"
    "../walkthrough/examples/mock/02_zero_code_change.py"
    "../walkthrough/examples/mock/03_parameter_mode.py"
    "../walkthrough/examples/mock/04_multi_objective.py"
    "../walkthrough/examples/mock/05_rag_parallel.py"
    "../walkthrough/examples/mock/06_custom_evaluator.py"
    "../walkthrough/examples/mock/07_multi_provider.py"
    "../walkthrough/examples/mock/08_privacy_modes.py"
)

print_header() {
    echo ""
    echo -e "${CYAN}=================================================================${NC}"
    echo -e "${CYAN}  Traigent Examples: $1 (${2} Mode)${NC}"
    echo -e "${CYAN}=================================================================${NC}"
    echo ""
}

run_example() {
    local example=$1
    local timeout_secs=$2
    local example_path="$SCRIPT_DIR/$example"

    if [ ! -f "$example_path" ]; then
        echo -e "${RED}  SKIP${NC} $example (file not found)"
        return 2
    fi

    local example_dir="$(dirname "$example_path")"
    local example_name="$(basename "$example_path")"

    echo -e "${BLUE}  RUN${NC}  $example"

    # Run from the example directory so relative paths work
    pushd "$example_dir" > /dev/null
    output=$(timeout "$timeout_secs" "$PYTHON" "$example_name" 2>&1)
    local exit_code=$?
    popd > /dev/null

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}  PASS${NC} $example"
        return 0
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}  TIMEOUT${NC} $example (exceeded ${timeout_secs}s)"
        if [ -n "$output" ]; then
            echo "$output" | tail -20 | sed 's/^/        /'
        fi
        return 1
    else
        echo -e "${RED}  FAIL${NC} $example (exit code: $exit_code)"
        if [ -n "$output" ]; then
            echo "$output" | tail -30 | sed 's/^/        /'
        fi
        return 1
    fi
}

run_category() {
    local category_name=$1
    shift
    local -a examples=("$@")

    if [ ${#examples[@]} -eq 0 ]; then
        echo -e "${YELLOW}  No examples in this category${NC}"
        return
    fi

    for example in "${examples[@]}"; do
        if run_example "$example" "$TIMEOUT"; then
            ((passed++))
        elif [ $? -eq 2 ]; then
            ((skipped++))
        else
            ((failed++))
            failed_list+=("$example")
        fi
    done
}

# Setup environment
passed=0
failed=0
skipped=0
failed_list=()

case "$MODE" in
    --mock)
        export TRAIGENT_MOCK_LLM=true
        export TRAIGENT_OFFLINE_MODE=true
        export TRAIGENT_PAUSE_ON_ERROR=false
        export TRAIGENT_BATCH_MODE=true
        export TRAIGENT_COST_APPROVED=true
        : "${TRAIGENT_RESULTS_FOLDER:=$SCRIPT_DIR/.traigent_local}"
        export TRAIGENT_RESULTS_FOLDER
        export JOBLIB_TEMP_FOLDER="$TRAIGENT_RESULTS_FOLDER/joblib"
        mkdir -p "$JOBLIB_TEMP_FOLDER"
        export TRAIGENT_DATASET_ROOT="$REPO_ROOT"
        TIMEOUT=90
        MODE_NAME="MOCK"
        ;;

    --real)
        if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
            echo -e "${RED}ERROR: No API key found${NC}"
            echo "Set ANTHROPIC_API_KEY or OPENAI_API_KEY"
            exit 1
        fi
        export TRAIGENT_COST_APPROVED=true
        export TRAIGENT_RUN_COST_LIMIT=${TRAIGENT_RUN_COST_LIMIT:-10}
        export TRAIGENT_BATCH_MODE=true
        : "${TRAIGENT_RESULTS_FOLDER:=$SCRIPT_DIR/.traigent_local}"
        export TRAIGENT_RESULTS_FOLDER
        export TRAIGENT_DATASET_ROOT="$REPO_ROOT"
        TIMEOUT=300
        MODE_NAME="REAL"
        ;;
esac

# Run requested category
case "$CATEGORY" in
    core)
        print_header "Core Examples" "$MODE_NAME"
        run_category "Core" "${CORE_EXAMPLES[@]}"
        ;;
    ragas)
        print_header "RAGAS Examples" "$MODE_NAME"
        run_category "RAGAS" "${RAGAS_EXAMPLES[@]}"
        ;;
    advanced)
        print_header "Advanced Examples" "$MODE_NAME"
        run_category "Advanced" "${ADVANCED_EXAMPLES[@]}"
        run_category "RAGAS" "${RAGAS_EXAMPLES[@]}"
        ;;
    docs)
        print_header "Documentation Examples" "$MODE_NAME"
        run_category "Documentation" "${DOCS_EXAMPLES[@]}"
        ;;
    walkthrough)
        print_header "Walkthrough Examples" "$MODE_NAME"
        run_category "Walkthrough" "${WALKTHROUGH_EXAMPLES[@]}"
        ;;
    all)
        print_header "All Examples" "$MODE_NAME"
        echo -e "${YELLOW}--- Core ---${NC}"
        run_category "Core" "${CORE_EXAMPLES[@]}"
        echo ""
        echo -e "${YELLOW}--- RAGAS ---${NC}"
        run_category "RAGAS" "${RAGAS_EXAMPLES[@]}"
        echo ""
        echo -e "${YELLOW}--- Documentation ---${NC}"
        run_category "Documentation" "${DOCS_EXAMPLES[@]}"
        echo ""
        echo -e "${YELLOW}--- Walkthrough ---${NC}"
        run_category "Walkthrough" "${WALKTHROUGH_EXAMPLES[@]}"
        ;;
esac

# Summary
echo ""
echo -e "${CYAN}=================================================================${NC}"
echo -e "${CYAN}  Summary${NC}"
echo -e "${CYAN}=================================================================${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $passed"
echo -e "  ${RED}Failed:${NC}  $failed"
echo -e "  ${YELLOW}Skipped:${NC} $skipped"
echo ""

if [ ${#failed_list[@]} -gt 0 ]; then
    echo -e "${RED}Failed examples:${NC}"
    for ex in "${failed_list[@]}"; do
        echo "  - $ex"
    done
    echo ""
fi

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}All examples completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}Some examples failed. See output above for details.${NC}"
    exit 1
fi
