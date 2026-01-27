#!/bin/bash

# Test Traigent walkthrough examples in mock or real mode
# Usage: ./test_all_examples.sh [--mock|--real]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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

# Default to mock mode
MODE="${1:---mock}"

# Example definitions
declare -A EXAMPLES=(
    ["01_tuning_qa.py"]="Basic QA Tuning|grid|10 trials x 20 examples|~0m 6s|~1m 34s"
    ["02_zero_code_change.py"]="Zero Code Change (Seamless)|random|10 trials x 20 examples|~0m 4s|~1m 18s"
    ["03_parameter_mode.py"]="Parameter Mode|random|10 trials x 20 examples|~0m 5s|~1m 16s"
    ["04_multi_objective.py"]="Multi-Objective|random|10 trials x 20 examples|~0m 11s|~1m 3s"
    ["05_rag_parallel.py"]="RAG Optimization (parallel eval)|random|10 trials x 20 examples|~0m 5s|~0m 55s"
    ["06_custom_evaluator.py"]="LLM-as-Judge|random|10 trials x 20 examples|~0m 4s|~1m 13s"
    ["07_privacy_modes.py"]="Privacy Modes (local-only)|mixed|20 trials x 20 examples|~0m 5s|~1m 44s"
)

EXAMPLE_ORDER=(
    "01_tuning_qa.py"
    "02_zero_code_change.py"
    "03_parameter_mode.py"
    "04_multi_objective.py"
    "05_rag_parallel.py"
    "06_custom_evaluator.py"
    "07_privacy_modes.py"
)

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Traigent Walkthrough Examples - ${1} Mode${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_example_info() {
    local example=$1
    local info="${EXAMPLES[$example]}"
    IFS='|' read -r name algo details mock_eta real_eta <<< "$info"

    echo -e "${BLUE}Example:${NC} $example"
    echo -e "${BLUE}Description:${NC} $name"
    echo -e "${BLUE}Algorithm:${NC} $algo"
    echo -e "${BLUE}Scope:${NC} $details"
    if [ "$MODE" == "--mock" ]; then
        echo -e "${BLUE}Estimated time:${NC} $mock_eta (mock)"
    else
        echo -e "${BLUE}Estimated time:${NC} $real_eta (real)"
    fi
}

confirm_example() {
    local example=$1

    if [ "$MODE" == "--mock" ]; then
        return 0  # No confirmation needed for mock mode
    fi

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    print_example_info "$example"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    read -p "Run this example? [Y=run / N=skip / Q=quit] " -n 1 -r
    echo ""

    case $REPLY in
        [Yy]) return 0 ;;
        [Nn]) return 1 ;;
        [Qq]) echo -e "${YELLOW}Exiting...${NC}"; exit 0 ;;
        *)
            echo -e "${YELLOW}Skipping (input was '${REPLY}'). Use Y to run, N to skip, Q to quit.${NC}"
            return 1
            ;;
    esac
}

run_example() {
    local example=$1
    local dir=$2
    local timeout_secs=$3

    echo -e "${BLUE}Running:${NC} $dir/$example"
    echo ""
    if [ -f "$dir/$example" ]; then
        # Run from the example directory so relative paths work
        pushd "$dir" > /dev/null
        # Avoid double-prompting under timeout for real Example 01; shell confirm happens first.
        if [ "$MODE" == "--real" ] && [ "$example" == "01_tuning_qa.py" ]; then
            TRAIGENT_REQUIRE_CONFIRM=false timeout "$timeout_secs" "$PYTHON" "$example" 2>&1
        else
            timeout "$timeout_secs" "$PYTHON" "$example" 2>&1
        fi
        local exit_code=${PIPESTATUS[0]}
        popd > /dev/null

        echo ""
        if [ $exit_code -eq 0 ]; then
            if [ "$MODE" != "--mock" ]; then
                echo -e "${GREEN}✅ $example - PASSED${NC}"
            fi
            return 0
        elif [ $exit_code -eq 124 ]; then
            echo -e "${RED}❌ $example - TIMEOUT${NC}"
            return 1
        else
            echo -e "${RED}❌ $example - FAILED (exit code: $exit_code)${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ $example - FILE NOT FOUND${NC}"
        return 1
    fi
}

# Main execution
case "$MODE" in
    --mock)
        print_header "MOCK"
        echo -e "${GREEN}Mock mode: No API keys needed, instant results${NC}"
        echo -e "${YELLOW}Info: For real LLM API calls, run the examples under walkthrough/examples/real/${NC}"
        echo ""

        export TRAIGENT_MOCK_LLM=true
        export TRAIGENT_OFFLINE_MODE=true
        export TRAIGENT_PAUSE_ON_ERROR=false
        export TRAIGENT_BATCH_MODE=true  # Skip estimated time in Python (shell shows it)
        : "${TRAIGENT_RESULTS_FOLDER:=$SCRIPT_DIR/.traigent_local}"
        export TRAIGENT_RESULTS_FOLDER
        export JOBLIB_TEMP_FOLDER="$TRAIGENT_RESULTS_FOLDER/joblib"
        mkdir -p "$JOBLIB_TEMP_FOLDER"
        export TRAIGENT_DATASET_ROOT="$SCRIPT_DIR"
        EXAMPLE_DIR="mock"
        TIMEOUT=60

        passed=0
        failed=0
        skipped=0

        for example in "${EXAMPLE_ORDER[@]}"; do
            echo ""
            echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}"
            print_example_info "$example"
            echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}"

            if run_example "$example" "$EXAMPLE_DIR" "$TIMEOUT"; then
                ((passed++))
            else
                ((failed++))
            fi
        done
        ;;

    --real)
        print_header "REAL"
        echo "Real mode: makes actual API calls (requires OPENAI_API_KEY for these examples)"
        echo ""

        # Check for API key
        if [ -f "real/.env" ]; then
            echo -e "${GREEN}Found .env file - loading environment...${NC}"
            set -a
            source real/.env
            set +a
        fi

        if [ -z "${OPENAI_API_KEY:-}" ]; then
            echo -e "${RED}ERROR: OPENAI_API_KEY not set${NC}"
            echo ""
            echo "Set it via environment variable or create real/.env with:"
            echo "  OPENAI_API_KEY=your-key-here"
            echo "  # Optional safety controls (not required to run):"
            echo "  TRAIGENT_COST_APPROVED=true   # skips cost confirmation prompts"
            echo "  TRAIGENT_RUN_COST_LIMIT=10    # soft spend cap in USD"
            echo ""
            echo "If you're not using real/.env, export in your terminal:"
            echo "  export OPENAI_API_KEY=your-key-here"
            echo "  # Optional safety controls (not required to run):"
            echo "  export TRAIGENT_COST_APPROVED=true   # skips cost confirmation prompts"
            echo "  export TRAIGENT_RUN_COST_LIMIT=10    # soft spend cap in USD"
            exit 1
        fi

        # Optional safety controls (skip prompts + set a soft spend cap)
        if [ -z "${TRAIGENT_COST_APPROVED:-}" ]; then
            export TRAIGENT_COST_APPROVED=true
        fi
        export TRAIGENT_RUN_COST_LIMIT=${TRAIGENT_RUN_COST_LIMIT:-10}
        export TRAIGENT_BATCH_MODE=true  # Skip estimated time in Python (shell shows it)
        : "${TRAIGENT_RESULTS_FOLDER:=$SCRIPT_DIR/.traigent_local}"
        export TRAIGENT_RESULTS_FOLDER
        export TRAIGENT_DATASET_ROOT="$SCRIPT_DIR"

        EXAMPLE_DIR="real"
        TIMEOUT=300  # 5 minutes for real API calls

        passed=0
        failed=0
        skipped=0

        for example in "${EXAMPLE_ORDER[@]}"; do
            if confirm_example "$example"; then
                echo ""
                if run_example "$example" "$EXAMPLE_DIR" "$TIMEOUT"; then
                    ((passed++))
                else
                    ((failed++))
                fi
            else
                echo -e "${YELLOW}⏭️  $example - SKIPPED${NC}"
                ((skipped++))
            fi
        done
        ;;

    *)
        echo "Usage: $0 [--mock|--real]"
        echo ""
        echo "  --mock  Run examples in mock mode (default, no API keys needed)"
        echo "  --real  Run examples with real API calls (requires OPENAI_API_KEY)"
        exit 1
        ;;
esac

# Summary
echo ""
if [ "$MODE" == "--mock" ]; then
    if [ $failed -eq 0 ]; then
        exit 0
    fi
    exit 1
fi

echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Summary${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $passed"
echo -e "  ${RED}Failed:${NC}  $failed"
if [ "$MODE" == "--real" ]; then
    echo -e "  ${YELLOW}Skipped:${NC} $skipped"
fi
echo ""

if [ $failed -eq 0 ] && [ $skipped -gt 0 ]; then
    echo -e "${YELLOW}No failures, but ${skipped} example(s) were skipped.${NC}"
    exit 0
elif [ $failed -eq 0 ]; then
    echo -e "${GREEN}All examples completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}Some examples failed. Check output above for details.${NC}"
    exit 1
fi
