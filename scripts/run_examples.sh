#!/bin/bash
#
# Interactive Example Runner for Traigent
# ========================================
# Runs example scripts with configurable mock/real and offline/online modes.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Example lists
QUICKSTART_EXAMPLES=(
    "examples/quickstart/01_simple_qa.py"
    "examples/quickstart/02_customer_support_rag.py"
    "examples/quickstart/03_custom_objectives.py"
)

CORE_EXAMPLES=(
    "examples/core/rag-optimization/run.py"
    "examples/core/simple-prompt/run.py"
    "examples/core/prompt-style-optimization/run.py"
    "examples/core/structured-output-json/run.py"
    "examples/core/safety-guardrails/run.py"
    "examples/core/tool-use-calculator/run.py"
    "examples/core/few-shot-classification/run.py"
    "examples/core/prompt-ab-test/run.py"
    "examples/core/token-budget-summarization/run.py"
    "examples/core/chunking-long-context/run.py"
)

ADVANCED_EXAMPLES=(
    "examples/advanced/metric-registry/run.py"
    "examples/advanced/execution-modes/ex01-local-basic/run.py"
    "examples/advanced/execution-modes/ex02-local-privacy/run.py"
    "examples/advanced/execution-modes/ex03-hybrid-basic/run.py"
    "examples/advanced/ragas/basics/run.py"
    "examples/advanced/ragas/with_llm/run.py"
    "examples/advanced/ragas/column_map/run.py"
    "examples/advanced/ai-engineering-tasks/p0_structured_output/main.py"
    "examples/advanced/ai-engineering-tasks/p0_context_engineering/main.py"
    "examples/advanced/ai-engineering-tasks/p0_few_shot_selection/main.py"
)

DOCS_EXAMPLES=(
    "examples/gallery/page-inline/by-goal/accuracy-optimization/simple.py"
    "examples/gallery/page-inline/by-goal/cost-reduction/simple.py"
    "examples/gallery/page-inline/by-goal/reliability-robustness/simple.py"
    "examples/gallery/page-inline/by-goal/speed-latency/simple.py"
    "examples/gallery/page-inline/cookbook-nlp/sentiment/simple.py"
    "examples/gallery/page-inline/cookbook-data/extraction/simple.py"
)

# Combine all examples
ALL_EXAMPLES=(
    "${QUICKSTART_EXAMPLES[@]}"
    "${CORE_EXAMPLES[@]}"
    "${ADVANCED_EXAMPLES[@]}"
    "${DOCS_EXAMPLES[@]}"
)

# Global settings
LLM_MODE=""
BACKEND_MODE=""
MAX_EXAMPLES=50

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║          Traigent Interactive Example Runner                   ║${NC}"
    echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_separator() {
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
}

select_llm_mode() {
    echo -e "${BOLD}Step 1: Select LLM Mode${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} Mock Mode - Use simulated LLM responses (no API costs)"
    echo -e "  ${YELLOW}2)${NC} Real Mode - Use actual LLM APIs (requires API keys, incurs costs)"
    echo ""
    read -p "Enter choice [1-2]: " choice

    case $choice in
        1)
            LLM_MODE="mock"
            echo -e "${GREEN}✓ Mock Mode selected${NC}"
            ;;
        2)
            LLM_MODE="real"
            echo -e "${YELLOW}✓ Real Mode selected - You will be prompted before each example${NC}"
            ;;
        *)
            echo -e "${RED}Invalid choice. Defaulting to Mock Mode.${NC}"
            LLM_MODE="mock"
            ;;
    esac
    echo ""
}

select_backend_mode() {
    echo -e "${BOLD}Step 2: Select Backend Mode${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} Online Mode - Send results to Traigent backend"
    echo -e "  ${CYAN}2)${NC} Offline Mode - Run locally only (no backend connection)"
    echo ""
    read -p "Enter choice [1-2]: " choice

    case $choice in
        1)
            BACKEND_MODE="online"
            echo -e "${GREEN}✓ Online Mode selected - Results will be sent to Traigent backend${NC}"
            ;;
        2)
            BACKEND_MODE="offline"
            echo -e "${CYAN}✓ Offline Mode selected - Running locally only${NC}"
            ;;
        *)
            echo -e "${RED}Invalid choice. Defaulting to Offline Mode.${NC}"
            BACKEND_MODE="offline"
            ;;
    esac
    echo ""
}

select_example_count() {
    echo -e "${BOLD}Step 3: How many examples to run?${NC}"
    echo ""
    echo -e "  Total available: ${#ALL_EXAMPLES[@]} examples"
    echo -e "  ${YELLOW}Note: Maximum is 50. For more, run examples manually.${NC}"
    echo ""
    read -p "Enter number of examples to run [1-50, or 'all' for all up to 50]: " count

    if [[ "$count" == "all" ]]; then
        if [[ ${#ALL_EXAMPLES[@]} -gt $MAX_EXAMPLES ]]; then
            EXAMPLE_COUNT=$MAX_EXAMPLES
            echo -e "${YELLOW}Running first $MAX_EXAMPLES examples (limit reached)${NC}"
        else
            EXAMPLE_COUNT=${#ALL_EXAMPLES[@]}
        fi
    elif [[ "$count" =~ ^[0-9]+$ ]]; then
        if [[ $count -gt $MAX_EXAMPLES ]]; then
            echo -e "${YELLOW}Limiting to $MAX_EXAMPLES examples. Run additional examples manually.${NC}"
            EXAMPLE_COUNT=$MAX_EXAMPLES
        elif [[ $count -lt 1 ]]; then
            echo -e "${RED}Invalid count. Setting to 1.${NC}"
            EXAMPLE_COUNT=1
        else
            EXAMPLE_COUNT=$count
        fi
    else
        echo -e "${RED}Invalid input. Running 5 examples.${NC}"
        EXAMPLE_COUNT=5
    fi

    echo -e "${GREEN}✓ Will run $EXAMPLE_COUNT example(s)${NC}"
    echo ""
}

show_example_list() {
    echo -e "${BOLD}Examples to run:${NC}"
    echo ""
    for i in $(seq 0 $((EXAMPLE_COUNT - 1))); do
        if [[ $i -lt ${#ALL_EXAMPLES[@]} ]]; then
            echo -e "  ${CYAN}$((i + 1)).${NC} ${ALL_EXAMPLES[$i]}"
        fi
    done
    echo ""
}

run_example() {
    local example_path="$1"
    local example_num="$2"
    local total="$3"

    print_separator
    echo -e "${BOLD}${BLUE}Example $example_num of $total: ${NC}${example_path}"
    print_separator

    # Check if file exists
    local full_path="$PROJECT_ROOT/$example_path"
    if [[ ! -f "$full_path" ]]; then
        echo -e "${RED}ERROR: File not found: $full_path${NC}"
        return 1
    fi

    # Build environment variables
    local env_vars=""

    # LLM Mode
    if [[ "$LLM_MODE" == "mock" ]]; then
        env_vars="TRAIGENT_MOCK_LLM=true"
    else
        # Real mode - ask for confirmation and trial count
        echo ""
        echo -e "${YELLOW}Real LLM Mode: This will use actual API calls and incur costs.${NC}"
        read -p "Run this example? [y/N]: " confirm

        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            echo -e "${CYAN}Skipping example.${NC}"
            return 0
        fi

        read -p "Number of trials/examples (max 50, default 5): " trials
        if [[ -z "$trials" ]]; then
            trials=5
        elif [[ ! "$trials" =~ ^[0-9]+$ ]] || [[ $trials -gt 50 ]]; then
            echo -e "${YELLOW}Invalid or too high. Using 5 trials.${NC}"
            trials=5
        fi

        # Note: Trial count would need to be passed to the script somehow
        # For now, we just set the mode
        env_vars="TRAIGENT_MOCK_LLM=false"
    fi

    # Backend Mode
    if [[ "$BACKEND_MODE" == "offline" ]]; then
        env_vars="$env_vars TRAIGENT_OFFLINE_MODE=true"
    fi

    echo ""
    echo -e "${GREEN}Running with: $env_vars${NC}"
    echo ""

    # Run the example
    cd "$PROJECT_ROOT"
    if [[ -n "$env_vars" ]]; then
        env $env_vars .venv/bin/python "$example_path" 2>&1
    else
        .venv/bin/python "$example_path" 2>&1
    fi

    local exit_code=$?

    echo ""
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✓ Example completed successfully${NC}"
    else
        echo -e "${RED}✗ Example failed with exit code $exit_code${NC}"
    fi

    return $exit_code
}

wait_for_continue() {
    echo ""
    echo -e "${CYAN}Press Enter to continue to next example (or Ctrl+C to exit)...${NC}"
    read
}

main() {
    cd "$PROJECT_ROOT"

    print_header

    # Check if venv exists
    if [[ ! -f ".venv/bin/python" ]]; then
        echo -e "${RED}ERROR: Virtual environment not found at .venv/bin/python${NC}"
        echo -e "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
        exit 1
    fi

    # Configuration
    select_llm_mode
    select_backend_mode
    select_example_count

    print_separator
    show_example_list
    print_separator

    echo -e "${BOLD}Ready to start!${NC}"
    read -p "Press Enter to begin running examples..."

    # Run examples
    local success_count=0
    local fail_count=0
    local skip_count=0

    for i in $(seq 0 $((EXAMPLE_COUNT - 1))); do
        if [[ $i -lt ${#ALL_EXAMPLES[@]} ]]; then
            local example="${ALL_EXAMPLES[$i]}"

            if run_example "$example" "$((i + 1))" "$EXAMPLE_COUNT"; then
                ((success_count++)) || true
            else
                if [[ $? -eq 0 ]]; then
                    ((skip_count++)) || true
                else
                    ((fail_count++)) || true
                fi
            fi

            # Wait for user if not the last example
            if [[ $i -lt $((EXAMPLE_COUNT - 1)) ]]; then
                wait_for_continue
            fi
        fi
    done

    # Summary
    echo ""
    print_separator
    echo -e "${BOLD}${BLUE}Run Summary${NC}"
    print_separator
    echo -e "  ${GREEN}Successful:${NC} $success_count"
    echo -e "  ${RED}Failed:${NC}     $fail_count"
    echo -e "  ${CYAN}Skipped:${NC}    $skip_count"
    echo -e "  ${BOLD}Total:${NC}      $EXAMPLE_COUNT"
    echo ""
    echo -e "${GREEN}Done!${NC}"
}

# Run main
main "$@"
