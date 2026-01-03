#!/bin/bash

# Traigent Interactive Walkthrough Script
# This script provides an interactive learning experience for Traigent SDK

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
README_FILE="$SCRIPT_DIR/README.md"
EXAMPLES_DIR="$SCRIPT_DIR/examples"
PROGRESS_FILE="$SCRIPT_DIR/.walkthrough_progress"
TRAIGENT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'

# Progress tracking variables
CURRENT_SECTION=1
TOTAL_SECTIONS=10
COMPLETED_EXAMPLES=""
SESSION_START=$(date +%s)

# ASCII Art Header
show_header() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ████████╗██████╗  █████╗ ██╗ ██████╗ ███████╗███╗   ██╗████████╗          ║
║  ╚══██╔══╝██╔══██╗██╔══██╗██║██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝          ║
║     ██║   ██████╔╝███████║██║██║  ███╗█████╗  ██╔██╗ ██║   ██║             ║
║     ██║   ██╔══██╗██╔══██║██║██║   ██║██╔══╝  ██║╚██╗██║   ██║             ║
║     ██║   ██║  ██║██║  ██║██║╚██████╔╝███████╗██║ ╚████║   ██║             ║
║     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝             ║
║                                                                              ║
║             🚀 Interactive Walkthrough - Learn by Doing! 🚀                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Progress bar display
show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local filled=$((percent / 5))
    local empty=$((20 - filled))

    echo -ne "${CYAN}Progress: ["
    printf '=%.0s' $(seq 1 $filled)
    echo -ne ">"
    printf ' %.0s' $(seq 1 $empty)
    echo -e "] ${percent}% - Chapter $current/$total${NC}"
}

# Load previous progress if exists
load_progress() {
    if [ -f "$PROGRESS_FILE" ]; then
        source "$PROGRESS_FILE"
        echo -e "${GREEN}✓ Previous progress loaded. Starting from Chapter $CURRENT_SECTION${NC}"
        echo -e "${YELLOW}  Completed examples: ${COMPLETED_EXAMPLES:-None}${NC}"
        echo ""
        read -p "Continue from Chapter $CURRENT_SECTION? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            CURRENT_SECTION=1
            COMPLETED_EXAMPLES=""
        fi
    else
        echo -e "${CYAN}ℹ Starting fresh walkthrough${NC}"
    fi
}

# Save current progress
save_progress() {
    cat > "$PROGRESS_FILE" << EOF
CURRENT_SECTION=$CURRENT_SECTION
COMPLETED_EXAMPLES="$COMPLETED_EXAMPLES"
LAST_RUN=$(date +%Y-%m-%dT%H:%M:%S)
SESSION_DURATION=$(($(date +%s) - SESSION_START))
EOF
    echo -e "${GREEN}✓ Progress saved${NC}"
}

# Extract and display a section from README
display_section() {
    local section_num=$1
    local section_content=""

    # Extract section content (between Chapter headers)
    section_content=$(awk "/^## 📖 Chapter $section_num:/, /^## 📖 Chapter $((section_num + 1)):/ {print}" "$README_FILE" | head -n -1)

    if [ -z "$section_content" ] && [ $section_num -eq $TOTAL_SECTIONS ]; then
        # Last section - extract until end of file or next major section
        section_content=$(awk "/^## 📖 Chapter $section_num:/, /^## 🎯 Quick Reference/ {print}" "$README_FILE" | head -n -1)
    fi

    # Display with pagination
    echo -e "${BOLD}${CYAN}"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    echo "$section_content" | less -R -P "Press SPACE to continue, q to finish reading..."
}

# Check for API keys
check_api_keys() {
    local has_keys=false

    echo -e "${CYAN}🔍 Checking for API keys...${NC}"
    echo ""

    if [ ! -z "$OPENAI_API_KEY" ]; then
        echo -e "${GREEN}✓ OpenAI API key detected${NC}"
        has_keys=true
    else
        echo -e "${YELLOW}⚠ OpenAI API key not found${NC}"
    fi

    if [ ! -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${GREEN}✓ Anthropic API key detected${NC}"
        has_keys=true
    else
        echo -e "${YELLOW}⚠ Anthropic API key not found${NC}"
    fi

    echo ""

    if [ "$has_keys" = false ]; then
        echo -e "${YELLOW}╔══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║                    ⚠️  NO API KEYS DETECTED              ║${NC}"
        echo -e "${YELLOW}╠══════════════════════════════════════════════════════════╣${NC}"
        echo -e "${YELLOW}║                                                          ║${NC}"
        echo -e "${YELLOW}║  You can still use MOCK mode to see how Traigent works! ║${NC}"
        echo -e "${YELLOW}║                                                          ║${NC}"
        echo -e "${YELLOW}║  To use real APIs, set your keys:                       ║${NC}"
        echo -e "${YELLOW}║    export OPENAI_API_KEY=\"sk-...\"                       ║${NC}"
        echo -e "${YELLOW}║    export ANTHROPIC_API_KEY=\"sk-ant-...\"                ║${NC}"
        echo -e "${YELLOW}║                                                          ║${NC}"
        echo -e "${YELLOW}╚══════════════════════════════════════════════════════════╝${NC}"
        echo ""
        return 1
    fi

    return 0
}

# Estimate cost for an example
estimate_cost() {
    local example_name=$1
    local dataset_size=10  # Default estimate
    local configs=9        # Default estimate

    # Rough cost estimates per model
    local gpt35_cost=0.002
    local gpt4mini_cost=0.008
    local gpt4_cost=0.04

    local total_cost=$(echo "scale=2; ($gpt35_cost + $gpt4mini_cost + $gpt4_cost) * $dataset_size" | bc)

    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    💰 COST ESTIMATE                          ║${NC}"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║  Example: $example_name                                      ║${NC}"
    echo -e "${CYAN}║  Dataset size: ~$dataset_size examples                       ║${NC}"
    echo -e "${CYAN}║  Configurations to test: $configs                            ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║  Estimated costs by model:                                   ║${NC}"
    echo -e "${CYAN}║    • GPT-3.5-turbo: ~\$$(printf "%.3f" $gpt35_cost)         ║${NC}"
    echo -e "${CYAN}║    • GPT-4o-mini:   ~\$$(printf "%.3f" $gpt4mini_cost)      ║${NC}"
    echo -e "${CYAN}║    • GPT-4o:        ~\$$(printf "%.3f" $gpt4_cost)          ║${NC}"
    echo -e "${CYAN}║  ────────────────────────────────────────────────           ║${NC}"
    echo -e "${YELLOW}║  TOTAL ESTIMATED: ~\$$total_cost                            ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${YELLOW}║  ⚠️  Actual costs may vary based on token usage              ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    read -p "Do you want to proceed with REAL API calls? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Run example in mock mode
run_example_mock() {
    local example_num=$1
    local example_file="$EXAMPLES_DIR/$(printf "%02d" $example_num)_*.py"

    # Check if example exists
    if ! ls $example_file 1> /dev/null 2>&1; then
        echo -e "${YELLOW}ℹ Example $example_num not yet available${NC}"
        return
    fi

    example_file=$(ls $example_file | head -1)
    local example_name=$(basename "$example_file" .py)

    echo -e "${GREEN}🔬 Running Example in MOCK mode: $example_name${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Run with mock mode enabled
    cd "$TRAIGENT_ROOT"
    TRAIGENT_MOCK_LLM=true python "$example_file" 2>&1 | while IFS= read -r line; do
        # Color code the output
        if [[ $line == *"Best"* ]] || [[ $line == *"✅"* ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ $line == *"Testing"* ]] || [[ $line == *"Running"* ]]; then
            echo -e "${CYAN}$line${NC}"
        elif [[ $line == *"Error"* ]] || [[ $line == *"Failed"* ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ $line == *"Warning"* ]]; then
            echo -e "${YELLOW}$line${NC}"
        else
            echo "$line"
        fi
    done

    # Mark as completed
    COMPLETED_EXAMPLES="$COMPLETED_EXAMPLES $example_num"

    echo ""
    echo -e "${GREEN}✓ Example completed!${NC}"
    echo ""
    read -p "Press ENTER to continue..."
}

# Run example with real API
run_example_real() {
    local example_num=$1
    local example_file="$EXAMPLES_DIR/$(printf "%02d" $example_num)_*.py"

    # Check if example exists
    if ! ls $example_file 1> /dev/null 2>&1; then
        echo -e "${YELLOW}ℹ Example $example_num not yet available${NC}"
        return
    fi

    example_file=$(ls $example_file | head -1)
    local example_name=$(basename "$example_file" .py)

    # Check API keys first
    if ! check_api_keys; then
        echo -e "${YELLOW}Falling back to MOCK mode...${NC}"
        sleep 2
        run_example_mock $example_num
        return
    fi

    # Show cost estimate
    if ! estimate_cost "$example_name"; then
        echo -e "${YELLOW}Cancelled. You can try MOCK mode instead.${NC}"
        return
    fi

    echo -e "${GREEN}🚀 Running Example with REAL APIs: $example_name${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Run with real APIs
    cd "$TRAIGENT_ROOT"
    python "$example_file" 2>&1 | while IFS= read -r line; do
        # Color code the output
        if [[ $line == *"Best"* ]] || [[ $line == *"✅"* ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ $line == *"Testing"* ]] || [[ $line == *"Running"* ]]; then
            echo -e "${CYAN}$line${NC}"
        elif [[ $line == *"cost"* ]] || [[ $line == *"$"* ]]; then
            echo -e "${YELLOW}$line${NC}"
        elif [[ $line == *"Error"* ]] || [[ $line == *"Failed"* ]]; then
            echo -e "${RED}$line${NC}"
        else
            echo "$line"
        fi
    done

    # Mark as completed
    COMPLETED_EXAMPLES="$COMPLETED_EXAMPLES $example_num"

    echo ""
    echo -e "${GREEN}✓ Example completed with real APIs!${NC}"
    echo ""
    read -p "Press ENTER to continue..."
}

# Show interactive menu
show_menu() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   What would you like to do?                 ║${NC}"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║  [1] 🧪 Run example in MOCK mode (safe, no API costs)        ║${NC}"
    echo -e "${CYAN}║  [2] 🚀 Run example with REAL APIs (costs apply)             ║${NC}"
    echo -e "${CYAN}║  [3] ⏭️  Skip to next section                                ║${NC}"
    echo -e "${CYAN}║  [4] 📚 Review previous section                              ║${NC}"
    echo -e "${CYAN}║  [5] 🎯 Jump to specific chapter (1-$TOTAL_SECTIONS)                     ║${NC}"
    echo -e "${CYAN}║  [6] 💾 Save progress and exit                               ║${NC}"
    echo -e "${CYAN}║  [7] ❌ Exit without saving                                  ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    read -p "Your choice [1-7]: " -n 1 -r choice
    echo ""

    case $choice in
        1)
            # Extract example number from current section
            local example_num=$CURRENT_SECTION
            if [ $example_num -le 10 ]; then
                run_example_mock $example_num
            else
                echo -e "${YELLOW}No example for this section${NC}"
            fi
            ;;
        2)
            # Extract example number from current section
            local example_num=$CURRENT_SECTION
            if [ $example_num -le 10 ]; then
                run_example_real $example_num
            else
                echo -e "${YELLOW}No example for this section${NC}"
            fi
            ;;
        3)
            if [ $CURRENT_SECTION -lt $TOTAL_SECTIONS ]; then
                CURRENT_SECTION=$((CURRENT_SECTION + 1))
            else
                echo -e "${GREEN}You've completed the walkthrough!${NC}"
                show_completion
                exit 0
            fi
            ;;
        4)
            if [ $CURRENT_SECTION -gt 1 ]; then
                CURRENT_SECTION=$((CURRENT_SECTION - 1))
            else
                echo -e "${YELLOW}Already at the beginning${NC}"
            fi
            ;;
        5)
            echo ""
            read -p "Enter chapter number (1-$TOTAL_SECTIONS): " -r chapter
            if [[ $chapter =~ ^[0-9]+$ ]] && [ $chapter -ge 1 ] && [ $chapter -le $TOTAL_SECTIONS ]; then
                CURRENT_SECTION=$chapter
            else
                echo -e "${RED}Invalid chapter number${NC}"
            fi
            ;;
        6)
            save_progress
            echo -e "${GREEN}Thanks for learning Traigent! See you next time! 👋${NC}"
            exit 0
            ;;
        7)
            echo -e "${YELLOW}Exiting without saving progress${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            ;;
    esac
}

# Show completion message
show_completion() {
    clear
    echo -e "${GREEN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        🎉 CONGRATULATIONS! YOU'VE COMPLETED THE TRAIGENT WALKTHROUGH! 🎉     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  You've learned:                                                            ║
║  ✅ How Traigent optimizes AI applications without code changes             ║
║  ✅ Different optimization modes (seamless vs parameter)                    ║
║  ✅ Multi-objective optimization strategies                                 ║
║  ✅ Privacy and execution modes                                             ║
║  ✅ RAG optimization techniques                                             ║
║  ✅ Custom evaluation metrics                                               ║
║  ✅ Cost control and performance optimization                               ║
║  ✅ Building production-ready AI agents                                     ║
║                                                                              ║
║  🚀 Next Steps:                                                             ║
║  • Explore more examples in the /examples directory                         ║
║  • Read the architecture documentation                                      ║
║  • Join the Traigent community                                              ║
║  • Start optimizing your own AI applications!                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    # Show statistics
    local session_duration=$(($(date +%s) - SESSION_START))
    local minutes=$((session_duration / 60))

    echo -e "${CYAN}📊 Your Learning Statistics:${NC}"
    echo -e "   Time spent: $minutes minutes"
    echo -e "   Examples completed: $COMPLETED_EXAMPLES"
    echo ""
}

# Check dependencies
check_dependencies() {
    echo -e "${CYAN}🔍 Checking dependencies...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
        exit 1
    fi

    # Check if Traigent is installed
    if ! python -c "import traigent" &> /dev/null; then
        echo -e "${YELLOW}⚠ Traigent not installed. Installing...${NC}"
        cd "$TRAIGENT_ROOT"
        pip install -e . > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Failed to install Traigent${NC}"
            echo "Please run: pip install -e ."
            exit 1
        fi
    fi

    echo -e "${GREEN}✓ All dependencies satisfied${NC}"
    echo ""
}

# Main loop
main_loop() {
    while true; do
        clear
        show_header
        show_progress $CURRENT_SECTION $TOTAL_SECTIONS
        echo ""

        echo -e "${BOLD}${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${BOLD}${MAGENTA}                    CHAPTER $CURRENT_SECTION OF $TOTAL_SECTIONS                     ${NC}"
        echo -e "${BOLD}${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
        echo ""

        # Display current section
        display_section $CURRENT_SECTION

        # Show menu
        show_menu
    done
}

# Main execution
main() {
    show_header
    echo -e "${CYAN}Welcome to the Traigent Interactive Walkthrough!${NC}"
    echo -e "${CYAN}This will guide you through learning Traigent step by step.${NC}"
    echo ""

    # Check dependencies
    check_dependencies

    # Load previous progress
    load_progress

    echo ""
    echo -e "${GREEN}Let's begin your journey! 🚀${NC}"
    echo ""
    read -p "Press ENTER to start..."

    # Start the main loop
    main_loop
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Interrupted. Saving progress...${NC}"; save_progress; exit 0' INT

# Run the main function
main
