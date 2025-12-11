#!/bin/bash
# Demo: Traigent Agent Configuration Hooks
set -e
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

clear
echo -e "${BLUE}${BOLD}┌────────────────────────────────────────────────────────────────┐${RESET}"
echo -e "${BLUE}${BOLD}│  🛡️  TraiGent: Git Hooks for Agent Validation                   │${RESET}"
echo -e "${BLUE}${BOLD}└────────────────────────────────────────────────────────────────┘${RESET}"
echo ""
sleep 1

echo -e "${DIM}Prevent expensive mistakes ${BOLD}BEFORE${RESET}${DIM} they reach production!${RESET}"
echo ""
sleep 0.5

echo -e "Available hooks:"
echo -e "  ${CYAN}traigent-validate${RESET}     ${DIM}─${RESET} Validates agent config against constraints"
echo -e "  ${CYAN}traigent-performance${RESET}  ${DIM}─${RESET} Ensures config meets performance baseline"
echo -e "  ${CYAN}traigent-cost${RESET}         ${DIM}─${RESET} Enforces cost budget limits"
echo ""
sleep 2

echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Step 1: Install Traigent hooks${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "$ ${BOLD}traigent hooks install${RESET}"
sleep 0.5
echo -e "Installing Traigent Git hooks..."
echo -e "  ${GREEN}✓${RESET} Installed: .git/hooks/pre-push (traigent-validate)"
echo -e "  ${GREEN}✓${RESET} Installed: .git/hooks/pre-commit (traigent-config-check)"
echo ""
echo -e "${GREEN}Hooks installed successfully!${RESET}"
sleep 2

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Step 2: Define constraints in traigent.yml${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "${DIM}# traigent.yml - Agent configuration constraints${RESET}"
echo -e "${CYAN}validation${RESET}:"
echo -e "  ${CYAN}hooks${RESET}:"
echo -e "    ${CYAN}pre-push${RESET}:"
echo -e "      - traigent-validate"
echo -e "      - traigent-performance"
echo ""
echo -e "${CYAN}constraints${RESET}:"
echo -e "  ${DIM}# Cost constraints${RESET}"
echo -e "  ${GREEN}max_cost_per_query${RESET}: ${YELLOW}0.05${RESET}       ${DIM}# \$0.05 max per query${RESET}"
echo -e "  ${GREEN}max_monthly_budget${RESET}: ${YELLOW}1000${RESET}       ${DIM}# \$1000/month limit${RESET}"
echo ""
echo -e "  ${DIM}# Performance constraints${RESET}"
echo -e "  ${GREEN}min_accuracy${RESET}: ${CYAN}0.85${RESET}             ${DIM}# 85% minimum accuracy${RESET}"
echo -e "  ${GREEN}max_latency_ms${RESET}: ${CYAN}500${RESET}            ${DIM}# 500ms max response time${RESET}"
echo ""
echo -e "  ${DIM}# Model constraints${RESET}"
echo -e "  ${GREEN}allowed_models${RESET}:"
echo -e "    - gpt-4o-mini"
echo -e "    - gpt-4o"
echo -e "    - claude-3-haiku"
echo -e "  ${RED}blocked_models${RESET}:"
echo -e "    - gpt-4-32k                ${DIM}# Too expensive${RESET}"
sleep 3

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}# Step 3: Push a VALID configuration${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "${CYAN}@traigent.optimize${RESET}("
echo -e "    configuration_space={"
echo -e "        ${GREEN}\"model\"${RESET}: [\"gpt-4o-mini\"],      ${DIM}# Allowed model ✓${RESET}"
echo -e "        ${GREEN}\"temperature\"${RESET}: [0.1, 0.3]"
echo -e "    }"
echo -e ")"
echo -e "${BLUE}def${RESET} ${YELLOW}support_agent${RESET}(query: str) -> str:"
echo -e "    llm = ChatOpenAI(model=${GREEN}\"gpt-4o-mini\"${RESET}, temperature=${CYAN}0.1${RESET})"
echo -e "    ${MAGENTA}return${RESET} llm.invoke(query).content"
sleep 2

echo ""
echo -e "$ ${BOLD}git commit -m \"Add support agent\"${RESET}"
echo -e "$ ${BOLD}git push origin main${RESET}"
echo ""
sleep 0.5

echo -e "Running Traigent validation hooks..."
echo ""
sleep 0.3

echo -e "[${CYAN}traigent-validate${RESET}] Checking agent configurations..."
echo -e "  Found 1 decorated function: ${BOLD}support_agent${RESET}"
echo -e "  Model: gpt-4o-mini ............. ${GREEN}✓ ALLOWED${RESET}"
echo -e "  Est. cost: \$0.008/query ........ ${GREEN}✓ WITHIN BUDGET${RESET}"
sleep 0.5
echo ""
echo -e "[${CYAN}traigent-performance${RESET}] Checking performance baseline..."
echo -e "  Accuracy: 0.89 ................. ${GREEN}✓ PASSES${RESET} (min: 0.85)"
echo -e "  Latency: 120ms ................. ${GREEN}✓ PASSES${RESET} (max: 500ms)"
sleep 0.5
echo ""
echo -e "${GREEN}${BOLD}✓ All Traigent hooks passed!${RESET}"
echo -e "${DIM}Pushing to origin/main...${RESET}"
sleep 2

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${RED}${BOLD}# Step 4: Push an INVALID configuration${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "${CYAN}@traigent.optimize${RESET}("
echo -e "    configuration_space={"
echo -e "        ${RED}\"model\"${RESET}: [\"gpt-4-32k\"],        ${RED}# BLOCKED model!${RESET}"
echo -e "        ${RED}\"temperature\"${RESET}: [0.9]"
echo -e "    }"
echo -e ")"
echo -e "${BLUE}def${RESET} ${YELLOW}expensive_agent${RESET}(query: str) -> str:"
echo -e "    llm = ChatOpenAI(model=${RED}\"gpt-4-32k\"${RESET}, temperature=${CYAN}0.9${RESET})"
echo -e "    ${MAGENTA}return${RESET} llm.invoke(query).content"
sleep 2

echo ""
echo -e "$ ${BOLD}git commit -m \"Add expensive agent\"${RESET}"
echo -e "$ ${BOLD}git push origin main${RESET}"
echo ""
sleep 0.5

echo -e "Running Traigent validation hooks..."
echo ""
sleep 0.3

echo -e "[${CYAN}traigent-validate${RESET}] Checking agent configurations..."
echo -e "  Found 1 decorated function: ${BOLD}expensive_agent${RESET}"
echo ""
sleep 0.3
echo -e "${RED}┌────────────────────────────────────────────────────────────────┐${RESET}"
echo -e "${RED}│  ${BOLD}VALIDATION ERRORS${RESET}${RED}                                              │${RESET}"
echo -e "${RED}├────────────────────────────────────────────────────────────────┤${RESET}"
echo -e "${RED}│${RESET}  ${RED}✗${RESET} Model '${BOLD}gpt-4-32k${RESET}' is in ${RED}blocked_models${RESET} list          ${RED}│${RESET}"
echo -e "${RED}│${RESET}    Reason: Too expensive for production                       ${RED}│${RESET}"
echo -e "${RED}│${RESET}                                                                ${RED}│${RESET}"
echo -e "${RED}│${RESET}  ${RED}✗${RESET} Est. cost ${YELLOW}\$0.12/query${RESET} exceeds ${GREEN}max_cost_per_query${RESET}      ${RED}│${RESET}"
echo -e "${RED}│${RESET}    Limit: \$0.05/query                                         ${RED}│${RESET}"
echo -e "${RED}└────────────────────────────────────────────────────────────────┘${RESET}"
echo ""
sleep 0.5
echo -e "${RED}${BOLD}✗ PUSH REJECTED${RESET}: Agent configuration violates constraints"
echo ""
echo -e "${DIM}To bypass (not recommended): git push --no-verify${RESET}"
echo ""
echo -e "${BOLD}To fix:${RESET}"
echo -e "  1. Use an allowed model: ${GREEN}gpt-4o-mini${RESET}, ${GREEN}gpt-4o${RESET}, ${GREEN}claude-3-haiku${RESET}"
echo -e "  2. Or request budget increase in traigent.yml"
sleep 3

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}# Step 5: Fix and push successfully${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "${CYAN}@traigent.optimize${RESET}("
echo -e "    configuration_space={"
echo -e "        ${GREEN}\"model\"${RESET}: [\"gpt-4o-mini\", \"gpt-4o\"],  ${DIM}# Fixed!${RESET}"
echo -e "        ${GREEN}\"temperature\"${RESET}: [0.1, 0.5]"
echo -e "    }"
echo -e ")"
echo -e "${BLUE}def${RESET} ${YELLOW}fixed_agent${RESET}(query: str) -> str: ..."
sleep 2

echo ""
echo -e "$ ${BOLD}git push origin main${RESET}"
echo ""
sleep 0.5

echo -e "[${CYAN}traigent-validate${RESET}] Checking agent configurations..."
echo -e "  Model: gpt-4o-mini ............. ${GREEN}✓ ALLOWED${RESET}"
echo -e "  Model: gpt-4o .................. ${GREEN}✓ ALLOWED${RESET}"
echo -e "  Est. cost: \$0.008/query ........ ${GREEN}✓ WITHIN BUDGET${RESET}"
echo ""
echo -e "${GREEN}${BOLD}✓ All Traigent hooks passed!${RESET}"
echo -e "${DIM}Pushing to origin/main...${RESET}"
sleep 2

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}Summary: Traigent Git Hooks${RESET}"
echo ""
echo -e "  ${GREEN}✓${RESET} ${CYAN}Validate${RESET}     Block pushes with invalid agent configs"
echo -e "  ${GREEN}✓${RESET} ${CYAN}Performance${RESET}  Ensure accuracy/latency meet baseline"
echo -e "  ${GREEN}✓${RESET} ${CYAN}Cost${RESET}         Enforce budget limits before production"
echo ""
echo -e "  ${YELLOW}${BOLD}Prevent expensive mistakes BEFORE they reach production!${RESET}"
sleep 2
