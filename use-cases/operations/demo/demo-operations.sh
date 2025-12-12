#!/bin/bash
# Demo: Operations Agent - Workflow Automation Optimization
set -e
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
echo -e "${BLUE}${BOLD}Ops workflows that match ground truth - hands-off${RESET}"
echo -e "${DIM}Operations Agent | Ground-Truth Comparison${RESET}"
echo ""
sleep 0.5

echo -e "$ ${YELLOW}python use-cases/operations/eval/evaluator.py${RESET}"
sleep 0.5
echo ""

echo -e "${CYAN}${BOLD}OPTIMIZATION RUN${RESET}: Testing Different Agent Configurations"
echo -e "Configs: ${BOLD}5${RESET} | Examples: ${BOLD}10${RESET} | Dataset: ${BOLD}209 task scenarios${RESET}"
echo ""
sleep 0.5

echo -e "${DIM}[1/5]${RESET} baseline  ... action=${YELLOW}0.72${RESET} escalation=${RED}78%${RESET}"
sleep 0.3
echo -e "${DIM}[2/5]${RESET} cautious  ... action=${YELLOW}0.79${RESET} escalation=${YELLOW}85%${RESET}"
sleep 0.3
echo -e "${DIM}[3/5]${RESET} minimal   ... action=${GREEN}0.88${RESET} escalation=${GREEN}96%${RESET} ${GREEN}<- best!${RESET}"
sleep 0.3
echo -e "${DIM}[4/5]${RESET} verbose   ... action=${YELLOW}0.81${RESET} escalation=${YELLOW}88%${RESET}"
sleep 0.3
echo -e "${DIM}[5/5]${RESET} strict    ... action=${YELLOW}0.83${RESET} escalation=${GREEN}92%${RESET}"
echo ""
sleep 0.5

echo -e "${BOLD}RESULTS TABLE${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo -e "Config      Action Acc    Escalation    Overall"
echo -e "${DIM}----------------------------------------------------------------${RESET}"
echo -e "baseline    ${DIM}██████████████░░░░░░${RESET} 0.72      78%         ${YELLOW}0.75${RESET}"
echo -e "minimal     ${GREEN}█████████████████░░░${RESET} 0.88      96%         ${GREEN}${BOLD}0.92${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo ""
sleep 0.5

echo -e "${GREEN}${BOLD}BEST: minimal${RESET} (score=${GREEN}0.92${RESET})"
echo -e "  Escalation ${GREEN}78% -> 96%${RESET} | Fewer steps, higher accuracy"
echo ""
