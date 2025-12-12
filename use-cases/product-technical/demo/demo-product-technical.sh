#!/bin/bash
# Demo: Product & Technical Agent - Code Generation Optimization
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
echo -e "${BLUE}${BOLD}Codegen that passes tests - then optimizes for elegance${RESET}"
echo -e "${DIM}Product & Technical Agent | Test Pass Rate + Code Quality${RESET}"
echo ""
sleep 0.5

echo -e "$ ${YELLOW}python use-cases/product-technical/eval/evaluator.py${RESET}"
sleep 0.5
echo ""

echo -e "${CYAN}${BOLD}OPTIMIZATION RUN${RESET}: Testing Different Code Generation Configs"
echo -e "Configs: ${BOLD}5${RESET} | Examples: ${BOLD}10${RESET} | Dataset: ${BOLD}126 coding tasks${RESET}"
echo ""
sleep 0.5

echo -e "${DIM}[1/5]${RESET} baseline  ... tests=${RED}2/4${RESET} quality=${YELLOW}0.70${RESET}"
sleep 0.3
echo -e "${DIM}[2/5]${RESET} verbose   ... tests=${YELLOW}3/4${RESET} quality=${YELLOW}0.75${RESET}"
sleep 0.3
echo -e "${DIM}[3/5]${RESET} minimal   ... tests=${GREEN}4/4${RESET} quality=${GREEN}0.84${RESET} ${GREEN}<- best!${RESET}"
sleep 0.3
echo -e "${DIM}[4/5]${RESET} defensive ... tests=${YELLOW}3/4${RESET} quality=${YELLOW}0.72${RESET}"
sleep 0.3
echo -e "${DIM}[5/5]${RESET} creative  ... tests=${YELLOW}3/4${RESET} quality=${YELLOW}0.78${RESET}"
echo ""
sleep 0.5

echo -e "${BOLD}RESULTS TABLE${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo -e "Config      Pass Rate    Quality    Overall"
echo -e "${DIM}----------------------------------------------------------------${RESET}"
echo -e "baseline    ${DIM}█████████████░░░░░░░${RESET}   68%       0.72       ${YELLOW}0.70${RESET}"
echo -e "minimal     ${GREEN}██████████████████░░${RESET}   92%       0.84       ${GREEN}${BOLD}0.90${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo ""
sleep 0.5

echo -e "${GREEN}${BOLD}BEST: minimal${RESET} (score=${GREEN}0.90${RESET})"
echo -e "  Tests ${GREEN}68% -> 92%${RESET} passing | Higher code quality"
echo ""
