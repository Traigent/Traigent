#!/bin/bash
# Demo: Customer Support Agent - ShopEasy Bot Optimization
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
echo -e "${BLUE}${BOLD}Support bots that resolve - and escalate only when needed${RESET}"
echo -e "${DIM}Customer Support Agent | Resolution + Tone + Escalation${RESET}"
echo ""
sleep 0.5

echo -e "$ ${YELLOW}python use-cases/customer-support/eval/evaluator.py${RESET}"
sleep 0.5
echo ""

echo -e "${RED}[FALSE ESCALATION]${RESET} Simple query - no escalation needed!"
sleep 0.3
echo ""

echo -e "${CYAN}${BOLD}OPTIMIZATION RUN${RESET}: Testing Different Support Agent Configs"
echo -e "Configs: ${BOLD}5${RESET} | Examples: ${BOLD}10${RESET} | Dataset: ${BOLD}309 scenarios (65 escalations)${RESET}"
echo ""
sleep 0.5

echo -e "${DIM}[1/5]${RESET} baseline    ... tone=${YELLOW}0.70${RESET} escalation=${RED}75%${RESET}"
sleep 0.3
echo -e "${DIM}[2/5]${RESET} formal      ... tone=${YELLOW}0.73${RESET} escalation=${YELLOW}79%${RESET}"
sleep 0.3
echo -e "${DIM}[3/5]${RESET} empathetic  ... tone=${GREEN}0.88${RESET} escalation=${GREEN}92%${RESET} ${GREEN}<- best!${RESET}"
sleep 0.3
echo -e "${DIM}[4/5]${RESET} concise     ... tone=${YELLOW}0.76${RESET} escalation=${YELLOW}83%${RESET}"
sleep 0.3
echo -e "${DIM}[5/5]${RESET} detailed    ... tone=${YELLOW}0.81${RESET} escalation=${YELLOW}87%${RESET}"
echo ""
sleep 0.5

echo -e "${BOLD}RESULTS TABLE${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo -e "Config       Tone     Escalation    Overall"
echo -e "${DIM}----------------------------------------------------------------${RESET}"
echo -e "baseline     ${DIM}██████████████░░░░░░${RESET} 0.70      75%         ${YELLOW}0.72${RESET}"
echo -e "empathetic   ${GREEN}█████████████████░░░${RESET} 0.88      92%         ${GREEN}${BOLD}0.90${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo ""
sleep 0.5

echo -e "${GREEN}${BOLD}BEST: empathetic${RESET} (score=${GREEN}0.90${RESET})"
echo -e "  Tone ${GREEN}0.70 -> 0.88${RESET} | Escalation ${GREEN}75% -> 92%${RESET}"
echo ""
