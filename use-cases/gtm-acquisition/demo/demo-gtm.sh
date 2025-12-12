#!/bin/bash
# Demo: GTM & Acquisition Agent - SDR Outbound Optimization
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
echo -e "${BLUE}${BOLD}Cold outreach -> high-quality emails, auto-optimized${RESET}"
echo -e "${DIM}GTM & Acquisition Agent | LLM-as-Judge Evaluation${RESET}"
echo ""
sleep 0.5

echo -e "$ ${YELLOW}python use-cases/gtm-acquisition/eval/evaluator.py${RESET}"
sleep 0.5
echo ""

echo -e "${CYAN}${BOLD}OPTIMIZATION RUN${RESET}: Testing Different Prompt Configurations"
echo -e "Configs: ${BOLD}5${RESET} | Examples: ${BOLD}10${RESET} | Dataset: ${BOLD}218 lead profiles${RESET}"
echo ""
sleep 0.5

echo -e "${DIM}[1/5]${RESET} baseline       ... quality=${YELLOW}0.67${RESET} compliance=${RED}84%${RESET}"
sleep 0.3
echo -e "${DIM}[2/5]${RESET} formal         ... quality=${YELLOW}0.71${RESET} compliance=${YELLOW}89%${RESET}"
sleep 0.3
echo -e "${DIM}[3/5]${RESET} empathy-first  ... quality=${GREEN}0.84${RESET} compliance=${GREEN}97%${RESET} ${GREEN}<- best!${RESET}"
sleep 0.3
echo -e "${DIM}[4/5]${RESET} aggressive     ... quality=${YELLOW}0.76${RESET} compliance=${RED}72%${RESET}"
sleep 0.3
echo -e "${DIM}[5/5]${RESET} concise        ... quality=${YELLOW}0.73${RESET} compliance=${GREEN}95%${RESET}"
echo ""
sleep 0.5

echo -e "${BOLD}RESULTS TABLE${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo -e "Config          Quality    Compliance    Overall"
echo -e "${DIM}----------------------------------------------------------------${RESET}"
echo -e "baseline        ${DIM}████████████░░░░░░░░${RESET} 0.67    84%         ${YELLOW}0.76${RESET}"
echo -e "empathy-first   ${GREEN}████████████████░░░░${RESET} 0.84    97%         ${GREEN}${BOLD}0.91${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo ""
sleep 0.5

echo -e "${GREEN}${BOLD}BEST: empathy-first${RESET} (score=${GREEN}0.91${RESET})"
echo -e "  ${GREEN}+0.15 overall${RESET} | Compliance ${GREEN}84% -> 97%${RESET}"
echo ""
