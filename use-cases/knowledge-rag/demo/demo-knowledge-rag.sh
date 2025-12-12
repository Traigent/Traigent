#!/bin/bash
# Demo: Knowledge & RAG Agent - Document Q&A Optimization
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
echo -e "${BLUE}${BOLD}RAG you can trust: grounded answers or clean abstain${RESET}"
echo -e "${DIM}Knowledge & RAG Agent | Grounded Accuracy + Abstention${RESET}"
echo ""
sleep 0.5

echo -e "$ ${YELLOW}python use-cases/knowledge-rag/eval/evaluator.py${RESET}"
sleep 0.5
echo ""

echo -e "${RED}[HALLUCINATION]${RESET} Answer not supported by any document!"
sleep 0.3
echo ""

echo -e "${CYAN}${BOLD}OPTIMIZATION RUN${RESET}: Testing Different RAG Configurations"
echo -e "Configs: ${BOLD}5${RESET} | Examples: ${BOLD}10${RESET} | Dataset: ${BOLD}203 Q&A pairs (56 unanswerable)${RESET}"
echo ""
sleep 0.5

echo -e "${DIM}[1/5]${RESET} baseline    ... acc=${YELLOW}0.63${RESET} abstention=${RED}71%${RESET}"
sleep 0.3
echo -e "${DIM}[2/5]${RESET} detailed    ... acc=${YELLOW}0.68${RESET} abstention=${YELLOW}76%${RESET}"
sleep 0.3
echo -e "${DIM}[3/5]${RESET} strict      ... acc=${GREEN}0.79${RESET} abstention=${GREEN}93%${RESET} ${GREEN}<- best!${RESET}"
sleep 0.3
echo -e "${DIM}[4/5]${RESET} citations   ... acc=${YELLOW}0.74${RESET} abstention=${YELLOW}84%${RESET}"
sleep 0.3
echo -e "${DIM}[5/5]${RESET} concise     ... acc=${YELLOW}0.71${RESET} abstention=${YELLOW}80%${RESET}"
echo ""
sleep 0.5

echo -e "${BOLD}RESULTS TABLE${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo -e "Config      Accuracy    Abstention    Overall"
echo -e "${DIM}----------------------------------------------------------------${RESET}"
echo -e "baseline    ${DIM}████████████░░░░░░░░${RESET} 0.63      71%         ${YELLOW}0.67${RESET}"
echo -e "strict      ${GREEN}███████████████░░░░░${RESET} 0.79      93%         ${GREEN}${BOLD}0.86${RESET}"
echo -e "${DIM}================================================================${RESET}"
echo ""
sleep 0.5

echo -e "${GREEN}${BOLD}BEST: strict${RESET} (score=${GREEN}0.86${RESET})"
echo -e "  Hallucinations ${RED}down${RESET} | Abstention ${GREEN}71% -> 93%${RESET}"
echo ""
