#!/bin/bash
# Demo: Traigent LLM Agent Optimization
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
echo -e "${BLUE}${BOLD}│  🚀 Traigent: Responsible LLM Optimization                     │${RESET}"
echo -e "${BLUE}${BOLD}└────────────────────────────────────────────────────────────────┘${RESET}"
echo ""
sleep 1

echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Step 1: Set cost limits (Responsible Optimization)${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "$ ${YELLOW}export TRAIGENT_RUN_COST_LIMIT=2.0${RESET}  ${DIM}# Max \$2 per optimization run${RESET}"
sleep 1
echo ""

echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Step 2: Define your LLM agent with tunable parameters${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "${MAGENTA}import${RESET} traigent"
echo -e "${MAGENTA}from${RESET} langchain_openai ${MAGENTA}import${RESET} ChatOpenAI"
echo ""
echo -e "${CYAN}@traigent.optimize${RESET}("
echo -e "    ${DIM}# Define TUNABLE PARAMETERS - what to optimize${RESET}"
echo -e "    configuration_space={"
echo -e "        ${GREEN}\"model\"${RESET}: [\"gpt-3.5-turbo\", \"gpt-4o-mini\", \"gpt-4o\"],"
echo -e "        ${GREEN}\"temperature\"${RESET}: [0.1, 0.5, 0.9],"
echo -e "        ${GREEN}\"max_tokens\"${RESET}: [500, 1000, 2000]"
echo -e "    },"
echo ""
echo -e "    ${DIM}# Define OBJECTIVES - what to optimize for${RESET}"
echo -e "    objectives=[${GREEN}\"accuracy\"${RESET}, ${GREEN}\"cost\"${RESET}],"
echo ""
echo -e "    ${DIM}# Evaluation dataset for testing${RESET}"
echo -e "    eval_dataset=${GREEN}\"data/qa_samples.jsonl\"${RESET}"
echo -e ")"
echo -e "${BLUE}def${RESET} ${YELLOW}qa_agent${RESET}(question: str) -> str:"
echo -e "    ${DIM}\"\"\"Q&A agent with tunable parameters\"\"\"${RESET}"
echo -e "    llm = ChatOpenAI("
echo -e "        model=${GREEN}\"gpt-3.5-turbo\"${RESET},    ${DIM}# Traigent will tune this${RESET}"
echo -e "        temperature=${CYAN}0.7${RESET}           ${DIM}# Traigent will tune this${RESET}"
echo -e "    )"
echo -e "    ${MAGENTA}return${RESET} llm.invoke(question).content"
sleep 3

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Step 3: Run the optimization${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "$ ${BOLD}python -c \"import asyncio; asyncio.run(qa_agent.optimize())\"${RESET}"
sleep 0.5
echo ""
echo -e "Starting optimization with ${CYAN}grid search${RESET}..."
echo -e "Objectives: ${GREEN}accuracy${RESET}, ${YELLOW}cost${RESET}"
echo -e "Configuration space: ${BOLD}27 combinations${RESET} (3 models × 3 temps × 3 tokens)"
echo -e "Cost limit: ${YELLOW}\$2.00${RESET}"
echo ""
sleep 1

echo -e "[${DIM} 1/27${RESET}] Testing: gpt-3.5-turbo, temp=0.1, tokens=500"
echo -e "        ${DIM}accuracy=${RESET}0.72  ${DIM}cost=${RESET}${YELLOW}\$0.002${RESET}"
sleep 0.3
echo -e "[${DIM} 2/27${RESET}] Testing: gpt-3.5-turbo, temp=0.1, tokens=1000"
echo -e "        ${DIM}accuracy=${RESET}0.75  ${DIM}cost=${RESET}${YELLOW}\$0.003${RESET}"
sleep 0.3
echo -e "[${DIM} 3/27${RESET}] Testing: gpt-3.5-turbo, temp=0.5, tokens=500"
echo -e "        ${DIM}accuracy=${RESET}0.68  ${DIM}cost=${RESET}${YELLOW}\$0.002${RESET}"
sleep 0.3
echo -e "${DIM}...${RESET}"
sleep 0.5
echo -e "[${BOLD}15/27${RESET}] Testing: gpt-4o-mini, temp=0.1, tokens=1000"
echo -e "        ${DIM}accuracy=${RESET}${GREEN}${BOLD}0.89${RESET}  ${DIM}cost=${RESET}${YELLOW}\$0.008${RESET}  ${GREEN}← New best!${RESET}"
sleep 0.3
echo -e "${DIM}...${RESET}"
sleep 0.5
echo -e "[${DIM}27/27${RESET}] Testing: gpt-4o, temp=0.9, tokens=2000"
echo -e "        ${DIM}accuracy=${RESET}0.91  ${DIM}cost=${RESET}${YELLOW}\$0.045${RESET}  ${DIM}(higher cost)${RESET}"
echo ""
sleep 1

echo -e "${GREEN}✓${RESET} Optimization complete! Total cost: ${YELLOW}\$0.42${RESET} (within \$2.00 limit)"
echo ""

echo -e "${GREEN}${BOLD}┌────────────────────────────────────────────────────────────────┐${RESET}"
echo -e "${GREEN}${BOLD}│  BEST CONFIGURATION FOUND                                      │${RESET}"
echo -e "${GREEN}${BOLD}└────────────────────────────────────────────────────────────────┘${RESET}"
echo -e "  model:       ${CYAN}${BOLD}gpt-4o-mini${RESET}"
echo -e "  temperature: ${CYAN}${BOLD}0.1${RESET}"
echo -e "  max_tokens:  ${CYAN}${BOLD}1000${RESET}"
echo ""
echo -e "  ${BOLD}METRICS:${RESET}"
echo -e "  accuracy:    ${GREEN}${BOLD}0.89 (89%)${RESET}"
echo -e "  cost:        ${YELLOW}${BOLD}\$0.008${RESET} per query"
echo ""
echo -e "  ${BOLD}Compared to default${RESET} (gpt-3.5-turbo, temp=0.7):"
echo -e "    ${GREEN}↑ +17% accuracy${RESET} improvement"
echo -e "    ${GREEN}↓ Best cost/accuracy trade-off${RESET}"
sleep 3

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Step 4: Apply and use the optimized agent${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "${DIM}# Apply the optimized configuration${RESET}"
echo -e "results = ${MAGENTA}await${RESET} qa_agent.optimize()"
echo -e "qa_agent.apply_best_config(results)"
echo ""
echo -e "${DIM}# Now qa_agent uses: gpt-4o-mini, temp=0.1, tokens=1000${RESET}"
echo -e "answer = qa_agent(${GREEN}\"What is machine learning?\"${RESET})"
echo -e "${DIM}# → Uses optimized parameters automatically!${RESET}"
sleep 2
