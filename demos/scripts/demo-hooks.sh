#!/bin/bash
# Demo: Traigent Hooks & Callbacks
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
echo -e "${BLUE}${BOLD}│  🔔 Traigent: Optimization Callbacks                           │${RESET}"
echo -e "${BLUE}${BOLD}└────────────────────────────────────────────────────────────────┘${RESET}"
echo ""
sleep 1

echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Lifecycle Hooks for Monitoring${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "Available callback events:"
echo -e "  ${CYAN}on_optimization_start${RESET}   ${DIM}─${RESET} Called when optimization begins"
echo -e "  ${CYAN}on_trial_start${RESET}          ${DIM}─${RESET} Called before each trial"
echo -e "  ${CYAN}on_trial_complete${RESET}       ${DIM}─${RESET} Called after each trial"
echo -e "  ${CYAN}on_optimization_complete${RESET} ${DIM}─${RESET} Called when optimization ends"
echo ""
sleep 2

echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Built-in Callbacks${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "${MAGENTA}from${RESET} traigent.utils.callbacks ${MAGENTA}import${RESET} ("
echo -e "    ${CYAN}ProgressBarCallback${RESET},     ${DIM}# Visual progress bar${RESET}"
echo -e "    ${CYAN}LoggingCallback${RESET},         ${DIM}# Log all events${RESET}"
echo -e "    ${CYAN}StatisticsCallback${RESET},      ${DIM}# Collect optimization stats${RESET}"
echo -e "    ${CYAN}DetailedProgressCallback${RESET}  ${DIM}# Comprehensive output${RESET}"
echo -e ")"
echo ""
echo -e "${CYAN}@traigent.optimize${RESET}("
echo -e "    configuration_space={${GREEN}\"model\"${RESET}: [\"gpt-4o-mini\", \"gpt-4o\"]},"
echo -e "    objectives=[${GREEN}\"accuracy\"${RESET}],"
echo -e "    callbacks=["
echo -e "        ProgressBarCallback(width=${CYAN}50${RESET}),"
echo -e "        StatisticsCallback()"
echo -e "    ]"
echo -e ")"
echo -e "${BLUE}def${RESET} ${YELLOW}my_agent${RESET}(query: str) -> str:"
echo -e "    ..."
sleep 3

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# ProgressBarCallback in Action${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "Starting optimization with ${CYAN}grid search${RESET}..."
echo -e "Objectives: ${GREEN}accuracy${RESET}"
echo -e "Configuration space: ${BOLD}5 combinations${RESET}"
echo ""
sleep 0.5

# Simulate colored progress bar
echo -e "[${GREEN}██████████${RESET}${DIM}░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░${RESET}]  20% (1/5) Best: 0.72"
sleep 0.6
echo -e "[${GREEN}████████████████████${RESET}${DIM}░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░${RESET}]  40% (2/5) Best: ${GREEN}0.78${RESET}"
sleep 0.6
echo -e "[${GREEN}██████████████████████████████${RESET}${DIM}░░░░░░░░░░░░░░░░░░░░${RESET}]  60% (3/5) Best: ${GREEN}0.85${RESET}"
sleep 0.6
echo -e "[${GREEN}████████████████████████████████████████${RESET}${DIM}░░░░░░░░░░${RESET}]  80% (4/5) Best: 0.85"
sleep 0.6
echo -e "[${GREEN}██████████████████████████████████████████████████${RESET}] 100% (5/5) Best: ${GREEN}${BOLD}0.89${RESET}"
echo ""
echo -e "${GREEN}✓${RESET} Optimization complete! Best score: ${GREEN}${BOLD}0.89${RESET}"
sleep 2

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Creating a Custom Callback${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "${MAGENTA}from${RESET} traigent.utils.callbacks ${MAGENTA}import${RESET} OptimizationCallback"
echo ""
echo -e "${BLUE}class${RESET} ${YELLOW}SlackNotificationCallback${RESET}(OptimizationCallback):"
echo -e "    ${DIM}\"\"\"Send Slack notifications on optimization events.\"\"\"${RESET}"
echo ""
echo -e "    ${BLUE}def${RESET} ${CYAN}on_optimization_start${RESET}(self, config_space, objectives, algorithm):"
echo -e "        slack.post(f${GREEN}\"🚀 Optimization started: {algorithm}\"${RESET})"
echo ""
echo -e "    ${BLUE}def${RESET} ${CYAN}on_trial_complete${RESET}(self, trial, progress):"
echo -e "        ${MAGENTA}if${RESET} progress.best_score ${MAGENTA}and${RESET} progress.best_score > ${CYAN}0.9${RESET}:"
echo -e "            slack.post(f${GREEN}\"🎯 New best score: {progress.best_score:.1%}\"${RESET})"
echo ""
echo -e "    ${BLUE}def${RESET} ${CYAN}on_optimization_complete${RESET}(self, result):"
echo -e "        slack.post(f${GREEN}\"✅ Done! Best: {result.best_config}\"${RESET})"
sleep 3

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}# Parameter Importance Analysis${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
sleep 0.5

echo -e "stats_callback = StatisticsCallback()"
echo ""
echo -e "${DIM}# After optimization...${RESET}"
echo -e "importance = stats_callback.get_parameter_importance()"
echo ""
echo -e "${BOLD}Parameter Importance:${RESET}"
echo -e "  model:       ${GREEN}████████░░${RESET} ${BOLD}0.85${RESET}  ${DIM}← Most impactful${RESET}"
echo -e "  temperature: ${YELLOW}████░░░░░░${RESET} ${BOLD}0.42${RESET}"
echo -e "  max_tokens:  ${DIM}██░░░░░░░░${RESET} ${BOLD}0.18${RESET}"
echo ""
echo -e "${DIM}→ Model choice has the most impact on results!${RESET}"
sleep 3

echo ""
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}Callbacks enable:${RESET}"
echo -e "  ${GREEN}✓${RESET} Real-time monitoring dashboards"
echo -e "  ${GREEN}✓${RESET} Slack/email notifications"
echo -e "  ${GREEN}✓${RESET} Custom logging and metrics"
echo -e "  ${GREEN}✓${RESET} Early stopping conditions"
echo -e "  ${GREEN}✓${RESET} Parameter importance analysis"
sleep 2
