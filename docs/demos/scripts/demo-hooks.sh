#!/bin/bash
# Demo: Traigent Hooks & Callbacks
set -e
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"

clear
echo "# Traigent Hooks & Callbacks"
echo ""
sleep 1

echo "# Traigent provides lifecycle hooks to monitor optimization"
echo ""
sleep 0.5

echo "Available callback events:"
echo "  on_optimization_start  - Called when optimization begins"
echo "  on_trial_start         - Called before each trial"
echo "  on_trial_complete      - Called after each trial"
echo "  on_optimization_complete - Called when optimization ends"
echo ""
sleep 2

echo "# Built-in Callbacks"
echo ""
sleep 0.5

cat << 'PYTHON'
from traigent.utils.callbacks import (
    ProgressBarCallback,    # Visual progress bar
    LoggingCallback,        # Log all events
    StatisticsCallback,     # Collect optimization stats
    DetailedProgressCallback # Comprehensive output
)

@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
    objectives=["accuracy"],
    callbacks=[
        ProgressBarCallback(width=50),
        StatisticsCallback()
    ]
)
def my_agent(query: str) -> str:
    ...
PYTHON
sleep 3

echo ""
echo "# Example: ProgressBarCallback output"
echo ""
sleep 0.5

echo "Starting optimization with grid..."
echo "Objectives: accuracy"
echo "Configuration space: 2 parameters"
echo ""
sleep 0.5

# Simulate progress bar
echo -n "[##########                                        ]  20% (1/5) Best: 0.72"
sleep 0.5
echo ""
echo -n "[####################                              ]  40% (2/5) Best: 0.78"
sleep 0.5
echo ""
echo -n "[##############################                    ]  60% (3/5) Best: 0.85"
sleep 0.5
echo ""
echo -n "[########################################          ]  80% (4/5) Best: 0.85"
sleep 0.5
echo ""
echo -n "[##################################################] 100% (5/5) Best: 0.89"
sleep 0.5
echo ""
echo ""
echo "Optimization complete! Best score: 0.89"
sleep 2

echo ""
echo "# Creating a Custom Callback"
echo ""
sleep 0.5

cat << 'PYTHON'
from traigent.utils.callbacks import OptimizationCallback

class SlackNotificationCallback(OptimizationCallback):
    """Send Slack notifications on optimization events."""

    def on_optimization_start(self, config_space, objectives, algorithm):
        slack.post(f"Optimization started: {algorithm}")

    def on_trial_complete(self, trial, progress):
        if progress.best_score and progress.best_score > 0.9:
            slack.post(f"New best score: {progress.best_score:.1%}")

    def on_optimization_complete(self, result):
        slack.post(f"Done! Best: {result.best_config}")

# Use it
@traigent.optimize(
    configuration_space={...},
    callbacks=[SlackNotificationCallback()]
)
def my_agent(query: str) -> str:
    ...
PYTHON
sleep 3

echo ""
echo "# StatisticsCallback: Analyze Results"
echo ""
sleep 0.5

cat << 'PYTHON'
stats_callback = StatisticsCallback()

@traigent.optimize(
    configuration_space={"model": [...], "temperature": [...]},
    callbacks=[stats_callback]
)
def agent(q): ...

await agent.optimize()

# Get parameter importance analysis
importance = stats_callback.get_parameter_importance()
print(importance)
# {'model': 0.85, 'temperature': 0.42}
# -> Model choice has more impact on results!
PYTHON
sleep 3

echo ""
echo "# Callbacks enable:"
echo "  - Real-time monitoring dashboards"
echo "  - Slack/email notifications"
echo "  - Custom logging and metrics"
echo "  - Early stopping conditions"
echo "  - Parameter importance analysis"
sleep 2
