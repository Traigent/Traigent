# Traigent SDK Demo Video Guide

This guide explains how to create terminal demo videos showcasing Traigent's LLM agent optimization capabilities. These animated recordings are ideal for GitHub READMEs and documentation. The core demo assets live in `docs/demos/`; TVL-specific scripts live in `demos/scripts/`.

## Overview

Create **three demo videos**:

1. **Core Optimization** - Tuning LLM agents with Traigent: defining tuned variables, objectives, and running evaluation-driven optimization
2. **Hooks & Callbacks** - Progress tracking, custom callbacks, and optimization lifecycle events
3. **GitHub Hooks** - Agent configuration validation and guardrails in CI

## Directory Structure

```
docs/demos/
├── README.md               # Demo overview
├── record-demos.sh         # Master generation script
├── scripts/
│   ├── generate-cast.py    # Python cast file generator
│   ├── demo-optimize.sh    # Video 1: Core optimization
│   ├── demo-hooks.sh       # Video 2: Hooks and callbacks
│   └── demo-github-hooks.sh # Video 3: GitHub hooks and validation
├── output/
│   ├── *.cast              # asciinema format
│   └── *.svg               # Animated SVGs for GitHub
└── test_agents/            # Sample agents + traigent.yml for hooks demo
```

---

## Video 1: Core Optimization Demo

**Purpose**: Show how Traigent optimizes LLM agents through evaluation-driven optimization.

**Key concepts to demonstrate**:
- `@traigent.optimize()` decorator
- **Tuned Variables** (configuration_space): model, temperature, max_tokens, k
- **Objectives**: accuracy, cost, latency
- **Evaluation dataset**: JSONL test cases
- Running optimization and viewing results

### Demo Script: `docs/demos/scripts/demo-optimize.sh`

```bash
#!/bin/bash
# Demo: Traigent LLM Agent Optimization
set -e
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"

clear
echo "# Traigent: Evaluation-Driven LLM Optimization"
echo ""
sleep 1

echo "# Step 1: Define your LLM agent with tuned variables"
echo ""
sleep 0.5

cat << 'PYTHON'
import traigent
from langchain_openai import ChatOpenAI

@traigent.optimize(
    # Define TUNED VARIABLES - parameters to optimize
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "max_tokens": [500, 1000, 2000]
    },

    # Define OBJECTIVES - what to optimize for
    objectives=["accuracy", "cost"],

    # Evaluation dataset for testing
    eval_dataset="data/qa_samples.jsonl"
)
def qa_agent(question: str) -> str:
    """Q&A agent with tunable parameters"""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",    # Traigent will tune this
        temperature=0.7           # Traigent will tune this
    )
    return str(llm.invoke(question).content)
PYTHON
sleep 3

echo ""
echo "# Step 2: Prepare evaluation dataset (data/qa_samples.jsonl)"
echo ""
sleep 0.5

cat << 'JSONL'
{"input": {"question": "What is Python?"}, "output": "A programming language"}
{"input": {"question": "What is 2+2?"}, "output": "4"}
{"input": {"question": "Capital of France?"}, "output": "Paris"}
JSONL
sleep 2

echo ""
echo "# Step 3: Run the optimization"
echo ""
sleep 0.5

echo '$ python -c "import asyncio; asyncio.run(qa_agent.optimize())"'
sleep 0.5
echo ""
echo "Starting optimization with grid search..."
echo "Objectives: accuracy, cost"
echo "Configuration space: 27 combinations (3 models x 3 temps x 3 tokens)"
echo ""
sleep 1

echo "[1/27] Testing: gpt-3.5-turbo, temp=0.1, tokens=500"
echo "       Score: accuracy=0.72, cost=\$0.002"
sleep 0.3
echo "[2/27] Testing: gpt-3.5-turbo, temp=0.1, tokens=1000"
echo "       Score: accuracy=0.75, cost=\$0.003"
sleep 0.3
echo "[3/27] Testing: gpt-3.5-turbo, temp=0.5, tokens=500"
echo "       Score: accuracy=0.68, cost=\$0.002"
sleep 0.3
echo "..."
sleep 0.5
echo "[15/27] Testing: gpt-4o-mini, temp=0.1, tokens=1000"
echo "       Score: accuracy=0.89, cost=\$0.008  <- New best!"
sleep 0.3
echo "..."
sleep 0.5
echo "[27/27] Testing: gpt-4o, temp=0.9, tokens=2000"
echo "       Score: accuracy=0.91, cost=\$0.045"
echo ""
sleep 1

echo "Optimization complete!"
echo ""
echo "BEST CONFIGURATION FOUND:"
echo "  model: gpt-4o-mini"
echo "  temperature: 0.1"
echo "  max_tokens: 1000"
echo ""
echo "METRICS:"
echo "  accuracy: 0.89 (89%)"
echo "  cost: \$0.008 per query"
echo ""
echo "Compared to default (gpt-3.5-turbo, temp=0.7):"
echo "  +17% accuracy improvement"
echo "  Best cost/accuracy trade-off"
sleep 3

echo ""
echo "# Step 4: Apply best configuration"
echo ""
sleep 0.5

cat << 'PYTHON'
# Apply the optimized configuration
results = await qa_agent.optimize()
qa_agent.apply_best_config(results)

# Now qa_agent uses: gpt-4o-mini, temp=0.1, tokens=1000
answer = qa_agent("What is machine learning?")
PYTHON
sleep 2
```

---

## Video 2: Hooks & Callbacks Demo

**Purpose**: Show Traigent's callback system for monitoring and customizing optimization.

**Key concepts to demonstrate**:
- `OptimizationCallback` base class
- Built-in callbacks: `ProgressBarCallback`, `LoggingCallback`, `StatisticsCallback`
- Lifecycle hooks: `on_optimization_start`, `on_trial_start`, `on_trial_complete`, `on_optimization_complete`
- Custom callback implementation

### Demo Script: `docs/demos/scripts/demo-hooks.sh`

```bash
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
```

---

## Video 3: GitHub Hooks Demo

**Purpose**: Show Traigent's Git hooks for validating agent configs before push.

**Key concepts to demonstrate**:
- `traigent hooks install`
- `traigent.yml` constraints (cost, performance, model allowlist)
- `traigent-validate` and `traigent-performance` hooks
- Sample agents in `docs/demos/test_agents/`

### Demo Script: `docs/demos/scripts/demo-github-hooks.sh`

---

## Implementation Steps

### Step 1: Update the demo scripts

Scripts live in `docs/demos/scripts/`:
- `demo-optimize.sh`
- `demo-hooks.sh`
- `demo-github-hooks.sh`

### Step 2: Update the cast generator list

Edit `docs/demos/scripts/generate-cast.py` to add or remove demos in the `demos` list.

### Step 3: Regenerate casts and SVGs

Run `docs/demos/record-demos.sh`. Cast files go to `docs/demos/output/`. SVG output is optional and requires `svg-term`.

---

## Generate the Demos

```bash
cd docs/demos

# Optional (for SVG output)
npm install -g svg-term-cli  # Requires Node.js

# Generate demos
chmod +x record-demos.sh scripts/*.sh
./record-demos.sh
```

---

## Embed in README

```markdown
## See Traigent in Action

### LLM Agent Optimization
![Traigent Optimization](docs/demos/output/optimize.svg)

### Hooks & Callbacks
![Traigent Hooks](docs/demos/output/hooks.svg)

### GitHub Hooks & Validation
![Traigent GitHub Hooks](docs/demos/output/github-hooks.svg)
```

---

## Key Messages to Convey

### Video 1 (Optimization)
- **Zero code changes** - just add the decorator
- **Tuned variables** = parameters that affect agent behavior (model, temperature, etc.)
- **Objectives** = what you want to optimize (accuracy, cost, latency)
- **Evaluation-driven** = uses test datasets to measure improvement
- **Results** = find the best configuration automatically

### Video 2 (Hooks)
- **Lifecycle hooks** for monitoring optimization progress
- **Built-in callbacks** for common use cases
- **Custom callbacks** for notifications, logging, dashboards
- **Statistics** for parameter importance analysis

### Video 3 (GitHub Hooks)
- **Pre-push validation** for agent configs
- **Cost and performance guardrails** in CI
- **Model allowlist/blocklist** enforcement
- **Clear failure messages** to fix before pushing

---

## Reference

See `docs/demos/README.md` for the demo index and `demos/scripts/TVL_VIDEO_SCRIPTS.md` for the TVL-specific videos.
