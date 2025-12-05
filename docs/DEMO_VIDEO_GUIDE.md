# Traigent SDK Demo Video Guide

This guide explains how to create terminal demo videos showcasing Traigent's LLM agent optimization capabilities. These animated recordings are ideal for GitHub READMEs and documentation.

## Overview

Create **two demo videos**:

1. **Core Optimization** - Tuning LLM agents with Traigent: defining tuned variables, objectives, and running evaluation-driven optimization
2. **Hooks & Callbacks** - Progress tracking, custom callbacks, and optimization lifecycle events

## Directory Structure

```
demos/
├── .gitignore           # Exclude .venv/, node_modules/
├── README.md            # Documentation
├── record-demos.sh      # Master generation script
├── mock-cli/
│   └── traigent         # Mock traigent CLI
├── scripts/
│   ├── generate-cast.py # Python cast file generator
│   ├── demo-optimize.sh # Video 1: Core optimization
│   └── demo-hooks.sh    # Video 2: Hooks & callbacks
└── output/
    ├── *.cast           # asciinema format
    └── *.svg            # Animated SVGs for GitHub
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

### Demo Script: `demos/scripts/demo-optimize.sh`

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
    eval_dataset="qa_samples.jsonl"
)
def qa_agent(question: str) -> str:
    """Q&A agent with tunable parameters"""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",    # Traigent will tune this
        temperature=0.7           # Traigent will tune this
    )
    return llm.invoke(question).content
PYTHON
sleep 3

echo ""
echo "# Step 2: Prepare evaluation dataset (qa_samples.jsonl)"
echo ""
sleep 0.5

cat << 'JSONL'
{"input": {"question": "What is Python?"}, "expected": "A programming language"}
{"input": {"question": "What is 2+2?"}, "expected": "4"}
{"input": {"question": "Capital of France?"}, "expected": "Paris"}
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

### Demo Script: `demos/scripts/demo-hooks.sh`

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

## Implementation Steps

### Step 1: Create Directory Structure

```bash
mkdir -p demos/{mock-cli,scripts,output}
```

### Step 2: Create the Python Generator

Save as `demos/scripts/generate-cast.py`:

```python
#!/usr/bin/env python3
"""Generate asciinema cast files from demo scripts."""

import json
import subprocess
import time
import os

def generate_cast(script_path: str, output_path: str, title: str):
    """Generate an asciinema cast file from a script."""

    env = os.environ.copy()
    env['TERM'] = 'xterm-256color'
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))

    result = subprocess.run(
        ['bash', script_path],
        capture_output=True,
        text=True,
        env=env,
        cwd=script_dir
    )

    output = result.stdout + result.stderr

    header = {
        "version": 2,
        "width": 100,
        "height": 30,
        "timestamp": int(time.time()),
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
        "title": title
    }

    with open(output_path, 'w') as f:
        f.write(json.dumps(header) + '\n')

        current_time = 0.0
        lines = output.split('\n')

        for line in lines:
            if line.startswith('$') or line.startswith('#'):
                # Type character by character for commands/comments
                for char in line:
                    f.write(json.dumps([current_time, "o", char]) + '\n')
                    current_time += 0.03
                current_time += 0.3
                f.write(json.dumps([current_time, "o", "\r\n"]) + '\n')
                current_time += 0.5
            else:
                # Output lines appear instantly
                if line:
                    f.write(json.dumps([current_time, "o", line + "\r\n"]) + '\n')
                    current_time += 0.05
                else:
                    f.write(json.dumps([current_time, "o", "\r\n"]) + '\n')
                    current_time += 0.02

        f.write(json.dumps([current_time + 1.0, "o", ""]) + '\n')

    print(f"  Generated {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demos_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(demos_dir, 'output')

    os.makedirs(output_dir, exist_ok=True)

    demos = [
        ('demo-optimize.sh', 'optimize.cast', 'Traigent LLM Optimization'),
        ('demo-hooks.sh', 'hooks.cast', 'Traigent Hooks & Callbacks'),
    ]

    print("Generating asciinema cast files...")
    print()

    for script, output, title in demos:
        script_path = os.path.join(script_dir, script)
        output_path = os.path.join(output_dir, output)

        if os.path.exists(script_path):
            print(f"-> {title}")
            try:
                generate_cast(script_path, output_path, title)
            except Exception as e:
                print(f"   Error: {e}")
        else:
            print(f"-> {title} (skipped - {script} not found)")

    print()
    print("Done!")

if __name__ == '__main__':
    main()
```

### Step 3: Create Master Script

Save as `demos/record-demos.sh`:

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TERM="${TERM:-xterm-256color}"

mkdir -p output

echo "=================================="
echo "  Traigent Demo Generator"
echo "=================================="
echo ""

chmod +x scripts/*.sh 2>/dev/null || true

# Step 1: Generate cast files
echo "Step 1: Generating cast files..."
python3 scripts/generate-cast.py

# Step 2: Convert to SVG (if svg-term available)
if command -v svg-term &> /dev/null; then
    echo ""
    echo "Step 2: Converting to SVG..."
    for f in output/*.cast; do
        name=$(basename "$f" .cast)
        svg-term --in "$f" --out "output/${name}.svg" --window --width 100 --height 30
        echo "  output/${name}.svg"
    done
else
    echo ""
    echo "Note: svg-term not found. Install with: npm install -g svg-term-cli"
fi

echo ""
echo "Done! Files in output/"
ls -la output/
```

### Step 4: Create .gitignore

```bash
# demos/.gitignore
.venv/
venv/
node_modules/
*.tmp
*.log
```

---

## Generate the Demos

```bash
cd demos

# One-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install asciinema
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
![Traigent Optimization](demos/output/optimize.svg)

### Hooks & Callbacks
![Traigent Hooks](demos/output/hooks.svg)
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

---

## Reference

See `TraigentPaper/tvl/demos/` for the technical implementation pattern used to generate these demos.
