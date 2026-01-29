# Traigent Agent Demo

A demonstration of how a JavaScript agent can be optimized using Traigent, supporting both standalone JS execution and Python-orchestrated parallel execution.

## Overview

This demo shows:
- A mock sentiment classification agent
- Configuration optimization across model, temperature, and prompt type
- **Two execution modes:**
  - Standalone JS execution (original demo)
  - Python-orchestrated parallel execution (new!)

## Configuration Space

| Parameter | Values |
|-----------|--------|
| Model | gpt-3.5-turbo, gpt-4o-mini, gpt-4o |
| Temperature | 0.0, 0.3, 0.5, 0.7, 1.0 |
| System Prompt | concise, detailed, cot (chain-of-thought) |

---

## Option 1: Standalone JS Execution

Run the optimization loop entirely in Node.js:

```bash
# Install dependencies
npm install

# Run the optimization (uses mock LLM calls)
npm run dev

# Or build and run
npm run build
npm start
```

This mode runs trials sequentially and submits results to the Traigent backend.

---

## Option 2: Python-Orchestrated Parallel Execution (Recommended)

Use the Python SDK's `@traigent.optimize` decorator to orchestrate JS trial execution with **parallel processing**:

### Benefits

- **Parallel Execution**: Run 4+ Node.js processes concurrently
- **Cost Budget Enforcement**: Automatic early stopping when budget is reached
- **Advanced Optimization**: TPE, CMA-ES, and other algorithms via Optuna
- **Unified Telemetry**: Integrated with Traigent cloud for analytics

### Setup

```bash
# 1. Build the JS demo
cd demos/agent-app
npm install
npm run build

# 2. Install Python SDK (from Traigent directory)
cd ../../../Traigent
pip install -e .

# 3. Run the Python orchestrator
cd ../traigent-js/demos/agent-app
python run_with_python.py
```

### How It Works

```
Python Orchestrator (traigent.optimize)
    │
    ├─ Spawns 4 Node.js worker processes (JSProcessPool)
    │
    ├─ Sends trial configs via NDJSON protocol
    │   ├─ Worker 1: {"trial_id": "t1", "config": {"model": "gpt-4o", ...}}
    │   ├─ Worker 2: {"trial_id": "t2", "config": {"model": "gpt-3.5-turbo", ...}}
    │   ├─ Worker 3: {"trial_id": "t3", "config": {"model": "gpt-4o-mini", ...}}
    │   └─ Worker 4: {"trial_id": "t4", "config": {"model": "gpt-4o", ...}}
    │
    ├─ Collects results and updates Optuna
    │
    └─ Returns best configuration
```

### Example Python Code

```python
import traigent

traigent.initialize(execution_mode="edge_analytics")

@traigent.optimize(
    execution={
        "runtime": "node",
        "js_module": "./dist/trial.js",
        "js_function": "runTrial",
        "js_parallel_workers": 4,  # 4 Node.js processes
    },
    parallel_trials=4,
    max_trials=20,
    search_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.5, 0.7, 1.0],
        "system_prompt": ["concise", "detailed", "cot"],
    },
    objectives={
        "accuracy": {"goal": "maximize"},
        "cost": {"goal": "minimize"},
    },
    cost_budget=0.50,
)
def sentiment_agent(text: str) -> str:
    pass  # Actual implementation in agent.ts

# Run optimization
result = sentiment_agent.optimize()
print(f"Best config: {result.best_config}")
```

### Trial Function (trial.ts)

The `trial.ts` module exports a `runTrial` function compatible with the traigent-js CLI runner:

```typescript
import { TrialContext, type TrialConfig, type TrialResult } from '@anthropic/traigent';

export async function runTrial(trialConfig: TrialConfig): Promise<TrialResult> {
  const agentConfig = {
    model: trialConfig.config.model,
    temperature: trialConfig.config.temperature,
    system_prompt: trialConfig.config.system_prompt,
  };

  const result = await runSentimentAgent(examples, agentConfig);

  return {
    metrics: {
      accuracy: result.accuracy,
      cost: result.total_cost,
    },
  };
}
```

---

## Environment Variables

Copy `.env` and configure:

```bash
# For both modes
TRAIGENT_API_URL=http://localhost:5000/api/v1
TRAIGENT_API_KEY=your_api_key

# For Python orchestrator
TRAIGENT_EXECUTION_MODE=edge_analytics  # or "mock" for testing
```

---

## Output Examples

### Standalone JS Mode

```
##################################################################
# TRIAL 1 (trial_1234567890_1)
##################################################################
[TRIAL] Evaluating 10 examples from dataset

============================================================
AGENT CONFIGURATION:
  Model:       gpt-4o-mini
  Temperature: 0.3
  Prompt Type: detailed
============================================================
  [OK] "This product is amazing! Exceeded..." => positive (expected: positive)
  [X]  "Terrible experience, very disappoi..." => neutral (expected: negative)
  ...

RESULTS:
  Accuracy:    80.0% (8/10)
  Total Cost:  $0.000110
  Avg Latency: 312ms
============================================================
```

### Python Orchestrator Mode

```
======================================================================
  TRAIGENT PYTHON ORCHESTRATOR - PARALLEL JS EXECUTION DEMO
======================================================================

JS Module: ./dist/trial.js
Parallel Workers: 4
Max Trials: 12
Cost Budget: $0.10

Search Space:
  - Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o
  - Temperature: 0.0, 0.3, 0.5, 0.7, 1.0
  - Prompt Type: concise, detailed, cot

----------------------------------------------------------------------
Starting parallel optimization...

[Trial 1] Running with config: {"model":"gpt-4o","temperature":0.3,"system_prompt":"cot"}
[Trial 2] Running with config: {"model":"gpt-4o-mini","temperature":0.0,"system_prompt":"detailed"}
[Trial 3] Running with config: {"model":"gpt-3.5-turbo","temperature":0.5,"system_prompt":"concise"}
[Trial 4] Running with config: {"model":"gpt-4o","temperature":0.0,"system_prompt":"detailed"}
...

======================================================================
  OPTIMIZATION COMPLETE
======================================================================

Best Configuration Found:
  Model:       gpt-4o
  Temperature: 0.0
  Prompt:      cot

Best Metrics:
  accuracy: 95.0%
  cost: $0.000255
  latency_ms: 512.00

Trials Completed: 12
Stop Reason: max_trials
Total Cost: $0.002456

Demo complete!
```

---

## File Structure

```
demos/agent-app/
├── src/
│   ├── agent.ts          # Sentiment classification agent
│   ├── dataset.ts        # Sample dataset
│   ├── trial.ts          # CLI runner trial function (for Python orchestration)
│   └── run-optimization.ts  # Standalone JS optimization loop
├── run_with_python.py    # Python orchestrator example
├── package.json
└── README.md
```

---

## Comparison

| Feature | Standalone JS | Python Orchestrator |
|---------|--------------|---------------------|
| Execution | Sequential | Parallel (4+ workers) |
| Optimization | Random sampling | TPE, CMA-ES, etc. |
| Budget enforcement | Manual | Automatic |
| Early stopping | Manual | Configurable |
| Analytics | Basic | Full Traigent integration |
| Best for | Prototyping | Production |
