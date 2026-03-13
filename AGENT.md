# Agent

This file documents how to wire a JS/TS agent to Traigent's trial context, how the Python SDK orchestrates JS trials,
and how to capture a WalkThought trace (a structured, human-readable reasoning trail) for later analysis.

## Python-to-JS Bridge Architecture

Traigent's Python SDK can orchestrate JavaScript agents using a subprocess bridge. The Python orchestrator:

1. Spawns a Node.js subprocess running the `@traigent/sdk` CLI
2. Sends trial configurations via NDJSON (newline-delimited JSON) on stdin
3. Receives trial results via NDJSON on stdout
4. Logs are sent to stderr to keep the protocol clean

### Enabling JS Runtime in Python

```python
from traigent import optimize

@optimize(
    execution={
        "runtime": "node",                 # Execute trials in Node.js
        "js_module": "./dist/my-agent.js", # Path to compiled JS module
        "js_function": "runTrial",         # Exported function name (default)
        "js_timeout": 300.0,               # Timeout in seconds (default: 5 min)
        "js_parallel_workers": 4,          # Parallel Node.js workers (optional)
    },
    configuration_space={                  # Note: 'configuration_space', not 'search_space'
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7, 1.0],
    },
    eval_dataset="./dataset.jsonl",        # Required for JS runtime
    objectives=["accuracy", "cost"],
    max_trials=20,
    plateau_window=5,                      # Stop if no improvement for N trials
)
def optimize_js_agent(text: str) -> str:
    # This function body is not executed when runtime="node"
    # The JS module handles trial execution
    pass

# Run optimization (async)
import asyncio
result = asyncio.run(optimize_js_agent.optimize())
```

### Protocol Actions

The NDJSON protocol supports these actions:

| Action      | Description                                  |
| ----------- | -------------------------------------------- |
| `run_trial` | Execute a trial with the given configuration |
| `ping`      | Health check / keepalive                     |
| `shutdown`  | Graceful termination                         |
| `cancel`    | Cancel an in-flight trial                    |

### Handling Cancellation

The Python orchestrator may cancel in-flight trials during early stopping (budget exceeded, plateau detected, timeout). Your trial function should handle cancellation cooperatively:

```typescript
import { TrialContext, getTrialParam } from '@traigent/sdk';

export async function runTrial(config: TrialConfig): Promise<TrialResult> {
  const metrics = { accuracy: 0, cost: 0, count: 0 };

  for (const example of dataset) {
    // Option 1: Check and return partial results
    if (TrialContext.isCancelled()) {
      return {
        metrics,
        metadata: { cancelled: true, examples_completed: metrics.count },
      };
    }

    // Option 2: Throw on cancellation (returns error status)
    TrialContext.checkCancellation();

    // Process example...
    const result = await processExample(example);
    metrics.accuracy += result.correct ? 1 : 0;
    metrics.cost += result.cost;
    metrics.count += 1;
  }

  return { metrics };
}
```

**Cancellation methods:**

- `TrialContext.isCancelled()`: Returns `true` if cancellation was requested
- `TrialContext.checkCancellation()`: Throws `CancellationError` if cancelled

### Session-Based API

When submitting results to the Traigent backend, use the session-based API:

```typescript
// 1. Create a session first
const session = await fetch(`${API_URL}/sessions`, {
  method: 'POST',
  headers: { 'X-API-Key': API_KEY },
  body: JSON.stringify({
    problem_statement: 'my_optimization',
    search_space: configurationSpace,
    optimization_config: { algorithm: 'random', max_trials: 10 },
  }),
});

// 2. Submit results to the session
await fetch(`${API_URL}/sessions/${sessionId}/results`, {
  method: 'POST',
  headers: { 'X-API-Key': API_KEY },
  body: JSON.stringify({
    trial_id: 'trial_001',
    status: 'COMPLETED',
    config: { model: 'gpt-4o', temperature: 0.3 },
    metrics: { accuracy: 0.92, cost: 0.015 },
  }),
});
```

## WalkThought usage examples

### Example 1: Minimal WalkThought helper

```ts
import { getTrialConfig } from '@traigent/sdk';

type WalkThoughtStep = {
  step: number;
  note: string;
  data?: unknown;
  ts: string;
};

function createWalkThought() {
  const steps: WalkThoughtStep[] = [];
  let i = 0;
  return {
    step(note: string, data?: unknown) {
      steps.push({ step: ++i, note, data, ts: new Date().toISOString() });
    },
    toMetadata() {
      return { walkthought: steps };
    },
  };
}

export async function runAgent(question: string) {
  const config = getTrialConfig();
  const walk = createWalkThought();

  walk.step('Loaded trial config', {
    model: config.model,
    temperature: config.temperature,
  });

  // ... agent logic ...
  const answer = `Echo: ${question}`;

  walk.step('Produced answer', { length: answer.length });

  return {
    metrics: { answer_length: answer.length },
    output: { answer },
    metadata: walk.toMetadata(),
  };
}
```

### Example 2: Preserve trial context through callbacks

```ts
import { bindContext, wrapCallback } from '@traigent/sdk';
import { EventEmitter } from 'node:events';

type WalkThoughtStep = { step: number; note: string; ts: string };

function createWalkThought() {
  const steps: WalkThoughtStep[] = [];
  let i = 0;
  return {
    step(note: string) {
      steps.push({ step: ++i, note, ts: new Date().toISOString() });
    },
    toMetadata() {
      return { walkthought: steps };
    },
  };
}

export async function runAgentWithEmitter(emitter: EventEmitter) {
  const walk = createWalkThought();
  const addStep = bindContext((note: string) => walk.step(note));

  emitter.on(
    'tool:start',
    wrapCallback(() => addStep('tool:start'))
  );
  emitter.on(
    'tool:end',
    wrapCallback(() => addStep('tool:end'))
  );

  // ... run your agent ...
  return { metrics: {}, metadata: walk.toMetadata() };
}
```
