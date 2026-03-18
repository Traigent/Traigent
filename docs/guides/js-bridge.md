# JavaScript Bridge Guide

This guide covers how to use Traigent's Python SDK to orchestrate JavaScript/TypeScript agents via the JS Bridge.

## Overview

The JS Bridge enables Python-orchestrated optimization of JavaScript agents by:

1. Spawning Node.js subprocesses to execute trials
2. Communicating via NDJSON (newline-delimited JSON) protocol
3. Supporting parallel execution via process pooling
4. Providing cooperative cancellation for early stopping

## Quick Start

### Python Side

```python
import traigent

@traigent.optimize(
    execution={
        "runtime": "node",
        "js_module": "./dist/my-agent.js",
        "js_function": "runTrial",
        "js_timeout": 300.0,
        "js_parallel_workers": 4,
    },
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7, 1.0],
    },
    evaluation={"eval_dataset": "./dataset.jsonl"},
    objectives=["accuracy", "cost"],
    max_trials=20,
)
def optimize_js_agent(text: str) -> str:
    # This function body is NOT executed when runtime="node"
    # The JS module handles all trial execution
    pass

# Run optimization
import asyncio
result = asyncio.run(optimize_js_agent.optimize())
print(f"Best config: {result.best_config}")
```

### JavaScript Side

```typescript
// my-agent.ts
import { getTrialConfig, getTrialParam } from '@traigent/sdk';

export async function runTrial(config: TrialConfig): Promise<TrialResult> {
  // Access configuration
  const model = getTrialParam('model', 'gpt-4o-mini');
  const temperature = getTrialParam('temperature', 0.7);

  // Run your agent logic
  const results = await runAgentOnDataset(model, temperature);

  // Return metrics
  return {
    metrics: {
      accuracy: results.accuracy,
      cost: results.totalCost,
      latency: results.avgLatency,
    },
    metadata: {
      examples_processed: results.count,
    },
  };
}
```

## Configuration Options

### ExecutionOptions for JS Runtime

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `runtime` | `str` | `"python"` | Set to `"node"` for JS execution |
| `js_module` | `str` | `None` | Path to JS module (required for node) |
| `js_function` | `str` | `"runTrial"` | Exported function name |
| `js_timeout` | `float` | `300.0` | Trial timeout in seconds |
| `js_parallel_workers` | `int` | `1` | Number of parallel Node.js processes |

### Example Configurations

**Sequential Execution (Single Worker)**:

```python
execution={
    "runtime": "node",
    "js_module": "./dist/agent.js",
}
```

**Parallel Execution (4 Workers)**:

```python
execution={
    "runtime": "node",
    "js_module": "./dist/agent.js",
    "js_parallel_workers": 4,
}
```

**With Custom Timeout**:

```python
execution={
    "runtime": "node",
    "js_module": "./dist/agent.js",
    "js_timeout": 600.0,  # 10 minutes per trial
}
```

## Protocol Reference

The Python SDK and JS runtime communicate via NDJSON over stdin/stdout.

### Protocol Actions

| Action | Direction | Description |
|--------|-----------|-------------|
| `run_trial` | Python → JS | Execute a trial with given configuration |
| `ping` | Python → JS | Health check / keepalive |
| `shutdown` | Python → JS | Graceful termination |
| `cancel` | Python → JS | Cancel an in-flight trial |

### Trial Configuration

The `run_trial` action sends:

```json
{
  "action": "run_trial",
  "request_id": "uuid-string",
  "payload": {
    "trial_id": "trial_001",
    "trial_number": 1,
    "experiment_run_id": "exp_abc123",
    "config": {
      "model": "gpt-4o",
      "temperature": 0.7
    },
    "dataset_subset": {
      "indices": [0, 1, 2, 3, 4],
      "total": 100
    }
  }
}
```

### Trial Response

```json
{
  "request_id": "uuid-string",
  "status": "completed",
  "payload": {
    "trial_id": "trial_001",
    "status": "completed",
    "metrics": {
      "accuracy": 0.92,
      "cost": 0.015,
      "latency": 1.23
    },
    "duration": 45.2,
    "metadata": {
      "examples_processed": 5
    }
  }
}
```

### Error Response

```json
{
  "request_id": "uuid-string",
  "status": "error",
  "payload": {
    "trial_id": "trial_001",
    "status": "failed",
    "error_message": "API rate limit exceeded",
    "error_code": "RATE_LIMIT",
    "retryable": true
  }
}
```

## Cancellation Support

The JS Bridge supports cooperative cancellation for early stopping scenarios.

### Python Side

Cancellation happens automatically when:
- Budget limit is reached
- Plateau detection triggers early stop
- Timeout is exceeded
- User interrupts (Ctrl+C)

### JavaScript Side

Check for cancellation in your trial function:

```typescript
import { TrialContext } from '@traigent/sdk';

export async function runTrial(config: TrialConfig): Promise<TrialResult> {
  const metrics = { accuracy: 0, cost: 0 };

  for (const example of dataset) {
    // Check if cancelled before each example
    if (TrialContext.isCancelled()) {
      return { metrics, metadata: { cancelled: true } };
    }

    // Or throw on cancellation
    TrialContext.checkCancellation(); // Throws if cancelled

    // Process example...
    const result = await processExample(example);
    metrics.accuracy += result.correct ? 1 : 0;
    metrics.cost += result.cost;
  }

  return { metrics };
}
```

## Process Pool Architecture

When `js_parallel_workers > 1`, Traigent uses a process pool:

```
Python Orchestrator
    │
    ├── JSProcessPool
    │       │
    │       ├── Worker 1 (Node.js subprocess)
    │       ├── Worker 2 (Node.js subprocess)
    │       ├── Worker 3 (Node.js subprocess)
    │       └── Worker 4 (Node.js subprocess)
    │
    └── Gather results from all workers
```

### Pool Behavior

- **Lazy Start**: Workers spawn on first trial
- **Worker Reuse**: Workers handle multiple trials sequentially
- **Health Checks**: Workers are pinged after each trial; unhealthy workers are replaced
- **Graceful Shutdown**: All workers receive shutdown signal on completion

### Memory Considerations

Each Node.js worker uses approximately 50MB of memory. Plan accordingly:

| Workers | Approximate Memory |
|---------|-------------------|
| 1 | ~50 MB |
| 4 | ~200 MB |
| 8 | ~400 MB |

## Error Handling

### Error Codes

| Code | Description | Retryable |
|------|-------------|-----------|
| `USER_FUNCTION_ERROR` | Error in trial function | No |
| `VALIDATION_ERROR` | Invalid trial config | No |
| `TIMEOUT` | Trial exceeded timeout | No |
| `CANCELLED` | Trial was cancelled | No |
| `RATE_LIMIT` | API rate limit hit | Yes |
| `PROCESS_DIED` | Node.js process crashed | Yes |

### Python Error Handling

```python
from traigent.bridges.js_bridge import JSBridgeError, JSTrialTimeoutError

try:
    result = await bridge.run_trial(config)
except JSTrialTimeoutError:
    # Trial timed out
    pass
except JSBridgeError as e:
    # Other bridge errors
    print(f"Bridge error: {e}")
```

## Troubleshooting

### Common Issues

**1. "Node.js not found"**

Ensure Node.js is installed and in PATH:
```bash
node --version  # Should show v18+
```

**2. "Module not found"**

Check that `js_module` path is correct relative to your Python working directory:
```python
execution={
    "js_module": "./dist/agent.js",  # Relative to cwd
}
```

**3. "Trial timeout"**

Increase timeout for long-running trials:
```python
execution={
    "js_timeout": 600.0,  # 10 minutes
}
```

**4. "stdout contamination"**

Your JS code must not write to stdout (it's used for protocol). Use stderr:
```typescript
// BAD - corrupts protocol
console.log("Debug info");

// GOOD - writes to stderr
console.error("Debug info");
```

### Debug Mode

Enable verbose logging:
```bash
export TRAIGENT_LOG_LEVEL=DEBUG
python examples/js_bridge/optimize_js_agent.py
```

## Best Practices

1. **Build Before Run**: Ensure your TypeScript is compiled before optimization
   ```bash
   export TRAIGENT_JS_MODULE=/abs/path/to/run-trial.js
   python examples/js_bridge/optimize_js_agent.py
   ```

2. **Use stderr for Logs**: Keep stdout clean for protocol communication

3. **Handle Cancellation**: Check `TrialContext.isCancelled()` for long trials

4. **Match Workers to Trials**: Set `js_parallel_workers` equal to `parallel_trials` for optimal parallelism

5. **Test Locally First**: Run a few trials with `max_trials=3` before full optimization

## Complete Example

A full working example is available in the repository:

**Python orchestration script**: `examples/js_bridge/optimize_js_agent.py`
**JS trial module**: provide a built `run-trial.js` via `TRAIGENT_JS_MODULE`

This repository does not vendor a built JS demo. To run the example, point
Traigent at a compiled JS trial module from your own app, from an installed
`traigent-js` package, or from a sibling checkout of the companion JS repo. If
you are not using an installed package, set `TRAIGENT_JS_RUNNER` to the local
CLI runner or keep the companion repo checked out next to this repo so the
Python example can auto-detect it.

```bash
export TRAIGENT_MOCK_LLM=true
export TRAIGENT_OFFLINE_MODE=true
export TRAIGENT_JS_MODULE=/abs/path/to/run-trial.js
export TRAIGENT_JS_RUNNER=/abs/path/to/runner.js  # Optional when not using npx
python examples/js_bridge/optimize_js_agent.py
```

The example demonstrates:
- Python SDK orchestrating JS agent optimization
- Grid/random search over model, temperature, and prompt configurations
- Parallel trial execution with 2 Node.js workers
- Dataset subset evaluation
- Result aggregation and best config selection

## Budget Guardrails

The JS Bridge integrates with Traigent's budget enforcement system to prevent runaway costs.

### How Budget Enforcement Works

1. **Pre-Trial Cost Check**: Before dispatching a trial to JS, the orchestrator reserves an estimated cost
2. **Post-Trial Tracking**: After JS returns, actual cost from metrics is tracked
3. **Budget Stop Condition**: When cumulative cost exceeds the limit, optimization stops

### Configuration

```python
@traigent.optimize(
    execution={
        "runtime": "node",
        "js_module": "./dist/agent.js",
    },
    budget={
        "max_cost_usd": 5.00,  # Stop when total cost exceeds $5
    },
    # ...
)
```

### Reporting Cost from JavaScript

Your trial function should return cost in the metrics:

```typescript
export async function runTrial(config: TrialConfig): Promise<TrialFunctionResult> {
  const result = await runAgent(config);

  return {
    metrics: {
      accuracy: result.accuracy,
      cost: result.totalCostUsd,  // Report actual cost
      latency_ms: result.latency,
    },
  };
}
```

### Budget Behavior in Sequential vs Parallel Modes

| Mode | Behavior |
|------|----------|
| Sequential (`js_parallel_workers=1`) | Each trial runs after budget check. Stops immediately when limit reached. |
| Parallel (`js_parallel_workers>1`) | Multiple trials may be in-flight. Stop triggers cancellation of running trials. |

### Cost Estimation

For the first few trials, Traigent estimates costs using an Exponential Moving Average (EMA) of observed costs. To improve estimation:

1. Run a few warm-up trials with `max_trials=3` to establish baseline
2. Use consistent cost metrics (e.g., always report in USD)
3. Consider setting conservative initial estimates via budget config

## Stop Conditions

Stop conditions trigger early termination when optimization goals are met.

### Available Stop Conditions

| Condition | Trigger | JS Behavior |
|-----------|---------|-------------|
| `max_trials` | Fixed number of trials completed | Normal completion |
| `plateau` | No improvement for N consecutive trials | In-flight trials may complete |
| `budget` | Cost limit exceeded | Cancellation sent to running trials |
| `timeout` | Wall-clock time exceeded | Cancellation sent to running trials |

### Configuration

```python
@traigent.optimize(
    execution={
        "runtime": "node",
        "js_module": "./dist/agent.js",
        "js_parallel_workers": 4,
    },
    max_trials=50,
    stop_conditions={
        "plateau_after_n": 10,      # Stop if no improvement in 10 trials
        "timeout_seconds": 3600,    # Stop after 1 hour
    },
    budget={
        "max_cost_usd": 10.00,
    },
)
```

### Cooperative Cancellation for Stop Conditions

When a stop condition triggers during parallel execution:

1. **No new trials dispatched**: Orchestrator stops queueing trials
2. **Cancel signal sent**: Running JS trials receive `cancel` action
3. **Graceful handling**: JS should check `TrialContext.isCancelled()` and return partial results

```typescript
export async function runTrial(config: TrialConfig): Promise<TrialFunctionResult> {
  const metrics = { accuracy: 0, cost: 0, processed: 0 };

  for (const example of dataset) {
    // Check cancellation periodically
    if (TrialContext.isCancelled()) {
      return {
        metrics,
        metadata: { cancelled: true, partial: true },
      };
    }

    const result = await evaluateExample(example);
    metrics.accuracy += result.correct ? 1 : 0;
    metrics.cost += result.cost;
    metrics.processed += 1;
  }

  return { metrics };
}
```

### Sequential vs Parallel Stop Condition Behavior

**Sequential Mode** (`js_parallel_workers=1`):
- Stop condition checked after each trial completes
- Clean stop without in-flight trials
- No cancellation needed

**Parallel Mode** (`js_parallel_workers>1`):
- Stop condition checked as trials complete
- Multiple trials may be running when stop triggers
- Cancel signals sent to all in-flight trials
- Results from completed trials are preserved

### Testing Stop Conditions

To verify stop conditions work correctly with your JS agent:

```python
# Test max_trials stop condition
@traigent.optimize(
    execution={"runtime": "node", "js_module": "./dist/agent.js"},
    max_trials=5,  # Should stop after exactly 5 trials
)
def test_max_trials(text: str): pass

# Test budget stop condition
@traigent.optimize(
    execution={"runtime": "node", "js_module": "./dist/agent.js"},
    budget={"max_cost_usd": 0.10},  # Low budget for testing
    max_trials=100,
)
def test_budget_stop(text: str): pass

# Test plateau stop condition
@traigent.optimize(
    execution={"runtime": "node", "js_module": "./dist/agent.js"},
    stop_conditions={"plateau_after_n": 3},
    max_trials=100,
)
def test_plateau_stop(text: str): pass
```

## Testing JavaScript Agents

### Local Development

For local testing without API calls:

```bash
export TRAIGENT_MOCK_LLM=true
export TRAIGENT_OFFLINE_MODE=true
export TRAIGENT_JS_MODULE=/abs/path/to/run-trial.js
export TRAIGENT_JS_RUNNER=/abs/path/to/runner.js  # Optional when not using npx
python examples/js_bridge/optimize_js_agent.py
```

### Unit Testing Trial Functions

Test your JS trial function directly:

```typescript
// test/trial.test.ts
import { runTrial } from '../src/run-trial';

describe('runTrial', () => {
  it('returns valid metrics', async () => {
    const config = {
      trial_id: 'test-1',
      trial_number: 1,
      config: { model: 'gpt-4o-mini', temperature: 0.7 },
      dataset_subset: { indices: [0, 1, 2], total: 10 },
    };

    const result = await runTrial(config);

    expect(result.metrics).toBeDefined();
    expect(typeof result.metrics.accuracy).toBe('number');
    expect(typeof result.metrics.cost).toBe('number');
  });
});
```

### Integration Testing with Python

```python
# tests/integration/bridges/test_js_bridge_e2e.py
import pytest
from traigent.bridges.js_bridge import JSBridge, JSBridgeConfig

@pytest.mark.asyncio
async def test_js_bridge_roundtrip():
    config = JSBridgeConfig(
        module_path="./dist/agent.js",
        function_name="runTrial",
    )
    bridge = JSBridge(config)

    try:
        await bridge.start()

        trial_config = {
            "trial_id": "test-1",
            "trial_number": 1,
            "config": {"temperature": 0.5},
            "dataset_subset": {"indices": [0], "total": 1},
        }

        result = await bridge.run_trial(trial_config)
        assert result.status == "completed"
        assert "accuracy" in result.metrics
    finally:
        await bridge.stop()
```

## Related Documentation

- [Decorator Reference](../api-reference/decorator-reference.md) - Full decorator options
- [Parallel Configuration Guide](./parallel-configuration.md) - Parallel execution details
- [traigent-js SDK](https://github.com/Traigent/traigent-js) - JavaScript SDK documentation
