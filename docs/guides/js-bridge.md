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
python my_optimization.py
```

## Best Practices

1. **Build Before Run**: Ensure your TypeScript is compiled before optimization
   ```bash
   npm run build && python optimize.py
   ```

2. **Use stderr for Logs**: Keep stdout clean for protocol communication

3. **Handle Cancellation**: Check `TrialContext.isCancelled()` for long trials

4. **Match Workers to Trials**: Set `js_parallel_workers` equal to `parallel_trials` for optimal parallelism

5. **Test Locally First**: Run a few trials with `max_trials=3` before full optimization

## Related Documentation

- [Decorator Reference](../api-reference/decorator-reference.md) - Full decorator options
- [Parallel Configuration Guide](./parallel-configuration.md) - Parallel execution details
- [traigent-js SDK](https://github.com/Traigent/traigent-js) - JavaScript SDK documentation
