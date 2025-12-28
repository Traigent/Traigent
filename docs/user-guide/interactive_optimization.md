# Interactive Optimization Guide

This guide explains how to use Traigent's interactive optimization model for client-side execution with remote guidance.

## Overview

The interactive optimization model enables a hybrid approach where:

- **Configuration suggestions** come from the Traigent Cloud Service
- **Function execution** happens on your local machine
- **Results** are reported back to guide the next suggestion

This approach is ideal when:

- Your function cannot be serialized or executed remotely
- Data privacy requirements prevent sending full datasets to the cloud
- You need low-latency execution
- You want to maintain full control over function execution

## Key Concepts

### 1. Session-Based Optimization

Unlike traditional optimization where everything runs in one call, interactive optimization uses sessions:

```python
# Create a session
session = await optimizer.initialize_session(
    function_name="my_llm_function",
    max_trials=50
)

# Run trials iteratively
while True:
    suggestion = await optimizer.get_next_suggestion(dataset_size)
    if not suggestion:
        break

    # Execute locally
    results = execute_with_config(suggestion.config)

    # Report back
    await optimizer.report_results(trial_id, metrics, duration)

# Finalize
final_results = await optimizer.finalize_optimization()
```

### 2. Dataset Subset Selection

The service intelligently suggests which subset of your data to use for each trial:

- **Early trials**: Small, diverse subsets for quick exploration
- **Middle trials**: Representative samples for better estimates
- **Late trials**: Larger subsets for high-confidence evaluation

```python
suggestion = await optimizer.get_next_suggestion(dataset_size=1000)
# suggestion.dataset_subset.indices = [5, 23, 67, 89, 156, ...]
# Use only these indices from your local dataset
```

### 3. Smart Configuration Suggestions

The service provides configurations based on:

- Previous trial results
- Exploration vs exploitation balance
- Your optimization objectives
- Resource budget constraints

## Step-by-Step Implementation

### Step 1: Setup

```python
from traigent.cloud.client import TraigentCloudClient
from traigent.optimizers.interactive_optimizer import InteractiveOptimizer

# Initialize cloud client
client = TraigentCloudClient(api_key="your-api-key")

# Create interactive optimizer
optimizer = InteractiveOptimizer(
    config_space={
        "temperature": (0.0, 1.0),
        "model": ["o4-mini", "GPT-4o"],
        "max_tokens": (50, 500)
    },
    objectives=["accuracy", "latency", "cost"],
    remote_service=client,
    dataset_metadata={
        "size": 10000,
        "type": "question_answering",
        "language": "english"
    }
)
```

### Step 2: Initialize Session

```python
session = await optimizer.initialize_session(
    function_name="optimize_qa_system",
    max_trials=100,
    user_id="team-alpha",
    billing_tier="professional"
)

print(f"Session ID: {session.session_id}")
print(f"Optimization strategy: {session.optimization_strategy}")
```

### Step 3: Optimization Loop

```python
async def run_optimization():
    while True:
        # Get next trial suggestion
        suggestion = await optimizer.get_next_suggestion(
            dataset_size=len(my_dataset)
        )

        if not suggestion:
            print("Optimization complete!")
            break

        # Extract suggested subset
        subset = [my_dataset[i] for i in suggestion.dataset_subset.indices]

        # Execute with suggested configuration
        metrics = await evaluate_function(
            function=my_llm_function,
            config=suggestion.config,
            data=subset
        )

        # Report results
        await optimizer.report_results(
            trial_id=suggestion.trial_id,
            metrics=metrics,
            duration=execution_time,
            status="completed"
        )

        # Check progress
        status = await optimizer.get_optimization_status()
        print(f"Progress: {status['progress']*100:.1f}%")
```

### Step 4: Finalize and Get Results

```python
# Finalize optimization
results = await optimizer.finalize_optimization(
    include_full_history=True
)

print(f"Best configuration: {results.best_config}")
print(f"Best metrics: {results.best_metrics}")
print(f"Total trials: {results.total_trials}")
print(f"Cost savings: {results.cost_savings*100:.1f}%")

# Use optimized configuration
optimized_function = lambda x: my_llm_function(x, **results.best_config)
```

## Advanced Features

### Custom Optimization Strategies

```python
optimizer = InteractiveOptimizer(
    config_space=config_space,
    objectives=objectives,
    remote_service=client,
    optimization_strategy={
        "exploration_ratio": 0.2,  # 20% exploration, 80% exploitation
        "min_examples_per_trial": 10,
        "max_examples_per_trial": 500,
        "adaptive_sampling": True,
        "early_stopping": True,
        "convergence_threshold": 0.001
    }
)
```

### Error Handling and Recovery

```python
async def robust_optimization():
    suggestion = await optimizer.get_next_suggestion(dataset_size)

    try:
        # Execute function
        results = await my_function(config=suggestion.config)
        metrics = compute_metrics(results)
        status = "completed"
        error_message = None

    except Exception as e:
        # Report failure
        metrics = {}
        status = "failed"
        error_message = str(e)

    await optimizer.report_results(
        trial_id=suggestion.trial_id,
        metrics=metrics,
        duration=duration,
        status=status,
        error_message=error_message
    )
```

### Monitoring and Logging

```python
# Get detailed status
status = await optimizer.get_optimization_status()

print(f"Session: {status['session_id']}")
print(f"Status: {status['status']}")
print(f"Completed trials: {status['completed_trials']}/{status['max_trials']}")
print(f"Best metrics so far: {status['best_metrics']}")
print(f"Trials per minute: {status['trials_per_minute']:.1f}")
```

## Best Practices

### 1. Dataset Metadata

Provide accurate metadata for better subset selection:

```python
dataset_metadata = {
    "size": 50000,
    "type": "classification",
    "class_distribution": {"positive": 0.3, "negative": 0.7},
    "avg_text_length": 150,
    "language": "english",
    "domain": "customer_support"
}
```

### 2. Efficient Local Execution

```python
async def evaluate_batch(function, config, data_batch):
    """Evaluate function on a batch of data efficiently."""
    results = await asyncio.gather(*[
        function(item, **config) for item in data_batch
    ])
    return results
```

### 3. Result Caching

```python
# Cache results to avoid re-computation
result_cache = {}

async def cached_execution(config, data):
    cache_key = (tuple(sorted(config.items())), tuple(data))

    if cache_key in result_cache:
        return result_cache[cache_key]

    result = await execute_function(config, data)
    result_cache[cache_key] = result
    return result
```

### 4. Progressive Evaluation

Start with fast approximate metrics, then switch to accurate ones:

```python
if suggestion.exploration_type == "exploration":
    # Use fast approximate metrics for exploration
    metrics = await fast_evaluation(config, subset)
else:
    # Use accurate metrics for exploitation
    metrics = await accurate_evaluation(config, subset)
```

## Comparison with Other Approaches

| Feature              | Interactive Optimization   | Full Remote          | Full Local       |
| -------------------- | -------------------------- | -------------------- | ---------------- |
| Data Privacy         | ✅ High (data stays local) | ❌ Low               | ✅ High          |
| Optimization Quality | ✅ High (cloud algorithms) | ✅ High              | ⚠️ Limited       |
| Execution Speed      | ✅ Fast (local execution)  | ⚠️ Network dependent | ✅ Fast          |
| Cost Efficiency      | ✅ 60-80% savings          | ❌ Full cost         | ✅ No cloud cost |
| Scalability          | ✅ High                    | ✅ High              | ❌ Limited       |

## Common Use Cases

### 1. LLM Parameter Tuning

```python
config_space = {
    "temperature": (0.0, 1.0),
    "top_p": (0.1, 1.0),
    "frequency_penalty": (-2.0, 2.0),
    "presence_penalty": (-2.0, 2.0),
    "model": ["o4-mini", "GPT-4o", "claude-2"]
}

objectives = ["response_quality", "latency", "cost_per_token"]
```

### 2. ML Model Hyperparameter Optimization

```python
config_space = {
    "learning_rate": (1e-5, 1e-1),
    "batch_size": [16, 32, 64, 128],
    "num_epochs": (1, 20),
    "optimizer": ["adam", "sgd", "rmsprop"],
    "dropout_rate": (0.0, 0.5)
}

objectives = ["validation_accuracy", "training_time", "model_size"]
```

### 3. System Configuration Optimization

```python
config_space = {
    "cache_size": (1, 1000),
    "num_workers": (1, 32),
    "timeout": (0.1, 10.0),
    "retry_strategy": ["exponential", "linear", "fixed"],
    "compression": ["none", "gzip", "zstd"]
}

objectives = ["throughput", "latency_p99", "error_rate"]
```

## Troubleshooting

### Session Timeout

```python
try:
    suggestion = await optimizer.get_next_suggestion(dataset_size)
except SessionError as e:
    if "expired" in str(e):
        # Recreate session
        session = await optimizer.initialize_session(...)
```

### Network Issues

```python
# Configure retry behavior
client = TraigentCloudClient(
    api_key="your-key",
    max_retries=5,
    timeout=60.0
)
```

### Handling Rate Limits

```python
# The client automatically handles rate limiting
# You can also add your own backoff
await asyncio.sleep(1.0)  # Add delay between trials if needed
```

## Next Steps

1. Check out the [complete example](../../examples/interactive_optimization.py)
2. Learn about [agent-based optimization](./agent_optimization.md)
3. Read the [API reference](../api/interactive_optimizer.md)
4. Join our [community forum](https://community.traigent.ai) for support

---

_For more information, visit [traigent.ai/docs](https://traigent.ai/docs)_
