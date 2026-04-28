---
name: traigent-analyze-results
description: "Analyze Traigent optimization results: best config, trial comparison, convergence, cost, and applying results to production. Use when reading results.best_config, comparing trials, checking stop_reason, calling apply_best_config(), accessing total_cost or total_tokens, or understanding why optimization stopped."
license: Apache-2.0
metadata:
  author: Traigent
  version: "1.0"
---

# Analyzing Traigent Optimization Results

## When to Use

Use this skill after `optimize()` returns an `OptimizationResult` object. This covers:

- Reading the best configuration and score
- Comparing individual trial results
- Understanding why optimization stopped (stop reasons)
- Checking cost and token usage
- Applying the best configuration for production use
- Reviewing optimization history across multiple runs

## Quick Results

After running optimization, the `OptimizationResult` object provides immediate access to the key outcomes:

```python
import traigent

@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"], "temperature": [0.0, 0.5, 1.0]},
    objectives=["accuracy"],
    max_trials=10,
)
def classify(text):
    config = traigent.get_config()
    # ... LLM call using config ...
    return result

results = classify.optimize(dataset="eval_data.jsonl")

# Top-level results
print(results.best_config)      # {"model": "gpt-4o", "temperature": 0.0}
print(results.best_score)       # 0.92 (float or None if no eligible trial)
print(results.stop_reason)      # "max_trials_reached"
print(results.duration)         # 45.3 (seconds, wall-clock)
print(results.status)           # OptimizationStatus.COMPLETED
print(results.algorithm)        # Name of optimization algorithm used
print(results.optimization_id)  # Unique ID for this run
print(results.objectives)       # ["accuracy"]
print(results.timestamp)        # datetime when optimization completed
```

`best_score` is `None` when no trial produced a valid score (e.g., all trials failed). Always check before comparing:

```python
if results.best_score is not None:
    print(f"Best accuracy: {results.best_score:.2%}")
else:
    print("No successful trials produced a score")
```

## Reading Trials

Each trial in `results.trials` is a `TrialResult` with full details about what happened:

```python
for trial in results.trials:
    print(f"Trial {trial.trial_id}")
    print(f"  Config: {trial.config}")
    print(f"  Status: {trial.status}")        # TrialStatus enum
    print(f"  Duration: {trial.duration:.1f}s")
    print(f"  Metrics: {trial.metrics}")       # {"accuracy": 0.85, "latency": 1.2}
    print(f"  Successful: {trial.is_successful}")
    print(f"  Timestamp: {trial.timestamp}")

    # Safe metric access with default
    accuracy = trial.get_metric("accuracy", default=0.0)
    latency = trial.get_metric("latency_ms", default=None)

    # Check for errors
    if trial.error_message:
        print(f"  Error: {trial.error_message}")

    # Trial metadata (additional context)
    if trial.metadata:
        print(f"  Metadata: {trial.metadata}")
```

### Filtering Trials

Use the built-in properties to filter trials by outcome:

```python
# Only successful trials
for trial in results.successful_trials:
    print(f"{trial.config} -> accuracy={trial.get_metric('accuracy')}")

# Only failed trials
for trial in results.failed_trials:
    print(f"FAILED: {trial.config} -> {trial.error_message}")

# Success rate
print(f"Success rate: {results.success_rate:.0%}")
# e.g., "Success rate: 80%"
```

### Comparing Trial Configurations

Find which configuration parameters matter most:

```python
# Sort trials by a specific metric
sorted_trials = sorted(
    results.successful_trials,
    key=lambda t: t.get_metric("accuracy", 0.0),
    reverse=True,
)

# Show top 3
for i, trial in enumerate(sorted_trials[:3], 1):
    print(f"#{i}: accuracy={trial.get_metric('accuracy'):.3f} config={trial.config}")

# Compare best vs worst
if len(sorted_trials) >= 2:
    best = sorted_trials[0]
    worst = sorted_trials[-1]
    for key in best.config:
        if best.config[key] != worst.config[key]:
            print(f"  {key}: best={best.config[key]}, worst={worst.config[key]}")
```

## Cost and Performance

Track what the optimization run cost in API spend and tokens:

```python
# Total cost across all trials (None if not tracked)
if results.total_cost is not None:
    print(f"Total cost: ${results.total_cost:.4f}")

# Total tokens consumed (None if not tracked)
if results.total_tokens is not None:
    print(f"Total tokens: {results.total_tokens:,}")

# Aggregated experiment statistics
stats = results.experiment_stats
print(f"Total duration: {stats.total_duration:.1f}s")
print(f"Total cost: ${stats.total_cost:.4f}")
print(f"Unique configurations tested: {stats.unique_configurations}")
print(f"Average trial duration: {stats.average_trial_duration:.1f}s")
print(f"Cost per configuration: ${stats.cost_per_configuration:.4f}")
print(f"Trial counts: {stats.trial_counts}")
# Trial counts: {"total": 10, "completed": 8, "failed": 2, ...}

# Best metrics from the winning trial
print(f"Best metrics: {results.best_metrics}")
# {"accuracy": 0.92, "latency": 0.8}
```

## Stop Reasons

The `stop_reason` field tells you why optimization ended. This is critical for deciding whether to run more trials:

| Stop Reason | Meaning | Action |
|---|---|---|
| `"max_trials_reached"` | Hit the `max_trials` limit | Increase `max_trials` if results are still improving |
| `"max_samples_reached"` | Hit the max samples/examples limit | Increase sample budget or reduce dataset size |
| `"timeout"` | Exceeded the timeout duration | Increase timeout or reduce config space |
| `"cost_limit"` | Hit the cost budget limit | Increase `cost_limit` or use cheaper models |
| `"optimizer"` | Optimizer decided to stop (search space exhausted) | Config space fully explored; results are final |
| `"plateau"` | No improvement detected | Results have converged; more trials unlikely to help |
| `"user_cancelled"` | User cancelled or declined cost approval | Review cost estimates, re-run if needed |
| `"condition"` | A generic stop condition triggered | Check convergence_info for details |
| `"error"` | Optimization failed due to an exception | Check failed trials for error messages |
| `"vendor_error"` | Provider error (rate limit, quota, service issue) | Check API keys, quotas, and provider status |
| `"network_error"` | Connectivity failure | Check network connection and retry |
| `None` | Stop reason not set | Typically means the run completed normally |

```python
if results.stop_reason == "max_trials_reached":
    print("Consider increasing max_trials for better results")
elif results.stop_reason == "plateau":
    print("Optimization converged - these are likely the best results")
elif results.stop_reason == "cost_limit":
    print(f"Budget exhausted at ${results.total_cost:.2f}")
elif results.stop_reason == "error":
    for trial in results.failed_trials:
        print(f"Error in trial {trial.trial_id}: {trial.error_message}")
```

## Applying Best Config

After optimization, apply the winning configuration so your function uses it in production:

```python
# Run optimization
results = classify.optimize(dataset="eval_data.jsonl")

# Apply the best configuration
classify.apply_best_config(results)

# Now every call uses the optimized config
# traigent.get_config() inside the function returns results.best_config
response = classify("What category is this email?")
```

`apply_best_config()` sets the configuration so that subsequent calls to `traigent.get_config()` inside the decorated function return the best configuration from the optimization run.

### Safety Check Before Applying

Verify results before applying:

```python
results = classify.optimize(dataset="eval_data.jsonl")

if results.best_score is not None and results.best_score >= 0.85:
    classify.apply_best_config(results)
    print(f"Applied config with score {results.best_score:.2%}")
else:
    print(f"Score {results.best_score} below threshold, not applying")
    # Use a known-good default instead
```

## Optimization History

Review results from previous optimization runs on the same function:

```python
history = classify.get_optimization_history()

for past_result in history:
    print(f"Run {past_result.optimization_id}")
    print(f"  Algorithm: {past_result.algorithm}")
    print(f"  Best score: {past_result.best_score}")
    print(f"  Best config: {past_result.best_config}")
    print(f"  Trials: {len(past_result.trials)}")
    print(f"  Duration: {past_result.duration:.1f}s")
    print(f"  Stop reason: {past_result.stop_reason}")
    print(f"  Timestamp: {past_result.timestamp}")
```

Compare across runs to see if optimization is improving over time:

```python
history = classify.get_optimization_history()
if len(history) >= 2:
    latest = history[-1]
    previous = history[-2]
    if latest.best_score is not None and previous.best_score is not None:
        improvement = latest.best_score - previous.best_score
        print(f"Improvement: {improvement:+.3f}")
```

## Complete Example

End-to-end workflow: optimize, analyze, decide, apply.

```python
import traigent

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [256, 512, 1024],
    },
    objectives=["accuracy"],
    max_trials=15,
)
def summarize(text):
    config = traigent.get_config()
    # ... your LLM summarization logic ...
    return summary

# 1. Run optimization
results = summarize.optimize(dataset="summarization_eval.jsonl")

# 2. Quick summary
print(f"Best config: {results.best_config}")
print(f"Best score: {results.best_score}")
print(f"Stop reason: {results.stop_reason}")
print(f"Duration: {results.duration:.1f}s")
print(f"Success rate: {results.success_rate:.0%}")

# 3. Cost analysis
if results.total_cost is not None:
    print(f"Total cost: ${results.total_cost:.4f}")
if results.total_tokens is not None:
    print(f"Total tokens: {results.total_tokens:,}")

# 4. Trial breakdown
print(f"\nTop 5 trials by accuracy:")
top_trials = sorted(
    results.successful_trials,
    key=lambda t: t.get_metric("accuracy", 0.0),
    reverse=True,
)[:5]
for trial in top_trials:
    print(f"  {trial.config} -> accuracy={trial.get_metric('accuracy'):.3f}")

# 5. Convergence check
if results.stop_reason == "plateau":
    print("\nOptimization converged naturally")
elif results.stop_reason == "max_trials_reached":
    print("\nMay benefit from more trials")

# 6. Apply if good enough
THRESHOLD = 0.80
if results.best_score is not None and results.best_score >= THRESHOLD:
    summarize.apply_best_config(results)
    print(f"\nApplied best config (score={results.best_score:.2%})")

    # Production usage
    output = summarize("Summarize this quarterly earnings report...")
else:
    print(f"\nScore {results.best_score} below threshold {THRESHOLD}, skipping apply")
```

## Reference Files

- [OptimizationResult and TrialResult API Reference](references/optimization-result-api.md)
- [Convergence Analysis Patterns](references/convergence-patterns.md)
