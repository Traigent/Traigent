# Convergence Analysis Patterns

## The convergence_info Dictionary

The `results.convergence_info` field is a dictionary containing statistics about how the optimization progressed over trials. Use it to understand whether the optimizer found a good solution or whether more trials might help.

```python
results = func.optimize(dataset="data.jsonl")
info = results.convergence_info
```

### Common Fields

The exact fields in `convergence_info` depend on the optimization algorithm, but common fields include:

| Field | Type | Description |
|---|---|---|
| `best_score_history` | `list[float]` | Best score at each trial step, showing the improvement curve. |
| `score_history` | `list[float]` | Score for each individual trial in order. |
| `improvement_rate` | `float` | Rate of improvement over recent trials. |
| `trials_since_improvement` | `int` | Number of consecutive trials with no improvement to best score. |
| `plateau_detected` | `bool` | Whether the optimizer detected a scoring plateau. |
| `exploration_ratio` | `float` | Fraction of trials that explored new regions vs. exploited known-good areas. |

## Interpreting Improvement Trends

### Healthy Convergence

A well-converging optimization shows rapid initial improvement that tapers off:

```python
info = results.convergence_info

if "best_score_history" in info:
    history = info["best_score_history"]

    # Check improvement in first half vs second half
    midpoint = len(history) // 2
    if midpoint > 0:
        first_half_gain = history[midpoint] - history[0]
        second_half_gain = history[-1] - history[midpoint]

        if second_half_gain < first_half_gain * 0.1:
            print("Optimization has converged - further trials unlikely to help")
        else:
            print("Still improving - consider running more trials")
```

### Stagnation

If the score is flat from the start, there may be a problem with the configuration space or evaluator:

```python
if "best_score_history" in info:
    history = info["best_score_history"]
    if len(history) > 3 and history[-1] == history[0]:
        print("No improvement at all - check:")
        print("  1. Is the configuration space meaningful?")
        print("  2. Does the evaluator differentiate between configs?")
        print("  3. Are all trials failing?")
        print(f"  Failed trials: {len(results.failed_trials)}/{len(results.trials)}")
```

### Late Improvement

If improvement comes late, the configuration space may be large and worth exploring further:

```python
if "best_score_history" in info:
    history = info["best_score_history"]
    if len(history) > 5:
        # Check if best score was found in last 20% of trials
        cutoff = int(len(history) * 0.8)
        late_best = max(history[cutoff:])
        early_best = max(history[:cutoff])
        if late_best > early_best:
            print("Best result found late - more trials may find even better configs")
```

## Plateau Detection

Traigent can stop optimization when it detects a plateau (no improvement over a window of trials). The stop reason will be `"plateau"`.

```python
if results.stop_reason == "plateau":
    print("Optimization stopped due to plateau detection")

    # Check how many trials ran without improvement
    if "trials_since_improvement" in results.convergence_info:
        stale = results.convergence_info["trials_since_improvement"]
        print(f"  {stale} trials with no improvement before stopping")

    # The result is likely near-optimal for this config space
    print(f"  Final best score: {results.best_score}")
```

## When to Increase max_trials

Use convergence information to decide whether more trials would help:

```python
def should_run_more_trials(results) -> bool:
    """Decide whether increasing max_trials is worthwhile."""

    # Already converged naturally
    if results.stop_reason == "plateau":
        return False

    # Optimizer exhausted the search space
    if results.stop_reason == "optimizer":
        return False

    # Error-based stops need fixing, not more trials
    if results.stop_reason in ("error", "vendor_error", "network_error"):
        return False

    # Hit trial limit - check if still improving
    if results.stop_reason == "max_trials_reached":
        info = results.convergence_info
        if "best_score_history" in info:
            history = info["best_score_history"]
            if len(history) >= 3:
                # Check last 3 scores for improvement
                recent = history[-3:]
                if recent[-1] > recent[0]:
                    return True  # Still improving
        # Default: yes, if we hit the limit, try more
        return True

    return False


results = func.optimize(dataset="data.jsonl")
if should_run_more_trials(results):
    print("Consider re-running with higher max_trials")
```

## Comparing Convergence Across Runs

Use optimization history to compare convergence patterns:

```python
history = func.get_optimization_history()

for run in history:
    info = run.convergence_info
    print(f"Run {run.optimization_id}:")
    print(f"  Algorithm: {run.algorithm}")
    print(f"  Trials: {len(run.trials)}")
    print(f"  Best score: {run.best_score}")
    print(f"  Stop reason: {run.stop_reason}")
    print(f"  Duration: {run.duration:.1f}s")

    if "best_score_history" in info:
        scores = info["best_score_history"]
        if scores:
            print(f"  Score range: {scores[0]:.3f} -> {scores[-1]:.3f}")
    print()
```

## Convergence Visualization

If you want to plot convergence (requires matplotlib):

```python
import matplotlib.pyplot as plt

info = results.convergence_info

if "best_score_history" in info:
    plt.figure(figsize=(10, 5))
    plt.plot(info["best_score_history"], label="Best score")
    if "score_history" in info:
        plt.scatter(
            range(len(info["score_history"])),
            info["score_history"],
            alpha=0.3,
            label="Individual trials",
        )
    plt.xlabel("Trial")
    plt.ylabel("Score")
    plt.title(f"Convergence (stop_reason={results.stop_reason})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence.png")
```
