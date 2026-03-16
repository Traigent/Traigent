# Accuracy / Score Logic Evidence (Internal)

This document captures exact code snippets from this repository proving how `accuracy` and `score` are computed in the Bazak run path.

## 1) Runtime Logic: Accuracy Derivation Priority

Source: `traigent/evaluators/hybrid_api.py:335-360`

```python
@classmethod
def _derive_accuracy_from_metrics(cls, metrics: dict[str, Any]) -> float | None:
    """Derive canonical accuracy from a metric dictionary.

    Preference order:
    1. explicit `accuracy`
    2. `overall_accuracy`
    3. mean of numeric `*_accuracy` metrics
    """
    explicit = cls._coerce_metric(metrics.get("accuracy"), default=-1.0)
    if explicit >= 0.0:
        return explicit

    overall = cls._coerce_metric(metrics.get("overall_accuracy"), default=-1.0)
    if overall >= 0.0:
        return overall

    split_accuracies = [
        float(v)
        for k, v in metrics.items()
        if k.endswith("_accuracy")
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    ]
    if split_accuracies:
        return float(sum(split_accuracies) / len(split_accuracies))
    return None
```

Interpretation:
- If explicit `accuracy` exists, it is used.
- Mean of split metrics (`tool_accuracy`, `param_accuracy`, `text_accuracy`, etc.) is fallback only.

## 2) Runtime Logic: Trial-Level Mean and Score Default

Source: `traigent/evaluators/hybrid_api.py:923-952`

```python
# Aggregate per-example metrics
metric_sums: dict[str, float] = {}
metric_counts: dict[str, int] = {}
...
# Compute means
for metric_name, total in metric_sums.items():
    count = metric_counts[metric_name]
    if count > 0:
        aggregated[metric_name] = total / count
...
if "accuracy" not in aggregated:
    derived_accuracy = self._derive_accuracy_from_metrics(aggregated)
    if derived_accuracy is not None:
        aggregated["accuracy"] = derived_accuracy

if "score" not in aggregated and "accuracy" in aggregated:
    aggregated["score"] = aggregated["accuracy"]
```

Interpretation:
- Trial metrics are computed as means over example-level metrics.
- `score` defaults to `accuracy` when explicit `score` is absent.

## 3) Bazak Run Objective Is Tool Accuracy

Source: `examples/bazak/run_optimization.py:273-276, 290-295`

```python
optimizer = GridSearchOptimizer(
    config_space=config_space,
    objectives=["tool_accuracy"],
    max_trials=max_trials,
)
...
orchestrator = OptimizationOrchestrator(
    optimizer=optimizer,
    evaluator=evaluator,
    max_trials=max_trials,
    objectives=["tool_accuracy"],
    config=traigent_config,
)
```

Interpretation:
- The optimization target in this Bazak script is explicitly `tool_accuracy`.

## 4) Evidence That This Is Documented In-Repo

### A) In-code documentation (docstring)
- `traigent/evaluators/hybrid_api.py:336-342`
- Explicitly documents the preference order and fallback behavior.

### B) In-report documentation (project artifact)
- `examples/bazak/CLIENT_VALIDATION_REPORT.md:62-67`

Current report text documents:
- Formula: `Accuracy = (1/N) * Σ(i=1..N) accuracy_i` (N=3 for this run)
- Clarification that this is **not** `(tool_accuracy + param_accuracy + text_accuracy)/3` for this run
- Split-metric mean used only as fallback when explicit `accuracy` is missing

## 5) Test Evidence In This Codebase

### Derive split-metric mean only when applicable
Source: `tests/unit/evaluators/test_hybrid_api_evaluator.py:822-825`

```python
def test_derive_accuracy_uses_numeric_split_accuracy(self) -> None:
    """Derive mean of numeric split accuracy keys only."""
    metrics = {"text_accuracy": 0.6, "tool_accuracy": 1.0, "aux_accuracy": "n/a"}
    assert HybridAPIEvaluator._derive_accuracy_from_metrics(metrics) == pytest.approx(0.8)
```

### Keep explicit accuracy (including 0.0)
Source: `tests/unit/evaluators/test_hybrid_api_evaluator.py:812-815`

```python
def test_derive_accuracy_preserves_explicit_zero(self) -> None:
    """Explicit accuracy=0.0 must not be treated as missing."""
    metrics = {"accuracy": 0.0, "overall_accuracy": 0.9, "text_accuracy": 1.0}
    assert HybridAPIEvaluator._derive_accuracy_from_metrics(metrics) == pytest.approx(0.0)
```

### Score uses accuracy when score absent
Source: `tests/unit/cloud/test_api_operations.py:1122-1125`

```python
# Only accuracy, no score - should use accuracy as score
result = await self.ops.update_config_run_measures(
    "config_123", {"accuracy": 0.95}
)
```
