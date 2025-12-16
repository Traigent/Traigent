# TraiGent CI/CD Integration

This example demonstrates how to integrate TraiGent optimization gates into your CI/CD pipeline using **only the SDK's native capabilities** - no custom evaluation code required.

## 🎯 Goals

1. **Regression Gate**: Prevent merging if configuration underperforms baseline
2. **Improvement Gate**: Alert when auto-tuning finds better configurations
3. **No Custom Scoring Functions**: Use TraiGent’s built-in evaluator + metrics (a small runner script wires it into CI)
4. **Specification-Driven**: Configure behavior through environment variables

## 📁 Structure

```
integrations/ci-cd/
├── math-qa/                    # Example: Arithmetic Q&A
│   ├── math_qa.py             # Decorated function
│   ├── math_qa.jsonl          # Dataset (expressions → answers)
│   ├── saved_config.json      # Current production config
│   └── run.py                 # CI runner (MODE=eval|tune)
├── hooks/                      # Git hooks
│   ├── pre-commit             # Quick validation
│   └── pre-push               # Optimization check
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Git Hooks (Optional)

```bash
# From repository root
cp examples/integrations/ci-cd/hooks/pre-commit .git/hooks/
cp examples/integrations/ci-cd/hooks/pre-push .git/hooks/
chmod +x .git/hooks/pre-*
```

### 2. Test Locally

```bash
cd examples/integrations/ci-cd/math-qa

# Evaluate current configuration
TRAIGENT_MOCK_MODE=true MODE=eval python run.py

# Run optimization to find better config
TRAIGENT_MOCK_MODE=true MODE=tune MAX_TRIALS=10 python run.py
```

### 3. GitHub Actions

The workflow (`.github/workflows/traigent-ci-gates.yml`) automatically:
1. Evaluates baseline configuration
2. Evaluates PR configuration
3. Runs optimization on PR
4. Compares metrics and enforces gates

## 🔑 Key Improvements Over ChatGPT5's Plan

### ✅ Proper SDK Usage

**Instead of singleton space hack:**
```python
# ❌ ChatGPT5's approach - inefficient
config_space = {param: [value] for param, value in config.items()}
results = func.optimize()  # Wasteful optimization of 1 config
```

**Our approach - direct evaluation:**
```python
# ✅ Proper SDK usage
evaluator = LocalEvaluator()
result = await evaluator.evaluate(func, config, dataset)
```

### ✅ Native Configuration Management

**Using SDK's `set_config()` method:**
```python
# Apply saved configuration
solve_arithmetic.set_config(saved_config)

# After optimization, apply best config
solve_arithmetic.apply_best_config(results)
```

### ✅ Python-Based Metric Comparison

**Instead of complex jq expressions**, we use a Python script that:
- Properly handles metric orientations (maximize/minimize)
- Provides clear error messages
- Supports custom thresholds

## 📊 Metrics & Orientations

Default tracked metrics:
- **accuracy** (maximize): Exact match percentage
- **cost** (minimize): LLM API cost per evaluation
- **response_time** (minimize): Average latency in ms

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `eval` | `eval` = evaluate config, `tune` = optimize |
| `MAX_TRIALS` | `10` | Maximum optimization trials |
| `TRAIGENT_MOCK_MODE` | `false` | Use mock LLM for testing |
| `REGRESSION_THRESHOLD` | `-0.01` | Max allowed degradation (1%) |
| `IMPROVEMENT_MIN_PCT` | `0.03` | Min improvement to trigger alert (3%) |

### saved_config.json

This file stores your current production configuration:

```json
{
  "model": "gpt-3.5-turbo",
  "temperature": 0.1,
  "prompt_style": "direct",
  "max_tokens": 50
}
```

When CI detects a better configuration, update this file with the optimized values.

## 🚦 CI Gates Logic

### Gate 1: Regression Detection

For each metric, calculate improvement:
- **Maximize metrics**: `(current - baseline) / baseline`
- **Minimize metrics**: `(baseline - current) / baseline`

If improvement < `REGRESSION_THRESHOLD`, **fail CI**.

### Gate 2: Missed Improvement

For each metric, calculate potential:
- **Maximize metrics**: `(tuned - current) / current`
- **Minimize metrics**: `(current - tuned) / tuned`

If potential ≥ `IMPROVEMENT_MIN_PCT` and configs differ, **warn or fail**.

## 🎭 Mock Mode

All CI runs use `TRAIGENT_MOCK_MODE=true` by default:
- Zero API costs
- Fast execution (~1 second)
- Deterministic results for testing

For production validation, set `TRAIGENT_MOCK_MODE=false` in protected branch workflows.

## 📝 Adding New Examples

1. **Create new directory**: `examples/integrations/ci-cd/your-example/`

2. **Add decorated function** (`your_example.py`):
```python
@traigent.optimize(
    configuration_space={...},
    eval_dataset="your_dataset.jsonl",
    objectives=["accuracy", "cost", "response_time"],
    execution_mode="edge_analytics"
)
def your_function(...):
    ...
```

3. **Add dataset** (`your_dataset.jsonl`):
```json
{"input": {...}, "output": "expected"}
```

4. **Copy and modify** `run.py` from math-qa example

5. **Add to CI matrix** in workflow YAML

## 🔍 Debugging

View detailed logs:
```bash
# Check evaluation details
cat results-ci/math/current.json

# Check optimization results
cat results-ci/math/tuned.json

# Compare configs
diff saved_config.json <(jq .best_config results-ci/math/tuned.json)
```

## 🏆 Best Practices

1. **Keep datasets small** (10-20 examples) for fast CI
2. **Use mock mode** in PRs, real mode in protected branches
3. **Set appropriate thresholds** based on your requirements
4. **Update saved_config.json** when improvements are found
5. **Version control configs** for rollback capability

## 📚 References

- [TraiGent Documentation](https://traigent.ai/docs)
- [GitHub Actions Workflow](.github/workflows/traigent-ci-gates.yml)
- [SDK Evaluator API](../../traigent/evaluators/base.py)

## 💡 Why This Approach?

1. **SDK-Native**: Uses TraiGent's actual APIs, not workarounds
2. **Efficient**: Direct evaluation without fake optimization
3. **Maintainable**: Clear Python code instead of complex shell scripts
4. **Scalable**: Easy to add new examples and metrics
5. **Production-Ready**: Handles real-world edge cases

This demonstrates the **proper way** to integrate TraiGent into CI/CD pipelines, leveraging the SDK's full capabilities while maintaining simplicity.
