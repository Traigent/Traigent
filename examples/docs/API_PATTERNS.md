# 🔌 Traigent API Patterns - Common Usage Examples

## Core Patterns

### 1. Basic Decorator Pattern
The simplest way to optimize a function.

```python
import traigent

@traigent.optimize(
    config_space={
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [100, 200, 500]
    },
    objectives=["accuracy", "cost"],
    num_trials=10
)
def my_llm_function(input_text: str) -> str:
    # Your LLM logic here
    return response
```

### 2. Dataset-Based Evaluation
For systematic evaluation across multiple examples.

```python
from traigent import Dataset, Example

# Create dataset
dataset = Dataset(
    examples=[
        Example(input="What is 2+2?", expected_output="4"),
        Example(input="Capital of France?", expected_output="Paris"),
    ]
)

# Optimize with dataset
results = my_llm_function.optimize(
    dataset=dataset,
    algorithm="bayesian",
    metrics=["accuracy", "f1_score"]
)
```

### 3. Custom Evaluator Pattern
Define custom evaluation logic for specific needs.

```python
def custom_evaluator(output: str, expected: str) -> dict:
    """Custom evaluation function."""
    # Your evaluation logic
    score = calculate_similarity(output, expected)
    return {
        "accuracy": score,
        "custom_metric": custom_calculation(output)
    }

results = my_function.optimize(
    evaluator=custom_evaluator,
    num_trials=20
)
```

### 4. Multi-Objective Optimization
Balance multiple goals simultaneously.

```python
@traigent.optimize(
    objectives=[
        ("accuracy", "maximize"),
        ("cost", "minimize"),
        ("latency", "minimize")
    ],
    constraints={
        "cost": {"max": 0.05},  # Max $0.05 per call
        "latency": {"max": 2.0}  # Max 2 seconds
    }
)
def optimized_function(text: str) -> str:
    # Implementation
    pass
```

### 5. Async Pattern
For concurrent optimization and execution.

```python
import asyncio

@traigent.optimize(
    config_space={"model": ["gpt-3.5", "gpt-4"]},
    async_mode=True
)
async def async_llm_function(text: str) -> str:
    # Async implementation
    response = await llm_client.generate(text)
    return response

# Run async optimization
results = await async_llm_function.optimize_async(
    dataset=dataset,
    num_workers=5
)
```

### 6. Batch Processing Pattern
Optimize batch operations efficiently.

```python
from traigent.config.parallel import ParallelConfig


@traigent.optimize(
    parallel_config=ParallelConfig(
        mode="parallel",
        example_concurrency=10,
    ),
    config_space={"model": ["gpt-3.5", "claude-3"]},
)
def batch_process(texts: List[str]) -> List[str]:
    # Process multiple inputs
    return [process(text) for text in texts]

# Optimize with batches
results = batch_process.optimize(
    dataset=large_dataset,
    batch_mode=True
)
```

### 7. Caching Pattern
Avoid redundant API calls during optimization.

```python
from traigent.cache import OptimizationCache

@traigent.optimize(
    config_space={"temperature": [0.0, 0.5, 1.0]},
    cache=OptimizationCache("./cache")
)
def cached_function(text: str) -> str:
    # Expensive operation
    return expensive_llm_call(text)
```

### 8. Progressive Optimization
Start simple and progressively increase complexity.

```python
# Stage 1: Quick exploration
results_1 = function.optimize(
    config_space={"model": ["gpt-3.5", "gpt-4"]},
    num_trials=5,
    algorithm="random"
)

# Stage 2: Refined search
best_model = results_1.best_config["model"]
results_2 = function.optimize(
    config_space={
        "model": [best_model],
        "temperature": np.linspace(0, 1, 10),
        "top_p": np.linspace(0.5, 1, 5)
    },
    num_trials=20,
    algorithm="bayesian"
)
```

### 9. A/B Testing Pattern
Compare different approaches systematically.

```python
@traigent.optimize(
    variants={
        "approach_a": {"prompt": "Be concise", "model": "gpt-3.5"},
        "approach_b": {"prompt": "Be detailed", "model": "gpt-4"}
    },
    split_test=True
)
def ab_test_function(text: str, variant: str) -> str:
    config = get_variant_config(variant)
    return process_with_config(text, config)
```

### 10. Constraint-Based Optimization
Optimize within specific boundaries.

```python
@traigent.optimize(
    config_space={
        "model": ["gpt-3.5", "gpt-4", "claude-3"],
        "max_tokens": range(50, 500, 50)
    },
    constraints={
        "token_usage": lambda config: config["max_tokens"] * 2 < 1000,
        "cost_per_call": lambda config: estimate_cost(config) < 0.10
    }
)
def constrained_function(text: str) -> str:
    # Implementation
    pass
```

## Configuration Space Patterns

### Discrete Choices
```python
config_space = {
    "model": ["gpt-3.5", "gpt-4", "claude-3"],
    "prompt_style": ["formal", "casual", "technical"]
}
```

### Continuous Ranges
```python
config_space = {
    "temperature": np.linspace(0, 1, 11),  # 0.0, 0.1, ..., 1.0
    "top_p": np.arange(0.5, 1.0, 0.1)      # 0.5, 0.6, ..., 0.9
}
```

### Mixed Types
```python
config_space = {
    "model": ["gpt-3.5", "gpt-4"],
    "temperature": [0.0, 0.3, 0.7, 1.0],
    "use_cot": [True, False],
    "max_retries": range(1, 4)
}
```

### Conditional Spaces
```python
def get_config_space(task_type):
    base = {"model": ["gpt-3.5", "gpt-4"]}

    if task_type == "creative":
        base["temperature"] = [0.7, 0.9, 1.0]
    else:
        base["temperature"] = [0.0, 0.3, 0.5]

    return base
```

## Evaluation Patterns

### Simple Accuracy
```python
def accuracy_evaluator(output: str, expected: str) -> float:
    return 1.0 if output.strip() == expected.strip() else 0.0
```

### Fuzzy Matching
```python
from difflib import SequenceMatcher

def fuzzy_evaluator(output: str, expected: str) -> float:
    return SequenceMatcher(None, output, expected).ratio()
```

### Multi-Metric Evaluation
```python
def comprehensive_evaluator(output: str, expected: str) -> dict:
    return {
        "exact_match": output == expected,
        "contains": expected in output,
        "length_ratio": len(output) / len(expected),
        "word_overlap": calculate_word_overlap(output, expected)
    }
```

### LLM-as-Judge
```python
def llm_judge_evaluator(output: str, expected: str) -> float:
    prompt = f"Rate similarity (0-1): Output: {output}, Expected: {expected}"
    score = llm_client.evaluate(prompt)
    return float(score)
```

## Algorithm Selection

### Grid Search
Best for small search spaces with discrete parameters.
```python
results = function.optimize(algorithm="grid", exhaustive=True)
```

### Random Search
Efficient for large search spaces.
```python
results = function.optimize(algorithm="random", num_trials=50)
```

### Bayesian Optimization
Smart exploration for expensive evaluations.
```python
results = function.optimize(
    algorithm="bayesian",
    acquisition_function="ei",  # Expected Improvement
    num_initial_points=10
)
```

### Custom Algorithm
```python
from traigent.algorithms import CustomOptimizer

optimizer = CustomOptimizer(
    strategy="genetic",
    population_size=20,
    generations=10
)
results = function.optimize(optimizer=optimizer)
```

## Result Analysis Patterns

### Best Configuration
```python
best_config = results.best_config
best_score = results.best_score
print(f"Best: {best_config} with score {best_score}")
```

### Performance Visualization
```python
import matplotlib.pyplot as plt

results.plot_convergence()  # Show optimization progress
results.plot_pareto_front()  # For multi-objective
results.plot_parameter_importance()  # Feature importance
```

### Export Results
```python
# Save to JSON
results.to_json("optimization_results.json")

# Save to CSV
results.to_csv("optimization_results.csv")

# Get DataFrame
df = results.to_dataframe()
```

### Statistical Analysis
```python
# Summary statistics
print(results.summary())

# Confidence intervals
ci = results.confidence_interval(metric="accuracy", confidence=0.95)

# Parameter sensitivity
sensitivity = results.parameter_sensitivity()
```

## Error Handling Patterns

### Retry Logic
```python
@traigent.optimize(
    retry_failed_trials=True,
    max_retries=3,
    retry_delay=1.0
)
def robust_function(text: str) -> str:
    # May fail occasionally
    pass
```

### Fallback Strategy
```python
@traigent.optimize(
    fallback_config={"model": "gpt-3.5", "temperature": 0.3}
)
def function_with_fallback(text: str) -> str:
    # Falls back to safe config on error
    pass
```

### Error Recovery
```python
@traigent.optimize(
    on_error="continue",  # or "stop", "retry"
    error_score=0.0      # Score for failed trials
)
def error_tolerant_function(text: str) -> str:
    # Continues optimization despite errors
    pass
```

## Performance Tips

1. **Start Small**: Begin with few trials and expand
2. **Use Caching**: Enable caching for expensive operations
3. **Parallelize**: Use async mode or batch processing
4. **Progressive Refinement**: Start broad, then narrow search
5. **Mock Mode**: Test with mock mode before using real APIs
6. **Monitor Resources**: Track token usage and costs
7. **Early Stopping**: Stop when convergence is reached

## Common Integration Patterns

### With LangChain
```python
from langchain import LLMChain
import traigent

@traigent.optimize(config_space={...})
def optimized_chain(prompt: str) -> str:
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(prompt)
```

### With OpenAI
```python
import openai
import traigent

@traigent.optimize(config_space={...})
def optimized_openai(text: str) -> str:
    response = openai.ChatCompletion.create(
        model=config["model"],
        messages=[{"role": "user", "content": text}],
        temperature=config["temperature"]
    )
    return response.choices[0].message.content
```

### With Custom Frameworks
```python
@traigent.optimize(config_space={...})
def optimized_custom(text: str) -> str:
    # Your custom framework integration
    return custom_framework.process(text, **config)
```

---

**Remember**: These patterns are composable - combine them as needed for your specific use case!
