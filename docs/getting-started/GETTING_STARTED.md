# Getting Started with TraiGent SDK

Welcome to TraiGent SDK - the first optimization platform that enhances your LLM applications **without requiring any code changes**.

## 🚀 Quick Start

### Installation

From source (recommended for examples):

```bash
python3 -m pip install -r requirements/requirements.txt \
                          -r requirements/requirements-integrations.txt -e .
python3 -m pip install langchain-anthropic
```

### Your First Optimization

```python
import traigent

@traigent.optimize(
    eval_dataset="examples/datasets/hello-world/evaluation_set.jsonl",
    objectives=["accuracy"],
    configuration_space={
        "model": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "temperature": [0.0, 0.3],
    },
)
def answer_question(question: str) -> str:
    cfg = traigent.get_config()  # Unified access during and after optimization
    # Call your LLM here using cfg["model"], cfg["temperature"]
    return "example"

# Prefer bundles when tweaking multiple related knobs
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

@traigent.optimize(
    configuration_space={"temperature": [0.1, 0.3, 0.5]},
    evaluation=EvaluationOptions(
        eval_dataset="examples/datasets/hello-world/evaluation_set.jsonl",
        scoring_function=lambda output, expected, _: float(output == expected),
    ),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def bundled_answer(question: str) -> str:
    ...

# Run optimization (async). Use asyncio.run() in sync contexts.
if __name__ == "__main__":
    import asyncio

    results = asyncio.run(answer_question.optimize())
    print({"best_config": results.best_config, "best_score": results.best_score})
```

### Config Access Lifecycle

| When | Use | Notes |
| --- | --- | --- |
| Inside the optimized function | `traigent.get_config()` | Unified access during trials and after `apply_best_config()`. |
| During optimization (strict) | `traigent.get_trial_config()` | Raises `OptimizationStateError` if called outside an active trial. |
| After optimization completes | `results.best_config` | Returned from `func.optimize()`. |
| When calling the function later | `answer_question.current_config` | Automatically set to the best config found. |

```python
# After the run finishes
results = await answer_question.optimize()
print(results.best_config)            # Best trial config
print(answer_question.current_config) # Same config applied to future calls
```

## 🎯 Core Concepts

### 1. The `@traigent.optimize` Decorator

This is your main entry point. It wraps your existing function and automatically finds the best configuration:

```python
@traigent.optimize(
    eval_dataset="path/to/data.jsonl",     # Your evaluation data
    objectives=["accuracy"],                # What to optimize for
    configuration_space={                   # Parameters to explore
        "model": ["o4-mini", "GPT-4o"],
        "temperature": (0.0, 1.0),
        "max_tokens": [100, 500, 1000]
    }
)
def my_function(input_text: str) -> str:
    config = traigent.get_config()  # Get config for the current call
    # Your code here
    return result
```

### 2. Seamless Framework Integration

TraiGent automatically optimizes your existing LangChain and OpenAI code:

```python
# Enable automatic framework override
@traigent.optimize(
    eval_dataset="data.jsonl",
    objectives=["accuracy", "cost"],
    auto_override_frameworks=True,  # Magic happens here!
    configuration_space={
        "model": ["o4-mini", "GPT-4o"],
        "temperature": (0.0, 1.0)
    }
)
def langchain_function(query: str) -> str:
    # Your existing LangChain code - no changes needed!
    chain = LLMChain(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        prompt=PromptTemplate(...)
    )
    return chain.run(query)
```

### 3. Dataset Format

Your evaluation dataset should be in JSONL format:

```jsonl
{"input": {"question": "What is AI?"}, "output": "Artificial Intelligence"}
{"input": {"question": "How does machine learning work?"}, "output": "Uses data and algorithms"}
{"input": {"question": "What does RAG stand for?"}, "output": "Retrieval Augmented Generation"}
```

## 🧪 Mock Mode and Examples

- Enable mock mode to run quickly without API keys:
  - `TRAIGENT_MOCK_MODE=true python examples/core/hello-world/run.py`
- Open the Examples Navigator:
  - `examples/index.html` (or `python -m http.server -d examples 8000` → http://localhost:8000)
- Use the Run button in the docs pages to copy exact commands.

### 4. Multiple Objectives

Optimize for multiple metrics simultaneously:

```python
from traigent.api.decorators import EvaluationOptions

@traigent.optimize(
    objectives=["accuracy", "cost", "latency"],  # Multi-objective optimization
    evaluation=EvaluationOptions(eval_dataset="data.jsonl"),
    configuration_space={
        "model": ["o4-mini", "GPT-4o", "gpt-4-turbo"],
        "temperature": (0.0, 1.0)
    }
)
def my_function(text: str) -> str:
    # TraiGent finds the best balance of accuracy, cost, and speed
    pass
```

## 🔧 Advanced Features

### Cloud Optimization

Use TraiGent locally by default (`execution_mode="edge_analytics"`). Cloud orchestration is gated/experimental; keep local mode unless your environment is configured for cloud.

```python
from traigent.api.decorators import ExecutionOptions

@traigent.optimize(
    eval_dataset="data.jsonl",
    objectives=["accuracy", "cost"],
    execution=ExecutionOptions(
        execution_mode="edge_analytics",  # local-first default
        parallel_config={"trial_concurrency": 2},
    ),
    algorithm="grid",
)
def locally_optimized_function(data):
    return process(data)
```

### Enterprise Security

For enterprise deployments:

```python
@traigent.optimize(
    eval_dataset="data.jsonl",
    objectives=["accuracy"],
    # Security features
    encrypt_data=True,
    audit_logging=True,
    tenant_id="my-organization"
)
def secure_function(data):
    return process_sensitive_data(data)
```

## 📊 Understanding Results

After optimization, you get comprehensive results:

```python
import asyncio

async def main():
    results = await my_function.optimize()
    print(f"Best config: {results.best_config}")
    print(f"Accuracy: {results.best_metrics['accuracy']}")
    print(f"Cost per call: {results.best_metrics['cost']}")
    print(f"Average latency: {results.best_metrics['latency']}")
    print(f"Total trials: {len(results.all_results)}")
    print(f"Optimization time: {results.optimization_duration}")

asyncio.run(main())
```

## 🛠️ CLI Tools

TraiGent includes powerful CLI tools:

```bash
# Show version/info
traigent info

# List available algorithms
traigent algorithms

# Run optimization on a file with @optimize
traigent optimize examples/core/hello-world/run.py -a grid --max-trials 5

# Validate a dataset or config
traigent validate examples/datasets/hello-world/evaluation_set.jsonl
traigent validate-config optimization_config.json

# List and plot stored results
traigent results
traigent plot my_run -p progress

# Generate starter templates
traigent generate --template langchain --output traigent_example.py
```

## 📚 Next Steps

1. **[Examples](../examples/)** - See complete working examples
2. **[API Reference](../api-reference/complete-function-specification.md)** - Detailed API documentation
3. **[Architecture Guide](../architecture/ARCHITECTURE.md)** - Understand the system design
4. **[User Guides](../user-guide/)** - LangChain, OpenAI, interactive/hybrid guides

## 🆘 Common Issues

### Import Error

```bash
# Make sure TraiGent is installed
pip install traigent

# For development
pip install -e .
```

### Dataset Format Error

```python
# Ensure your dataset is valid JSONL
# Each line should be a JSON object with 'input' and 'output'
```

### Configuration Issues

```python
# Use traigent.get_config() inside your function to read applied parameters
config = traigent.get_config()
model = config.get("model", "o4-mini")  # Provide defaults

# After optimization, access the best config via the result:
# import asyncio
# result = asyncio.run(my_function.optimize())
# print(result.best_config)
```

---

Ready to optimize your LLM applications? Check out our [examples](../examples/) for complete working code!
