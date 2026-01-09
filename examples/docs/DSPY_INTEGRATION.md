# DSPy Integration Guide

This guide explains how to use Traigent's DSPy integration for automatic prompt optimization alongside hyperparameter tuning.

## Overview

Traigent integrates with [DSPy](https://github.com/stanfordnlp/dspy) to provide:

- **Automatic prompt engineering** via MIPROv2 and BootstrapFewShot
- **Combined optimization** of prompts AND hyperparameters (model, temperature, etc.)
- **Best-of-both-worlds** approach leveraging each tool's strengths

| Tool | Strength |
|------|----------|
| **DSPy** | Prompt optimization (MIPROv2, BootstrapFewShot) |
| **Traigent** | Hyperparameter optimization (model, temperature, RAG params) |
| **Combined** | Optimized prompts + optimized configs |

## Installation

```bash
# Install Traigent with DSPy support
pip install traigent[dspy]

# Or install DSPy separately
pip install dspy-ai>=2.5.0
```

## Integration Patterns

### Pattern 1: DSPy Prompt Choices with Traigent Optimization

Use DSPy to create prompt variants that Traigent then optimizes alongside other hyperparameters.

```python
from traigent.integrations.dspy_adapter import DSPyPromptOptimizer
import traigent

@traigent.optimize(
    # DSPy generates prompt variants that Traigent explores
    prompt=DSPyPromptOptimizer.create_prompt_choices([
        "You are a helpful assistant. Answer concisely.",
        "You are an expert. Provide detailed, accurate answers.",
        "Think step by step before answering the question.",
        "Answer only with the most relevant information.",
    ]),
    model=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    temperature=[0.1, 0.3, 0.5, 0.7],
    objectives=["accuracy", "cost", "latency"],
    eval_dataset="validation.jsonl",  # Use validation set!
)
def qa_with_prompt_optimization(question: str) -> str:
    """Traigent optimizes both the prompt AND model/temperature."""
    config = traigent.get_config()
    llm = ChatOpenAI(model=config["model"], temperature=config["temperature"])
    return llm.invoke(config["prompt"] + "\n\nQuestion: " + question).content
```

### Pattern 2: Direct DSPy Prompt Optimization

Use DSPy's optimizers directly for advanced prompt engineering, then use the optimized module.

```python
from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

# Create optimizer
optimizer = DSPyPromptOptimizer(
    method="mipro",           # Use MIPROv2 for instruction optimization
    teacher_model="gpt-4o",   # Use GPT-4o to generate better prompts
    auto_setting="medium",    # Balance thoroughness vs speed
)

# Optimize a DSPy module's prompts
result = optimizer.optimize_prompt(
    module=my_dspy_module,    # Your DSPy module
    trainset=train_examples,  # Training data (small set, ~10-50 examples)
    metric=accuracy_metric,   # Evaluation function
)

print(f"Best score: {result.best_score}")
print(f"Num demos: {result.num_demos}")
optimized_module = result.optimized_module
```

### Pattern 3: Combined Workflow

First use DSPy to generate optimized prompts, then use those prompts in Traigent optimization.

```python
from traigent.integrations.dspy_adapter import DSPyPromptOptimizer

# Step 1: Use DSPy to generate optimized prompts on training data
optimizer = DSPyPromptOptimizer(method="mipro", teacher_model="gpt-4o")

result = optimizer.optimize_prompt(
    module=my_module,
    trainset=train_examples,  # Training split
    metric=accuracy_metric,
)

# Extract the optimized prompt/instructions
optimized_prompt = extract_instructions(result.optimized_module)

# Step 2: Use Traigent to optimize model/temperature with the optimized prompt
@traigent.optimize(
    prompt=[optimized_prompt],  # Use DSPy-optimized prompt
    model=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    temperature=[0.1, 0.3, 0.5],
    objectives=["accuracy", "cost", "latency"],
    eval_dataset="validation.jsonl",  # Validation split (different!)
)
def optimized_agent(question: str) -> str:
    config = traigent.get_config()
    # ... implementation
```

## Dataset Separation (Critical!)

**To avoid overfitting, use different data splits for each optimization stage.**

### Official DSPy Recommendations

From [DSPy's documentation](https://dspy.ai/learn/evaluation/overview/):
> "Even 20 input examples can be useful, though **200 goes a long way**."

From [DSPy Optimizers](https://dspy.ai/learn/optimization/optimizers/):
> "Ideally **200 examples or more to prevent overfitting** when running longer optimization runs."

### Recommended Dataset Sizes

| Dataset | Purpose | Size | Notes |
|---------|---------|------|-------|
| `trainset` | DSPy prompt optimization | 10-200 examples | BootstrapFewShot: ~10, MIPROv2: 50-200+ |
| `eval_dataset` | Traigent config optimization | Validation set | Separate from DSPy trainset |
| `test_dataset` | Final evaluation | Held-out | Never seen during any optimization |

### Per-Optimizer Data Requirements

| Optimizer | Minimum | Recommended | Official Notes |
|-----------|---------|-------------|----------------|
| **BootstrapFewShot** | ~10 | 10-50 | "Works with very few examples" |
| **MIPROv2 (light)** | ~20 | 50+ | For instruction-only optimization |
| **MIPROv2 (medium/heavy)** | 50+ | **200+** | "To prevent overfitting" |

### Why Separation Matters

If you use the same dataset for both:
1. DSPy optimizes prompts specifically for those examples
2. Traigent then evaluates configs on the same examples
3. **Result:** Overfitting - great scores on training data, poor generalization

### Correct Approach

```python
# Load data and split (following DSPy's pattern)
all_data = load_jsonl("data.jsonl")
train, validation, test = split_data(all_data, [0.3, 0.35, 0.35])

# DSPy uses training split (following their cheatsheet pattern)
result = optimizer.optimize_prompt(
    module=my_module,
    trainset=train,  # 30% of data (~60 examples from 200)
    metric=accuracy,
)

# Traigent uses validation split
@traigent.optimize(
    ...
    eval_dataset=validation,  # 35% of data (different!)
)
def my_agent(...): ...

# Final evaluation on test split
test_score = evaluate(my_agent, test)  # 35% of data (never seen!)
```

### DSPy's Official Split Pattern

From the [DSPy Cheatsheet](https://dspy.ai/cheatsheet/):
```python
# DSPy explicitly uses trainset and valset
compile(student=your_program, trainset=trainset, valset=devset)

# Common pattern for splitting
trainset=trainset[:some_num], valset=trainset[some_num:]
```

## DSPy Optimizer Options

### MIPROv2 (Recommended)

Best for instruction optimization and generating effective prompts.

```python
optimizer = DSPyPromptOptimizer(
    method="mipro",
    teacher_model="gpt-4o",     # Higher-quality teacher for prompt generation
    auto_setting="medium",      # "light", "medium", or "heavy"
)

result = optimizer.optimize_prompt(
    module=my_module,
    trainset=train_examples,
    metric=accuracy_metric,
    num_candidates=10,          # Number of prompt candidates to evaluate
    requires_permission_to_run=False,  # Skip confirmation prompts
)
```

### BootstrapFewShot

Best for generating few-shot examples from unlabeled data.

```python
optimizer = DSPyPromptOptimizer(
    method="bootstrap",
    teacher_model="gpt-4o",
)

result = optimizer.optimize_prompt(
    module=my_module,
    trainset=train_examples,
    metric=accuracy_metric,
    max_bootstrapped_demos=4,   # Max demos to generate
    max_labeled_demos=8,        # Max labeled demos to use
)
```

## API Reference

### DSPyPromptOptimizer

```python
class DSPyPromptOptimizer:
    def __init__(
        self,
        method: Literal["mipro", "bootstrap"] = "mipro",
        *,
        teacher_model: str | None = None,
        auto_setting: Literal["light", "medium", "heavy"] = "medium",
    ): ...

    def optimize_prompt(
        self,
        module: Any,
        trainset: list[Any],
        metric: Callable[[Any, Any], float],
        *,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 8,
        num_candidates: int = 10,
        requires_permission_to_run: bool = False,
    ) -> PromptOptimizationResult: ...

    @classmethod
    def create_prompt_choices(
        cls,
        base_prompts: list[str],
        *,
        trainset: list[Any] | None = None,
        metric: Callable[[Any, Any], float] | None = None,
        name: str = "prompt",
        return_choices: bool = True,
    ) -> Choices | list[str]: ...
```

### PromptOptimizationResult

```python
@dataclass
class PromptOptimizationResult:
    optimized_module: Any      # The optimized DSPy module
    method: str                # "mipro" or "bootstrap"
    num_demos: int             # Number of demonstrations in optimized module
    trainset_size: int         # Size of training set used
    best_score: float | None   # Best score achieved
    metadata: dict[str, Any]   # Additional optimization metadata
```

## When to Use Each Pattern

| Pattern | Use When |
|---------|----------|
| **Pattern 1** (create_prompt_choices) | You have a few prompt templates and want Traigent to pick the best |
| **Pattern 2** (DSPyPromptOptimizer) | You want DSPy to auto-generate/refine prompts with few-shot examples |
| **Pattern 3** (Combined) | You want the best of both - DSPy generates prompts, Traigent optimizes configs |

## Example: Complete Workflow

```python
import traigent
from traigent.integrations.dspy_adapter import DSPyPromptOptimizer
from langchain_openai import ChatOpenAI

# Load and split data
train_data = load_jsonl("train.jsonl")      # For DSPy
val_data = load_jsonl("validation.jsonl")   # For Traigent

# Define accuracy metric
def accuracy(example, pred):
    return float(example.answer.lower() in pred.answer.lower())

# Create prompt choices (Pattern 1)
@traigent.optimize(
    prompt=DSPyPromptOptimizer.create_prompt_choices([
        "Answer the following question concisely.",
        "You are an expert. Provide a detailed answer.",
        "Think step by step, then give your final answer.",
    ]),
    model=["gpt-3.5-turbo", "gpt-4o-mini"],
    temperature=[0.1, 0.3, 0.5],
    objectives=["accuracy", "cost", "latency"],
    eval_dataset=val_data,  # Validation set
    max_trials=10,
)
def qa_agent(question: str) -> str:
    config = traigent.get_config()
    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
    )
    prompt = config["prompt"] + f"\n\nQuestion: {question}"
    return str(llm.invoke(prompt).content)

# Run optimization
results = await qa_agent.optimize()

# Use best config
print(f"Best config: {results.best_config}")
print(f"Best score: {results.best_score}")
```

## Troubleshooting

### DSPy Not Available

```python
from traigent.integrations.dspy_adapter import DSPY_AVAILABLE

if not DSPY_AVAILABLE:
    print("DSPy is not installed. Install with: pip install dspy-ai>=2.5.0")
```

### Import Error

```bash
# Ensure you have the correct version
pip install "dspy-ai>=2.5.0"
```

### Optimization Taking Too Long

Use lighter settings for faster (but less thorough) optimization:

```python
optimizer = DSPyPromptOptimizer(
    method="mipro",
    auto_setting="light",  # Faster than "medium" or "heavy"
)

result = optimizer.optimize_prompt(
    module=my_module,
    trainset=train_examples[:20],  # Use fewer examples
    metric=accuracy,
    num_candidates=5,  # Fewer candidates to evaluate
)
```

## Further Reading

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [MIPROv2 Paper](https://arxiv.org/abs/2406.11695)
- [Traigent Quickstart](../quickstart/README.md)
- [Demo Walkthrough Notebook](../quickstart/demo_walkthrough.ipynb)
