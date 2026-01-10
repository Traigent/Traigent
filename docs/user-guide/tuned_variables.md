# Tuned Variables Guide

This guide covers Traigent's tuned variables system, which provides domain-aware parameter ranges, callable auto-discovery, and post-optimization variable analysis.

## Overview

Tuned variables in Traigent consist of two components:

1. **Core SDK** (`traigent.tuned_variables`): Callable auto-discovery from source code
2. **Plugin** (`traigent-tuned-variables`): Domain presets, variable analysis, and TunedCallable pattern

## Installation

```bash
# Core tuned variables (included with traigent)
pip install traigent

# Plugin with domain presets and analysis
pip install traigent-tuned-variables

# With DSPy integration
pip install traigent-tuned-variables[dspy]
```

---

## Part 1: Callable Auto-Discovery

The core SDK provides utilities to automatically discover callable functions from Python modules. This is useful when you have multiple implementations of a component (e.g., different retrieval strategies) and want to expose them as optimization choices.

### Basic Discovery

```python
import my_retrieval_module
from traigent.tuned_variables import discover_callables

# Discover all public functions
callables = discover_callables(my_retrieval_module)

for name, info in callables.items():
    print(f"{name}: {info.signature}")
```

### Discovery with Filters

```python
from traigent.tuned_variables import discover_callables

# Find functions matching a pattern
retrievers = discover_callables(
    my_module,
    pattern=r"^(retrieve|search)_",  # Regex pattern
    required_params=["query", "k"],   # Must have these parameters
    return_type=list,                 # Must return this type
)
```

### Discovery by Decorator

Mark functions with a decorator attribute for explicit registration:

```python
# In my_retrievers.py
def traigent_callable(func):
    """Decorator to mark functions as tunable callables."""
    func.__traigent_callable__ = True
    return func

@traigent_callable
def similarity_search(query: str, k: int = 5) -> list:
    """Similarity-based retrieval."""
    ...

@traigent_callable
def mmr_search(query: str, k: int = 5, lambda_mult: float = 0.5) -> list:
    """MMR-based retrieval with diversity."""
    ...
```

```python
# Discovery
from traigent.tuned_variables import discover_callables_by_decorator
import my_retrievers

callables = discover_callables_by_decorator(my_retrievers)
# Returns only functions with __traigent_callable__ attribute
```

### Using Discovered Callables with Traigent

```python
import traigent
from traigent.api.parameter_ranges import Choices
from traigent.tuned_variables import discover_callables
import my_retrievers

# Discover retrieval functions
retrievers = discover_callables(
    my_retrievers,
    pattern=r"^(similarity|mmr|bm25)_search$",
    required_params=["query"]
)

# Create Choices for optimization
retriever_choices = Choices(list(retrievers.keys()), name="retriever")

@traigent.optimize(
    retriever=retriever_choices,
    k=traigent.IntRange(3, 10),
    objectives=["accuracy", "latency"],
)
def rag_pipeline(query: str) -> str:
    config = traigent.get_config()

    # Get the selected retriever function
    retriever_fn = retrievers[config["retriever"]].callable
    docs = retriever_fn(query=query, k=config["k"])

    return generate_answer(query, docs)
```

### CallableInfo Attributes

Each discovered callable is wrapped in a `CallableInfo` dataclass:

```python
@dataclass
class CallableInfo:
    name: str                    # Function name
    callable: Callable           # The function itself
    signature: inspect.Signature # Function signature
    module: str                  # Module name
    docstring: str | None        # Function docstring
    tags: tuple[str, ...]        # Extracted tags

    def matches_params(self, required_params: list[str]) -> bool:
        """Check if callable has all required parameters."""

    def matches_return_type(self, expected_type: type) -> bool:
        """Check if return annotation matches expected type."""
```

### Signature Filtering

Filter callables to match a target signature:

```python
from traigent.tuned_variables import discover_callables, filter_by_signature
import inspect

# Define a reference function with the target signature
def _target_signature(query: str, k: int) -> list:
    ...

# Get signature from the reference function
target_sig = inspect.signature(_target_signature)

# Filter to compatible callables
all_funcs = discover_callables(my_module)
compatible = filter_by_signature(all_funcs, target_sig, strict=False)
```

---

## Part 2: Domain Presets (Plugin)

The `traigent-tuned-variables` plugin provides pre-configured parameter ranges that encode domain knowledge.

### LLM Presets

```python
from traigent_tuned_variables import LLMPresets
import traigent

@traigent.optimize(
    temperature=LLMPresets.temperature(creative=True),  # Range(0.7, 1.5)
    top_p=LLMPresets.top_p(),                          # Range(0.1, 1.0)
    max_tokens=LLMPresets.max_tokens(task="medium"),   # IntRange(256, 1024)
    model=LLMPresets.model(provider="openai", tier="balanced"),
    objectives=["accuracy", "cost"],
)
def my_agent(query: str) -> str:
    ...
```

**Available LLM Presets:**

| Method | Description | Default Range |
|--------|-------------|---------------|
| `temperature(conservative=False, creative=False)` | Temperature control | 0.0-1.0 (default), 0.0-0.5 (conservative), 0.7-1.5 (creative) |
| `top_p()` | Nucleus sampling | 0.1-1.0 |
| `frequency_penalty()` | Frequency penalty | 0.0-2.0 |
| `presence_penalty()` | Presence penalty | 0.0-2.0 |
| `max_tokens(task="short"\|"medium"\|"long")` | Token limit | short: 50-256, medium: 256-1024, long: 1024-4096 |
| `model(provider=None, tier="fast"\|"balanced"\|"quality")` | Model selection | Provider/tier dependent |

### RAG Presets

```python
from traigent_tuned_variables import RAGPresets

@traigent.optimize(
    k=RAGPresets.k_retrieval(max_k=10),      # IntRange(1, 10)
    chunk_size=RAGPresets.chunk_size(),       # IntRange(100, 1000, step=100)
    objectives=["accuracy"],
)
def rag_agent(query: str) -> str:
    ...
```

### Prompting Presets

```python
from traigent_tuned_variables import PromptingPresets

@traigent.optimize(
    strategy=PromptingPresets.strategy(),       # CoT, few-shot, etc.
    context_format=PromptingPresets.context_format(),
    objectives=["accuracy"],
)
def prompt_agent(query: str) -> str:
    ...
```

### Environment-Based Model Configuration

Model presets use environment variables for flexibility:

```bash
# Override model lists via environment
export TRAIGENT_MODELS_OPENAI_FAST="gpt-4o-mini"
export TRAIGENT_MODELS_OPENAI_BALANCED="gpt-4o-mini,gpt-4o"
export TRAIGENT_MODELS_ANTHROPIC_QUALITY="claude-3-opus-20240229"
```

---

## Part 3: TunedCallable Pattern (Plugin)

`TunedCallable` is a composition pattern for function-valued variables with per-callable parameters.

### Basic Usage

```python
from traigent_tuned_variables import TunedCallable
from traigent.api.parameter_ranges import Range
import traigent

# Define callables with per-callable parameters
retrievers = TunedCallable(
    name="retriever",
    callables={
        "similarity": similarity_search,
        "mmr": mmr_search,
        "bm25": bm25_search,
    },
    parameters={
        "mmr": {"lambda_mult": Range(0.0, 1.0)},
        "bm25": {"b": Range(0.0, 1.0), "k1": Range(0.5, 2.0)},
    },
)

@traigent.optimize(
    retriever=retrievers.as_choices(),  # Converts to Choices
    objectives=["accuracy"],
)
def my_agent(query: str) -> str:
    config = traigent.get_config()

    # Invoke the selected callable
    docs = retrievers.invoke(config["retriever"], query=query)
    return generate_answer(query, docs)
```

### Full Configuration Space

For per-callable parameters, use `get_full_space()`:

```python
# Get full config space including per-callable params
space = retrievers.get_full_space()
# Returns: {
#     "retriever": Choices(["similarity", "mmr", "bm25"]),
#     "retriever.mmr.lambda_mult": Range(0.0, 1.0),
#     "retriever.bm25.b": Range(0.0, 1.0),
#     "retriever.bm25.k1": Range(0.5, 2.0),
# }

@traigent.optimize(**space, objectives=["accuracy"])
def my_agent(query: str) -> str:
    config = traigent.get_config()

    # Extract only params for selected callable
    params = retrievers.extract_callable_params(config)
    # If retriever="mmr", returns: {"lambda_mult": 0.7}

    fn = retrievers.get_callable(config["retriever"])
    docs = fn(query=query, **params)
    return generate_answer(query, docs)
```

### Pre-built Callables

The plugin includes pre-built `Retrievers` and `ContextFormatters`:

```python
from traigent_tuned_variables import Retrievers, ContextFormatters

@traigent.optimize(
    retriever=Retrievers.as_choices(),
    context_format=ContextFormatters.as_choices(),
    objectives=["accuracy"],
)
def rag_agent(query: str) -> str:
    config = traigent.get_config()

    docs = Retrievers.invoke(config["retriever"], vector_store, query)
    formatted = ContextFormatters.invoke(config["context_format"], docs)

    return llm(f"{formatted}\n\nQuestion: {query}")
```

---

## Part 4: Variable Analysis (Plugin)

After optimization, use `VariableAnalyzer` to understand which variables matter and which values to prune.

### Basic Analysis

```python
from traigent_tuned_variables import VariableAnalyzer

# After optimization completes
result = my_agent.optimize()

# Analyze for a specific objective
analyzer = VariableAnalyzer(result)
analysis = analyzer.analyze("accuracy")

# View elimination suggestions
for suggestion in analysis.elimination_suggestions:
    print(f"{suggestion.variable}: {suggestion.action.value}")
    print(f"  Reason: {suggestion.reason}")
    print(f"  Importance: {suggestion.importance_score:.3f}")
```

### Multi-Objective Analysis

```python
# Analyze across multiple objectives
analysis = analyzer.analyze_multi_objective(
    objectives=["accuracy", "cost", "latency"],
    aggregation="mean",  # or "max", "min"
)

# Get Pareto-dominated values
for var_name, frontier_values in analysis.pareto_frontier_values.items():
    print(f"{var_name}: Keep {frontier_values}")
```

### Refined Configuration Space

Get a pruned configuration space for the next optimization round:

```python
# Get refined space with dominated values removed
refined_space = analyzer.get_refined_space(
    objectives=["accuracy", "cost"],
    prune_low_importance=True,    # Remove low-importance variables
    prune_dominated_values=True,  # Remove dominated categorical values
    narrow_ranges=True,           # Narrow numeric ranges
)

# Use refined space for next optimization
@traigent.optimize(**refined_space, objectives=["accuracy", "cost"])
def my_agent_v2(query: str) -> str:
    ...
```

### Variable Importance

```python
# Get importance scores
importance = analyzer.get_variable_importance("accuracy")
for var, score in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"{var}: {score:.3f}")
```

### Value Rankings

```python
# Rank categorical values by performance
rankings = analyzer.get_value_rankings("model", "accuracy")
for ranking in rankings:
    status = " (dominated)" if ranking.is_dominated else ""
    print(f"{ranking.value}: {ranking.mean_score:.3f}{status}")
```

### Objective Direction Support

The analyzer respects optimization direction for each objective:

```python
analyzer = VariableAnalyzer(
    result,
    directions={
        "accuracy": "maximize",
        "cost": "minimize",
        "latency": "minimize",
    }
)
```

---

## Complete Example

```python
import traigent
from traigent.tuned_variables import discover_callables
from traigent_tuned_variables import (
    LLMPresets,
    RAGPresets,
    VariableAnalyzer,
    TunedCallable,
)
from traigent.api.parameter_ranges import Range
import my_retrievers

# 1. Discover callables from module
discovered = discover_callables(
    my_retrievers,
    pattern=r"^search_",
    required_params=["query"],
)

# 2. Create TunedCallable with per-callable params
retrievers = TunedCallable(
    name="retriever",
    callables={name: info.callable for name, info in discovered.items()},
    parameters={
        "search_mmr": {"lambda_mult": Range(0.0, 1.0)},
    },
)

# 3. Define optimization with domain presets
@traigent.optimize(
    retriever=retrievers.as_choices(),
    k=RAGPresets.k_retrieval(),
    temperature=LLMPresets.temperature(),
    model=LLMPresets.model(tier="balanced"),
    objectives=["accuracy", "cost"],
)
def optimized_rag(query: str) -> str:
    config = traigent.get_config()

    docs = retrievers.invoke(config["retriever"], query=query, k=config["k"])
    return generate_with_llm(query, docs, config["temperature"], config["model"])

# 4. Run optimization
result = optimized_rag.optimize()

# 5. Analyze results
analyzer = VariableAnalyzer(
    result,
    directions={"accuracy": "maximize", "cost": "minimize"},
)
analysis = analyzer.analyze_multi_objective(["accuracy", "cost"])

# 6. Get refined space for next iteration
refined = analyzer.get_refined_space(["accuracy", "cost"])
print(f"Refined space has {len(refined)} variables (down from original)")
```

---

## API Reference

### traigent.tuned_variables

| Function | Description |
|----------|-------------|
| `discover_callables(module, *, pattern=None, include_private=False, required_params=None, return_type=None)` | Auto-discover callables from a module |
| `discover_callables_by_decorator(module, decorator_attr="__traigent_callable__", *, include_private=False)` | Discover decorated callables |
| `filter_by_signature(callables, target_signature, *, strict=False)` | Filter callables by signature |
| `CallableInfo` | Dataclass with callable metadata |

### traigent_tuned_variables

| Class | Description |
|-------|-------------|
| `LLMPresets` | Pre-configured LLM parameter ranges |
| `RAGPresets` | Pre-configured RAG parameter ranges |
| `PromptingPresets` | Pre-configured prompting parameter ranges |
| `TunedCallable` | Composition pattern for function-valued variables |
| `VariableAnalyzer` | Post-optimization variable analysis |
| `Retrievers` | Pre-built retriever callables |
| `ContextFormatters` | Pre-built context formatting callables |

---

## See Also

- [DSPy Integration Guide](../../examples/docs/DSPY_INTEGRATION.md) - Prompt optimization with DSPy
- [Configuration Spaces](./configuration-spaces.md) - Parameter ranges and constraints
- [Evaluation Guide](./evaluation_guide.md) - Metrics and evaluation
