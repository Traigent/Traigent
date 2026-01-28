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

## Part 5: Reasoning & Chain-of-Thought TVARs

Modern LLMs support native reasoning capabilities (extended thinking, chain-of-thought). Traigent provides tuned variables for optimizing these parameters across providers.

### Overview

| Provider | Feature | TVARs |
|----------|---------|-------|
| **OpenAI** (o1/o3/GPT-5+) | Reasoning Effort | `Choices.reasoning_effort()`, `IntRange.reasoning_tokens()` |
| **Anthropic** (Claude 4+) | Extended Thinking | `Choices.extended_thinking()`, `IntRange.thinking_budget()` |
| **Gemini 3** | Thinking Level | `Choices.thinking_level()` |
| **Gemini 2.5** | Thinking Budget | `IntRange.gemini_thinking_budget()` |
| **All Providers** | Generic | `Choices.reasoning_mode()`, `IntRange.reasoning_budget()` |

### Generic Provider-Agnostic TVARs

Use these when optimizing across multiple providers:

```python
from traigent.api import Choices, IntRange
import traigent

@traigent.optimize(
    model=Choices(["o3", "claude-sonnet-4-5", "gemini-3-pro"]),
    reasoning_mode=Choices.reasoning_mode(),       # "none", "standard", "deep"
    reasoning_budget=IntRange.reasoning_budget(),  # 0-128000 tokens
    objectives=["accuracy", "cost"],
)
def multi_provider_reasoning(query: str) -> str:
    # reasoning_mode auto-translates to:
    # - OpenAI: reasoning_effort (none→minimal, standard→medium, deep→high)
    # - Anthropic: thinking.type + budget_tokens
    # - Gemini 3: thinking_level (none→MINIMAL, standard→low, deep→high)
    # - Gemini 2.5: thinking_budget (none→0, standard→8192, deep→32768)
    ...
```

**Generic TVARs:**

| Factory | Parameter | Range | Default |
|---------|-----------|-------|---------|
| `Choices.reasoning_mode(default="standard")` | `reasoning_mode` | `"none"`, `"standard"`, `"deep"` | `"standard"` |
| `IntRange.reasoning_budget()` | `reasoning_budget` | 0-128,000 tokens | 8,000 |

### OpenAI-Specific TVARs (o1/o3/GPT-5+)

```python
from traigent.api import Choices, IntRange
import traigent

@traigent.optimize(
    model=Choices(["o3", "o3-mini", "gpt-5"]),
    reasoning_effort=Choices.reasoning_effort(),   # "minimal" to "xhigh"
    max_completion_tokens=IntRange.reasoning_tokens(),  # 1024-128000
    objectives=["accuracy", "latency"],
)
def openai_reasoning(query: str) -> str:
    ...
```

**OpenAI TVARs:**

| Factory | Parameter | Range | Default | Notes |
|---------|-----------|-------|---------|-------|
| `Choices.reasoning_effort(default="medium")` | `reasoning_effort` | `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` | `"medium"` | `"minimal"`: GPT-5+ only; `"xhigh"`: GPT-5.1-codex-max only |
| `IntRange.reasoning_tokens()` | `max_completion_tokens` | 1,024-128,000 | 32,000 | Replaces `max_tokens` for reasoning models |

### Anthropic-Specific TVARs (Claude 4+)

```python
from traigent.api import Choices, IntRange
import traigent

@traigent.optimize(
    model=Choices(["claude-sonnet-4-5", "claude-opus-4-5"]),
    extended_thinking=Choices.extended_thinking(),  # True/False
    thinking_budget_tokens=IntRange.thinking_budget(),  # 1024-128000
    objectives=["accuracy", "cost"],
)
def anthropic_reasoning(query: str) -> str:
    ...
```

**Anthropic TVARs:**

| Factory | Parameter | Range | Default | Notes |
|---------|-----------|-------|---------|-------|
| `Choices.extended_thinking(default=False)` | `extended_thinking` | `True`, `False` | `False` | Toggle extended thinking |
| `IntRange.thinking_budget()` | `thinking_budget_tokens` | 1,024-128,000 | 8,000 | Min 1,024 enforced by API |

**Note:** Providing only `thinking_budget_tokens` (without `extended_thinking`) auto-enables thinking.

### Gemini-Specific TVARs

Gemini 3 and 2.5 use different parameters:

```python
from traigent.api import Choices, IntRange
import traigent

# Gemini 3: uses thinking_level
@traigent.optimize(
    model=Choices(["gemini-3-pro", "gemini-3-flash"]),
    thinking_level=Choices.thinking_level(),  # "MINIMAL", "low", "high"
    objectives=["accuracy"],
)
def gemini_3_reasoning(query: str) -> str:
    ...

# Gemini 2.5: uses thinking_budget
@traigent.optimize(
    model=Choices(["gemini-2.5-pro", "gemini-2.5-flash"]),
    thinking_budget=IntRange.gemini_thinking_budget(),  # 0-32768
    objectives=["accuracy"],
)
def gemini_2_5_reasoning(query: str) -> str:
    ...
```

**Gemini TVARs:**

| Factory | Parameter | Range | Default | Models |
|---------|-----------|-------|---------|--------|
| `Choices.thinking_level(default="high")` | `thinking_level` | `"MINIMAL"`, `"low"`, `"high"` | `"high"` | Gemini 3 |
| `IntRange.gemini_thinking_budget()` | `thinking_budget` | 0-32,768 | 8,192 | Gemini 2.5 |

### Model Capability Detection

Use these utilities to check reasoning support:

```python
from traigent.integrations.utils.model_capabilities import (
    supports_reasoning,
    get_reasoning_effort_levels,
    is_gemini_3,
    get_provider_from_model,
)

# Check if model supports reasoning
supports_reasoning("o3", "openai")           # True
supports_reasoning("gpt-4o", "openai")       # False
supports_reasoning("claude-sonnet-4-5", "anthropic")  # True
supports_reasoning("claude-3-opus", "anthropic")      # False
supports_reasoning("gemini-3-pro", "gemini")  # True
supports_reasoning("gemini-1.5-pro", "gemini")  # False

# Get available effort levels for OpenAI models
get_reasoning_effort_levels("o3")      # ["low", "medium", "high"]
get_reasoning_effort_levels("gpt-5")   # ["minimal", "low", "medium", "high"]
get_reasoning_effort_levels("gpt-5.1-codex-max")  # [..., "xhigh"]

# Check Gemini version
is_gemini_3("gemini-3-pro")    # True (uses thinking_level)
is_gemini_3("gemini-2.5-pro")  # False (uses thinking_budget)

# Detect provider from model name
get_provider_from_model("o3")          # "openai"
get_provider_from_model("claude-3")    # "anthropic"
get_provider_from_model("gemini-pro")  # "gemini"
```

### Translation Rules

| Generic Param | OpenAI | Anthropic | Gemini 3 | Gemini 2.5 |
|---------------|--------|-----------|----------|------------|
| `reasoning_mode="none"` | `reasoning_effort="minimal"` | *(omit thinking)* | `thinking_level="MINIMAL"` | `thinking_budget=0` |
| `reasoning_mode="standard"` | `reasoning_effort="medium"` | `thinking.type="enabled"` + default budget | `thinking_level="low"` | `thinking_budget=8192` |
| `reasoning_mode="deep"` | `reasoning_effort="high"` | `thinking.type="enabled"` + budget | `thinking_level="high"` | `thinking_budget=32768` |
| `reasoning_budget=N` | `max_completion_tokens=N` | `thinking.budget_tokens=N` | *(ignored)* | `thinking_budget=min(N,32768)` |

### Precedence Rules

1. **Provider-specific params win over generic params**
   ```python
   # reasoning_effort takes precedence over reasoning_mode
   @traigent.optimize(
       reasoning_mode=Choices.reasoning_mode(),
       reasoning_effort=Choices.reasoning_effort(),  # This wins
   )
   ```

2. **Non-reasoning models strip all reasoning params**
   - `gpt-4o`, `claude-3-opus`, `gemini-1.5-pro` will have reasoning params removed

3. **Effort level fallback**
   - If `reasoning_effort="minimal"` but model doesn't support it (e.g., o3), falls back to `"low"`
   - If `reasoning_effort="xhigh"` but model doesn't support it, falls back to `"high"`

### Complete Example

```python
import traigent
from traigent.api import Choices, IntRange

# Cross-provider optimization with automatic translation
@traigent.optimize(
    model=Choices([
        "o3",              # OpenAI reasoning
        "claude-sonnet-4-5",  # Anthropic extended thinking
        "gemini-3-pro",    # Gemini 3 thinking
    ]),
    reasoning_mode=Choices.reasoning_mode(),
    reasoning_budget=IntRange.reasoning_budget(),
    objectives=["accuracy", "cost", "latency"],
)
def optimized_reasoning_agent(query: str) -> str:
    config = traigent.get_config()
    model = config["model"]

    # The framework automatically translates reasoning_mode/reasoning_budget
    # to the correct provider-specific parameters
    return call_llm(model=model, query=query)
```

### API Reference - Reasoning TVARs

**Choices Factory Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `Choices.reasoning_mode(default="standard")` | `Choices[str]` | Generic reasoning mode across all providers |
| `Choices.reasoning_effort(default="medium")` | `Choices[str]` | OpenAI-specific reasoning effort level |
| `Choices.extended_thinking(default=False)` | `Choices[bool]` | Anthropic extended thinking toggle |
| `Choices.thinking_level(default="high")` | `Choices[str]` | Gemini 3 thinking level |

**IntRange Factory Methods:**

| Method | Returns | Range | Description |
|--------|---------|-------|-------------|
| `IntRange.reasoning_budget()` | `IntRange` | 0-128,000 | Generic reasoning token budget |
| `IntRange.reasoning_tokens()` | `IntRange` | 1,024-128,000 | OpenAI max_completion_tokens |
| `IntRange.thinking_budget()` | `IntRange` | 1,024-128,000 | Anthropic thinking budget |
| `IntRange.gemini_thinking_budget()` | `IntRange` | 0-32,768 | Gemini 2.5 thinking budget |

**Model Capabilities Module (`traigent.integrations.utils.model_capabilities`):**

| Function | Returns | Description |
|----------|---------|-------------|
| `supports_reasoning(model, provider)` | `bool` | Check if model supports native reasoning |
| `get_reasoning_effort_levels(model)` | `list[str]` | Get available reasoning effort levels |
| `is_gemini_3(model)` | `bool` | Check if Gemini 3 (vs 2.5) |
| `get_provider_from_model(model)` | `str \| None` | Detect provider from model name |

---

## See Also

- [DSPy Integration Guide](../../examples/docs/DSPY_INTEGRATION.md) - Prompt optimization with DSPy
- [Configuration Spaces](./configuration-spaces.md) - Parameter ranges and constraints
- [Evaluation Guide](./evaluation_guide.md) - Metrics and evaluation
