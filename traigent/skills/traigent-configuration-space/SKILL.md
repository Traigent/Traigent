---
name: traigent-configuration-space
description: "Define tuned variables and configuration spaces for Traigent optimization. Use when setting up parameter search spaces, choosing models/temperatures/prompts to optimize, using Range/IntRange/Choices/LogRange types, adding constraints between parameters, or using factory presets like Range.temperature()."
license: Apache-2.0
metadata:
  author: Traigent
  version: "1.0"
---

# Traigent Configuration Space

## When to Use

Use this skill when:

- Defining which parameters to optimize (model, temperature, max_tokens, prompts, etc.)
- Choosing between dict-based and typed parameter definitions
- Using `Range`, `IntRange`, `Choices`, or `LogRange` for search spaces
- Adding constraints between parameters (e.g., "if model is gpt-4, temperature must be low")
- Using factory presets like `Range.temperature()` or `Choices.model()`
- Bundling parameters and constraints into a `ConfigSpace` object

## Quick Start

The simplest way to define a configuration space is with a dictionary passed to `configuration_space`:

```python
import traigent

@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "max_tokens": [256, 512, 1024],
    },
)
def my_function(query: str) -> str:
    config = traigent.get_config()
    # config["model"], config["temperature"], config["max_tokens"]
    ...
```

Lists create categorical choices. Tuples of two numbers create continuous ranges:

```python
configuration_space={
    "model": ["gpt-4o-mini", "gpt-4o"],       # Categorical: pick one
    "temperature": (0.0, 1.0),                  # Continuous float range
    "max_tokens": (100, 4096),                  # Continuous int range (both ints)
}
```

## SE-Friendly Typed Parameters

For stronger typing, validation, and constraint support, use the parameter range classes:

```python
from traigent import Range, IntRange, Choices, LogRange
```

Pass them as keyword arguments directly to the decorator:

```python
import traigent
from traigent import Range, IntRange, Choices

@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy", "cost"],
    model=Choices(["gpt-4o-mini", "gpt-4o"]),
    temperature=Range(0.0, 1.0),
    max_tokens=IntRange(100, 4096),
)
def my_function(query: str) -> str:
    config = traigent.get_config()
    ...
```

### Parameter Types

| Class      | Use Case                    | Example                                      |
| ---------- | --------------------------- | -------------------------------------------- |
| `Range`    | Continuous float values     | `Range(0.0, 1.0)`                            |
| `IntRange` | Integer values              | `IntRange(100, 4096)`                         |
| `Choices`  | Categorical selection       | `Choices(["gpt-4o-mini", "gpt-4o"])`         |
| `LogRange` | Log-scale float values      | `LogRange(1e-5, 1e-1)`                        |

### Range Options

```python
# Basic float range
temperature = Range(0.0, 1.0)

# With step size (discretized)
temperature = Range(0.0, 1.0, step=0.1)

# With log-scale sampling
learning_rate = Range(1e-5, 1e-1, log=True)

# With a default value
temperature = Range(0.0, 1.0, default=0.7)
```

### IntRange Options

```python
# Basic integer range
max_tokens = IntRange(100, 4096)

# With step size
batch_size = IntRange(16, 256, step=16)

# With a default
max_tokens = IntRange(100, 4096, default=512)
```

### LogRange

Convenience class for `Range(low, high, log=True)`. Use for parameters that vary over orders of magnitude:

```python
learning_rate = LogRange(1e-5, 1e-1)
regularization = LogRange(0.001, 10.0)
```

### Choices Options

```python
# String choices
model = Choices(["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"])

# Boolean choices
use_cot = Choices([True, False], default=True)

# Numeric choices (discrete set)
temperature = Choices([0.0, 0.3, 0.7, 1.0])

# With a default
model = Choices(["gpt-4o-mini", "gpt-4o"], default="gpt-4o-mini")
```

See `references/parameter-types.md` for complete API details.

## Factory Presets

Each parameter type offers factory methods with sensible defaults for common LLM parameters.

### Range Presets

```python
from traigent import Range

# Temperature
temp = Range.temperature()                       # [0.0, 1.0], default 0.7
temp = Range.temperature(conservative=True)      # [0.0, 0.5], default 0.2
temp = Range.temperature(creative=True)          # [0.7, 1.5], default 1.0

# Other LLM parameters
top_p = Range.top_p()                            # [0.1, 1.0], default 0.9
freq_pen = Range.frequency_penalty()             # [0.0, 2.0], default 0.0
pres_pen = Range.presence_penalty()              # [0.0, 2.0], default 0.0

# RAG parameters
threshold = Range.similarity_threshold()         # [0.0, 1.0], default 0.5
mmr = Range.mmr_lambda()                         # [0.0, 1.0], default 0.5
overlap = Range.chunk_overlap_ratio()            # [0.0, 0.5], default 0.1
```

### IntRange Presets

```python
from traigent import IntRange

# Token limits
tokens = IntRange.max_tokens()                   # [256, 1024], step 64, default 512
tokens = IntRange.max_tokens(task="short")       # [50, 256], step 64, default 128
tokens = IntRange.max_tokens(task="long")        # [1024, 4096], step 64, default 2048

# RAG parameters
k = IntRange.k_retrieval()                       # [1, 10], default 3
k = IntRange.k_retrieval(max_k=20)               # [1, 20], default 3
chunk = IntRange.chunk_size()                     # [100, 1000], step 100, default 500
overlap = IntRange.chunk_overlap()                # [0, 200], step 25, default 50
few_shot = IntRange.few_shot_count()              # [0, 10], default 3
batch = IntRange.batch_size()                     # [1, 64], default 16
```

### Choices Presets

```python
from traigent import Choices

# Model selection
model = Choices.model()                                   # Balanced: gpt-4o-mini, gpt-4o, claude-3-5-sonnet
model = Choices.model(provider="openai", tier="fast")     # Fast OpenAI: gpt-4o-mini
model = Choices.model(provider="anthropic", tier="quality")  # Quality Anthropic: claude-3-opus

# Prompting and RAG
strategy = Choices.prompting_strategy()           # direct, chain_of_thought, react, self_consistency
ctx_fmt = Choices.context_format()                # bullet, numbered, xml, markdown, json
retriever = Choices.retriever_type()              # similarity, mmr, bm25, hybrid
embedding = Choices.embedding_model()             # text-embedding-3-small, text-embedding-3-large
reranker = Choices.reranker_model()               # none, cohere-rerank-v3, cross-encoder, llm-rerank
```

## Constraints

Constraints define valid parameter combinations. They prevent the optimizer from exploring invalid configurations.

### Lambda Constraints

The simplest form -- pass lambda functions that return `True` for valid configs:

```python
@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "max_tokens": [256, 512, 1024],
    },
    constraints=[
        # If model is gpt-4o, temperature must be below 0.8
        lambda config: config["temperature"] < 0.8 if config["model"] == "gpt-4o" else True,
        # Max tokens must be at least 512
        lambda config: config["max_tokens"] >= 512,
    ],
)
def my_function(query: str) -> str:
    ...
```

Lambda constraints can also receive metrics from past trials:

```python
constraints=[
    lambda config, metrics: metrics.get("cost", 0) <= 0.10,
]
```

### Builder-Style Constraints

For typed parameters, use the builder methods on `Range`, `IntRange`, and `Choices` objects combined with the `implies()` function:

```python
from traigent import Range, IntRange, Choices, implies

model = Choices(["gpt-4o-mini", "gpt-4o"])
temperature = Range(0.0, 1.0)
max_tokens = IntRange(100, 4096)

@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy"],
    model=model,
    temperature=temperature,
    max_tokens=max_tokens,
    constraints=[
        # If model is gpt-4o, temperature must be <= 0.7
        implies(model.equals("gpt-4o"), temperature.lte(0.7)),
        # If model is gpt-4o-mini, max_tokens must be >= 256
        implies(model.equals("gpt-4o-mini"), max_tokens.gte(256)),
    ],
)
def my_function(query: str) -> str:
    ...
```

### Three Syntax Styles for Constraints

All three produce equivalent `Constraint` objects:

```python
from traigent import Range, Choices, implies, when

model = Choices(["gpt-4o-mini", "gpt-4o"])
temp = Range(0.0, 1.0)

# 1. Functional (canonical)
implies(model.equals("gpt-4o"), temp.lte(0.7))

# 2. Operator-based (concise)
model.equals("gpt-4o") >> temp.lte(0.7)

# 3. Fluent (readable)
when(model.equals("gpt-4o")).then(temp.lte(0.7))
```

### Combining Conditions

Use `&` (and), `|` (or), `~` (not) operators to combine conditions:

```python
from traigent import Range, Choices, implies

model = Choices(["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"])
temp = Range(0.0, 1.5)
max_tokens = IntRange(100, 4096)

constraints = [
    # If model is gpt-4o AND temperature is high, require many tokens
    implies(
        model.equals("gpt-4o") & temp.gte(0.8),
        max_tokens.gte(1024),
    ),
    # If model is NOT gpt-4o-mini, temperature must be <= 1.0
    implies(
        ~model.equals("gpt-4o-mini"),
        temp.lte(1.0),
    ),
    # If model is gpt-4o OR claude, limit temperature
    implies(
        model.equals("gpt-4o") | model.equals("claude-3-5-sonnet-20241022"),
        temp.lte(0.9),
    ),
]
```

**Operator precedence warning:** Python precedence is `~` > `>>` > `&` > `|`. Always use parentheses when combining `&`/`|` with `>>`:

```python
# Correct
(model.equals("gpt-4o") & temp.lte(0.7)) >> max_tokens.gte(1000)

# Wrong -- evaluates as model.equals("gpt-4o") & (temp.lte(0.7) >> max_tokens.gte(1000))
model.equals("gpt-4o") & temp.lte(0.7) >> max_tokens.gte(1000)
```

See `references/constraints.md` for the full constraint system reference.

## ConfigSpace Object

For complex setups, bundle parameters and constraints into a `ConfigSpace`:

```python
from traigent import Range, IntRange, Choices, implies
from traigent.api.config_space import ConfigSpace

# Define parameters
temperature = Range(0.0, 1.0, name="temperature", unit="ratio")
max_tokens = IntRange(100, 4096, name="max_tokens", unit="tokens")
model = Choices(["gpt-4o-mini", "gpt-4o"], name="model")

# Define constraints
constraints = [
    implies(model.equals("gpt-4o"), temperature.lte(0.7)),
]

# Bundle into ConfigSpace
space = ConfigSpace(
    tvars={"temperature": temperature, "max_tokens": max_tokens, "model": model},
    constraints=constraints,
    description="QA optimization space",
)

# Validate a configuration
result = space.validate({"temperature": 0.5, "max_tokens": 2000, "model": "gpt-4o"})
print(result.is_valid)  # True

# Check satisfiability (are there any valid configs?)
sat = space.check_satisfiability()
print(sat)

# Use with decorator
@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy"],
    configuration_space=space,
)
def my_function(query: str) -> str:
    config = traigent.get_config()
    ...
```

## Common Patterns

### Model Selection with Cost Awareness

```python
from traigent import Range, Choices, implies

model = Choices(["gpt-4o-mini", "gpt-4o"])
temperature = Range(0.0, 1.0)

@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy", "cost"],
    model=model,
    temperature=temperature,
    constraints=[
        implies(model.equals("gpt-4o"), temperature.lte(0.7)),
    ],
)
def answer(question: str) -> str:
    config = traigent.get_config()
    ...
```

### RAG Pipeline Tuning

```python
from traigent import Range, IntRange, Choices

@traigent.optimize(
    eval_dataset="rag_evals.jsonl",
    objectives=["accuracy"],
    model=Choices.model(provider="openai"),
    temperature=Range.temperature(conservative=True),
    k=IntRange.k_retrieval(),
    chunk_size=IntRange.chunk_size(),
    chunk_overlap=IntRange.chunk_overlap(),
    retriever=Choices.retriever_type(),
)
def rag_query(question: str) -> str:
    config = traigent.get_config()
    ...
```

### Multi-Agent Parameters

Assign parameters to specific agents using the `agent` keyword:

```python
from traigent import Range, Choices

@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy"],
    planner_model=Choices(["gpt-4o-mini", "gpt-4o"], agent="planner"),
    planner_temperature=Range(0.0, 0.5, agent="planner"),
    executor_model=Choices(["gpt-4o-mini", "gpt-4o"], agent="executor"),
    executor_temperature=Range(0.5, 1.0, agent="executor"),
)
def multi_agent_workflow(task: str) -> str:
    config = traigent.get_config()
    ...
```

### Temperature Tuning Only

The minimal configuration -- just optimize temperature:

```python
import traigent
from traigent import Range

@traigent.optimize(
    eval_dataset="evals.jsonl",
    objectives=["accuracy"],
    temperature=Range.temperature(),
)
def my_function(query: str) -> str:
    config = traigent.get_config()
    ...
```
