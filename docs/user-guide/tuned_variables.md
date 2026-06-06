# Tuned Variables Guide

This guide covers Traigent's tuned variables system, which provides domain-aware parameter ranges, callable auto-discovery, and post-optimization variable analysis.

## Overview

Tuned variables in Traigent consist of two components:

1. **Core SDK** (`traigent.tuned_variables`): Callable auto-discovery from source code
2. **Recommendation catalog**: `traigent recommend` plus `recommend_configuration_space()` and `list_recommendation_agent_types()` for evidence-backed configuration-space suggestions

## Installation

```bash
pip install "traigent[recommended]"
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

## Part 2: Recommendation Catalog

Traigent 0.12.0 exposes configuration-space recommendations through the core SDK and CLI. There is no separate `traigent-tuned-variables` package to install for this path.

### CLI

```bash
traigent recommend --list-types
traigent recommend rag
traigent recommend code_gen --min-impact medium
```

The recommendations are advisory catalog entries. They include task-local evidence notes and suggested ranges, but they are not a statistical certificate for your dataset. Validate returned knobs on your own evaluation dataset.

### Python API

```python
from traigent import (
    list_recommendation_agent_types,
    recommend_configuration_space,
)

agent_types = list_recommendation_agent_types()
recommendations = recommend_configuration_space("rag", min_impact="medium")
configuration_space = recommendations["configuration_space"]
```

You can pass the returned `configuration_space` into `@traigent.optimize(...)` after reviewing it for your task and provider constraints.

### Provider Override Variables

SDK provider override variables such as `TRAIGENT_MODELS_OPENAI_FAST`, `TRAIGENT_MODELS_OPENAI_BALANCED`, and `TRAIGENT_MODELS_ANTHROPIC_QUALITY` are available when using integration/provider presets. They are not tied to a separate tuned-variables plugin.

---

## Part 3: Tuned Variable Detection (SDK)

The SDK includes a static analysis engine that automatically detects likely tunable variables in your existing Python functions — before you write any optimization code. It combines AST-based name matching, data-flow analysis, and optional LLM heuristics.

### Basic Detection

```python
from traigent.tuned_variables import TunedVariableDetector

detector = TunedVariableDetector()

def my_agent(query: str) -> str:
    temperature = 0.7
    model = "gpt-4o"
    max_tokens = 1024
    response = llm(query, temperature=temperature, model=model, max_tokens=max_tokens)
    return response

result = detector.detect_from_callable(my_agent)

print(f"Detected {result.count} candidates in '{result.function_name}':")
for candidate in result.candidates:
    print(f"  {candidate.name}: {candidate.candidate_type} ({candidate.confidence})")
    if candidate.suggested_range:
        print(f"    Suggested range: {candidate.suggested_range.to_parameter_range_code()}")
```

### Detection from Source Code

```python
# Detect from a source string directly
source = '''
def rag_pipeline(query: str) -> str:
    temperature = 0.5
    chunk_size = 512
    k = 5
    model = "gpt-4o-mini"
    ...
'''

result = detector.detect_from_source(source, function_name="rag_pipeline")
```

### Detection from a File

```python
# Scan a file for all tunable variable candidates
result = detector.detect_from_file("my_agent.py", function_name="my_agent")
```

### Detection Strategies

The detector ships with three strategies:

| Strategy | Description |
| -------- | ----------- |
| `ASTDetectionStrategy` | Name matching via AST, no LLM cost. Detects literal assignments of known LLM params (`temperature`, `model`, `max_tokens`, etc.) at **HIGH** confidence; fuzzy-matches variants like `temp`, `model_name`. |
| `DataFlowDetectionStrategy` | Backward program slicing via def-use chains, no LLM cost. Traces all variables that flow into statistical call sites (LLM invoke, retrieval, embedding). Finds parameters that name-matching misses. |
| `LLMDetectionStrategy` | LLM-assisted analysis for higher recall at extra cost. Returns **MEDIUM** or **LOW** confidence results. |

The default detector uses AST + DataFlow (both zero-cost):

```python
from traigent.tuned_variables import (
    TunedVariableDetector,
    ASTDetectionStrategy,
    DataFlowDetectionStrategy,
    LLMDetectionStrategy,
)

# Default: AST + DataFlow (zero LLM cost)
detector = TunedVariableDetector()
# equivalent to:
detector = TunedVariableDetector(strategies=[
    ASTDetectionStrategy(),
    DataFlowDetectionStrategy(),
])

# All three strategies for maximum recall
detector = TunedVariableDetector(strategies=[
    ASTDetectionStrategy(),
    DataFlowDetectionStrategy(),
    LLMDetectionStrategy(model="gpt-4o-mini"),
])
```

### Data-Flow Detection

The `DataFlowDetectionStrategy` uses **backward program slicing** to find variables that flow into LLM invocations, retrieval queries, or embedding calls — even if they don't have recognizable names like `temperature`.

**Constructor tracing** — detects parameters passed through LLM client constructors:

```python
def my_agent(query: str) -> str:
    custom_temp = 0.7                                  # no "temperature" in name
    llm = ChatOpenAI(temperature=custom_temp, model="gpt-4o")
    return llm.invoke(query)
    # DataFlow detects: custom_temp (HIGH — 1 hop to ChatOpenAI.__init__)
```

**Transitive chains** — follows variable assignments backward:

```python
def my_agent(query: str) -> str:
    base = 0.5
    adjusted = base + 0.2      # traces through BinOp
    llm.invoke(temperature=adjusted)
    # DataFlow detects: base (MEDIUM — 2 hops)
```

**Statistical call sites (sinks)** — 30 method patterns across three families:

| Family | Methods |
| ------ | ------- |
| LLM | `invoke`, `ainvoke`, `create`, `chat`, `generate`, `complete`, `completion`, `stream`, `batch`, ... |
| Retrieval | `search`, `similarity_search`, `query`, `retrieve`, `get_relevant_documents`, ... |
| Embedding | `embed`, `embed_query`, `embed_documents`, `get_text_embedding` |

**Custom sinks** — extend detection for your own frameworks:

```python
result = detector.detect_from_source(source, "my_func", context={
    "extra_sinks": [
        {"method": "predict", "category": "ml"},
        {"method": "classify", "category": "ml"},
    ],
})
```

For a deeper technical explanation, see [Data-Flow Detection](../features/dataflow-detection.md).

### Confidence Levels and Candidate Types

**Confidence:**

| Level | AST Strategy | DataFlow Strategy | LLM Strategy |
| ----- | ------------ | ----------------- | ------------ |
| `DetectionConfidence.HIGH` | Exact name match | Hop 0-1 (direct or one assignment away) | — |
| `DetectionConfidence.MEDIUM` | Fuzzy match, dict key | Hop 2-3 (transitive chain), constructor kwargs, function params | LLM suggestion |
| `DetectionConfidence.LOW` | — | Hop 4+ | Weak LLM heuristic |

When two strategies detect the same variable, confidence is **upgraded** (e.g., LOW+LOW becomes MEDIUM).

**Candidate types:**

| Type | Examples |
| ---- | -------- |
| `CandidateType.NUMERIC_CONTINUOUS` | `temperature`, `top_p`, `frequency_penalty` |
| `CandidateType.NUMERIC_INTEGER` | `max_tokens`, `top_k`, `seed` |
| `CandidateType.CATEGORICAL` | `model`, `stop`, `strategy` |
| `CandidateType.BOOLEAN` | `stream`, `use_cache` |

### Convert to Configuration Space

`DetectionResult.to_configuration_space()` can emit either normalized dict values
or typed `Range`/`Choices` objects.

```python
result = detector.detect_from_callable(my_agent)

# Get optimize-ready Range/Choices objects from detected candidates
config_space = result.to_configuration_space(format="ranges")
print(config_space)
# {'temperature': Range(...), 'model': Choices(...), ...}

@traigent.optimize(**config_space, objectives=["accuracy", "cost"])
def my_agent(query: str) -> str:
    ...
```

### Seamless Decorator Auto-Setup

`@traigent.optimize` can auto-build `configuration_space` directly from source
detection when no explicit config space is provided:

```python
@traigent.optimize(
    objectives=["accuracy", "cost"],
    auto_detect_tvars_mode="apply",          # off | suggest | apply
    auto_detect_tvars_min_confidence="medium",
    auto_detect_tvars_include={"temperature", "model"},  # optional
    auto_detect_tvars_exclude={"debug_flag"},            # optional
)
def my_agent(query: str) -> str:
    temperature = 0.7
    model = "gpt-4o-mini"
    return run_agent(query, temperature=temperature, model=model)
```

Behavior:

- `off`: do nothing (default)
- `suggest`: log suggested tvars only (backward-compatible with `auto_detect_tvars=True`)
- `apply`: auto-populate `configuration_space` before `OptimizedFunction` is created

If you provide `configuration_space` manually, it always takes precedence and
auto-apply is skipped.

### Excluding Already-Known Tunables

Pass `existing_tvars` in the context to avoid re-detecting variables you've already defined:

```python
result = detector.detect_from_callable(
    my_agent,
    context={"existing_tvars": {"temperature", "model"}},
)
# Only returns candidates not already in existing_tvars
```

### DetectionResult Attributes

```python
@dataclass
class DetectionResult:
    function_name: str                              # Analyzed function name
    candidates: tuple[TunedVariableCandidate, ...]  # All detected candidates
    warnings: tuple[str, ...]                       # Non-blocking warnings
    source_hash: str                                # Hash of analyzed source (for caching)
    detection_strategies_used: tuple[str, ...]      # Strategies that ran

    # Computed properties
    count: int                                      # Total candidate count
    high_confidence: tuple[TunedVariableCandidate, ...]  # HIGH confidence only

    def to_configuration_space(
        self,
        *,
        format: Literal["normalized", "ranges"] = "normalized",
        min_confidence: DetectionConfidence | str = DetectionConfidence.MEDIUM,
        include: Collection[str] | None = None,
        exclude: Collection[str] | None = None,
    ) -> dict: ...
```

### TunedVariableCandidate Attributes

```python
@dataclass
class TunedVariableCandidate:
    name: str                          # Variable name in source
    candidate_type: CandidateType      # Inferred parameter type
    confidence: DetectionConfidence    # Detection confidence
    location: SourceLocation           # Line/col in source
    current_value: Any | None          # Literal value if detectable
    suggested_range: SuggestedRange | None  # Suggested ParameterRange
    detection_source: str              # "ast", "dataflow", "llm", or "combined"
    reasoning: str                     # Human-readable explanation
    canonical_name: str | None         # Canonical LLM param (e.g., "model")
```

---

## API Reference

### traigent.tuned_variables

| Symbol | Description |
| ------ | ----------- |
| `discover_callables(module, *, pattern=None, include_private=False, required_params=None, return_type=None)` | Auto-discover callables from a module |
| `discover_callables_by_decorator(module, decorator_attr="__traigent_callable__", *, include_private=False)` | Discover decorated callables |
| `filter_by_signature(callables, target_signature, *, strict=False)` | Filter callables by target signature |
| `CallableInfo` | Frozen dataclass with callable metadata |
| `TunedVariableDetector` | Detects tuned variable candidates in Python functions |
| `ASTDetectionStrategy` | Fast AST name-matching strategy (no LLM cost) |
| `DataFlowDetectionStrategy` | Backward program slicing strategy (no LLM cost) |
| `LLMDetectionStrategy` | LLM-based detection strategy (higher recall) |
| `DetectionStrategy` | Protocol for custom detection strategies |
| `DetectionResult` | Frozen result dataclass from detection |
| `TunedVariableCandidate` | Frozen dataclass representing one detected candidate |
| `DetectionConfidence` | Enum: `HIGH`, `MEDIUM`, `LOW` |
| `CandidateType` | Enum: `NUMERIC_CONTINUOUS`, `NUMERIC_INTEGER`, `CATEGORICAL`, `BOOLEAN` |
| `SourceLocation` | Frozen dataclass: line/col position in source |
| `SuggestedRange` | Frozen dataclass with `range_type` and `kwargs`; `.to_parameter_range_code()` |

### Recommendation APIs

| Symbol | Description |
| ------ | ----------- |
| `traigent recommend` | CLI for listing agent types and showing recommendation catalog entries |
| `list_recommendation_agent_types()` | Return valid agent/task types for recommendation queries |
| `recommend_configuration_space(agent_type, *, min_impact=None, min_confidence=None)` | Return advisory configuration-space recommendations for an agent/task type |

---

### CLI: detect-tvars

Scan files from the command line without writing any Python:

```bash
# Scan a single file
traigent detect-tvars my_agent.py

# Scan a directory recursively
traigent detect-tvars src/agents/

# Filter to high confidence only
traigent detect-tvars my_agent.py --min-confidence high

# Machine-readable JSON output
traigent detect-tvars my_agent.py --json

# Analyze a specific function
traigent detect-tvars my_agent.py --function answer_question
```

---

## See Also

- [Data-Flow Detection (technical deep dive)](../features/dataflow-detection.md) - Algorithm details, sink catalog, plugin packaging

- [Configuration Spaces](./configuration-spaces.md) - Parameter ranges and constraints
- [Evaluation Guide](./evaluation_guide.md) - Metrics and evaluation
