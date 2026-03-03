# Data-Flow Tuned Variable Detection

The `DataFlowDetectionStrategy` uses **intraprocedural backward program slicing** to find all variables on a data-flow path to a statistical call site. Unlike AST name matching, it does not require variables to have recognizable names — any variable that influences an LLM invocation, retrieval query, or embedding call is a candidate.

## Why Data Flow?

Consider this function:

```python
def answer(query: str) -> str:
    t = 0.7
    n = 512
    llm = ChatOpenAI(temperature=t, max_tokens=n)
    return llm.invoke(query)
```

The AST name-matching strategy detects nothing here — `t` and `n` are not recognizable parameter names. But both flow directly into `ChatOpenAI`, which is a statistical component. The data-flow strategy traces this chain backward:

```
llm.invoke(query)       ← sink: LLM invocation
  └─ llm = ChatOpenAI(temperature=t, max_tokens=n)
       ├─ t = 0.7       ← candidate (1 hop, HIGH confidence)
       └─ n = 512       ← candidate (1 hop, HIGH confidence)
```

## How It Works

### 1. Parse and Find Target Function

The strategy parses the source string into an AST and locates the target `FunctionDef` (or `AsyncFunctionDef`) by name.

### 2. Build Def-Use Map

A single-pass AST walk collects all variable definitions within the function body:

- Simple assignments: `x = 1`
- Annotated assignments: `x: int = 1`
- Augmented assignments: `x += 1`
- Tuple unpacking: `a, b = func()`
- Walrus operator: `if (x := val):`
- For-loop targets: `for i in items:`
- With-as targets: `with open(f) as x:`
- Function parameters: `def f(x, y):`

Nested function definitions are skipped (intraprocedural scope).

### 3. Identify Sinks (Statistical Call Sites)

The strategy recognizes 30 method patterns across three families:

**LLM family:**
`invoke`, `ainvoke`, `create`, `complete`, `chat`, `generate`, `generate_content`, `completion`, `acompletion`, `stream`, `astream`, `batch`, `abatch`, `run`, `arun`

**Retrieval family:**
`search`, `similarity_search`, `get_relevant_documents`, `aget_relevant_documents`, `retrieve`, `query`

**Embedding family:**
`embed`, `embed_query`, `embed_documents`, `get_text_embedding`

Generic names (`run`, `query`, `create`, `search`, `chat`, `batch`) are only matched as attribute calls (`obj.run(...)`) to avoid false positives like `subprocess.run()`.

### 4. Constructor Tracing

When a sink call's receiver was constructed from a known class, the constructor kwargs are also treated as sink arguments:

```python
llm = ChatOpenAI(temperature=0.7)   # constructor kwargs → sink args
llm.invoke(prompt)                    # sink call
```

Recognized constructors include: `OpenAI`, `AsyncOpenAI`, `ChatOpenAI`, `ChatAnthropic`, `Anthropic`, `ChatMistralAI`, `FAISS`, `Chroma`, `Pinecone`, `OpenAIEmbeddings`, and 20+ more (derived from `traigent.integrations.mappings`).

### 5. Backward Walk (BFS)

From each sink argument, BFS walks backward through the def-use map:

```
worklist = [(var_name, hop=0, sink_info)]
while worklist:
    pop (var, hop, sink)
    if var visited or hop > max_hops: skip
    for each definition of var:
        if value is literal → emit TunedVariableCandidate
        if value contains Name refs → add them at hop+1
```

BFS (FIFO) guarantees shortest-hop-first traversal, so each variable gets its best (lowest hop) confidence level.

### 6. Confidence Model

| Hops from sink | Confidence | Example |
| -------------- | ---------- | ------- |
| 0 (literal directly in call) | HIGH | `llm.invoke(temperature=0.7)` |
| 1 (one assignment away) | HIGH | `t = 0.7; llm.invoke(temperature=t)` |
| 2-3 (transitive chain) | MEDIUM | `x = 0.7; t = x; llm.invoke(temperature=t)` |
| 4+ | LOW | Long chains through helper calls |
| Constructor kwargs | MEDIUM | `ChatOpenAI(temperature=val)` |
| Function parameters | MEDIUM | `def f(temp): llm.invoke(temperature=temp)` |

When `DataFlowDetectionStrategy` and `ASTDetectionStrategy` both detect the same variable, `TunedVariableDetector` upgrades the confidence (e.g., two LOWs become MEDIUM).

## Configuration

### Custom Sink Patterns

Extend detection for your own frameworks by passing `extra_sinks` in the context:

```python
result = detector.detect_from_source(source, "my_func", context={
    "extra_sinks": [
        {"method": "predict", "category": "ml"},
        {"method": "classify", "category": "ml"},
    ],
})
```

### Max Hops

Control how far backward the slicing walks (default: 5):

```python
from traigent.tuned_variables import DataFlowDetectionStrategy

strategy = DataFlowDetectionStrategy(max_hops=3)  # Shorter chains only
```

### Constructor Class Hints

Add your own constructor class names:

```python
from traigent.tuned_variables.dataflow_strategy import (
    DataFlowDetectionStrategy,
    CONSTRUCTOR_CLASS_HINTS,
)

my_hints = CONSTRUCTOR_CLASS_HINTS | {"MyCustomLLM", "MyEmbedder"}
strategy = DataFlowDetectionStrategy(constructor_hints=my_hints)
```

## Architecture

### Plugin Design

The `DataFlowDetectionStrategy` implements the `DetectionStrategy` protocol:

```python
class DetectionStrategy(Protocol):
    def detect(
        self,
        source: str,
        function_name: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> list[TunedVariableCandidate]: ...
```

It is fully composable — the `TunedVariableDetector` orchestrator runs all strategies and merges results with deduplication and confidence upgrading.

### Dependency Graph

```
dataflow_strategy.py
  ├── detection_types.py    (TunedVariableCandidate, enums)
  ├── detection_strategies.py (shared helpers: _extract_literal_value, etc.)
  └── stdlib only (ast, logging, collections.deque, dataclasses)

detection_strategies.py
  └── detection_types.py

detector.py
  ├── dataflow_strategy.py
  ├── detection_strategies.py
  └── detection_types.py

detect_tvars_command.py (CLI)
  ├── detection_types.py
  ├── detector.py
  └── click, rich (optional)
```

**Zero dependencies on `traigent` core.** The entire `tuned_variables` package uses only the Python standard library plus its own types.

## Extracting as a Standalone Package

The `tuned_variables` module can be extracted as a separate pip-installable package with minimal effort:

### What to extract

```
traigent/tuned_variables/
    __init__.py
    dataflow_strategy.py
    detection_strategies.py
    detection_types.py
    detector.py
    discovery.py
    py.typed
traigent/cli/detect_tvars_command.py   (optional CLI)
```

### Dependencies

- **Core logic:** Python stdlib only (ast, dataclasses, logging, hashlib, inspect, pathlib)
- **CLI (optional):** `click`, `rich`

### Suggested pyproject.toml for standalone package

```toml
[project]
name = "traigent-detect-tvars"
version = "0.1.0"
requires-python = ">=3.10"
# No runtime dependencies for core

[project.optional-dependencies]
cli = ["click>=8.0", "rich>=13.0"]

[project.scripts]
traigent-detect-tvars = "traigent_detect_tvars.cli:detect_tvars"
```

### Re-integration with Traigent

The main `traigent` package would add it as an optional dependency:

```toml
[project.optional-dependencies]
tvar-detect = ["traigent-detect-tvars"]
```

And conditionally import in `traigent/cli/main.py`:

```python
try:
    from traigent_detect_tvars.cli import detect_tvars
    cli.add_command(detect_tvars)
except ImportError:
    pass
```

## Limitations

- **Intraprocedural only** — analyzes a single function body. Cross-function data flow (e.g., `config = get_config(); my_func(config)`) is not traced.
- **No type resolution** — matches method names (`.invoke()`, `.create()`), not receiver types. `subprocess.run()` is filtered via the generic-name safeguard, but custom `.invoke()` methods may be false positives.
- **No `**kwargs` unpacking** — `llm.invoke(**config)` does not trace keys inside `config`.
- **No `self.x` attribute chains** — `self.temperature = 0.7` in `__init__` is not traced to `self.temperature` used in another method.

These are fundamental limits of intraprocedural AST-only analysis. They can be addressed by future interprocedural or type-aware strategies.

## See Also

- [Tuned Variables Guide](../user-guide/tuned_variables.md) — Full guide including discovery, presets, and detection
- [Configuration Spaces](../user-guide/configuration-spaces.md) — Parameter ranges and constraints
