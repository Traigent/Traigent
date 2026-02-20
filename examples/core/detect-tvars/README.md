# Detect Tunable Variables

Automatically find tunable parameters in your Python functions using static analysis — no API keys or LLM calls required.

## What This Example Shows

- **AST name-matching**: Detects variables named `temperature`, `model`, `max_tokens`, etc.
- **Data-flow slicing**: Traces variables that flow into LLM invocations, retrieval queries, or embedding calls — even when they have non-standard names like `t` or `n`.
- **Constructor tracing**: Follows parameters through `ChatOpenAI(temperature=t)` constructors.
- **Configuration space generation**: Converts detected candidates into a ready-to-use `@traigent.optimize` config space.

## Running the Example

```bash
# Programmatic API
python examples/core/detect-tvars/run.py

# CLI — table output
traigent detect-tvars examples/core/detect-tvars/sample_agent.py

# CLI — JSON output for tooling
traigent detect-tvars examples/core/detect-tvars/sample_agent.py --json

# CLI — high confidence only
traigent detect-tvars examples/core/detect-tvars/sample_agent.py --min-confidence high
```

No API keys or environment variables needed — detection is purely static analysis.

## Key Code Patterns

### Programmatic Detection

```python
from traigent.tuned_variables import TunedVariableDetector

detector = TunedVariableDetector()
results = detector.detect_from_file("my_agent.py")

for result in results:
    for candidate in result.high_confidence:
        print(f"{candidate.name}: {candidate.suggested_range.to_parameter_range_code()}")
```

### From Detection to Optimization

```python
result = detector.detect_from_callable(my_agent)
config_space = result.to_configuration_space()

@traigent.optimize(**config_space, objectives=["accuracy", "cost"])
def my_agent(query: str) -> str:
    ...
```

## Sample Agent

`sample_agent.py` contains three functions demonstrating different detection patterns:

| Function | Pattern | What's Detected |
| -------- | ------- | --------------- |
| `answer_question` | Direct name match | `temperature`, `model_name`, `max_tokens`, `k`, `chunk_size` |
| `summarize_document` | Constructor tracing | `t` (flows to ChatOpenAI temperature), `n` (flows to max_tokens) |
| `multi_step_agent` | Transitive chains | `base_temp` (2 hops), `adjusted_temp` (1 hop), `model` |

## Output Interpretation

Each candidate includes:

- **Confidence**: HIGH (direct match or 0-1 hops), MEDIUM (fuzzy match or 2-3 hops), LOW (weak signal)
- **Detection source**: `ast` (name matching), `dataflow` (program slicing), or `combined` (both strategies agree)
- **Suggested range**: A `ParameterRange` constructor you can copy into your `@traigent.optimize` decorator

## Learn More

- [Tuned Variables Guide](../../../docs/user-guide/tuned_variables.md)
- [Data-Flow Detection (technical)](../../../docs/features/dataflow-detection.md)
