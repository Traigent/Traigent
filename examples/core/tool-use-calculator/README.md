# Tool Use Calculator

Optimize function calling and tool usage for math problems.

## Quick Start

```bash
export TRAIGENT_MOCK_MODE=true
python examples/core/tool-use-calculator/run.py
```

## Configuration Space

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `use_tool` | true, false | Enable calculator tool |
| `max_tool_calls` | 1, 2 | Tool invocation limit |
| `temperature` | 0.0, 0.2 | Response variance |

## What It Optimizes

- Whether tool use improves accuracy
- Optimal number of tool calls
- Tool vs native LLM math capability

## Expected Output

```
Best config: {'use_tool': True, 'max_tool_calls': 1, 'temperature': 0.0}
Best score: 1.0
```

## Key Concepts

- **Parameter injection mode**: Explicit config passing
- **Tool integration**: Calculator function
- **Strategy API**: Custom algorithm and parallel workers

```python
strategy = traigent.set_strategy(algorithm="random", parallel_workers=2)
result = await solve_math.optimize(strategy=strategy)
```

## Use Cases

- Math and calculation agents
- Data processing pipelines
- Hybrid LLM + tool workflows

## Next Steps

- [structured-output-json](../structured-output-json/) - Structured tool outputs
- [multi-objective-tradeoff](../multi-objective-tradeoff/) - Balance tool cost
