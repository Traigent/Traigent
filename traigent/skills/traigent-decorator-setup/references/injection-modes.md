# Injection Modes Reference

Injection mode controls how Traigent delivers the optimized configuration to your function during trials and production use.

```python
from traigent.api.decorators import InjectionOptions
```

## Mode Comparison

| Mode | Code Changes | Thread-Safe | Best For |
|---|---|---|---|
| `"context"` (default) | Add `traigent.get_config()` call | Yes (contextvars) | Most use cases. Clean, explicit config access. |
| `"parameter"` | Add a config parameter to function signature | Yes (per-call argument) | When you want the config as a visible function argument. |
| `"seamless"` | None | Yes (AST transform) | Existing codebases where you cannot modify the function body. |

## Context Mode (Default)

Uses Python's `contextvars` to store the trial configuration. Access it with `traigent.get_config()` anywhere inside the function (including nested calls).

```python
@traigent.optimize(
    # injection_mode defaults to "context", no InjectionOptions needed
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": [0.1, 0.5, 0.9],
    },
)
def answer_question(question: str) -> str:
    cfg = traigent.get_config()
    return call_llm(
        model=cfg["model"],
        temperature=cfg["temperature"],
        prompt=question,
    )
```

### Thread Safety

Context mode uses `contextvars`, which are natively thread-safe in Python 3.7+. Each thread (and each asyncio task) gets its own copy of the context. When Traigent runs parallel trials, each trial sees its own configuration without interference.

**Important**: If you spawn your own threads inside an optimized function, the context does not propagate automatically. Use `traigent.config.context.copy_context_to_thread()` to capture and restore context in worker threads:

```python
from traigent.config.context import copy_context_to_thread

@traigent.optimize(
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def parallel_processing(data_batch: list) -> list:
    cfg = traigent.get_config()

    # Capture context before spawning threads
    snapshot = copy_context_to_thread()

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for item in data_batch:
            # Each worker inherits the trial config
            future = executor.submit(snapshot.run, process_item, item, cfg)
            futures.append(future)
        return [f.result() for f in futures]
```

## Parameter Mode

Passes the trial configuration directly as a function parameter. You must specify `config_param` to name the parameter.

```python
@traigent.optimize(
    injection=InjectionOptions(
        injection_mode="parameter",
        config_param="config",
    ),
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": [0.1, 0.5, 0.9],
    },
)
def answer_question(question: str, config: dict = None) -> str:
    return call_llm(
        model=config["model"],
        temperature=config["temperature"],
        prompt=question,
    )
```

### When to Use Parameter Mode

- When you want the configuration dependency to be explicit in the function signature
- When your function is already designed to accept a config dict
- In testing scenarios where you want to pass config directly without Traigent context

### Notes

- The `config` parameter must have a default value (typically `None`) so the function can be called normally outside of optimization.
- During optimization, Traigent injects the trial config as this parameter.
- After `apply_best_config()`, calling the function without a config argument uses the applied best config.

## Seamless Mode

Zero code change required. Traigent uses AST (Abstract Syntax Tree) transformation to automatically inject configuration values into LLM API calls inside your function.

```python
@traigent.optimize(
    injection=InjectionOptions(injection_mode="seamless"),
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": [0.1, 0.5, 0.9],
    },
)
def answer_question(question: str) -> str:
    # Traigent will override 'model' and 'temperature' in this call
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content
```

### How It Works

1. At decoration time, Traigent inspects the function's AST.
2. It identifies LLM API calls (OpenAI, Anthropic, LiteLLM, etc.).
3. During trials, it transforms the AST to inject trial config values into matching keyword arguments.
4. The original function source is never modified on disk.

### When to Use Seamless Mode

- Migrating an existing codebase to Traigent without touching function bodies
- Quick prototyping where you want zero friction
- Functions with straightforward, single-call LLM usage

### Limitations

- Works best with direct API calls (e.g., `openai.chat.completions.create(...)`)
- May not detect LLM calls that are deeply nested or dynamically constructed
- Context mode gives more explicit control and is recommended for production

## InjectionOptions Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `injection_mode` | `str \| InjectionMode` | `"context"` | How to deliver config: `"context"`, `"parameter"`, or `"seamless"`. |
| `config_param` | `str \| None` | `None` | Parameter name for `injection_mode="parameter"`. Required when using parameter mode. |
| `auto_override_frameworks` | `bool` | `False` | Auto-override framework calls (LangChain, LlamaIndex, etc.). Requires `traigent-integrations` plugin. |
| `framework_targets` | `list[str] \| None` | `None` | Specific frameworks to target for auto-override (e.g., `["langchain", "llamaindex"]`). |

## Removed Modes

The `"attribute"` and `"decorator"` injection modes were removed in Traigent v2.x due to thread-safety issues. If you pass either of these, Traigent raises a `ValueError` with migration guidance. Use `"context"` (recommended) or `"seamless"` instead.

## Framework Auto-Override

When using `auto_override_frameworks=True`, Traigent intercepts calls to supported LLM frameworks and applies the trial configuration automatically. This requires the `traigent-integrations` plugin.

```python
@traigent.optimize(
    injection=InjectionOptions(
        injection_mode="context",
        auto_override_frameworks=True,
        framework_targets=["langchain"],
    ),
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def my_chain(query: str) -> str:
    # LangChain calls are automatically intercepted
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # Will be overridden
    chain = prompt | llm | output_parser
    return chain.invoke({"query": query})
```
