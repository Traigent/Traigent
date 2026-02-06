# Configuration Injection Modes - Complete Guide

## Overview

Traigent provides four powerful configuration injection modes to seamlessly integrate optimization into your existing code. Each mode offers different trade-offs between simplicity, flexibility, and control.

## Quick Comparison

| Mode | Code Changes | Type Safety | Best Use Case |
|------|--------------|-------------|---------------|
| **Context (Default)** | Minimal | No | Most cases; dynamic configs |
| **Seamless** | None | No | Zero-touch variable overrides |
| **Parameter** | Function signature | Yes | Type-safe apps, team projects |
| **Attribute** | None | No | External monitoring, debugging |

## Parallel & Threading Notes

- **Context + Seamless**: Use `contextvars` under the hood. Traigent propagates
  context in its own evaluators, but if you spawn your own `ThreadPoolExecutor`
  inside an optimized function, use `traigent.copy_context_to_thread()` (or pass
  config explicitly) so worker threads can read `get_config()`.
- **Parameter**: Safe for parallel trials, but treat the injected config object
  as read-only, especially when `example_concurrency > 1`.
- **Attribute**: Unsafe for parallel trials; see the warning in the Attribute
  section below.

## 1. Context Mode (Default)

Access configuration through Traigent's context system — flexible, thread/async-safe, and works everywhere.

### Basic Example

```python
@traigent.optimize(
    injection_mode="context",
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0]
    }
)
def generate_text(prompt: str) -> str:
    cfg = traigent.get_config()  # Works during optimization and after apply_best_config()
    return ai_client.generate(
        model=cfg.get("model", "gpt-3.5-turbo"),
        temperature=cfg.get("temperature", 0.7),
        prompt=prompt
    )
```

### Advanced Example

```python
@traigent.optimize(
    injection_mode="context",
    configuration_space={
        "strategy": ["conservative", "balanced", "aggressive"],
        "risk_factor": [0.1, 0.5, 0.9],
        "max_exposure": [1000, 5000, 10000]
    }
)
def trading_algorithm(market_data: pd.DataFrame) -> List[Trade]:
    cfg = traigent.get_config()  # Get the config applied to this invocation
    strategy = cfg.get("strategy", "balanced")
    risk_factor = cfg.get("risk_factor", 0.5)
    max_exposure = cfg.get("max_exposure", 5000)
    # ... existing logic ...
    return compute_positions(market_data, strategy, risk_factor, max_exposure)
```

### How It Works

1. Traigent sets a thread/async-local config context
2. Your function reads config via `traigent.get_config()` (or `get_trial_config()` during optimization)
3. Optimizer updates the active config per trial
4. Your code stays unchanged apart from the decorator and get call

> **Note**: Use `traigent.get_config()` for unified access. `get_trial_config()` should only be called during an active optimization trial. After optimization, access the best config via `result.best_config` or `func.current_config`.

### Pros & Cons

**Pros:**
- ✅ Most natural coding style
- ✅ Minimal code changes to existing logic
- ✅ Clean, readable code
- ✅ No imports or API calls needed

**Cons:**

- ❌ Requires explicit `get_config()` call during optimization
- ❌ Implicit context may be less explicit than parameters

### When to Use

- Starting new projects
- Simple configuration needs
- Want the cleanest code possible
- Have full control over source code

## 2. Seamless Mode

Traigent automatically injects configuration values into simple variable assignments using safe AST transformation (no exec).

### Basic Example

```python
@traigent.optimize(
    injection_mode="seamless",
    configuration_space={
        "model": ["claude-2", "gpt-4", "gemini-pro"],
        "max_retries": [1, 3, 5],
        "timeout": [10, 30, 60]
    }
)
def robust_api_call(query: str) -> str:
    # These assignments are overridden by Traigent during optimization
    model = "gpt-4"
    max_retries = 3
    timeout = 30

    for attempt in range(max_retries):
        try:
            response = api_client.query(
                model=model,
                query=query,
                timeout=timeout
            )
            return response
        except TimeoutError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
```

### Advanced Example: Multi-Parameter Optimization

```python
@traigent.optimize(
    injection_mode="seamless",
    configuration_space={
        "chunk_size": [500, 1000, 2000],
        "overlap": [50, 100, 200],
        "top_k": [3, 5, 10]
    }
)
def rag_pipeline(documents: List[str], query: str) -> str:
    # These variable assignments are overridden by Traigent
    chunk_size = 1000
    overlap = 100
    top_k = 5

    # Chunk documents using optimized parameters
    chunks = []
    for doc in documents:
        chunks.extend(create_chunks(doc, chunk_size, overlap))

    # Search using optimized top_k
    embeddings = embed_texts(chunks)
    query_embedding = embed_texts([query])[0]
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = np.argsort(similarities)[-top_k:]

    relevant_chunks = [chunks[i] for i in top_indices]
    return generate_answer(query, relevant_chunks)
```

### How It Works

1. Traigent parses your function's AST (Abstract Syntax Tree)
2. Finds variable assignments matching config keys (e.g., `chunk_size = 1000`)
3. Replaces the assigned values with optimized config values at runtime
4. No `exec()` used — safe AST transformation with validation

### Pros & Cons (Seamless Mode)

**Pros:**

- ✅ Zero code changes to existing logic
- ✅ Natural variable usage
- ✅ Best developer experience
- ✅ Works with existing codebases

**Cons:**

- ❌ Variable names must match config keys exactly
- ❌ Only works with simple assignments (not computed values)
- ❌ Requires source code access (won't work with compiled code)
- ❌ Less explicit than other modes

### When to Use (Seamless Mode)

- Optimizing existing code without modifications
- Quick prototyping and experimentation
- Code where you want minimal Traigent-specific changes
- Functions with straightforward variable assignments

## 3. Parameter Mode

Explicit configuration injection with full type safety.

### Basic Example

```python
from traigent import TraigentConfig

@traigent.optimize(
    injection_mode="parameter",
    config_param="config",  # Parameter name
    configuration_space={
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "optimizer": ["adam", "sgd", "rmsprop"]
    }
)
def train_model(
    data: np.ndarray,
    labels: np.ndarray,
    config: TraigentConfig  # Traigent injects here
) -> Model:
    # Type-safe configuration access
    model = create_model()

    optimizer = get_optimizer(
        config["optimizer"],
        lr=config["learning_rate"]
    )

    for epoch in range(100):
        for i in range(0, len(data), config["batch_size"]):
            batch_data = data[i:i + config["batch_size"]]
            batch_labels = labels[i:i + config["batch_size"]]

            loss = train_step(model, batch_data, batch_labels, optimizer)

    return model
```

> **Note:** `TraigentConfig` exposes built-in fields (e.g., `model`, `temperature`) as attributes and custom parameters via dict-style access (`config["learning_rate"]` or `config.get(...)`). Use a wrapper dataclass for stricter typing.

### Advanced Example with Custom Config Types

```python
from dataclasses import dataclass
from typing import Literal
from traigent import TraigentConfig

@dataclass
class TransformerConfig:
    """Strongly typed configuration."""
    n_layers: int
    n_heads: int
    d_model: int
    dropout: float
    activation: Literal["relu", "gelu", "silu"]

    @classmethod
    def from_traigent(cls, config: TraigentConfig) -> "TransformerConfig":
        return cls(
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            d_model=config["d_model"],
            dropout=config["dropout"],
            activation=config["activation"]
        )

@traigent.optimize(
    injection_mode="parameter",
    config_param="traigent_config",
    configuration_space={
        "n_layers": [6, 12, 24],
        "n_heads": [8, 16, 32],
        "d_model": [512, 768, 1024],
        "dropout": [0.1, 0.2, 0.3],
        "activation": ["relu", "gelu", "silu"]
    }
)
def build_transformer(
    vocab_size: int,
    traigent_config: TraigentConfig
) -> nn.Module:
    # Convert to strongly typed config
    config = TransformerConfig.from_traigent(traigent_config)

    # Now we have full type safety and IDE support
    return Transformer(
        vocab_size=vocab_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        dropout=config.dropout,
        activation=get_activation(config.activation)
    )
```

### Dependency Injection Pattern

```python
@traigent.optimize(
    injection_mode="parameter",
    config_param="config",
    configuration_space={
        "db_pool_size": [10, 50, 100],
        "cache_ttl": [60, 300, 3600],
        "rate_limit": [10, 100, 1000]
    }
)
def create_application(
    config: TraigentConfig
) -> Application:
    # Use config to initialize all components
    db = Database(pool_size=config["db_pool_size"])
    cache = Cache(ttl=config["cache_ttl"])
    rate_limiter = RateLimiter(limit=config["rate_limit"])

    return Application(
        database=db,
        cache=cache,
        rate_limiter=rate_limiter
    )
```

### Pros & Cons

**Pros:**
- ✅ Full type safety
- ✅ IDE autocomplete support
- ✅ Clear dependencies
- ✅ Works well with DI patterns

**Cons:**
- ❌ Changes function signature
- ❌ Not suitable for callbacks
- ❌ Requires refactoring existing code
- ❌ Config must be passed through call chain

### When to Use

- Building type-safe applications
- Team projects with clear interfaces
- Using dependency injection
- Want IDE support and autocomplete

## 4. Attribute Mode

Store configuration as a function attribute for external access.

### Basic Example

```python
@traigent.optimize(
    injection_mode="attribute",
    configuration_space={
        "threshold": [0.5, 0.7, 0.9],
        "min_confidence": [0.6, 0.8, 0.95]
    }
)
def classify_image(image: np.ndarray) -> str:
    # Access config via function attribute
    config = classify_image.current_config

    predictions = model.predict(image)
    max_prob = np.max(predictions)

    if max_prob < config["min_confidence"]:
        return "uncertain"

    if max_prob > config["threshold"]:
        return "high_confidence"
    else:
        return "low_confidence"

# Access configuration externally
print(f"Current threshold: {classify_image.current_config['threshold']}")
```

### Monitoring and Debugging Example

```python
@traigent.optimize(
    injection_mode="attribute",
    configuration_space={
        "model_name": ["bert-base", "roberta-base", "distilbert"],
        "max_length": [128, 256, 512],
        "temperature": [0.7, 0.9, 1.0]
    }
)
def text_generation_api(prompt: str) -> str:
    config = text_generation_api.current_config

    response = generate_text(
        prompt=prompt,
        model=config["model_name"],
        max_length=config["max_length"],
        temperature=config["temperature"]
    )

    return response

# Monitoring system
def log_api_metrics():
    """Log current configuration for monitoring."""
    config = text_generation_api.current_config

    metrics.gauge("api.model", config["model_name"])
    metrics.gauge("api.max_length", config["max_length"])
    metrics.gauge("api.temperature", config["temperature"])

    logger.info(f"API Config: {config}")

# A/B testing
def get_variant():
    """Get current configuration variant for A/B testing."""
    return hash(str(text_generation_api.current_config)) % 2
```

### Stateful Configuration Example

```python
@traigent.optimize(
    injection_mode="attribute",
    configuration_space={
        "buffer_size": [100, 1000, 10000],
        "flush_interval": [1, 5, 10],
        "compression": ["none", "gzip", "lz4"]
    }
)
def buffered_writer(data: bytes) -> None:
    config = buffered_writer.current_config

    # Initialize buffer if needed
    if not hasattr(buffered_writer, 'buffer'):
        buffered_writer.buffer = []
        buffered_writer.last_flush = time.time()

    # Add to buffer
    if config["compression"] != "none":
        data = compress(data, config["compression"])

    buffered_writer.buffer.append(data)

    # Check if we should flush
    should_flush = (
        len(buffered_writer.buffer) >= config["buffer_size"] or
        time.time() - buffered_writer.last_flush > config["flush_interval"]
    )

    if should_flush:
        flush_buffer(buffered_writer.buffer)
        buffered_writer.buffer = []
        buffered_writer.last_flush = time.time()

# External buffer management
def force_flush():
    """Force flush the buffer."""
    if hasattr(buffered_writer, 'buffer'):
        flush_buffer(buffered_writer.buffer)
        buffered_writer.buffer = []

def get_buffer_stats():
    """Get current buffer statistics."""
    return {
        "size": len(getattr(buffered_writer, 'buffer', [])),
        "config": buffered_writer.current_config,
        "compression_enabled": buffered_writer.current_config["compression"] != "none"
    }
```

### Pros & Cons

**Pros:**
- ✅ Configuration accessible externally
- ✅ Good for monitoring/debugging
- ✅ Can maintain state
- ✅ No function signature changes

**Cons:**
- ❌ Less intuitive than other modes
- ❌ Config tied to function object
- ❌ Potential for external mutations
- ❌ Not ideal for pure functions
- ❌ **Not safe for parallel trials** (see warning below)

> ⚠️ **Parallel Execution Blocked**: Attribute mode is **not compatible** with
> parallel trials (`trial_concurrency > 1`). The SDK raises `ValueError` if you
> attempt this combination - no workarounds exist.
>
> **Why?** Attribute mode fundamentally relies on a shared mutable function
> attribute (`my_func.current_config`) that cannot be made thread-safe:
>
> - Concurrent trials overwrite the same attribute simultaneously
> - There is no isolation between trials - all share the same function object
> - Race conditions cause silent data corruption
>
> **Alternatives for parallel execution:**
> - Use `injection_mode="context"` (recommended) - fully thread-safe
> - Use `injection_mode="parameter"` - explicit config passing
> - Use sequential trials: `parallel_config={"trial_concurrency": 1}`
>
> ```python
> # ✅ CORRECT: Use context mode for parallel trials
> @traigent.optimize(
>     injection_mode="context",  # Thread-safe
>     parallel_config={"trial_concurrency": 4},
>     ...
> )
> def my_func(query: str) -> str:
>     config = traigent.get_config()  # Thread-safe access
>     return process(query, config)
>
> # ❌ BLOCKED: Attribute mode + parallel raises ValueError
> @traigent.optimize(
>     injection_mode="attribute",
>     parallel_config={"trial_concurrency": 2},  # ValueError!
>     ...
> )
> ```

### When to Use

- Need external configuration access
- Building monitoring/debugging tools
- Stateful optimizations
- A/B testing scenarios

## Best Practices

### 1. Start Simple
Context mode is the default and recommended starting point for most use cases:

```python
# Start with this (context mode is default)
@traigent.optimize(
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4o"]}
)
def my_function(prompt: str) -> str:
    cfg = traigent.get_config()  # Get config for current call
    return call_llm(model=cfg.get("model"), prompt=prompt)
```

### 2. Use Seamless for Zero-Touch Migration

For existing code where you don't want to add `get_config()` calls:

```python
# Seamless mode: Traigent overrides variable assignments automatically
@traigent.optimize(
    injection_mode="seamless",
    configuration_space={"threshold": [0.5, 0.7, 0.9]}
)
def process():
    threshold = 0.5  # Traigent will override this value during optimization
    return apply_threshold(data, threshold)
```

### 3. Use Type Safety When It Matters
For production systems, consider parameter mode:

```python
# Development
@traigent.optimize(injection_mode="seamless", ...)
def train():
    learning_rate = 0.01

# Production
@traigent.optimize(injection_mode="parameter", ...)
def train(data, config: TraigentConfig):
    learning_rate = config.learning_rate  # Type-safe!
```

### 4. Monitor in Production

Use attribute mode for production monitoring and A/B testing:

```python
@traigent.optimize(
    injection_mode="attribute",
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4o"]}
)
def api_endpoint(query: str) -> str:
    config = api_endpoint.current_config
    return call_llm(model=config["model"], query=query)

# External monitoring can access config without calling the function
current_config = api_endpoint.current_config
alert_if_config_drift(current_config)
```

## Common Patterns

### Multi-Mode Usage
You can use different modes for different functions:

```python
# Seamless for simple functions
@traigent.optimize()
def preprocess(text):
    max_length = 512
    return truncate(text, max_length)

# Context for complex pipelines
@traigent.optimize(injection_mode="context")
def pipeline(data):
    config = traigent.get_config()  # Get config for current call
    # Complex logic with config

# Parameter for type-safe components
@traigent.optimize(injection_mode="parameter")
def train(data, config: TraigentConfig):
    # Type-safe training
```

### Configuration Validation
Add validation regardless of mode:

```python
@traigent.optimize(...)
def process(data):
    config = get_config_somehow()  # Depends on mode

    # Validate
    assert config["threshold"] >= 0
    assert config["model"] in SUPPORTED_MODELS

    # Continue processing
```

## Summary

| Mode | Best For |
|------|----------|
| **Context** (default) | Most cases; dynamic config access via `get_config()` inside your function |
| **Seamless** | Zero-code-change; existing code with matching variable names |
| **Parameter** | Type safety; explicit dependencies; team projects |
| **Attribute** | External monitoring; A/B testing; debugging |

Choose based on your specific needs. Context mode is the default and works for most use cases.

## Configuration Access Lifecycle

Understanding **when** to use each config access method is critical:

| Lifecycle Phase | Access Method | Description |
|-----------------|---------------|-------------|
| **During/After Optimization** | `traigent.get_config()` | Unified accessor inside your optimized function. Works during trials and after `apply_best_config()`. |
| **During Optimization** | `traigent.get_trial_config()` | Returns the config being tested in the current trial. Deprecated; prefer `traigent.get_config()` unless you need explicit trial-only access. |
| **After Optimization** | `result.best_config` | The best configuration found, returned in `OptimizationResult`. Recommended for most post-optimization use. |
| **After Optimization** | `func.current_config` | The config currently applied to the function (same as `best_config` after optimization). |

### Post-Optimization Example

```python
import traigent

@traigent.optimize(
    eval_dataset="data.jsonl",
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0]
    }
)
def my_function(prompt: str) -> str:
    cfg = traigent.get_config()  # ✅ Unified access during/after optimization
    return call_llm(model=cfg["model"], temperature=cfg["temperature"], prompt=prompt)

# Run optimization
result = my_function.optimize()

# Access best config AFTER optimization
print(f"Best model: {result.best_config['model']}")       # ✅ From OptimizationResult
print(f"Best temp: {result.best_config['temperature']}")

# The function now uses best_config automatically
response = my_function("Hello!")  # Uses {"model": "gpt-4o", "temperature": 0.5}

# Can also access via function attribute
print(f"Applied config: {my_function.current_config}")    # ✅ Same as best_config
```

> **Warning**: Calling `get_trial_config()` outside an active optimization trial raises `OptimizationStateError`. Prefer `traigent.get_config()` inside your function for lifecycle-safe access.
