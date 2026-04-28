# Parameter Types API Reference

Detailed API documentation for `Range`, `IntRange`, `Choices`, and `LogRange`.

All classes are importable from the top-level package:

```python
from traigent import Range, IntRange, Choices, LogRange
```

---

## Range

Continuous float range for optimization. Inherits from `ParameterRange` and `NumericConstraintBuilderMixin`.

### Constructor

```python
Range(
    low: float,
    high: float,
    step: float | None = None,
    log: bool = False,
    default: float | None = None,
    name: str | None = None,
    unit: str | None = None,
    agent: str | None = None,
)
```

**Parameters:**

| Parameter | Type             | Description                                                   |
| --------- | ---------------- | ------------------------------------------------------------- |
| `low`     | `float`          | Lower bound (inclusive). Must be less than `high`.            |
| `high`    | `float`          | Upper bound (inclusive).                                       |
| `step`    | `float` or None  | Step size for discretization. Cannot combine with `log=True`. |
| `log`     | `bool`           | Log-scale sampling. Requires `low > 0`. Cannot combine with `step`. |
| `default` | `float` or None  | Default value. Must be within `[low, high]`.                  |
| `name`    | `str` or None    | TVAR name. Auto-assigned from decorator kwarg if not set.     |
| `unit`    | `str` or None    | Unit of measurement (e.g., `"ratio"`, `"seconds"`).          |
| `agent`   | `str` or None    | Agent identifier for multi-agent experiments.                 |

**Raises:** `ValueError` if `low >= high`, `step <= 0`, `log` with non-positive `low`, `log` combined with `step`, or `default` outside range.

### Factory Methods

| Method                                | Returns          | Range            | Default |
| ------------------------------------- | ---------------- | ---------------- | ------- |
| `Range.temperature()`                 | `Range`          | [0.0, 1.0]       | 0.7     |
| `Range.temperature(conservative=True)`| `Range`          | [0.0, 0.5]       | 0.2     |
| `Range.temperature(creative=True)`    | `Range`          | [0.7, 1.5]       | 1.0     |
| `Range.top_p()`                       | `Range`          | [0.1, 1.0]       | 0.9     |
| `Range.frequency_penalty()`           | `Range`          | [0.0, 2.0]       | 0.0     |
| `Range.presence_penalty()`            | `Range`          | [0.0, 2.0]       | 0.0     |
| `Range.similarity_threshold()`        | `Range`          | [0.0, 1.0]       | 0.5     |
| `Range.mmr_lambda()`                  | `Range`          | [0.0, 1.0]       | 0.5     |
| `Range.chunk_overlap_ratio()`         | `Range`          | [0.0, 0.5]       | 0.1     |

### to_config_value()

Returns `tuple[float, float]` for simple ranges, or `dict` with `type`, `low`, `high`, `step`, `log` keys when step or log is set.

---

## IntRange

Integer range for optimization. Inherits from `ParameterRange` and `NumericConstraintBuilderMixin`.

### Constructor

```python
IntRange(
    low: int,
    high: int,
    step: int | None = None,
    log: bool = False,
    default: int | None = None,
    name: str | None = None,
    unit: str | None = None,
    agent: str | None = None,
)
```

**Parameters:**

| Parameter | Type            | Description                                                   |
| --------- | --------------- | ------------------------------------------------------------- |
| `low`     | `int`           | Lower bound (inclusive). Must be an integer, less than `high`.|
| `high`    | `int`           | Upper bound (inclusive). Must be an integer.                   |
| `step`    | `int` or None   | Step size. Cannot combine with `log=True`.                    |
| `log`     | `bool`          | Log-scale sampling. Requires `low > 0`. Cannot combine with `step`. |
| `default` | `int` or None   | Default value. Must be within `[low, high]`.                  |
| `name`    | `str` or None   | TVAR name. Auto-assigned from decorator kwarg if not set.     |
| `unit`    | `str` or None   | Unit of measurement (e.g., `"tokens"`, `"count"`).           |
| `agent`   | `str` or None   | Agent identifier for multi-agent experiments.                 |

**Raises:** `TypeError` if `low`/`high` are not integers. `ValueError` if `low >= high`, `step <= 0`, or invalid `log`/`step` combination.

### Factory Methods

| Method                                     | Returns    | Range            | Step | Default |
| ------------------------------------------ | ---------- | ---------------- | ---- | ------- |
| `IntRange.max_tokens()`                    | `IntRange` | [256, 1024]      | 64   | 512     |
| `IntRange.max_tokens(task="short")`        | `IntRange` | [50, 256]        | 64   | 128     |
| `IntRange.max_tokens(task="long")`         | `IntRange` | [1024, 4096]     | 64   | 2048    |
| `IntRange.k_retrieval()`                   | `IntRange` | [1, 10]          | --   | 3       |
| `IntRange.k_retrieval(max_k=20)`           | `IntRange` | [1, 20]          | --   | 3       |
| `IntRange.chunk_size()`                    | `IntRange` | [100, 1000]      | 100  | 500     |
| `IntRange.chunk_overlap()`                 | `IntRange` | [0, 200]         | 25   | 50      |
| `IntRange.few_shot_count()`                | `IntRange` | [0, 10]          | --   | 3       |
| `IntRange.few_shot_count(max_examples=5)`  | `IntRange` | [0, 5]           | --   | 3       |
| `IntRange.batch_size()`                    | `IntRange` | [1, 64]          | --   | 16      |

### to_config_value()

Returns `tuple[int, int]` for simple ranges, or `dict` with `type`, `low`, `high`, `step`, `log` keys when step or log is set.

---

## Choices

Categorical choices for optimization. Inherits from `ParameterRange` and `CategoricalConstraintBuilderMixin`. Generic over element type `T`.

### Constructor

```python
Choices(
    values: Sequence[T],
    default: T | None = None,
    name: str | None = None,
    unit: str | None = None,
    agent: str | None = None,
    enforce_type: bool = True,
)
```

**Parameters:**

| Parameter      | Type              | Description                                                        |
| -------------- | ----------------- | ------------------------------------------------------------------ |
| `values`       | `Sequence[T]`     | Allowed values (list or tuple). Must not be empty. Must not be str/bytes. |
| `default`      | `T` or None       | Default value. Must be present in `values`.                        |
| `name`         | `str` or None     | TVAR name. Auto-assigned from decorator kwarg if not set.          |
| `unit`         | `str` or None     | Unit of measurement (rarely needed for categorical).               |
| `agent`        | `str` or None     | Agent identifier for multi-agent experiments.                      |
| `enforce_type` | `bool`            | Validate all values have the same type (default `True`). Set `False` for mixed types. |

**Raises:** `TypeError` if `values` is a string or bytes, or if `enforce_type=True` and values contain mixed types. `ValueError` if `values` is empty or `default` is not in `values`.

### Factory Methods

| Method                                           | Returns         | Values                                                |
| ------------------------------------------------ | --------------- | ----------------------------------------------------- |
| `Choices.model()`                                | `Choices[str]`  | gpt-4o-mini, gpt-4o, claude-3-5-sonnet-20241022      |
| `Choices.model(provider="openai", tier="fast")`  | `Choices[str]`  | gpt-4o-mini                                           |
| `Choices.model(provider="openai", tier="quality")`| `Choices[str]` | gpt-4o, o1-preview                                    |
| `Choices.model(provider="anthropic", tier="quality")` | `Choices[str]` | claude-3-opus-20240229                           |
| `Choices.prompting_strategy()`                   | `Choices[str]`  | direct, chain_of_thought, react, self_consistency     |
| `Choices.context_format()`                       | `Choices[str]`  | bullet, numbered, xml, markdown, json                 |
| `Choices.retriever_type()`                       | `Choices[str]`  | similarity, mmr, bm25, hybrid                         |
| `Choices.embedding_model()`                      | `Choices[str]`  | text-embedding-3-small, text-embedding-3-large        |
| `Choices.reranker_model()`                       | `Choices[str]`  | none, cohere-rerank-v3, cross-encoder/..., llm-rerank |

The `Choices.model()` factory respects the `TRAIGENT_MODELS_{PROVIDER}_{TIER}` environment variable. Set it to a comma-separated list to override the defaults.

### to_config_value()

Returns `list[T]` of the values.

### Container Methods

`Choices` supports iteration (`for v in choices`), `len(choices)`, and `in` membership (`"gpt-4o" in choices`).

---

## LogRange

Log-scale float range. Convenience class equivalent to `Range(low, high, log=True)`. Inherits from `ParameterRange` and `NumericConstraintBuilderMixin`.

### Constructor

```python
LogRange(
    low: float,
    high: float,
    default: float | None = None,
    name: str | None = None,
    unit: str | None = None,
    agent: str | None = None,
)
```

**Parameters:**

| Parameter | Type            | Description                                           |
| --------- | --------------- | ----------------------------------------------------- |
| `low`     | `float`         | Lower bound (must be positive).                       |
| `high`    | `float`         | Upper bound (must be positive, greater than `low`).   |
| `default` | `float` or None | Default value. Must be within `[low, high]`.          |
| `name`    | `str` or None   | TVAR name. Auto-assigned from decorator kwarg if not set. |
| `unit`    | `str` or None   | Unit of measurement.                                  |
| `agent`   | `str` or None   | Agent identifier for multi-agent experiments.         |

**Raises:** `ValueError` if bounds are not positive or `low >= high`.

### to_config_value()

Returns `dict` with `type: "float"`, `low`, `high`, and `log: True`.

---

## Constraint Builder Methods

All numeric types (`Range`, `IntRange`, `LogRange`) provide these builder methods via `NumericConstraintBuilderMixin`:

| Method                   | Condition Created         | Example                               |
| ------------------------ | ------------------------- | ------------------------------------- |
| `.equals(value)`         | `param == value`          | `temp.equals(0.5)`                    |
| `.not_equals(value)`     | `param != value`          | `temp.not_equals(0.5)`               |
| `.gt(value)`             | `param > value`           | `temp.gt(0.5)`                        |
| `.gte(value)`            | `param >= value`          | `temp.gte(0.5)`                       |
| `.lt(value)`             | `param < value`           | `temp.lt(0.5)`                        |
| `.lte(value)`            | `param <= value`          | `temp.lte(0.5)`                       |
| `.in_range(low, high)`   | `low <= param <= high`    | `temp.in_range(0.3, 0.7)`           |
| `.is_in(values)`         | `param in values`         | `temp.is_in([0.0, 0.5, 1.0])`       |
| `.not_in(values)`        | `param not in values`     | `temp.not_in([0.0, 1.0])`           |

`Choices` provides these builder methods via `CategoricalConstraintBuilderMixin`:

| Method                   | Condition Created         | Example                               |
| ------------------------ | ------------------------- | ------------------------------------- |
| `.equals(value)`         | `param == value`          | `model.equals("gpt-4o")`             |
| `.not_equals(value)`     | `param != value`          | `model.not_equals("gpt-4o")`        |
| `.is_in(values)`         | `param in values`         | `model.is_in(["gpt-4o", "gpt-4o-mini"])` |
| `.not_in(values)`        | `param not in values`     | `model.not_in(["gpt-4o"])`          |

All builder methods return a `Condition` object (a `BoolExpr`) that can be used with `implies()`, `require()`, `when().then()`, or combined with `&`, `|`, `~` operators.
