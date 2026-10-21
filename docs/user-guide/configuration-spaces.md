# Configuration Spaces

This guide documents the SE-friendly tuned variable types and constraint builders
implemented in `traigent/api/parameter_ranges.py` and `traigent/api/constraints.py`.

## Overview

Configuration spaces define which parameters Traigent can tune. You can express them
with SE-friendly `ParameterRange` objects (`Range`, `IntRange`, `LogRange`, `Choices`)
or with primitive tuples/lists in `configuration_space`. ParameterRange objects also
enable constraint builder methods.

## ParameterRange Base Class

All tuned variable types inherit from `ParameterRange`, which defines:

- `to_config_value()` -> internal tuple/list/dict representation
- `get_default()` -> default value or `None`

Common fields supported by the concrete types include `name` (used for constraints
and TVL export), `unit`, and `agent` for multi-agent grouping.

## Tuned Variable Types

| Type | Domain | Common fields | Notes |
| --- | --- | --- | --- |
| `Range` | Continuous float `[low, high]` | `low`, `high`, `step`, `log`, `default`, `name`, `unit`, `agent` | `log` requires positive bounds; `log` and `step` are mutually exclusive |
| `IntRange` | Discrete int `[low, high]` | `low`, `high`, `step`, `log`, `default`, `name`, `unit`, `agent` | bounds must be ints; `log` and `step` are mutually exclusive |
| `LogRange` | Log-scale float (positive bounds) | `low`, `high`, `default`, `name`, `unit`, `agent` | convenience for `Range(..., log=True)` |
| `Choices[T]` | Categorical values | `values`, `default`, `name`, `unit`, `agent`, `enforce_type` | `enforce_type=True` keeps types consistent (set `False` for mixed types) |

## Constraint Builder Methods

### Numeric Types (`Range`, `IntRange`, `LogRange`)

| Method | Operator | Example |
| --- | --- | --- |
| `.equals(v)` | `==` | `temp.equals(0.5)` |
| `.not_equals(v)` | `!=` | `temp.not_equals(0.0)` |
| `.gt(v)` | `>` | `temp.gt(0.5)` |
| `.gte(v)` | `>=` | `temp.gte(0.5)` |
| `.lt(v)` | `<` | `temp.lt(0.5)` |
| `.lte(v)` | `<=` | `temp.lte(0.5)` |
| `.in_range(low, high)` | `low <= x <= high` | `temp.in_range(0.3, 0.7)` |

### Categorical Type (`Choices`)

| Method | Operator | Example |
| --- | --- | --- |
| `.equals(v)` | `==` | `model.equals("gpt-4.1")` |
| `.not_equals(v)` | `!=` | `model.not_equals("gpt-4.1-mini")` |
| `.is_in(values)` | `in` | `model.is_in(["gpt-4.1", "gpt-4o"])` |
| `.not_in(values)` | `not in` | `model.not_in(["gpt-4.1-mini"])` |

### Boolean Combinators (`BoolExpr`)

Once you have `Condition` objects, combine them with:

- `A >> B` (implication)
- `A & B` (and)
- `A | B` (or)
- `~A` (not)

See `docs/features/constraint-dsl.md` for full syntax and precedence details.

## Factory Presets (Domain Helpers)

### Range Presets

- `Range.temperature(conservative=False, creative=False)`
- `Range.top_p()`
- `Range.frequency_penalty()`
- `Range.presence_penalty()`
- `Range.similarity_threshold()`
- `Range.mmr_lambda()`
- `Range.chunk_overlap_ratio()`

### IntRange Presets

- `IntRange.max_tokens(task="short"|"medium"|"long")`
- `IntRange.k_retrieval(max_k=10)`
- `IntRange.chunk_size()`
- `IntRange.chunk_overlap()`
- `IntRange.few_shot_count(max_examples=10)`
- `IntRange.batch_size()`
- `IntRange.thinking_budget_tokens(min_tokens=1024, max_tokens=32768, default=8192)`

### Choices Presets

- `Choices.model(provider=None, tier="balanced")`
- `Choices.prompting_strategy()`
- `Choices.context_format()`
- `Choices.retriever_type()`
- `Choices.embedding_model(provider=None)`
- `Choices.reranker_model()`
- `Choices.reasoning_effort(levels=("minimal","low","medium","high"), default="medium")`

### Reasoning knobs across providers

Put `reasoning_effort` and/or `thinking_budget_tokens` in the config space; the
provider plugins translate them to each provider's real parameter:

| Provider | `reasoning_effort` | `thinking_budget_tokens` |
|----------|--------------------|--------------------------|
| OpenAI | `reasoning_effort` (o-series: `minimal` becomes `low`) | skipped (no equivalent) |
| Anthropic | `thinking={"type": "enabled", "budget_tokens": N}` (minimal/low/medium/high = 1024/2048/8192/24576) | `budget_tokens` (>= 1024 and < `max_tokens`) |
| Gemini 3 | `thinking_config.thinking_level` | skipped |
| Gemini 2.5 | `thinking_config.thinking_budget` | `thinking_budget` |

Models known not to reason (for example `gpt-4o`, `gemini-2.0-flash`) skip the
knob with a warning; unknown model ids are passed through unchanged. Invalid
values raise a `ValueError` that does not echo the value. A raw `reasoning_effort`,
`thinking` or `thinking_config` you pass directly to the provider call still works
and is never rewritten. The capability table lives in
`traigent/integrations/utils/model_capabilities.py`.

## Notes

- Constraint builder methods are implemented per class; numeric builders are
  duplicated across `Range`, `IntRange`, and `LogRange`.
- `traigent/tuned_variables` currently exposes callable discovery utilities
  (`traigent/tuned_variables/discovery.py`); there is no `TunedCallable` class yet.
