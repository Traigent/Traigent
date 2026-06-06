# Strategy Presets

Strategy presets select completed trials using advisory accuracy and cost rules.
They are helpers for choosing from existing results, not optimizers themselves.

## Public API

```python
from traigent.api.strategy_presets import (
    VALID_PRESET_NAMES,
    normalize_strategy_preset,
    select_strategy_preset,
)

print(VALID_PRESET_NAMES)

preset = normalize_strategy_preset(
    "max_accuracy_then_cheapest_within_epsilon",
    {"epsilon": 0.02},
)
selection = select_strategy_preset(preset, trials)
```

## Valid Presets

`VALID_PRESET_NAMES` contains:

- `max_accuracy_then_cheapest_within_epsilon`
- `quality_floor_min_cost`
- `pareto_frontier`

`max_accuracy_then_cheapest_within_epsilon` requires:

```python
{"epsilon": 0.02}
```

It selects the lowest-cost completed trial within the preset accuracy band.

`quality_floor_min_cost` requires:

```python
{"floor": 0.85}
```

It selects the lowest-cost completed trial satisfying the quality floor.

`pareto_frontier` accepts no params and returns completed trials on the advisory
accuracy-cost Pareto frontier.

## Copy-Paste Example

```python
from traigent.api.strategy_presets import (
    normalize_strategy_preset,
    select_strategy_preset,
)

preset = normalize_strategy_preset(
    "quality_floor_min_cost",
    {"floor": 0.90},
)

selection = select_strategy_preset(preset, result.trials)
print(selection.selected_config)
print(selection.selection_grade)
```

## Honesty Note

These presets are advisory and selection-only over completed trials. They are
not a statistical certificate, do not prove significance, and do not certify
that the selected configuration will generalize beyond the task-local trials and
metrics you already ran.
