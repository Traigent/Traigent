# Weighted Score Frontend Contract (Bazak)

This guide defines the payload fields and validation checks needed to render weighted objective scores in frontend views.

## Data Source

Configuration runs payload from backend (experiment runs API), where each run includes:

1. `summary_stats`
2. `experiment_parameters`
3. `metadata` (with backward-compat `metadata.config`)

## Required Fields per Configuration Run

The frontend should read:

1. `summary_stats.weighted_score` (number)
2. `summary_stats.multi_objective_analysis.objective_weights` (object)
3. `summary_stats.multi_objective_analysis.normalization_ranges` (object, optional for tooltips)
4. `metadata.config` (resolved trial config values)

## Minimal Example

```json
{
  "id": "trial_123",
  "summary_stats": {
    "weighted_score": 0.8421,
    "multi_objective_analysis": {
      "weighted_score": 0.8421,
      "objective_weights": {
        "accuracy": 0.7,
        "cost": 0.2,
        "latency": 0.1
      },
      "normalization_ranges": {
        "accuracy": {"min": 0.4, "max": 0.95},
        "cost": {"min": 0.0002, "max": 0.0025}
      }
    }
  },
  "metadata": {
    "config": {
      "model": "gpt-4o-mini",
      "temperature": 0.3
    }
  }
}
```

## Frontend Rendering Rules

1. Show `weighted_score` with fixed precision (recommended: 4 decimals).
2. Show objective weights as labeled chips or table rows.
3. Correlate score row with run config (`metadata.config`) in same card/row.
4. If `weighted_score` is missing, show `"N/A"` and do not compute client-side fallback.

## Mock Checks (Backend -> Front)

Before enabling UI:

1. At least one run has numeric `summary_stats.weighted_score`.
2. `objective_weights` exists and contains at least two objectives for multi-objective runs.
3. `metadata.config` is present and non-empty.
4. Sorting by `weighted_score` is stable and deterministic in frontend list.

## Acceptance Criteria for #182

Close `#182` when all are true:

1. Weighted score is visible per run in Bazak frontend.
2. Objective weights are visible and match payload values.
3. Run configuration shown next to the weighted score.
4. Missing field cases render gracefully (`N/A`, no crash).
