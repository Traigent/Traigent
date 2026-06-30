"""Smart-pruning wire contract helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError


class SmartPruningOptions(BaseModel):
    """Cloud smart-pruning profile sent on session creation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    label: Literal["aggressive", "balanced", "conservative"]
    min_completed_trials: int | None = Field(default=None, ge=1)
    warmup_steps: int | None = Field(default=None, ge=0)
    epsilon: float | None = Field(default=None, ge=0)
    cost_threshold: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, gt=0, lt=1)
    min_samples_per_config: int | None = Field(default=None, ge=1)
    warmup_trials: int | None = Field(default=None, ge=0)


def normalize_smart_pruning_options(
    value: Any,
    *,
    field_name: str = "smart_pruning",
) -> dict[str, Any] | None:
    """Normalize smart-pruning config to the schema-shaped wire dict."""
    if value is None:
        return None
    if isinstance(value, SmartPruningOptions):
        model = value
    elif isinstance(value, Mapping):
        try:
            model = SmartPruningOptions.model_validate(dict(value))
        except PydanticValidationError as exc:
            extra_keys = sorted(
                str(error["loc"][0])
                for error in exc.errors()
                if error.get("type") == "extra_forbidden" and error.get("loc")
            )
            extra_message = (
                f" contains unsupported key(s): {', '.join(extra_keys)};"
                if extra_keys
                else ""
            )
            raise ValueError(
                f"{field_name}{extra_message} "
                f"does not match smart_pruning_schema.json: {exc}"
            ) from exc
    else:
        raise TypeError(
            f"{field_name} must be a mapping or SmartPruningOptions, "
            f"got {type(value).__name__}"
        )
    return cast(dict[str, Any], model.model_dump(exclude_none=True))


_INTERMEDIATE_REPORT_KEYS = frozenset(
    {
        "session_id",
        "trial_id",
        "running_score",
        "examples_attempted",
        "objective_name",
        "partial_cost_usd",
    }
)


def normalize_intermediate_report_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an allowlisted intermediate-report payload."""
    unknown_keys = sorted(set(payload) - _INTERMEDIATE_REPORT_KEYS)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(
            f"intermediate report payload contains unsupported key(s): {joined}"
        )
    return {key: payload[key] for key in payload}
