"""Tests for public trial serialization helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent import serialize_trials
from traigent.api.types import ExampleResult, TrialError, TrialResult, TrialStatus


class _NestedTimestampedPayload:
    """Test helper exposing a no-arg ``to_dict`` with nested datetimes."""

    def to_dict(self) -> dict[str, object]:
        return {
            "nested_at": datetime(2026, 3, 13, 11, 58, tzinfo=UTC),
            "status": "ok",
        }


class _FormatAwarePayload:
    """Test helper exposing a ``to_dict`` that accepts ``datetime_format``."""

    def to_dict(self, *, datetime_format: str = "iso") -> dict[str, object]:
        ts = datetime(2026, 3, 13, 11, 57, tzinfo=UTC)
        return {
            "nested_at": ts.isoformat() if datetime_format == "iso" else ts.timestamp()
        }


def _build_trial_result() -> TrialResult:
    return TrialResult(
        trial_id="trial-001",
        config={"model": "gpt-4o-mini", "temperature": 0.2},
        metrics={"accuracy": 0.91, "cost": 0.02},
        status=TrialStatus.COMPLETED,
        duration=1.25,
        timestamp=datetime(2026, 3, 13, 12, 0, tzinfo=UTC),
        metadata={
            "started_at": datetime(2026, 3, 13, 11, 59, tzinfo=UTC),
            "tags": {"sdk", "export"},
            "example_result": ExampleResult(
                example_id="ex-1",
                input_data={"question": "What is 2+2?"},
                expected_output="4",
                actual_output="4",
                metrics={"accuracy": 1.0},
                execution_time=0.05,
                success=True,
            ),
        },
    )


def test_trial_result_to_dict_defaults() -> None:
    trial = _build_trial_result()

    serialized = trial.to_dict()

    assert serialized["trial_id"] == "trial-001"
    assert serialized["status"] == "completed"
    assert serialized["duration"] == 1.25
    assert serialized["timestamp"] == "2026-03-13T12:00:00+00:00"
    assert serialized["config"] == {"model": "gpt-4o-mini", "temperature": 0.2}
    assert serialized["metrics"] == {"accuracy": 0.91, "cost": 0.02}
    assert serialized["metadata"]["started_at"] == "2026-03-13T11:59:00+00:00"
    assert set(serialized["metadata"]["tags"]) == {"sdk", "export"}
    assert serialized["metadata"]["example_result"]["example_id"] == "ex-1"


def test_trial_result_to_dict_can_filter_optional_fields() -> None:
    trial = _build_trial_result()

    serialized = trial.to_dict(
        include_config=False,
        include_metrics=False,
        include_metadata=False,
    )

    assert "config" not in serialized
    assert "metrics" not in serialized
    assert "metadata" not in serialized
    assert serialized["status"] == "completed"


def test_trial_result_to_dict_supports_epoch_timestamps() -> None:
    trial = _build_trial_result()

    serialized = trial.to_dict(datetime_format="epoch")

    assert serialized["timestamp"] == pytest.approx(
        datetime(2026, 3, 13, 12, 0, tzinfo=UTC).timestamp()
    )
    assert serialized["metadata"]["started_at"] == pytest.approx(
        datetime(2026, 3, 13, 11, 59, tzinfo=UTC).timestamp()
    )


def test_trial_result_to_dict_serializes_structured_error_context() -> None:
    trial = _build_trial_result()
    trial.status = TrialStatus.FAILED
    trial.error_message = "Connection timeout"
    trial.error = TrialError(
        message="Connection timeout",
        error_type="TimeoutError",
        traceback="Traceback (most recent call last): ...",
        timestamp=datetime(2026, 3, 13, 11, 56, tzinfo=UTC),
        config={"model": "gpt-4o-mini", "temperature": 0.2},
    )

    serialized = trial.to_dict(datetime_format="epoch")

    assert serialized["error_message"] == "Connection timeout"
    assert serialized["error"]["message"] == "Connection timeout"
    assert serialized["error"]["error_type"] == "TimeoutError"
    assert serialized["error"]["timestamp"] == pytest.approx(
        datetime(2026, 3, 13, 11, 56, tzinfo=UTC).timestamp()
    )
    assert serialized["error"]["config"] == {
        "model": "gpt-4o-mini",
        "temperature": 0.2,
    }


def test_trial_result_to_dict_serializes_nested_to_dict_datetimes() -> None:
    trial = _build_trial_result()
    trial.metadata["nested_payload"] = _NestedTimestampedPayload()

    serialized = trial.to_dict(datetime_format="epoch")

    assert serialized["metadata"]["nested_payload"]["nested_at"] == pytest.approx(
        datetime(2026, 3, 13, 11, 58, tzinfo=UTC).timestamp()
    )


def test_trial_result_to_dict_forwards_datetime_format_to_nested_to_dict() -> None:
    trial = _build_trial_result()
    trial.metadata["format_aware_payload"] = _FormatAwarePayload()

    serialized = trial.to_dict(datetime_format="epoch")

    assert serialized["metadata"]["format_aware_payload"]["nested_at"] == pytest.approx(
        datetime(2026, 3, 13, 11, 57, tzinfo=UTC).timestamp()
    )


def test_serialize_trials_applies_requested_options() -> None:
    trials = [_build_trial_result(), _build_trial_result()]

    serialized = serialize_trials(
        trials,
        include_metadata=False,
        datetime_format="epoch",
    )

    assert len(serialized) == 2
    assert all("metadata" not in item for item in serialized)
    assert all(item["status"] == "completed" for item in serialized)
    assert all(isinstance(item["timestamp"], float) for item in serialized)


def test_serialize_trials_rejects_unknown_datetime_format() -> None:
    trial = _build_trial_result()

    with pytest.raises(ValueError, match="datetime_format"):
        serialize_trials([trial], datetime_format="rfc3339")  # type: ignore[arg-type]


def test_serialize_trials_rejects_unknown_datetime_format_for_empty_input() -> None:
    with pytest.raises(ValueError, match="datetime_format"):
        serialize_trials([], datetime_format="rfc3339")  # type: ignore[arg-type]
