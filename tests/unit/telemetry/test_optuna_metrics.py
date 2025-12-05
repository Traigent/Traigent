"""Tests for Optuna telemetry helpers."""

from __future__ import annotations

from traigent.security.enterprise import MetricsCollector
from traigent.telemetry.optuna_metrics import OptunaMetricsEmitter, sanitize_config


def test_sanitize_config_strips_private_keys():
    config = {"model": "gpt-4", "_optuna_trial_id": 10}
    sanitized = sanitize_config(config)
    assert sanitized == {"model": "gpt-4"}


def test_emitter_records_events_and_notifies_listeners():
    collector = MetricsCollector()
    captured: list[dict[str, object]] = []

    def listener(event: dict[str, object]) -> None:
        captured.append(event)

    emitter = OptunaMetricsEmitter(metrics_collector=collector, listeners=[listener])
    message = emitter.emit_trial_update(
        event="trial_suggested",
        trial_id=1,
        study_name="study",
        payload={"config": {"model": "gpt-4"}},
    )

    assert message["event"] == "trial_suggested"
    assert captured[0]["trial_id"] == 1

    events = collector.get_optuna_events()
    assert len(events) == 1
    assert events[0]["event"] == "trial_suggested"
    assert events[0]["trial_id"] == 1
