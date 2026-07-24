from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from walkthrough.utils.helpers import (
    _build_metric_overrides,
    _get_mock_latency_for_trial,
    maybe_run_mock_example,
)
from walkthrough.utils.mock_answers import get_mock_cost, get_mock_latency

REPO_ROOT = Path(__file__).resolve().parents[3]


def _create_example_tree(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    real_dir = tmp_path / "walkthrough" / "real"
    mock_dir = tmp_path / "walkthrough" / "mock"
    real_dir.mkdir(parents=True)
    mock_dir.mkdir(parents=True)
    example_path = real_dir / "01_tuning_qa.py"
    example_path.write_text("print('real')\n")
    (mock_dir / "01_tuning_qa.py").write_text("print('mock')\n")
    return example_path


def test_real_examples_fail_without_provider_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example_path = _create_example_tree(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        maybe_run_mock_example(str(example_path))

    message = str(exc_info.value)
    assert "OPENAI_API_KEY environment variable is required" in message
    assert "python walkthrough/real/01_tuning_qa.py" in message
    assert "python walkthrough/mock/01_tuning_qa.py" in message


def test_real_examples_reject_mock_fallback_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example_path = _create_example_tree(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    with pytest.raises(SystemExit) as exc_info:
        maybe_run_mock_example(str(example_path))

    message = str(exc_info.value)
    assert "do not fall back to mock mode" in message
    assert "python walkthrough/mock/01_tuning_qa.py" in message


def test_real_examples_continue_when_provider_key_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example_path = _create_example_tree(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)

    maybe_run_mock_example(str(example_path))


@pytest.fixture
def walkthrough_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let helpers.py resolve its lazy ``from utils.mock_answers import ...``."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "walkthrough"))


def _trial(model: str, metrics: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(config={"model": model}, metrics=metrics)


def _result(trials: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(trials=trials)


def test_metric_overrides_applied_when_example_reports_nothing(
    walkthrough_on_path: None,
) -> None:
    """Metrics the example never simulates fall back to the static-table estimate."""
    result = _result(
        [
            _trial("gpt-3.5-turbo", {"accuracy": 0.75, "cost": 0.0, "latency": 0.0}),
            _trial("gpt-4o-mini", {"accuracy": 0.82, "cost": 0.0, "latency": 0.0}),
        ]
    )

    overrides = _build_metric_overrides(
        result, is_mock=True, task_type="classification", dataset_size=20
    )

    assert overrides is not None
    assert overrides["cost"] == [
        get_mock_cost("gpt-3.5-turbo", "classification", 20),
        get_mock_cost("gpt-4o-mini", "classification", 20),
    ]
    assert overrides["latency"] == [
        pytest.approx(get_mock_latency("gpt-3.5-turbo", "classification") * 1000.0),
        pytest.approx(get_mock_latency("gpt-4o-mini", "classification") * 1000.0),
    ]


def test_wall_clock_latency_is_still_overridden(walkthrough_on_path: None) -> None:
    """Varying values are not proof of signal: 07/demo sleep ``mock_latency * 0.01``.

    Those sub-millisecond wall-clock readings vary run to run, so a
    "constant means inert" guard would leave them on screen underneath a footer
    that promises static-table latency. Only ``reported_metrics`` suppresses the
    override.
    """
    result = _result(
        [
            _trial("gpt-3.5-turbo", {"accuracy": 0.75, "cost": 0.0, "latency": 3.7}),
            _trial("gpt-4o-mini", {"accuracy": 0.82, "cost": 0.0, "latency": 2.9}),
        ]
    )

    overrides = _build_metric_overrides(
        result, is_mock=True, task_type="simple_qa", dataset_size=20
    )

    assert overrides is not None
    assert overrides["latency"] == [
        pytest.approx(get_mock_latency("gpt-3.5-turbo", "simple_qa") * 1000.0),
        pytest.approx(get_mock_latency("gpt-4o-mini", "simple_qa") * 1000.0),
    ]


def test_metric_overrides_skipped_for_reported_metrics(
    walkthrough_on_path: None,
) -> None:
    """Metrics the example simulates itself drove selection and are shown as-is."""
    result = _result(
        [
            _trial(
                "gpt-3.5-turbo", {"accuracy": 0.75, "cost": 0.0011, "latency": 280.0}
            ),
            _trial(
                "gpt-4o-mini", {"accuracy": 0.82, "cost": 0.00036, "latency": 200.0}
            ),
        ]
    )

    overrides = _build_metric_overrides(
        result,
        is_mock=True,
        task_type="classification",
        dataset_size=20,
        reported_metrics=("cost", "latency"),
    )

    assert overrides is None


def test_metric_overrides_are_per_metric(walkthrough_on_path: None) -> None:
    """Only the metric the example does not report is overridden."""
    result = _result(
        [
            _trial("gpt-3.5-turbo", {"cost": 0.0, "latency": 280.0}),
            _trial("gpt-4o-mini", {"cost": 0.0, "latency": 200.0}),
        ]
    )

    overrides = _build_metric_overrides(
        result,
        is_mock=True,
        task_type="classification",
        dataset_size=20,
        reported_metrics=("latency",),
    )

    assert overrides is not None
    assert set(overrides) == {"cost"}


def test_mock_latency_override_is_milliseconds(walkthrough_on_path: None) -> None:
    """The latency column renders ``{val:.0f}ms``, so the override must be ms."""
    value = _get_mock_latency_for_trial(_trial("gpt-4o-mini", {}), "classification")

    assert value == pytest.approx(
        get_mock_latency("gpt-4o-mini", "classification") * 1000.0
    )
    assert value > 1.0
