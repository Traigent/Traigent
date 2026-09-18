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


def test_every_real_example_has_an_estimated_time() -> None:
    """``print_estimated_time`` is a silent no-op for an unlisted example.

    ``print_estimated_time`` looks the file up in ``EXAMPLE_ESTIMATED_TIMES``
    and only prints on a truthy hit, so a new ``walkthrough/real/NN_*.py`` that
    calls it without adding its entry prints nothing and the omission is
    invisible. Example 09 shipped exactly that way (Traigent/Traigent#1544
    review). Pin the set so the next one fails a test instead.
    """
    from walkthrough.utils.helpers import EXAMPLE_ESTIMATED_TIMES

    real_dir = REPO_ROOT / "walkthrough" / "real"
    examples = sorted(p.name for p in real_dir.glob("[0-9][0-9]_*.py"))
    assert examples, f"no numbered real examples found under {real_dir}"

    missing = [
        name
        for name in examples
        if name in _example_names_calling_estimated_time(real_dir)
        and name not in EXAMPLE_ESTIMATED_TIMES
    ]
    assert not missing, (
        "these real examples call print_estimated_time() but have no "
        f"EXAMPLE_ESTIMATED_TIMES entry, so it prints nothing: {missing}"
    )

    stale = sorted(set(EXAMPLE_ESTIMATED_TIMES) - set(examples))
    assert not stale, (
        f"EXAMPLE_ESTIMATED_TIMES lists examples that no longer exist: {stale}"
    )


def _example_names_calling_estimated_time(real_dir: Path) -> set[str]:
    return {
        path.name
        for path in real_dir.glob("[0-9][0-9]_*.py")
        if "print_estimated_time(" in path.read_text(encoding="utf-8")
    }


def test_printed_cost_estimate_matches_the_real_dataset_size() -> None:
    """A literal ``dataset_size=`` must equal the example's own dataset.

    ``print_cost_estimate`` multiplies by whatever it is handed, so a literal
    that drifts from the ``eval_dataset`` file silently overstates the spend a
    user is about to approve -- the whole point of the estimate. Examples 05 and
    09 both said 20 against a 13-row ``rag_questions.jsonl`` (~1.5x), which is
    why both now count the file instead. Catch the next one here.
    """
    import re

    real_dir = REPO_ROOT / "walkthrough" / "real"
    datasets_dir = REPO_ROOT / "walkthrough" / "datasets"
    dataset_re = re.compile(r'DATASETS\s*/\s*"([^"]+\.jsonl)"')
    literal_size_re = re.compile(r"dataset_size\s*=\s*(\d+)\s*,")

    mismatches: list[str] = []
    for path in sorted(real_dir.glob("[0-9][0-9]_*.py")):
        source = path.read_text(encoding="utf-8")
        dataset_match = dataset_re.search(source)
        size_match = literal_size_re.search(source)
        # A derived size (e.g. EVAL_DATASET_SIZE) cannot drift, so only literals
        # are checked; an example with no jsonl dataset is out of scope.
        if not dataset_match or not size_match:
            continue

        dataset_path = datasets_dir / dataset_match.group(1)
        assert dataset_path.is_file(), (
            f"{path.name} references a missing {dataset_path}"
        )
        actual = sum(
            1
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        declared = int(size_match.group(1))
        if declared != actual:
            mismatches.append(
                f"{path.name}: dataset_size={declared} but "
                f"{dataset_match.group(1)} has {actual} rows"
            )

    assert not mismatches, (
        f"printed cost estimates disagree with their own datasets: {mismatches}"
    )
