"""Tests for the degenerate-configuration-space warning (issue #2021).

``configuration_space={"temperature": [0.7]}`` describes a search with nothing
to search: exactly one configuration exists, so the "best" one the run reports
was never compared against anything. ``@traigent.optimize`` used to accept that
silently.

The predicate is about the space *as a whole*. Pinning one knob while sweeping
others is a documented, supported pattern, so a per-parameter check would nag on
correct code - see ``TestPinnedKnobIsNotDegenerate``.
"""

import logging
import warnings

import pytest

import traigent
from traigent.api.decorators import _count_configurations, optimize
from traigent.utils.exceptions import TraigentWarning

_MARKER = "pins every tuned variable to a single value"


def _degenerate_warnings(records: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in records if _MARKER in str(w.message)]


class TestDegenerateSpaceWarns:
    """A space that enumerates exactly one configuration must warn."""

    def test_single_pinned_knob_warns_on_decorator(self) -> None:
        with pytest.warns(TraigentWarning, match=_MARKER) as record:

            @optimize(
                configuration_space={"temperature": [0.7]},
                objectives=["accuracy"],
                algorithm="grid",
            )
            def generate(question):
                return traigent.get_config().get("temperature")

        messages = _degenerate_warnings(list(record))
        assert len(messages) == 1
        # Names the function (see TestWarningIsNotDeduplicated) and the values.
        assert "'generate'" in messages[0]
        assert "'temperature': 0.7" in messages[0]

    def test_every_knob_pinned_warns(self) -> None:
        """Several knobs, each pinned - still exactly one configuration."""
        with pytest.warns(TraigentWarning, match=_MARKER) as record:

            @optimize(
                configuration_space={"temperature": [0.0], "model": ["gpt-4"]},
                objectives=["accuracy"],
                algorithm="grid",
            )
            def generate(question):
                return traigent.get_config().get("model")

        assert len(_degenerate_warnings(list(record))) == 1

    def test_warns_once_per_declaration(self) -> None:
        """Internal re-validation must not repeat the warning."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            @optimize(
                configuration_space={"temperature": [0.7]},
                objectives=["accuracy"],
                algorithm="grid",
            )
            def generate(question):
                return traigent.get_config().get("temperature")

        assert len(_degenerate_warnings(record)) == 1

    def test_warning_is_mirrored_to_the_logger(self) -> None:
        """Users who silence warnings still get the signal in logs."""
        with pytest.warns(TraigentWarning):
            with _capture_traigent_logs() as logged:

                @optimize(
                    configuration_space={"temperature": [0.7]},
                    objectives=["accuracy"],
                    algorithm="grid",
                )
                def generate(question):
                    return traigent.get_config().get("temperature")

        assert any(_MARKER in message for message in logged)


class TestPinnedKnobIsNotDegenerate:
    """Regression: a pinned knob alongside a varying one is a REAL search.

    Each space here comes from SDK-owned code or docs that a per-parameter check
    nagged on.
    """

    def test_rag_optimization_example_is_silent(self) -> None:
        """examples/core/rag-optimization/run.py - 2*1*2*3 = 12 configurations."""
        space = {
            "model": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            "temperature": [0.0],
            "use_rag": [True, False],
            "top_k": [1, 2, 3],
        }
        assert _count_configurations(space) == 12

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            @optimize(
                configuration_space=space,
                objectives=["accuracy"],
                algorithm="grid",
            )
            def answer_question(question):
                return traigent.get_config().get("model")

        assert _degenerate_warnings(record) == []

    def test_safety_guardrails_example_is_silent(self) -> None:
        """examples/core/safety-guardrails/run.py - 3*2*1 = 6 configurations."""
        space = {
            "safety_strength": ["low", "medium", "high"],
            "refusal_style": ["brief", "policy_cite"],
            "temperature": [0.0],
        }
        assert _count_configurations(space) == 6

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            @optimize(
                configuration_space=space,
                objectives=["accuracy"],
                algorithm="grid",
            )
            def respond_safely(prompt_input):
                return traigent.get_config().get("safety_strength")

        assert _degenerate_warnings(record) == []

    def test_quick_reference_docs_pattern_is_silent(self) -> None:
        """docs/examples/QUICK_REFERENCE.md - one model, two temperatures."""
        space = {"model": ["claude-3-haiku-20240307"], "temperature": [0.0, 0.7]}
        assert _count_configurations(space) == 2

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            @optimize(
                configuration_space=space,
                objectives=["accuracy"],
                algorithm="grid",
            )
            def summarize(text):
                return traigent.get_config().get("model")

        assert _degenerate_warnings(record) == []


class TestWarningIsNotDeduplicated:
    """Regression: the warning must survive Python's DEFAULT warning filter.

    The default filter dedups on ``(message, category, module, lineno)``. When
    the warning was raised from a fixed SDK frame with a message that named only
    the parameter, a second decorated function pinning the same knob produced an
    identical key and was silently swallowed.

    These tests deliberately do NOT use ``simplefilter("always")`` - that
    disables the very filter under test.
    """

    def test_two_functions_pinning_the_same_knob_both_warn(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("default")

            @optimize(
                configuration_space={"temperature": [0.7]},
                objectives=["accuracy"],
                algorithm="grid",
            )
            def alpha(question):
                return traigent.get_config().get("temperature")

            @optimize(
                configuration_space={"temperature": [0.7]},
                objectives=["accuracy"],
                algorithm="grid",
            )
            def beta(question):
                return traigent.get_config().get("temperature")

            @optimize(
                configuration_space={"temperature": [0.7]},
                objectives=["accuracy"],
                algorithm="grid",
            )
            def gamma(question):
                return traigent.get_config().get("temperature")

        messages = _degenerate_warnings(record)
        assert len(messages) == 3
        assert [
            name for name in ("alpha", "beta", "gamma") if _named(name, messages)
        ] == [
            "alpha",
            "beta",
            "gamma",
        ]

    def test_warning_points_at_the_user_decoration_not_the_sdk(self) -> None:
        """The reported location must be this test file, not a traigent module."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("default")

            @optimize(
                configuration_space={"temperature": [0.7]},
                objectives=["accuracy"],
                algorithm="grid",
            )
            def generate(question):
                return traigent.get_config().get("temperature")

        emitted = [w for w in record if _MARKER in str(w.message)]
        assert len(emitted) == 1
        assert emitted[0].filename == __file__


class TestCountConfigurations:
    """The predicate itself: countable spaces only, 0 means "unknown"."""

    @pytest.mark.parametrize(
        ("space", "expected"),
        [
            ({"a": [1]}, 1),
            ({"a": [1], "b": [2]}, 1),
            ({"a": [1, 2]}, 2),
            ({"a": [1], "b": [2, 3]}, 2),
            ({"a": {1}}, 1),
            # A (min, max) range varies continuously - never degenerate.
            ({"a": [1], "b": (0.0, 1.0)}, 2),
            # Unknowable / invalid shapes report 0 so no warning is emitted.
            ({}, 0),
            ({"a": []}, 0),
            ({"a": {"type": "float", "low": 0.0, "high": 1.0}}, 0),
            (None, 0),
        ],
    )
    def test_count(self, space, expected) -> None:
        assert _count_configurations(space) == expected

    def test_empty_list_parameter_does_not_warn_or_crash(self) -> None:
        """An empty list is invalid, not degenerate - the validator owns it."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            with pytest.raises(Exception):  # noqa: B017 - validator's own error

                @optimize(
                    configuration_space={"temperature": []},
                    objectives=["accuracy"],
                    algorithm="grid",
                )
                def generate(question):
                    return traigent.get_config().get("temperature")

        assert _degenerate_warnings(record) == []


def _named(func_name: str, messages: list[str]) -> bool:
    return any(f"'{func_name}'" in message for message in messages)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _capture_traigent_logs:
    """Capture WARNING records from the decorator module's own logger."""

    def __enter__(self) -> list[str]:
        self._logger = logging.getLogger("traigent.api.decorators")
        self._handler = _ListHandler()
        self._previous_level = self._logger.level
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.WARNING)
        return self._handler.messages

    def __exit__(self, *exc_info: object) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._previous_level)
