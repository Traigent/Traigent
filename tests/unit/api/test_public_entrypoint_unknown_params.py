"""Contract: public entrypoints reject unknown keyword arguments (Phase D, #1727).

Companion to
``tests/unit/core/optimized_function_tests/test_optimize_calltime_kwarg_rejection.py``
(call-time kwarg rejection on ``OptimizedFunction.optimize``/``optimize_sync``
for decorator-only params). This file pins down the sibling surface: unknown
kwargs must be rejected loudly at the *decoration* boundary
(``@traigent.optimize(bogus=...)``) and by every grouped options bundle
model, not silently swallowed into ``**runtime_overrides`` or dropped on the
floor. Per the no-silent-legacy policy (#1720 silent-failure audit, Phase D),
these are structural regressions, not just bugs -- hence pinned here as a
standing contract test rather than left to incidental coverage.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from traigent.api.decorators import (
    EvaluationOptions,
    ExecutionOptions,
    ExternalServiceEvaluator,
    HybridAPIOptions,
    InjectionOptions,
    optimize,
)
from traigent.cli.main import cli

OPTIONS_BUNDLE_CLASSES = (
    EvaluationOptions,
    InjectionOptions,
    HybridAPIOptions,
    ExecutionOptions,
    ExternalServiceEvaluator,
)


class TestOptimizeDecoratorUnknownKwargRejection:
    """``@traigent.optimize(<bogus>=...)`` must raise, not silently swallow."""

    def test_unknown_toplevel_kwarg_raises_typeerror(self):
        with pytest.raises(TypeError, match=r"[Uu]nknown"):
            optimize(
                configuration_space={"x": [1, 2]},
                definitely_not_a_real_kwarg=True,
            )

    def test_unknown_kwarg_message_names_the_bad_key(self):
        with pytest.raises(TypeError) as excinfo:
            optimize(configuration_space={"x": [1, 2]}, totally_bogus_option="value")
        assert "totally_bogus_option" in str(excinfo.value)

    def test_multiple_unknown_kwargs_are_all_named(self):
        with pytest.raises(TypeError) as excinfo:
            optimize(configuration_space={"x": [1, 2]}, bogus_one=1, bogus_two=2)
        message = str(excinfo.value)
        assert "bogus_one" in message
        assert "bogus_two" in message

    def test_unknown_field_inside_evaluation_bundle_dict_raises(self):
        """Unknown keys nested in a dict-form bundle must not be swallowed
        into the bundle either -- pydantic's ``extra="forbid"`` surfaces it
        as a ``ValidationError`` when the decorator constructs the model."""
        with pytest.raises((TypeError, ValidationError)):
            optimize(
                configuration_space={"x": [1, 2]},
                evaluation={"not_a_real_field": True},
            )


class TestOptionsBundlesRejectUnknownFields:
    """Grouped decorator options bundles are ``extra="forbid"``.

    Pin this so a future refactor cannot quietly reopen the door to unknown
    fields being accepted and silently dropped.
    """

    @pytest.mark.parametrize("options_cls", OPTIONS_BUNDLE_CLASSES)
    def test_options_bundle_declares_extra_forbid(self, options_cls):
        assert options_cls.model_config.get("extra") == "forbid", (
            f"{options_cls.__name__} must declare extra='forbid' so unknown "
            f"fields raise instead of being silently accepted"
        )

    @pytest.mark.parametrize("options_cls", OPTIONS_BUNDLE_CLASSES)
    def test_options_bundle_rejects_unknown_field(self, options_cls):
        with pytest.raises(ValidationError):
            options_cls(not_a_real_field_on_this_bundle=True)


class TestCliRejectsUnknownOption:
    """Light assertion: Click already rejects unknown CLI options.

    No live backend/network needed -- ``CliRunner`` invokes the command
    in-process and Click's own argument parser is what rejects the option,
    before any command body runs.
    """

    def test_auth_status_rejects_unknown_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "status", "--definitely-not-a-real-flag"])

        assert result.exit_code != 0
        assert "no such option" in result.output.lower()
