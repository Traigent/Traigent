"""Characterization tests pinning metrics_tracker._calculate_cost_for_metrics.

Pricing collaborators, the module logger, the strict-accounting switch and the
environment are replaced by recorders writing into one shared event trace, so
the order of checks, the attribute reads and the exact log calls are pinned
together with every outcome (return, raise, swallow).
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import pytest

from traigent.evaluators import metrics_tracker as mt
from traigent.utils.cost_calculator import UnknownModelError

UNNAMED_MESSAGE = (
    "Cost accounting is strict for this run and tokens were "
    "captured (in={inp}, out={out}) but no model name was "
    "available to price them, so the call is not recorded as $0. "
    "Fix by choosing one of: 1) expose the model name on the LLM "
    "response (a 'model' or 'model_name' attribute or key, or "
    "LangChain response_metadata/llm_output), 2) include a "
    "'model' key in the optimization configuration, 3) set "
    "TRAIGENT_STRICT_COST_ACCOUNTING=false to accept $0 for "
    "calls that cannot be priced."
)


class _Tokens:
    def __init__(
        self, trace: list[Any], estimated: Any, input_tokens: Any, output_tokens: Any
    ) -> None:
        self._trace = trace
        self._estimated = estimated
        self._input = input_tokens
        self._output = output_tokens

    @property
    def estimated(self) -> Any:
        self._trace.append(("estimated",))
        return self._estimated

    @property
    def input_tokens(self) -> Any:
        self._trace.append(("input_tokens",))
        if isinstance(self._input, BaseException):
            raise self._input
        return self._input

    @property
    def output_tokens(self) -> Any:
        self._trace.append(("output_tokens",))
        return self._output


class _Cost:
    def __init__(self, trace: list[Any], explicit: Any, total: Any) -> None:
        self._trace = trace
        self._explicit = explicit
        self.total_value = total

    @property
    def cost_explicit(self) -> Any:
        self._trace.append(("cost_explicit",))
        return self._explicit

    @property
    def total_cost(self) -> Any:
        self._trace.append(("total_cost",))
        return self.total_value


class _Metrics:
    def __init__(
        self,
        trace: list[Any],
        *,
        estimated: Any = False,
        input_tokens: Any = 10,
        output_tokens: Any = 5,
        explicit: Any = False,
        total: Any = 0.0,
    ) -> None:
        self._trace = trace
        self._tokens = _Tokens(trace, estimated, input_tokens, output_tokens)
        self._cost = _Cost(trace, explicit, total)

    @property
    def tokens(self) -> _Tokens:
        self._trace.append(("tokens",))
        return self._tokens

    @property
    def cost(self) -> _Cost:
        self._trace.append(("cost",))
        return self._cost


class _Logger:
    def __init__(self, trace: list[Any]) -> None:
        self._trace = trace

    def debug(self, *args: Any) -> None:
        self._trace.append(("log_debug", *args))

    def warning(self, *args: Any) -> None:
        self._trace.append(("log_warning", *args))

    def exception(self, *args: Any) -> None:
        self._trace.append(("log_exception", *args))


class _Env:
    def __init__(self, trace: list[Any], values: dict[str, str]) -> None:
        self._trace = trace
        self._values = values

    def get(self, key: str, default: str = "") -> str:
        self._trace.append(("env", key, default))
        return self._values.get(key, default)


class _Harness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        strict: bool = False,
        env: dict[str, str] | None = None,
        compute_error: BaseException | None = None,
        compute_effect: Callable[[], None] | None = None,
    ) -> None:
        self.trace: list[Any] = []
        trace = self.trace

        def fake_strict() -> bool:
            trace.append(("strict_check",))
            return strict

        def fake_mock(metrics: Any, prompt_length: Any, response_length: Any) -> None:
            trace.append(("mock_mode", prompt_length, response_length))

        def fake_reconcile(metrics: Any, model_name: Any) -> None:
            trace.append(("reconcile", model_name))

        def fake_compute(
            metrics: Any,
            model_name: Any,
            original_prompt: Any,
            response_text: Any,
            *,
            strict_cost_accounting: bool,
        ) -> None:
            trace.append(
                (
                    "compute",
                    model_name,
                    original_prompt,
                    response_text,
                    strict_cost_accounting,
                )
            )
            if compute_effect is not None:
                compute_effect()
            if compute_error is not None:
                raise compute_error

        monkeypatch.setattr(
            "traigent.utils.env_config.is_strict_cost_accounting", fake_strict
        )
        monkeypatch.setattr(mt, "_handle_mock_mode", fake_mock)
        monkeypatch.setattr(mt, "_reconcile_reported_cost_with_tokens", fake_reconcile)
        monkeypatch.setattr(mt, "_compute_cost", fake_compute)
        monkeypatch.setattr(mt, "logger", _Logger(trace))
        monkeypatch.setattr(
            mt, "os", types.SimpleNamespace(environ=_Env(trace, env or {}))
        )

    def run(
        self,
        metrics: Any,
        model_name: Any = "model-a",
        prompt: Any = "P",
        response: Any = "R",
        prompt_length: Any = None,
        response_length: Any = None,
    ) -> None:
        mt._calculate_cost_for_metrics(
            metrics, model_name, prompt, response, prompt_length, response_length
        )


def _names(trace: list[Any]) -> list[Any]:
    return [event[0] for event in trace]


# --------------------------------------------------------------- estimated guard


@pytest.mark.parametrize("estimated", [True, 1, "yes", [0]])
def test_estimated_tokens_are_never_priced(
    monkeypatch: pytest.MonkeyPatch, estimated: Any
) -> None:
    harness = _Harness(monkeypatch, env={"TRAIGENT_GENERATE_MOCKS": "true"})
    harness.run(_Metrics(harness.trace, estimated=estimated))
    assert harness.trace == [("tokens",), ("estimated",)]


def test_estimated_truth_is_tested_once(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)
    truths: list[str] = []

    class Flag:
        def __bool__(self) -> bool:
            truths.append("t")
            return True

    harness.run(_Metrics(harness.trace, estimated=Flag()))
    assert truths == ["t"]


# ------------------------------------------------------------------- mock mode


@pytest.mark.parametrize("raw", ["true", "TRUE", "True", "tRuE"])
def test_generate_mocks_env_routes_to_mock_mode(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    harness = _Harness(monkeypatch, env={"TRAIGENT_GENERATE_MOCKS": raw})
    harness.run(_Metrics(harness.trace), prompt_length=40, response_length=8)
    assert harness.trace == [
        ("tokens",),
        ("estimated",),
        ("strict_check",),
        ("env", "TRAIGENT_GENERATE_MOCKS", ""),
        ("mock_mode", 40, 8),
    ]


@pytest.mark.parametrize("raw", ["", "false", "1", "yes", " true", "true ", "on"])
def test_other_env_values_do_not_route_to_mock_mode(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    harness = _Harness(monkeypatch, env={"TRAIGENT_GENERATE_MOCKS": raw})
    harness.run(_Metrics(harness.trace, total=0.5))
    assert "mock_mode" not in _names(harness.trace)
    assert ("reconcile", "model-a") in harness.trace


def test_mock_mode_returns_before_reading_any_cost_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, env={"TRAIGENT_GENERATE_MOCKS": "true"})
    harness.run(_Metrics(harness.trace, explicit=True, total=0.0))
    assert "cost" not in _names(harness.trace)
    assert "compute" not in _names(harness.trace)


# -------------------------------------------------------------- explicit zero


@pytest.mark.parametrize("total", [0.0, -0.5, -1e-300])
def test_explicit_non_positive_provider_cost_is_kept_and_logged(
    monkeypatch: pytest.MonkeyPatch, total: float
) -> None:
    harness = _Harness(monkeypatch)
    harness.run(
        _Metrics(
            harness.trace, explicit=True, total=total, input_tokens=7, output_tokens=3
        )
    )
    assert harness.trace == [
        ("tokens",),
        ("estimated",),
        ("strict_check",),
        ("env", "TRAIGENT_GENERATE_MOCKS", ""),
        ("cost",),
        ("cost_explicit",),
        ("cost",),
        ("total_cost",),
        ("tokens",),
        ("input_tokens",),
        ("tokens",),
        ("output_tokens",),
        (
            "log_debug",
            "Keeping explicit provider-reported $0 cost for model %r "
            "(tokens: in=%d, out=%d); price-table estimate suppressed.",
            "model-a",
            7,
            3,
        ),
    ]


def test_explicit_cost_boundary_just_above_zero_goes_to_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    harness.run(_Metrics(harness.trace, explicit=True, total=1e-300))
    assert "log_debug" not in _names(harness.trace)
    assert harness.trace[-1] == ("reconcile", "model-a")


def test_explicit_zero_does_not_reach_reconcile_or_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    harness.run(_Metrics(harness.trace, explicit=True, total=0.0))
    assert not {"reconcile", "compute"} & set(_names(harness.trace))


# ---------------------------------------------------------- reported positive


@pytest.mark.parametrize("total", [1e-300, 0.01, 12.5])
def test_positive_reported_cost_is_reconciled_and_returns(
    monkeypatch: pytest.MonkeyPatch, total: float
) -> None:
    harness = _Harness(monkeypatch, strict=True)
    harness.run(_Metrics(harness.trace, total=total), model_name=None)
    assert harness.trace == [
        ("tokens",),
        ("estimated",),
        ("strict_check",),
        ("env", "TRAIGENT_GENERATE_MOCKS", ""),
        ("cost",),
        ("cost_explicit",),
        ("cost",),
        ("total_cost",),
        ("reconcile", None),
    ]


# ---------------------------------------------------------------- no model name


@pytest.mark.parametrize("name", [None, ""])
@pytest.mark.parametrize(("inp", "out"), [(1, 0), (0, 1), (5, 5)])
def test_strict_unnamed_model_with_tokens_raises_the_closed_message(
    monkeypatch: pytest.MonkeyPatch, name: Any, inp: int, out: int
) -> None:
    harness = _Harness(monkeypatch, strict=True)
    with pytest.raises(UnknownModelError) as info:
        harness.run(
            _Metrics(harness.trace, input_tokens=inp, output_tokens=out),
            model_name=name,
        )
    assert info.value.args == (UNNAMED_MESSAGE.format(inp=inp, out=out),)
    assert not {"log_warning", "compute", "reconcile"} & set(_names(harness.trace))


def test_strict_unnamed_model_reads_input_then_output_then_message_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, strict=True)
    with pytest.raises(UnknownModelError):
        harness.run(
            _Metrics(harness.trace, input_tokens=0, output_tokens=2), model_name=""
        )
    tail = harness.trace[harness.trace.index(("total_cost",)) + 1 :]
    assert tail == [
        ("tokens",),
        ("input_tokens",),
        ("tokens",),
        ("output_tokens",),
        ("tokens",),
        ("input_tokens",),
        ("tokens",),
        ("output_tokens",),
    ]


@pytest.mark.parametrize("name", [None, ""])
def test_strict_unnamed_model_without_tokens_only_warns(
    monkeypatch: pytest.MonkeyPatch, name: Any
) -> None:
    harness = _Harness(monkeypatch, strict=True)
    harness.run(
        _Metrics(harness.trace, input_tokens=0, output_tokens=0), model_name=name
    )
    tail = harness.trace[harness.trace.index(("total_cost",)) + 1 :]
    assert tail == [
        ("tokens",),
        ("input_tokens",),
        ("tokens",),
        ("output_tokens",),
        ("tokens",),
        ("input_tokens",),
        ("tokens",),
        ("output_tokens",),
        (
            "log_warning",
            "Cost calculation skipped: model_name is None/empty. "
            "Ensure the optimization config includes a 'model' key or "
            "the LLM response exposes the model name. "
            "tokens=(in=%d, out=%d)",
            0,
            0,
        ),
    ]
    assert "compute" not in _names(harness.trace)


def test_strict_unnamed_model_with_negative_tokens_only_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, strict=True)
    harness.run(
        _Metrics(harness.trace, input_tokens=-1, output_tokens=-1), model_name=None
    )
    assert harness.trace[-1][0] == "log_warning"
    assert harness.trace[-1][2:] == (-1, -1)


@pytest.mark.parametrize("name", [None, ""])
def test_lenient_unnamed_model_warns_and_reads_tokens_only_for_the_message(
    monkeypatch: pytest.MonkeyPatch, name: Any
) -> None:
    harness = _Harness(monkeypatch, strict=False)
    harness.run(
        _Metrics(harness.trace, input_tokens=9, output_tokens=4), model_name=name
    )
    tail = harness.trace[harness.trace.index(("total_cost",)) + 1 :]
    assert tail == [
        ("tokens",),
        ("input_tokens",),
        ("tokens",),
        ("output_tokens",),
        (
            "log_warning",
            "Cost calculation skipped: model_name is None/empty. "
            "Ensure the optimization config includes a 'model' key or "
            "the LLM response exposes the model name. "
            "tokens=(in=%d, out=%d)",
            9,
            4,
        ),
    ]
    assert "compute" not in _names(harness.trace)


# ---------------------------------------------------------------------- pricing


def _set_total(metrics: _Metrics, value: float) -> Callable[[], None]:
    def effect() -> None:
        metrics._cost.total_value = value

    return effect


def test_nominal_priced_path_event_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_holder: list[Any] = []
    metrics = _Metrics(trace_holder, total=0.0)
    harness = _Harness(monkeypatch, compute_effect=_set_total(metrics, 0.25))
    metrics._trace = harness.trace
    metrics._tokens._trace = harness.trace
    metrics._cost._trace = harness.trace
    harness.run(metrics, model_name="m1", prompt="PP", response="RR")
    assert harness.trace == [
        ("tokens",),
        ("estimated",),
        ("strict_check",),
        ("env", "TRAIGENT_GENERATE_MOCKS", ""),
        ("cost",),
        ("cost_explicit",),
        ("cost",),
        ("total_cost",),
        ("compute", "m1", "PP", "RR", False),
        ("cost",),
        ("total_cost",),
    ]


def test_pricing_receives_the_strict_flag_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, strict=True)
    harness.run(
        _Metrics(harness.trace, total=0.0), model_name="m1", prompt="PP", response="RR"
    )
    assert ("compute", "m1", "PP", "RR", True) in harness.trace


def test_non_explicit_zero_cost_does_not_take_the_explicit_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    harness.run(_Metrics(harness.trace, explicit=False, total=0.0))
    assert "log_debug" not in _names(harness.trace)
    assert _names(harness.trace).count("cost_explicit") == 1
    assert "compute" in _names(harness.trace)


@pytest.mark.parametrize(("inp", "out"), [(1, 0), (0, 1), (3, 3)])
def test_zero_priced_cost_with_tokens_warns(
    monkeypatch: pytest.MonkeyPatch, inp: int, out: int
) -> None:
    harness = _Harness(monkeypatch)
    harness.run(
        _Metrics(harness.trace, total=0.0, input_tokens=inp, output_tokens=out),
        model_name="m1",
    )
    assert harness.trace[-1] == (
        "log_warning",
        "Cost is $0.00 despite non-zero tokens for model %r "
        "(in=%d, out=%d). Model may be missing from pricing tables.",
        "m1",
        inp,
        out,
    )


@pytest.mark.parametrize(("inp", "out"), [(0, 0), (-1, 0), (0, -2)])
def test_zero_priced_cost_without_positive_tokens_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, inp: int, out: int
) -> None:
    harness = _Harness(monkeypatch)
    harness.run(_Metrics(harness.trace, total=0.0, input_tokens=inp, output_tokens=out))
    assert not {"log_warning", "log_exception"} & set(_names(harness.trace))


def test_zero_cost_diagnostic_reads_input_then_output_only_while_input_is_not_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    harness.run(_Metrics(harness.trace, total=0.0, input_tokens=5, output_tokens=0))
    after = harness.trace[
        harness.trace.index(("compute", "model-a", "P", "R", False)) :
    ]
    assert after[:5] == [
        after[0],
        ("cost",),
        ("total_cost",),
        ("tokens",),
        ("input_tokens",),
    ]
    assert after[5] == ("tokens",) and after[6] == ("input_tokens",)


@pytest.mark.parametrize(
    ("total", "warns"),
    [
        (0.0, True),
        (1e-12, True),
        (-1e-12, True),
        (2e-12, False),
        (-2e-12, False),
        (1.0000000000000002e-12, False),
        (0.5, False),
    ],
)
def test_zero_cost_tolerance_boundary(
    monkeypatch: pytest.MonkeyPatch, total: float, warns: bool
) -> None:
    metrics = _Metrics([], total=0.0)
    harness = _Harness(monkeypatch, compute_effect=_set_total(metrics, total))
    metrics._trace = harness.trace
    metrics._tokens._trace = harness.trace
    metrics._cost._trace = harness.trace
    harness.run(metrics)
    assert ("log_warning" in _names(harness.trace)) is warns


def test_zero_cost_diagnostic_error_is_handled_like_a_pricing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    metrics = _Metrics(
        harness.trace, total=0.0, input_tokens=RuntimeError("tokens failed")
    )
    with pytest.raises(RuntimeError, match="tokens failed"):
        harness.run(metrics, model_name="m1")
    # the handler reads the failing accessor again while building its log call
    assert "log_exception" not in _names(harness.trace)


# ------------------------------------------------------------- failure handling


def test_pricing_failure_is_logged_and_swallowed_when_lenient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("price table missing")
    harness = _Harness(monkeypatch, compute_error=error)
    harness.run(
        _Metrics(harness.trace, input_tokens=4, output_tokens=2), model_name="m1"
    )
    idx = harness.trace.index(("compute", "m1", "P", "R", False))
    assert harness.trace[idx + 1 :] == [
        ("tokens",),
        ("input_tokens",),
        ("tokens",),
        ("output_tokens",),
        (
            "log_exception",
            "Cost calculation failed for model %s: %s (tokens: in=%d, out=%d)",
            "m1",
            error,
            4,
            2,
        ),
        ("env", "TRAIGENT_DEBUG", ""),
    ]


def test_pricing_failure_reraises_when_strict_without_reading_debug_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("boom")
    harness = _Harness(monkeypatch, strict=True, compute_error=error)
    with pytest.raises(RuntimeError) as info:
        harness.run(_Metrics(harness.trace), model_name="m1")
    assert info.value is error
    assert ("env", "TRAIGENT_DEBUG", "") not in harness.trace
    assert _names(harness.trace).count("log_exception") == 1


@pytest.mark.parametrize("raw", ["true", "TRUE", "True"])
def test_pricing_failure_reraises_with_debug_env(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    error = ValueError("boom")
    harness = _Harness(monkeypatch, compute_error=error, env={"TRAIGENT_DEBUG": raw})
    with pytest.raises(ValueError) as info:
        harness.run(_Metrics(harness.trace), model_name="m1")
    assert info.value is error
    assert _names(harness.trace).count("log_exception") == 1


@pytest.mark.parametrize("raw", ["", "false", "1", "yes", " true", "on"])
def test_pricing_failure_is_swallowed_for_other_debug_values(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    harness = _Harness(
        monkeypatch, compute_error=RuntimeError("x"), env={"TRAIGENT_DEBUG": raw}
    )
    harness.run(_Metrics(harness.trace), model_name="m1")
    assert _names(harness.trace).count("log_exception") == 1


def test_base_exception_from_pricing_is_not_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, compute_error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        harness.run(_Metrics(harness.trace), model_name="m1")
    assert "log_exception" not in _names(harness.trace)


# ------------------------------------------------------------ raising accessors


@pytest.mark.parametrize("name", ["estimated", "cost", "tokens"])
def test_accessor_errors_propagate_before_any_later_step(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    harness = _Harness(monkeypatch)
    metrics = _Metrics(harness.trace)
    if name == "estimated":
        metrics._tokens._estimated = None
        monkeypatch.setattr(
            _Tokens,
            "estimated",
            property(
                lambda self: (_ for _ in ()).throw(RuntimeError("estimated failed"))
            ),
        )
    elif name == "cost":
        monkeypatch.setattr(
            _Metrics,
            "cost",
            property(lambda self: (_ for _ in ()).throw(RuntimeError("cost failed"))),
        )
    else:
        monkeypatch.setattr(
            _Metrics,
            "tokens",
            property(lambda self: (_ for _ in ()).throw(RuntimeError("tokens failed"))),
        )
    with pytest.raises(RuntimeError, match=f"{name} failed"):
        harness.run(metrics)
    assert not {"compute", "reconcile", "mock_mode"} & set(_names(harness.trace))


def test_strict_check_error_propagates_before_the_env_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)

    def boom() -> bool:
        raise RuntimeError("strict failed")

    monkeypatch.setattr("traigent.utils.env_config.is_strict_cost_accounting", boom)
    with pytest.raises(RuntimeError, match="strict failed"):
        harness.run(_Metrics(harness.trace))
    assert harness.trace == [("tokens",), ("estimated",)]


def test_mock_mode_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch, env={"TRAIGENT_GENERATE_MOCKS": "true"})

    def boom(*args: Any) -> None:
        raise RuntimeError("mock failed")

    monkeypatch.setattr(mt, "_handle_mock_mode", boom)
    with pytest.raises(RuntimeError, match="mock failed"):
        harness.run(_Metrics(harness.trace))


def test_reconcile_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)

    def boom(*args: Any) -> None:
        raise RuntimeError("reconcile failed")

    monkeypatch.setattr(mt, "_reconcile_reported_cost_with_tokens", boom)
    with pytest.raises(RuntimeError, match="reconcile failed"):
        harness.run(_Metrics(harness.trace, total=0.5))
