"""Characterization of ObservabilityConfig numeric limit validation."""

from __future__ import annotations

import pytest

from traigent.observability.config import ObservabilityConfig

ORIGIN = "https://portal.traigent.ai"

# Independent literals (deliberately not read from the production constants).
CASES = [
    # field, upper limit (as it appears in the message), max valid value, over value
    ("batch_size", "10000", 10_000, 10_001),
    ("max_buffer_age", "3600.0", 3600.0, 3600.5),
    ("max_queue_size", "1000000", 1_000_000, 1_000_001),
    ("max_batch_bytes", str(10 * 1024 * 1024), 10 * 1024 * 1024, 10 * 1024 * 1024 + 1),
    ("flush_timeout", "600.0", 600.0, 600.5),
    ("request_timeout", "600.0", 600.0, 600.5),
]


def _make(**kw) -> ObservabilityConfig:
    return ObservabilityConfig(backend_origin=ORIGIN, api_key="k", **kw)


@pytest.mark.parametrize("name,limit,ok,over", CASES)
class TestRangeChecks:
    def test_value_at_upper_limit_accepted(self, name, limit, ok, over):
        assert getattr(_make(**{name: ok}), name) == ok

    def test_value_above_limit_denied_with_exact_message(self, name, limit, ok, over):
        with pytest.raises(ValueError) as exc:
            _make(**{name: over})
        assert str(exc.value) == f"{name} must be less than or equal to {limit}"

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_denied_with_exact_message(self, name, limit, ok, over, bad):
        with pytest.raises(ValueError) as exc:
            _make(**{name: bad})
        assert str(exc.value) == f"{name} must be greater than 0"

    def test_smallest_positive_accepted(self, name, limit, ok, over):
        assert getattr(_make(**{name: 0.5 if isinstance(ok, float) else 1}), name) > 0


class TestCheckOrder:
    def test_first_invalid_field_wins_in_declaration_order(self):
        # every field invalid: batch_size is reported first
        with pytest.raises(ValueError) as exc:
            _make(
                batch_size=0,
                max_buffer_age=0,
                max_queue_size=0,
                max_batch_bytes=0,
                flush_timeout=0,
                request_timeout=0,
                per_trace_timeout=-1,
            )
        assert str(exc.value) == "batch_size must be greater than 0"

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            (
                {"max_buffer_age": 0, "max_queue_size": 0},
                "max_buffer_age must be greater than 0",
            ),
            (
                {"max_queue_size": 0, "max_batch_bytes": 0},
                "max_queue_size must be greater than 0",
            ),
            (
                {"max_batch_bytes": 0, "flush_timeout": 0},
                "max_batch_bytes must be greater than 0",
            ),
            (
                {"flush_timeout": 0, "request_timeout": 0},
                "flush_timeout must be greater than 0",
            ),
            (
                {"request_timeout": 0, "per_trace_timeout": -1},
                "request_timeout must be greater than 0",
            ),
        ],
    )
    def test_adjacent_fields_report_earlier_one(self, kwargs, message):
        with pytest.raises(ValueError) as exc:
            _make(**kwargs)
        assert str(exc.value) == message

    def test_lower_bound_checked_before_upper_bound_per_field(self):
        with pytest.raises(ValueError) as exc:
            _make(batch_size=-5)
        assert str(exc.value) == "batch_size must be greater than 0"


class _Probe:
    """Numeric stand-in recording which comparisons are evaluated, in order."""

    def __init__(self, le_zero: bool, gt_max: bool) -> None:
        self.log: list[str] = []
        self._le, self._gt = le_zero, gt_max

    def __float__(self) -> float:
        return 1.0

    def __le__(self, other):
        self.log.append("le")
        return self._le

    def __gt__(self, other):
        self.log.append("gt")
        return self._gt


class TestComparisonOrder:
    def test_lower_bound_compared_first_and_short_circuits(self):
        probe = _Probe(le_zero=True, gt_max=True)
        with pytest.raises(ValueError) as exc:
            _make(batch_size=probe)
        assert str(exc.value) == "batch_size must be greater than 0"
        assert probe.log == ["le"]

    def test_upper_bound_compared_second(self):
        probe = _Probe(le_zero=False, gt_max=True)
        with pytest.raises(ValueError) as exc:
            _make(batch_size=probe)
        assert str(exc.value) == "batch_size must be less than or equal to 10000"
        assert probe.log == ["le", "gt"]


class TestPerTraceTimeout:
    def test_zero_accepted(self):
        assert _make(per_trace_timeout=0).per_trace_timeout == 0

    def test_negative_denied(self):
        with pytest.raises(ValueError) as exc:
            _make(per_trace_timeout=-0.001)
        assert str(exc.value) == "per_trace_timeout must be greater than or equal to 0"

    def test_has_no_upper_limit(self):
        assert _make(per_trace_timeout=1e9).per_trace_timeout == 1e9


class TestMalformedInputs:
    @pytest.mark.parametrize("name", [c[0] for c in CASES])
    def test_nan_denied(self, name):
        with pytest.raises(ValueError) as exc:
            _make(**{name: float("nan")})
        assert str(exc.value) == f"{name} must be a finite number"

    @pytest.mark.parametrize("name,limit", [(c[0], c[1]) for c in CASES])
    def test_positive_infinity_denied_by_upper_bound(self, name, limit):
        with pytest.raises(ValueError) as exc:
            _make(**{name: float("inf")})
        assert str(exc.value) == f"{name} must be less than or equal to {limit}"

    @pytest.mark.parametrize("name", [c[0] for c in CASES])
    def test_negative_infinity_denied_by_lower_bound(self, name):
        with pytest.raises(ValueError) as exc:
            _make(**{name: float("-inf")})
        assert str(exc.value) == f"{name} must be greater than 0"

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_per_trace_timeout_non_finite_denied(self, value):
        with pytest.raises(ValueError) as exc:
            _make(per_trace_timeout=value)
        assert str(exc.value) == "per_trace_timeout must be a finite number"

    @pytest.mark.parametrize("name", [c[0] for c in CASES])
    def test_non_numeric_raises_type_error(self, name):
        with pytest.raises(TypeError):
            _make(**{name: "5"})

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            _make(batch_size=None)


class TestIdentityNormalisation:
    def test_tenant_and_project_ids_are_stripped(self):
        cfg = _make(tenant_id="  t1 ", project_id="  ")
        assert cfg.tenant_id == "t1"
        assert cfg.project_id is None
