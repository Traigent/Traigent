import pytest

from traigent.connectors.testing import (
    CASES,
    ConformanceFailure,
    ConnectorConformanceSuite,
    ConnectorHarness,
)

from .fixtures.dummy_harness import (
    DummyHarness,
    EchoingGateHarness,
    LeakyGateHarness,
    NoIdempotencyHarness,
    NoResumeHarness,
    SharedConnectionHarness,
    StaleCursorHarness,
    SwallowedFaultHarness,
)


class TestDummyConformance(ConnectorConformanceSuite):
    @staticmethod
    def make_harness() -> ConnectorHarness:
        return DummyHarness()


# Each case paired with a deliberately broken harness it must catch.
BROKEN = [
    ("lost_response", NoIdempotencyHarness),
    ("lost_response", SwallowedFaultHarness),
    ("crash_resume", NoResumeHarness),
    ("duplicate_rows", NoIdempotencyHarness),
    ("expired_cursor", StaleCursorHarness),
    ("partial_batch", NoIdempotencyHarness),
    ("partial_batch", SwallowedFaultHarness),
    ("connection_collision", SharedConnectionHarness),
    ("content_rejection", LeakyGateHarness),
    ("content_rejection", EchoingGateHarness),
]


@pytest.mark.parametrize(("case", "harness_cls"), BROKEN)
def test_broken_connector_fails_case(case, harness_cls):
    with pytest.raises(ConformanceFailure):
        CASES[case](harness_cls())


def test_every_case_has_a_negative_test():
    assert {case for case, _ in BROKEN} == set(CASES)


def test_failure_messages_do_not_echo_canary():
    from traigent.connectors.testing import CANARY

    with pytest.raises(ConformanceFailure) as info:
        CASES["content_rejection"](LeakyGateHarness())
    assert CANARY not in str(info.value)
