import pytest

from traigent.connectors.testing import (
    CANARY,
    CASES,
    ConformanceFailure,
    ConnectorConformanceSuite,
    ConnectorHarness,
)

from .fixtures import dummy_harness as fx


class TestDummyConformance(ConnectorConformanceSuite):
    @staticmethod
    def make_harness() -> ConnectorHarness:
        return fx.DummyHarness()


def run_all(harness_cls):
    """Run every case; map case name to its failure reason (absent when passing)."""
    failures = {}
    for name, case in CASES.items():
        try:
            case(harness_cls())
        except ConformanceFailure as exc:
            failures[name] = exc.reason
    return failures


# Each broken variant carries one defect and must fail exactly these cases,
# each for the pinned reason, and no others.
EXPECTED = {
    # blocker 1: crash/resume
    fx.NoOpCrashHarness: {"crash_resume": "crash_not_real"},
    fx.InMemoryCursorHarness: {"crash_resume": "resume_failed"},
    # blocker 2: distinct write faults
    fx.FailBeforeRecordsKeyHarness: {"lost_response": "rows_mismatch"},
    fx.LostAckNotRecordedHarness: {"lost_response": "rows_mismatch"},
    fx.PartialNotRecordedHarness: {"partial_batch": "rows_mismatch"},
    fx.SwallowedFaultHarness: {
        "lost_response": "fault_not_injected",
        "partial_batch": "fault_not_injected",
    },
    fx.NoIdempotencyHarness: {
        "lost_response": "rows_mismatch",
        "duplicate_rows": "rows_mismatch",
        "partial_batch": "rows_mismatch",
    },
    # blocker 3: content integrity, not just ids
    fx.CorruptsOnWriteHarness: {
        "lost_response": "rows_mismatch",
        "duplicate_rows": "rows_mismatch",
        "partial_batch": "rows_mismatch",
        "connection_collision": "idem_collision",
    },
    fx.CorruptsOnReadHarness: {
        "crash_resume": "rows_mismatch",
        "overlapping_pages": "rows_mismatch",
        "expired_cursor": "rows_mismatch",
    },
    # read-side duplicates and cursors
    fx.OverlapLosesRowHarness: {"overlapping_pages": "rows_mismatch"},
    fx.ConflictingOverlapHarness: {"overlapping_pages": "conflicting_duplicate"},
    fx.StaleCursorHarness: {"expired_cursor": "expiry_not_enforced"},
    # blocker 4/5: the connector's own gate, every field
    fx.LeakyGateHarness: {"content_rejection": "gate_accepts_content"},
    fx.EchoingGateHarness: {"content_rejection": "gate_echoes_content"},
    fx.WrongPointerGateHarness: {"content_rejection": "gate_wrong_pointer"},
    fx.UncheckedStatusGateHarness: {"content_rejection": "gate_accepts_content"},
    # blocker 6: connection isolation
    fx.SharedCredentialHarness: {"connection_collision": "credential_shared"},
    fx.SharedMinterHarness: {"connection_collision": "token_shared"},
    fx.CachedCredentialHarness: {"connection_collision": "credential_cross_use"},
    fx.GlobalIdempotencyHarness: {"connection_collision": "idem_collision"},
    fx.ForeignTokenAcceptedHarness: {"connection_collision": "foreign_token_accepted"},
}


def test_correct_dummy_passes_every_case():
    assert run_all(fx.DummyHarness) == {}


@pytest.mark.parametrize("harness_cls", list(EXPECTED), ids=lambda c: c.__name__)
def test_broken_variant_fails_only_intended_cases(harness_cls):
    assert run_all(harness_cls) == EXPECTED[harness_cls]


def test_every_case_is_caught_by_some_variant():
    assert {case for failed in EXPECTED.values() for case in failed} == set(CASES)


def test_failure_messages_do_not_echo_canary():
    with pytest.raises(ConformanceFailure) as info:
        CASES["content_rejection"](fx.LeakyGateHarness())
    assert CANARY not in str(info.value)
