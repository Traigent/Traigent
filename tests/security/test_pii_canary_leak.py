"""End-to-end PII canary leak test for the optimization pipeline.

This is the P0-1 safety net for the SDK result-redaction plan.

It seeds a set of *canary* PII patterns (fake email, CC, SSN, API key) into an
optimization run and, after the run completes, scans every persistence target
the SDK might write to for any trace of the canaries. Any hit is a leak.

The test is written to fail closed: any raw canary in public trial results,
serialized exports, stdout/stderr, or enabled sinks fails CI.

Canary patterns are intentionally easy for humans and tooling to recognize so
that a hit in prod logs during manual review is obvious.
"""

# Traceability: CONC-Layer-Security CONC-Quality-Privacy FUNC-ORCH-LIFECYCLE
# See traigent.security.redaction for the public-result redaction helper.

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.observability import (
    ObservabilityClient,
    ObservabilityConfig,
    ObservationType,
)
from traigent.security.audit import AuditEventType, AuditLogger

# ---------------------------------------------------------------------------
# Canary catalogue
# ---------------------------------------------------------------------------

# Each canary is a (label, value, regex) triple. The regex is what the scanner
# uses — the value is what we plant. Keeping them distinct lets us add "loose"
# patterns that would catch redaction that accidentally preserves structure
# (e.g. replacing digits with X but leaving the format recognizable as a CC).

CANARIES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "email",
        "canary.leak.alpha@redteam.traigent.invalid",
        re.compile(r"canary\.leak\.alpha@redteam\.traigent\.invalid"),
    ),
    (
        "credit_card",
        "4111111111111234",
        re.compile(r"4111[- ]?1111[- ]?1111[- ]?1234"),
    ),
    (
        "ssn",
        "123-45-6789",
        re.compile(r"123[- ]?45[- ]?6789"),
    ),
    (
        "api_key",
        "sk-ant-canary-DO-NOT-USE-123456789abcdef",  # pragma: allowlist secret
        re.compile(r"sk-ant-canary-DO-NOT-USE-[A-Za-z0-9]+"),
    ),
    (
        "bearer_token",
        "Bearer canary.jwt.header.payload.signature",
        re.compile(r"canary\.jwt\.header\.payload\.signature"),
    ),
)

CANARY_VALUES: tuple[str, ...] = tuple(value for _, value, _ in CANARIES)
CANARY_REGEXES: tuple[re.Pattern[str], ...] = tuple(rx for _, _, rx in CANARIES)


def _contains_canary(blob: str) -> list[str]:
    """Return names of canaries found in ``blob`` (empty list = clean)."""
    hits: list[str] = []
    for label, _value, rx in CANARIES:
        if rx.search(blob):
            hits.append(label)
    return hits


# ---------------------------------------------------------------------------
# Echo evaluator — simulates an LLM that dutifully echoes its input
# ---------------------------------------------------------------------------
#
# Using an echo evaluator keeps the test deterministic and offline, while still
# exercising the pipeline as if a real model had produced canary-laden output.
# The real threat model is: adversary crafts input that a model then echoes or
# that any downstream log/trace pipeline records verbatim.


class _EchoCanaryEvaluator(BaseEvaluator):
    """Evaluator that returns the dataset's canary-laden content as the output."""

    def __init__(self) -> None:
        self.call_count = 0

    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: Any = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
        **_kwargs: Any,
    ) -> EvaluationResult:
        self.call_count += 1
        outputs: list[str] = []
        for index, example in enumerate(dataset.examples):
            # Echo the input verbatim — this is what a prompt-compliant model
            # would do if the attacker placed canaries in the prompt.
            outputs.append(str(example.input_data))
            if progress_callback:
                progress_callback(index, {"success": True})
        metrics = {"accuracy": 1.0, "examples_attempted": len(outputs)}
        return EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=len(outputs),
            successful_examples=len(outputs),
            duration=0.0,
            metrics=metrics,
            outputs=outputs,
            errors=[None] * len(outputs),
        )


# ---------------------------------------------------------------------------
# Scan targets
# ---------------------------------------------------------------------------
#
# Each scanner checks a public or enabled persistence surface for raw canaries.


class _ScanReport:
    """Accumulator for canary scan results across persistence targets."""

    def __init__(self) -> None:
        self.hits: list[tuple[str, str, list[str]]] = []  # (target, loc, canaries)
        self.gaps: list[tuple[str, str]] = []  # (target, reason)

    def record_hit(self, target: str, location: str, canaries: list[str]) -> None:
        self.hits.append((target, location, canaries))

    def record_gap(self, target: str, reason: str) -> None:
        self.gaps.append((target, reason))

    def format_hits(self) -> str:
        if not self.hits:
            return "(none)"
        return "\n".join(
            f"  - {target}:{location} -> {', '.join(canaries)}"
            for target, location, canaries in self.hits
        )

    def format_gaps(self) -> str:
        if not self.gaps:
            return "(none)"
        return "\n".join(f"  - {target}: {reason}" for target, reason in self.gaps)


def _scan_trial_results(trials: Iterable[TrialResult], report: _ScanReport) -> None:
    """Scan in-memory TrialResult objects for canaries.

    This is the most-local surface: if canaries survive to here, every
    downstream persistence path will leak them. This scanner must be
    a hard fail once redaction lands.
    """
    for trial in trials:
        # Check every string-serializable field that could carry a prompt
        # or completion, including outputs and arbitrary metadata.
        blobs: list[tuple[str, str]] = []
        if getattr(trial, "outputs", None):
            for i, output in enumerate(trial.outputs or []):
                blobs.append((f"trial[{trial.trial_id}].outputs[{i}]", str(output)))
        if getattr(trial, "metadata", None):
            blobs.append((f"trial[{trial.trial_id}].metadata", str(trial.metadata)))
        # Raw config stays available in memory so callers can replay or apply the
        # selected configuration. The external serialization scanner below owns
        # the public leak boundary for config payloads.
        for location, blob in blobs:
            hits = _contains_canary(blob)
            if hits:
                report.record_hit("in_memory_trial_result", location, hits)


def _scan_serialized_trial_results(
    trials: Iterable[TrialResult], report: _ScanReport
) -> None:
    """Scan the public TrialResult serialization contract."""
    for trial in trials:
        serialized = trial.to_dict()
        hits = _contains_canary(str(serialized))
        if hits:
            report.record_hit(
                "serialized_trial_result",
                f"trial[{trial.trial_id}].to_dict",
                hits,
            )


def _scan_audit_log(events: Iterable[Any], report: _ScanReport) -> None:
    """Scan the audit chain (traigent.security.audit) for canaries.

    The canary test exercises an AuditLogger sink explicitly so this scanner
    stays a hard assertion instead of a placeholder.
    """
    for index, event in enumerate(events):
        payload = event.to_dict() if hasattr(event, "to_dict") else event
        hits = _contains_canary(str(payload))
        if hits:
            report.record_hit("audit_log", f"event[{index}]", hits)


def _scan_observability_traces(
    trace_batches: Iterable[Iterable[dict[str, Any]]], report: _ScanReport
) -> None:
    """Scan observability traces for canaries.

    The canary test exercises ObservabilityClient with an in-memory sender so
    trace payload redaction is covered before data leaves the SDK.
    """
    for batch_index, trace_batch in enumerate(trace_batches):
        for trace_index, trace_payload in enumerate(trace_batch):
            hits = _contains_canary(str(trace_payload))
            if hits:
                report.record_hit(
                    "observability_trace",
                    f"batch[{batch_index}].trace[{trace_index}]",
                    hits,
                )


def _scan_stdout_capture(
    capsys: pytest.CaptureFixture[str], report: _ScanReport
) -> None:
    """Scan captured stdout/stderr for canaries.

    This one IS live today — any print() or logger.* in the pipeline will
    flow through pytest's capture. If a canary appears here, even the
    simplest logging is leaking.
    """
    captured = capsys.readouterr()
    for stream_name, stream in (("stdout", captured.out), ("stderr", captured.err)):
        hits = _contains_canary(stream)
        if hits:
            report.record_hit("stdout_capture", stream_name, hits)


def _exercise_audit_sink() -> list[Any]:
    """Write canaries through AuditLogger and return stored events for scanning."""
    audit_logger = AuditLogger("CanaryAuditSecret-1234567890!abcdef")
    audit_logger.log_event(
        AuditEventType.DATA_READ,
        user_id=CANARY_VALUES[0],
        session_id=f"session-{CANARY_VALUES[1]}",
        tenant_id=f"tenant-{CANARY_VALUES[2]}",
        resource_id=f"resource-{CANARY_VALUES[3]}",
        resource_type="dataset",
        action="read",
        message=f"Audit canary payload {CANARY_VALUES[4]}",
        details={
            "input": CANARY_VALUES[0],
            "expected": CANARY_VALUES[1],
            "token": CANARY_VALUES[3],
        },
    )
    return audit_logger.get_events()


def _exercise_observability_sink() -> list[list[dict[str, Any]]]:
    """Write canaries through ObservabilityClient and return captured batches."""
    sent_batches: list[list[dict[str, Any]]] = []

    def sender(traces: list[dict[str, Any]]) -> None:
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=1,
            max_buffer_age=999.0,
            max_queue_size=10,
        ),
        sender=sender,
    )
    trace_id = client.start_trace(
        "pii-canary-trace",
        trace_id="trace_pii_canary",
        user_id=CANARY_VALUES[0],
        metadata={"credit_card": CANARY_VALUES[1]},
        input_data={"ssn": CANARY_VALUES[2]},
    )
    client.record_observation(
        trace_id,
        name="pii-canary-observation",
        observation_type=ObservationType.GENERATION,
        input_data={"api_key": CANARY_VALUES[3]},
        output_data={"bearer": CANARY_VALUES[4]},
        metadata={"email": CANARY_VALUES[0]},
    )
    client.end_trace(
        trace_id,
        output_data={"answer": CANARY_VALUES[3]},
        metadata={"ssn": CANARY_VALUES[2]},
    )
    client.flush()
    client.close()
    return sent_batches


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------


@pytest.fixture
def canary_dataset() -> Dataset:
    """Dataset whose every example is canary-laden."""
    examples = [
        EvaluationExample(
            input_data={"question": f"Question containing {value}"},
            expected_output=f"Expected answer with {value}",
        )
        for _label, value, _rx in CANARIES
    ]
    return Dataset(
        examples,
        name="pii_canary_dataset",
        description="Canary PII seeded for leak detection",
    )


@pytest.fixture
def canary_evaluator() -> _EchoCanaryEvaluator:
    return _EchoCanaryEvaluator()


@pytest.mark.asyncio
async def test_canaries_do_not_leak_through_optimization_pipeline(
    canary_dataset: Dataset,
    canary_evaluator: _EchoCanaryEvaluator,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: seed canaries, run pipeline, scan every known sink."""
    from tests.shared.mocks.optimizers import MockOptimizer

    optimizer = MockOptimizer(
        {"canary_param": [f"config-holding-{CANARY_VALUES[0]}"]},
        ["accuracy"],
    )
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=canary_evaluator,
        max_trials=1,
    )

    result = await orchestrator.optimize(lambda x: x, canary_dataset)
    assert result.trials, "optimization produced no trials; canary test vacuous"
    audit_events = _exercise_audit_sink()
    observability_trace_batches = _exercise_observability_sink()

    trials = [t for t in result.trials if t.status == TrialStatus.COMPLETED]
    assert trials, "no completed trials; cannot assess canary leakage"

    report = _ScanReport()
    _scan_trial_results(trials, report)
    _scan_serialized_trial_results(trials, report)
    _scan_audit_log(audit_events, report)
    _scan_observability_traces(observability_trace_batches, report)
    _scan_stdout_capture(capsys, report)

    # Assertion: zero leaks, no unresolved gaps.
    failure_message = (
        "Canary PII leak detected.\n\n"
        f"Hits:\n{report.format_hits()}\n\n"
        f"Unscanned persistence gaps (must be wired for P0-1 close-out):\n"
        f"{report.format_gaps()}\n"
    )
    assert not report.hits and not report.gaps, failure_message


def test_canary_regexes_match_their_own_values() -> None:
    """Sanity: each canary regex matches its own seeded value.

    Protects against someone editing CANARIES in a way that silently
    breaks the scanner (e.g., typo in the regex that no longer matches
    the value we're planting). If this ever fails, the main test becomes
    meaningless even if it passes.
    """
    for label, value, rx in CANARIES:
        assert rx.search(value), (
            f"canary regex for {label!r} does not match its seed value"
        )


def test_contains_canary_detects_partial_embeddings() -> None:
    """_contains_canary must detect canaries even when wrapped in other text.

    Redaction that merely truncates or pads around the canary does not count
    as protection. The scanner has to catch embedded occurrences.
    """
    for label, value, _rx in CANARIES:
        blob = f"some prefix [{value}] suffix"
        assert label in _contains_canary(blob), (
            f"scanner missed {label!r} when embedded in surrounding text"
        )


def test_contains_canary_is_quiet_on_clean_text() -> None:
    """Scanner must not false-positive on ordinary strings."""
    clean = "The quick brown fox jumps over the lazy dog. Model temperature 0.7."
    assert _contains_canary(clean) == [], "scanner false-positive on clean text"
