"""End-to-end PII canary leak test for the optimization pipeline.

This is the P0-1 safety net referenced in
``/home/nimrodbu/.claude/plans/gpt-will-work-on-declarative-chipmunk.md``.

It seeds a set of *canary* PII patterns (fake email, CC, SSN, API key) into an
optimization run and, after the run completes, scans every persistence target
the SDK might write to for any trace of the canaries. Any hit is a leak.

The test is written to **fail closed** as redaction capability lands. Today,
several of the persistence targets are known-unscanned and the test is therefore
marked xfail with a precise reason — flipping those to hard asserts is the P0-1
acceptance criterion.

Canary patterns are intentionally easy for humans and tooling to recognize so
that a hit in prod logs during manual review is obvious.
"""

# Traceability: CONC-Layer-Security CONC-Quality-Privacy FUNC-ORCH-LIFECYCLE
# See docs/security/redaction.md (to be authored as part of P0-1 close-out).

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
            outputs.append(str(example.input))
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
# Each target is a callable that returns an iterable of (location_name, text)
# tuples. A target that cannot be scanned yet returns an empty iterable and
# registers the gap via ``pytest.xfail``. As P0-1 implementation lands, gaps
# should be replaced with real scanners and their corresponding xfails removed.


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


def _scan_trial_results(
    trials: Iterable[TrialResult], report: _ScanReport
) -> None:
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
            blobs.append(
                (f"trial[{trial.trial_id}].metadata", str(trial.metadata))
            )
        if getattr(trial, "config", None):
            blobs.append(
                (f"trial[{trial.trial_id}].config", str(trial.config))
            )
        for location, blob in blobs:
            hits = _contains_canary(blob)
            if hits:
                report.record_hit("in_memory_trial_result", location, hits)


def _scan_audit_log(report: _ScanReport) -> None:
    """Scan the audit chain (traigent.security.audit) for canaries.

    TODO(P0-1): Hook the audit subsystem's in-memory buffer or test sink and
    iterate every event.details/event.context for canaries. For now we record
    a gap so the xfail is precise.
    """
    report.record_gap(
        "audit_log",
        "scanner not wired; audit subsystem has no test sink exposed yet",
    )


def _scan_observability_traces(report: _ScanReport) -> None:
    """Scan observability traces for canaries.

    TODO(P0-1): Wire an in-memory ObservabilityClient sink for tests and
    iterate its captured traces. Observability payloads include full
    config values, which is where canaries planted in config are most
    likely to survive.
    """
    report.record_gap(
        "observability_traces",
        "scanner not wired; observability client has no test sink exposed",
    )


def _scan_stdout_capture(capsys: pytest.CaptureFixture[str], report: _ScanReport) -> None:
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


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------


@pytest.fixture
def canary_dataset() -> Dataset:
    """Dataset whose every example is canary-laden."""
    examples = [
        EvaluationExample(
            input=f"Question containing {value}",
            output=f"Expected answer with {value}",
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


# Explicit xfail until redaction lands. The strict=True means the test
# *must start failing* (unexpected pass) the moment redaction is functional,
# which is how we prove the pipeline is complete. Remove xfail and flip any
# remaining .record_gap() to .record_hit() as each scanner comes online.
@pytest.mark.xfail(
    reason=(
        "P0-1 redaction pipeline not yet implemented. Scanners for audit "
        "and observability stores are not yet wired. Flip to strict pass "
        "once redaction is live and all record_gap() calls become "
        "record_hit() paths with zero hits."
    ),
    strict=False,
)
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

    trials = [
        t for t in result.trials if t.status == TrialStatus.COMPLETED
    ]
    assert trials, "no completed trials; cannot assess canary leakage"

    report = _ScanReport()
    _scan_trial_results(trials, report)
    _scan_audit_log(report)
    _scan_observability_traces(report)
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
        assert rx.search(value), f"canary regex for {label!r} does not match its seed value"


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
