# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""Phase 8 P3b: the client-attested certified-selection finalize report.

The builder mirrors the SERVER's fail-closed coverage rule client-side:
a report exists only when every governed CVAR carries a CERTIFIED
certificate with an issued hash — any gap means NO report (honest
no-winner), never a partial one. Content-free (RFC 0001 P8).
"""

from __future__ import annotations

import json
import types
from unittest.mock import Mock

import pytest

from traigent.cloud.governance import build_certified_selection
from traigent.knobs.canonical import canonical_hash
from traigent.knobs.certificates import (
    CertificateDecision,
    FreshnessContext,
    TargetProperty,
    issue_certificate,
)


def _target() -> TargetProperty:
    return TargetProperty(name="accuracy", mode="require_calibration")


def _ctx(cvar: str = "theta") -> FreshnessContext:
    return FreshnessContext(
        cvar_name=cvar,
        tuned_parent_values=(("model", "a"),),
        calibration_source_id="margin_calibration",
        signal_spec_hash=canonical_hash({"signal": "margin"}),
        calibrator_id="cal",
        calibrator_version="1",
        calibrator_params_hash=canonical_hash({}),
        dataset_hash="ds_v1",
        evidence_n=20,
        calibration_split="cal",
        eval_split="eval",
        target=_target(),
    )


def _certified(cvar: str = "theta"):
    return issue_certificate(cvar, "float", 0.5, _ctx(cvar))


class TestBuildCertifiedSelection:
    def test_full_coverage_builds_report(self):
        cert = _certified()
        report = build_certified_selection("trial_42", {"theta": cert})
        assert report == {
            "trial_id": "trial_42",
            "certificates": [
                {
                    "cvar_name": "theta",
                    "decision": "CERTIFIED_SELECTION",
                    "freshness_hash": cert.issued_hash,
                }
            ],
            "attestation": "sdk_client_attested",
        }

    def test_non_certified_decision_withholds_report(self):
        cert = _certified()
        uncertified = types.SimpleNamespace(
            decision=CertificateDecision.NO_DECISION,
            issued_hash=cert.issued_hash,
        )
        assert (
            build_certified_selection("trial_42", {"theta": cert, "phi": uncertified})
            is None
        )

    def test_missing_issued_hash_withholds_report(self):
        broken = types.SimpleNamespace(
            decision=CertificateDecision.CERTIFIED, issued_hash=""
        )
        assert build_certified_selection("trial_42", {"theta": broken}) is None

    def test_empty_inputs_withhold_report(self):
        assert build_certified_selection("trial_42", {}) is None
        assert build_certified_selection("", {"theta": _certified()}) is None
        assert build_certified_selection(None, None) is None

    def test_report_is_content_free(self):
        """P8 canary: the calibrated VALUE's hash, evidence counts, pool
        hashes, and target details must never serialize."""
        cert = _certified()
        blob = json.dumps(build_certified_selection("trial_42", {"theta": cert}))
        assert cert.subject_value_hash not in blob
        assert "evidence" not in blob
        assert "ds_v1" not in blob
        assert "accuracy" not in blob
        assert "0.5" not in blob


class _AckManagerStub:
    """Stand-in for BackendSessionManager's acknowledgment ledger."""

    def __init__(self, acknowledged: set[tuple[str, str]]):
        self._acked = acknowledged

    def is_trial_backend_acknowledged(self, session_id, trial_id):
        return (session_id, trial_id) in self._acked


class _OrchestratorStub:
    """Minimal host for the real _build_certified_selection_report method."""

    def __init__(
        self,
        *,
        strict,
        incumbent,
        resolver,
        promotions=1,
        session_manager=None,
        session_id=None,
    ):
        self._strict = strict
        self._best_trial_cached = incumbent
        self.knob_resolver = resolver
        self._certified_promotions = promotions
        # Default: no backend manager (local-only host) ⇒ ack guard is a
        # no-op so the OTHER report conditions can be tested in isolation.
        if session_manager is not None:
            self.backend_session_manager = session_manager
        if session_id is not None:
            self._active_session_id = session_id

    def _is_strict_evidence_mode(self):
        return self._strict


def _bind_report_method(stub):
    from traigent.core.orchestrator import OptimizationOrchestrator

    return types.MethodType(
        OptimizationOrchestrator._build_certified_selection_report, stub
    )


def _resolver(*, governed: bool, with_certificate: bool):
    from traigent.api.parameter_ranges import Choices
    from traigent.knobs.bindings import Calibrated, Knob, Tuned
    from traigent.knobs.signals import SignalSpec

    binding = Calibrated(
        signal=SignalSpec(
            name="margin",
            version="1",
            score_function="sf",
            score_function_version="1",
            comparator="ge",
            comparator_version="1",
        ),
        target=_target(),
        value_type="float",
        require_calibration=governed,
    )
    space = types.SimpleNamespace(
        knobs={
            "model": Knob(name="model", binding=Tuned(range=Choices(["a", "b"]))),
            "theta": Knob(name="theta", binding=binding),
        }
    )
    calibrated_inputs = {}
    if with_certificate:
        calibrated_inputs["theta"] = types.SimpleNamespace(
            certificate=_certified("theta")
        )
    return types.SimpleNamespace(_space=space, _calibrated_inputs=calibrated_inputs)


class TestOrchestratorReportConditions:
    def test_report_built_when_all_conditions_hold(self):
        stub = _OrchestratorStub(
            strict=True,
            incumbent=types.SimpleNamespace(trial_id="trial_7"),
            resolver=_resolver(governed=True, with_certificate=True),
        )
        report = _bind_report_method(stub)()
        assert report is not None
        assert report["trial_id"] == "trial_7"
        assert report["certificates"][0]["cvar_name"] == "theta"

    def test_no_report_when_not_strict(self):
        stub = _OrchestratorStub(
            strict=False,
            incumbent=types.SimpleNamespace(trial_id="trial_7"),
            resolver=_resolver(governed=True, with_certificate=True),
        )
        assert _bind_report_method(stub)() is None

    def test_no_report_without_incumbent(self):
        stub = _OrchestratorStub(
            strict=True,
            incumbent=None,
            resolver=_resolver(governed=True, with_certificate=True),
        )
        assert _bind_report_method(stub)() is None

    def test_no_report_when_governed_cvar_lacks_certificate(self):
        """Fail closed: a coverage gap means NO report, never a partial one."""
        stub = _OrchestratorStub(
            strict=True,
            incumbent=types.SimpleNamespace(trial_id="trial_7"),
            resolver=_resolver(governed=True, with_certificate=False),
        )
        assert _bind_report_method(stub)() is None

    def test_no_report_without_certified_promotion(self):
        """Review round 2 (codex): the FIRST trial seeds the incumbent as
        comparison initialization, NOT as certification — terminal strict
        selection requires _certified_promotions > 0 before naming a winner
        (test_fail_closed_promotion pins it). The wire report must mirror
        that guard or it overclaims a winner the SDK result itself refuses
        to certify."""
        stub = _OrchestratorStub(
            strict=True,
            incumbent=types.SimpleNamespace(trial_id="trial_7"),
            resolver=_resolver(governed=True, with_certificate=True),
            promotions=0,
        )
        assert _bind_report_method(stub)() is None

    def test_no_report_without_resolver_trial_binding(self):
        """backend_guided risk: no resolver/space ⇒ no report (fail closed)."""
        stub = _OrchestratorStub(
            strict=True,
            incumbent=types.SimpleNamespace(trial_id="trial_7"),
            resolver=None,
        )
        assert _bind_report_method(stub)() is None

    def test_report_built_when_incumbent_backend_acknowledged(self):
        """With a backend session manager present, a report is built ONLY when
        the incumbent's trial id was minted+acknowledged by the backend."""
        stub = _OrchestratorStub(
            strict=True,
            incumbent=types.SimpleNamespace(trial_id="be_trial_7"),
            resolver=_resolver(governed=True, with_certificate=True),
            session_manager=_AckManagerStub({("sess-1", "be_trial_7")}),
            session_id="sess-1",
        )
        report = _bind_report_method(stub)()
        assert report is not None
        assert report["trial_id"] == "be_trial_7"

    def test_no_report_when_incumbent_not_backend_acknowledged(self):
        """Fail closed (risk-register unbindable case): if the incumbent's
        trial id was NEVER acknowledged by the backend, the winner cannot be
        bound to a server record — send NO report, even though every other
        certified-selection condition holds."""
        stub = _OrchestratorStub(
            strict=True,
            incumbent=types.SimpleNamespace(trial_id="client_hash_unbound"),
            resolver=_resolver(governed=True, with_certificate=True),
            session_manager=_AckManagerStub(set()),  # nothing acknowledged
            session_id="sess-1",
        )
        assert _bind_report_method(stub)() is None

    def test_no_report_when_session_id_missing_under_backend_manager(self):
        """A backend manager with no active session id cannot have acknowledged
        any trial ⇒ withhold (fail closed)."""
        stub = _OrchestratorStub(
            strict=True,
            incumbent=types.SimpleNamespace(trial_id="be_trial_7"),
            resolver=_resolver(governed=True, with_certificate=True),
            session_manager=_AckManagerStub({("sess-1", "be_trial_7")}),
            session_id=None,
        )
        assert _bind_report_method(stub)() is None


class TestWireThreading:
    @pytest.mark.asyncio
    async def test_finalize_body_carries_report_top_level(self, monkeypatch):
        """The report rides the finalize body at the TOP level (never under
        metadata) and is omitted entirely when absent."""
        from traigent.cloud.session_operations import SessionOperations

        ops = SessionOperations.__new__(SessionOperations)
        ops.client = Mock()
        captured: dict = {}

        class FakeResponse:
            status = 200

            async def json(self):
                return {"session_id": "s-1", "best_config": {"m": 1}}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def post(self, url, *, json=None, headers=None, timeout=None):
                captured["body"] = json
                return FakeResponse()

        async def fake_headers(base):
            return base

        ops.client.auth_manager.augment_headers = fake_headers
        ops.client.backend_config.api_base_url = "http://localhost:5000/api/v1"
        monkeypatch.setattr(
            "traigent.cloud.session_operations.aiohttp",
            types.SimpleNamespace(
                ClientSession=FakeSession,
                ClientTimeout=lambda total=None: None,
                ClientError=Exception,
            ),
        )
        monkeypatch.setattr("traigent.cloud.session_operations.AIOHTTP_AVAILABLE", True)

        report = {
            "trial_id": "t1",
            "certificates": [],
            "attestation": "sdk_client_attested",
        }
        await ops._finalize_session_via_api("s-1", "r-1", certified_selection=report)
        assert captured["body"]["certified_selection"] == report
        assert "certified_selection" not in captured["body"].get("metadata", {})

        captured.clear()
        await ops._finalize_session_via_api("s-1", "r-1")
        assert "certified_selection" not in captured["body"]

    @pytest.mark.asyncio
    async def test_finalize_body_carries_session_aggregation_top_level(
        self, monkeypatch
    ):
        """Traigent#1720/#1724 (g2:agg-summary): session_aggregation rides the
        finalize body at the TOP level (never under metadata), mirroring
        certified_selection exactly, and is omitted entirely when absent."""
        from traigent.cloud.session_operations import SessionOperations

        ops = SessionOperations.__new__(SessionOperations)
        ops.client = Mock()
        captured: dict = {}

        class FakeResponse:
            status = 200

            async def json(self):
                return {"session_id": "s-1", "best_config": {"m": 1}}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def post(self, url, *, json=None, headers=None, timeout=None):
                captured["body"] = json
                return FakeResponse()

        async def fake_headers(base):
            return base

        ops.client.auth_manager.augment_headers = fake_headers
        ops.client.backend_config.api_base_url = "http://localhost:5000/api/v1"
        monkeypatch.setattr(
            "traigent.cloud.session_operations.aiohttp",
            types.SimpleNamespace(
                ClientSession=FakeSession,
                ClientTimeout=lambda total=None: None,
                ClientError=Exception,
            ),
        )
        monkeypatch.setattr("traigent.cloud.session_operations.AIOHTTP_AVAILABLE", True)

        agg = {"selection_mode": "aggregated_mean", "sdk_version": "0.0.0"}
        await ops._finalize_session_via_api("s-1", "r-1", session_aggregation=agg)
        # Egress boundary re-sanitizes (Codex round 6): the payload is carried
        # TOP-LEVEL (never under metadata) with its valid fields preserved.
        carried = captured["body"]["session_aggregation"]
        assert carried["selection_mode"] == "aggregated_mean"
        assert carried["sdk_version"] == "0.0.0"
        assert "session_aggregation" not in captured["body"].get("metadata", {})

        captured.clear()
        await ops._finalize_session_via_api("s-1", "r-1")
        assert "session_aggregation" not in captured["body"]

    def test_manager_drops_report_on_failed_runs(self):
        """A failed optimization must never carry a certified winner."""
        from traigent.api.types import OptimizationStatus
        from traigent.core.backend_session_manager import BackendSessionManager

        manager = BackendSessionManager.__new__(BackendSessionManager)
        manager._backend_tracking_enabled = True
        client = Mock(spec=["finalize_session_sync"])
        client.finalize_session_sync = Mock(return_value={"ok": True})
        manager._backend_client = client

        report = {"trial_id": "t1"}
        manager.finalize_session(
            "s-1", OptimizationStatus.FAILED, certified_selection=report
        )
        assert (
            client.finalize_session_sync.call_args.kwargs["certified_selection"] is None
        )

        manager.finalize_session(
            "s-1", OptimizationStatus.COMPLETED, certified_selection=report
        )
        assert (
            client.finalize_session_sync.call_args.kwargs["certified_selection"]
            == report
        )

    def test_manager_threads_session_aggregation_regardless_of_status(self):
        """session_aggregation is not gated on completion status the way
        certified_selection is — it rides whatever finalize call is made."""
        from traigent.api.types import OptimizationStatus
        from traigent.core.backend_session_manager import BackendSessionManager

        manager = BackendSessionManager.__new__(BackendSessionManager)
        manager._backend_tracking_enabled = True
        client = Mock(spec=["finalize_session_sync"])
        client.finalize_session_sync = Mock(return_value={"ok": True})
        manager._backend_client = client

        agg = {"selection_mode": "aggregated_mean"}
        manager.finalize_session(
            "s-1", OptimizationStatus.COMPLETED, session_aggregation=agg
        )
        assert (
            client.finalize_session_sync.call_args.kwargs["session_aggregation"] == agg
        )
