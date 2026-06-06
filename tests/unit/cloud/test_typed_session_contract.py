# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""Phase 8: the typed session-create contract + content-free governance wire.

Red-first regressions for the scope-defining finding: the cloud client posted
the LEGACY shape (problem_statement/search_space) with NO promotion_policy —
strict governance was unreachable from the Python SDK. Contract source of
truth: TraigentSchema sdk_tuning (typed create) + promotion_policy_schema +
tvl_governance_schema (v4.5.0).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import CloudServiceError
from traigent.cloud.governance import build_tvl_governance, promotion_policy_to_wire
from traigent.cloud.models import SessionCreationRequest

STRICT_POLICY = {
    "dominance": "epsilon_pareto",
    "alpha": 0.05,
    "require_calibration": {"enabled": True, "hash_covered_context": ["model_versions"]},
}

GOVERNANCE = {"cvars": [{"name": "retriever.k", "type": "int", "governed": True}]}


def _request(**overrides) -> SessionCreationRequest:
    base = {
        "function_name": "answer_question",
        "configuration_space": {
            "model": {"type": "categorical", "choices": ["cheap", "strong"]}
        },
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 12, "privacy_mode": True},
        "max_trials": 5,
        "metadata": {"evaluation_set": "dev"},
    }
    base.update(overrides)
    return SessionCreationRequest(**base)


def _ops() -> ApiOperations:
    return ApiOperations(Mock())


class TestContractGate:
    def test_default_contract_builds_typed_payload(self, monkeypatch):
        monkeypatch.delenv("TRAIGENT_SESSION_CONTRACT", raising=False)
        payload = _ops()._build_session_payload(
            _request(promotion_policy=STRICT_POLICY, tvl_governance=GOVERNANCE), 5
        )
        # typed selectors at TOP level — this is what routes the backend's
        # governed path (is_typed_create_request)
        assert payload["function_name"] == "answer_question"
        assert payload["configuration_space"]["model"]["choices"] == ["cheap", "strong"]
        assert payload["objectives"] == ["accuracy"]
        assert payload["dataset_metadata"]["size"] == 12
        assert payload["promotion_policy"] == STRICT_POLICY
        assert payload["tvl_governance"] == GOVERNANCE
        # the legacy selectors must be GONE
        assert "problem_statement" not in payload
        assert "search_space" not in payload

    def test_legacy_contract_refuses_governed_sessions(self, monkeypatch):
        """The Phase 7 laundering bug must be impossible to reintroduce via
        the compatibility flag: legacy CANNOT carry strict mode."""
        monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "legacy")
        with pytest.raises(CloudServiceError, match="launder|legacy"):
            _ops()._build_session_payload(
                _request(promotion_policy=STRICT_POLICY), 5
            )

    def test_legacy_contract_for_ungoverned_keeps_old_shape(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "legacy")
        payload = _ops()._build_session_payload(_request(), 5)
        assert payload["problem_statement"] == "answer_question"
        assert "configuration_space" not in payload
        assert "promotion_policy" not in payload

    def test_invalid_contract_value_fails_loud(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "yolo")
        with pytest.raises(CloudServiceError, match="auto|typed|legacy"):
            _ops()._build_session_payload(_request(), 5)

    def test_typed_dataset_size_clamped_positive(self, monkeypatch):
        """The backend's typed path requires a positive dataset size."""
        monkeypatch.delenv("TRAIGENT_SESSION_CONTRACT", raising=False)
        payload = _ops()._build_session_payload(
            _request(dataset_metadata={"size": 0, "privacy_mode": True}), 5
        )
        assert payload["dataset_metadata"]["size"] == 1


class TestAutoFallback:
    @pytest.mark.asyncio
    async def test_ungoverned_typed_failure_falls_back_to_legacy_once(
        self, monkeypatch
    ):
        monkeypatch.delenv("TRAIGENT_SESSION_CONTRACT", raising=False)
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        ops = _ops()
        ops.client.auth_manager.augment_headers = AsyncMock(return_value={})
        attempts: list[dict] = []

        async def fake_post(payload, headers, connector):
            attempts.append(payload)
            if "problem_statement" not in payload:
                raise CloudServiceError("typed create rejected")
            return ("s-1", "e-1", "r-1")

        monkeypatch.setattr(ops, "_post_session_creation", fake_post)
        monkeypatch.setattr(ops, "_build_connector", lambda: None)

        result = await ops.create_traigent_session_via_api(_request())
        assert result == ("s-1", "e-1", "r-1")
        assert len(attempts) == 2
        assert "problem_statement" in attempts[1]

    @pytest.mark.asyncio
    async def test_governed_typed_failure_raises_no_laundering(self, monkeypatch):
        monkeypatch.delenv("TRAIGENT_SESSION_CONTRACT", raising=False)
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        ops = _ops()
        ops.client.auth_manager.augment_headers = AsyncMock(return_value={})
        attempts: list[dict] = []

        async def fake_post(payload, headers, connector):
            attempts.append(payload)
            raise CloudServiceError("typed create rejected")

        monkeypatch.setattr(ops, "_post_session_creation", fake_post)
        monkeypatch.setattr(ops, "_build_connector", lambda: None)

        with pytest.raises(CloudServiceError):
            await ops.create_traigent_session_via_api(
                _request(promotion_policy=STRICT_POLICY)
            )
        assert len(attempts) == 1  # NO legacy retry for governed sessions


class TestPayloadChokePoint:
    def test_typed_payload_filters_policy_at_the_wire(self, monkeypatch):
        """Review round 2 (codex): promotion_policy_to_wire is allowlist-style
        but the typed payload sent the request's policy VERBATIM — a direct
        SessionCreationRequest caller could serialize value/evidence-shaped
        blobs. The serializer must sit on the actual request-body choke
        point, not only on the orchestrator path."""
        monkeypatch.delenv("TRAIGENT_SESSION_CONTRACT", raising=False)
        leaky_policy = {
            **STRICT_POLICY,
            "internal_debug_blob": {"signals": [0.42, 0.43]},
            "chance_constraints": [
                {
                    "metric": "accuracy",
                    "threshold": 0.9,
                    "confidence": 0.95,
                    "examples": ["SECRET_PROMPT_b7e1"],
                }
            ],
        }
        payload = _ops()._build_session_payload(
            _request(promotion_policy=leaky_policy), 5
        )
        blob = json.dumps(payload["promotion_policy"])
        assert "internal_debug_blob" not in blob
        assert "SECRET_PROMPT_b7e1" not in blob
        assert "examples" not in blob
        # the contract-known fields survive intact
        assert payload["promotion_policy"]["chance_constraints"] == [
            {"metric": "accuracy", "threshold": 0.9, "confidence": 0.95}
        ]
        assert payload["promotion_policy"]["require_calibration"]["enabled"] is True


class TestGovernedNoSilentFallback:
    """Review round 2 (codex): SessionOperations.create_session caught every
    CloudServiceError into a LOCAL fallback — masking laundering refusals,
    invalid contract env values, and governed typed-create rejection from an
    old BE as a quiet local-only run. Governed/contract failures fail LOUD."""

    @staticmethod
    def _session_ops(create_error):
        from traigent.cloud.session_operations import SessionOperations

        ops = SessionOperations.__new__(SessionOperations)
        ops.client = Mock()
        ops.client.auth_manager.has_api_key = Mock(return_value=True)
        ops.client._create_traigent_session_via_api = AsyncMock(
            side_effect=create_error
        )
        ops._create_local_fallback_session = Mock(return_value="local_fallback_1")
        return ops

    def test_governed_create_failure_raises_never_falls_back(self, monkeypatch):
        monkeypatch.delenv("TRAIGENT_SESSION_CONTRACT", raising=False)
        ops = self._session_ops(CloudServiceError("typed create rejected"))
        with pytest.raises(CloudServiceError):
            ops.create_session(
                function_name="answer_question",
                search_space={"model": {"choices": ["a"]}},
                promotion_policy=STRICT_POLICY,
            )
        ops._create_local_fallback_session.assert_not_called()

    def test_contract_error_raises_even_for_ungoverned(self, monkeypatch):
        from traigent.cloud.client import SessionContractError

        monkeypatch.delenv("TRAIGENT_SESSION_CONTRACT", raising=False)
        ops = self._session_ops(
            SessionContractError("TRAIGENT_SESSION_CONTRACT must be one of ...")
        )
        with pytest.raises(SessionContractError):
            ops.create_session(
                function_name="answer_question",
                search_space={"model": {"choices": ["a"]}},
            )
        ops._create_local_fallback_session.assert_not_called()

    def test_ungoverned_backend_failure_still_falls_back(self, monkeypatch):
        """The availability fallback is FOR non-governed sessions — keep it."""
        monkeypatch.delenv("TRAIGENT_SESSION_CONTRACT", raising=False)
        ops = self._session_ops(CloudServiceError("backend unavailable"))
        result = ops.create_session(
            function_name="answer_question",
            search_space={"model": {"choices": ["a"]}},
        )
        assert result.session_id == "local_fallback_1"
        ops._create_local_fallback_session.assert_called_once()


class TestGovernanceBuilders:
    def test_promotion_policy_dataclass_to_wire(self):
        from traigent.tvl.models import PromotionPolicy, RequireCalibration

        policy = PromotionPolicy(
            alpha=0.05,
            min_effect={"accuracy": 0.01},
            require_calibration=RequireCalibration(
                enabled=True, hash_covered_context=["model_versions"]
            ),
        )
        wire = promotion_policy_to_wire(policy)
        assert wire["require_calibration"] == {
            "enabled": True,
            "hash_covered_context": ["model_versions"],
        }
        assert wire["min_effect"] == {"accuracy": 0.01}
        assert wire["dominance"] == "epsilon_pareto"

    def test_dict_policy_passthrough_filters_unknown_keys(self):
        wire = promotion_policy_to_wire(
            {**STRICT_POLICY, "internal_debug_blob": {"signals": [1, 2, 3]}}
        )
        assert "internal_debug_blob" not in wire
        assert wire["require_calibration"]["enabled"] is True

    def test_none_policy_is_none(self):
        assert promotion_policy_to_wire(None) is None

    @staticmethod
    def _calibrated(*, governed: bool, signal_name: str = "margin_signal"):
        from traigent.knobs.bindings import Calibrated
        from traigent.knobs.certificates import TargetProperty
        from traigent.knobs.signals import SignalSpec

        return Calibrated(
            signal=SignalSpec(
                name=signal_name,
                version="1",
                score_function="sf",
                score_function_version="1",
                comparator="ge",
                comparator_version="1",
            ),
            target=TargetProperty(name="accuracy", mode="require_calibration"),
            value_type="int",
            require_calibration=governed,
        )

    def test_build_tvl_governance_from_declared_bindings(self):
        from traigent.api.parameter_ranges import Choices
        from traigent.knobs.bindings import Fixed, Knob, Tuned

        class Space:
            knobs = {
                "model": Knob(name="model", binding=Tuned(range=Choices(["a", "b"]))),
                "retriever.k": Knob(
                    name="retriever.k", binding=self._calibrated(governed=True)
                ),
                "timeout": Knob(name="timeout", binding=Fixed(value=30)),
            }

        governance = build_tvl_governance(Space())
        assert governance == {
            "cvars": [{"name": "retriever.k", "type": "int", "governed": True}]
        }

    def test_governance_is_content_free(self):
        """P8 canary: no signal/source/target detail may serialize."""
        from traigent.knobs.bindings import Knob

        class Space:
            knobs = {
                "k": Knob(
                    name="k",
                    binding=self._calibrated(
                        governed=True, signal_name="SECRET_SIGNAL_a8f2"
                    ),
                ),
            }

        blob = json.dumps(build_tvl_governance(Space()))
        assert "SECRET_SIGNAL_a8f2" not in blob
        assert "accuracy" not in blob  # target detail never crosses

    def test_ungoverned_space_yields_none(self):
        from traigent.api.parameter_ranges import Choices
        from traigent.knobs.bindings import Knob, Tuned

        class Space:
            knobs = {
                "model": Knob(name="model", binding=Tuned(range=Choices(["a", "b"])))
            }

        assert build_tvl_governance(Space()) is None
