"""Client-side boolean configuration_space validation tests (issue #1488).

These tests verify that:
  - boolean values in configuration_space raise a clear, field-named
    ValidationException BEFORE any network call is attempted
  - plain int/float values (0/1) are NOT rejected
  - the same guard is present in both SessionOperations.create_session and
    TraigentCloudService._validate_request
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.cloud.service import OptimizationRequest, TraigentCloudService
from traigent.cloud.session_operations import SessionOperations
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ValidationError as ValidationException

# ---------------------------------------------------------------------------
# Shared helpers / stubs
# ---------------------------------------------------------------------------


class FakeAuthManager:
    def __init__(self) -> None:
        self.auth = SimpleNamespace(get_headers=AsyncMock(return_value={}))

    def has_api_key(self) -> bool:  # pragma: no cover - not reached on bool reject
        return True


class FakeClient:
    """Minimal BackendIntegratedClient stub for SessionOperations tests."""

    def __init__(self) -> None:
        self._active_sessions: dict[str, Any] = {}
        self._active_sessions_lock = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )
        self._max_active_sessions = 5
        self.session_bridge = SimpleNamespace(
            create_session_mapping=MagicMock(),
            get_session_mapping=MagicMock(return_value=None),
        )
        self.backend_config = SimpleNamespace(api_base_url=None, backend_base_url=None)
        self.auth_manager = FakeAuthManager()
        self._register_security_session = MagicMock()
        self._revoke_security_session = MagicMock()
        self.local_storage = None
        # Track HTTP calls
        self._create_traigent_session_via_api = AsyncMock(
            return_value=("session-1", "experiment-1", "run-1")
        )
        self._url_invalid = False
        self.no_egress = False


def _sample_dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(
                input_data={"prompt": "Hello"},
                expected_output="Hi",
                metadata={},
            )
        ]
    )


# ---------------------------------------------------------------------------
# SessionOperations tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _offline_off(monkeypatch):
    """Ensure offline mode is off so create_session actually validates."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")


class TestSessionOperationsBoolValidation:
    """Boolean validation in SessionOperations.create_session (issue #1488)."""

    def test_boolean_in_choice_list_raises_validation_exception(self):
        """A config space with True/False values must raise ValidationException
        naming the offending field, without any network call."""
        client = FakeClient()
        ops = SessionOperations(client)

        with pytest.raises(ValidationException) as exc_info:
            ops.create_session(
                "my_function",
                {"include_schema": [True, False], "model": ["gpt-4o"]},
                metadata={"max_trials": 5},
            )

        msg = str(exc_info.value)
        assert "include_schema" in msg, f"offending field name missing from: {msg}"
        assert "boolean" in msg.lower(), f"'boolean' missing from: {msg}"
        # Workaround hint must be present
        assert '"true"' in msg or "true" in msg.lower() or "0/1" in msg, (
            f"workaround hint missing from: {msg}"
        )

        # The HTTP call must NOT have been reached
        client._create_traigent_session_via_api.assert_not_called()

    def test_multiple_bool_knobs_all_named_in_error(self):
        """All offending parameter names must appear in the error message."""
        client = FakeClient()
        ops = SessionOperations(client)

        with pytest.raises(ValidationException) as exc_info:
            ops.create_session(
                "my_function",
                {
                    "include_schema": [True, False],
                    "use_cache": [True, False],
                    "model": ["gpt-4o"],
                },
                metadata={"max_trials": 5},
            )

        msg = str(exc_info.value)
        assert "include_schema" in msg
        assert "use_cache" in msg
        client._create_traigent_session_via_api.assert_not_called()

    def test_int_and_float_values_pass_validation(self, monkeypatch):
        """Integer and float choice lists (including 0/1) must NOT be rejected."""
        client = FakeClient()
        ops = SessionOperations(client)

        # Patch the actual API call so we don't need a real backend
        async def fake_api(_req):
            return ("session-ok", "experiment-ok", "run-ok")

        monkeypatch.setattr(client, "_create_traigent_session_via_api", fake_api)

        # Should not raise — 0/1 are ints, not bools
        result = ops.create_session(
            "my_function",
            {
                "top_k": [1, 3, 5],
                "temperature": [0.1, 0.5, 0.9],
                "use_flag": [0, 1],  # int 0/1 must be allowed
                "model": ["gpt-4o", "gpt-3.5-turbo"],
            },
            metadata={"max_trials": 5},
        )
        assert result is not None

    def test_str_and_int_values_do_not_raise(self, monkeypatch):
        """A typical string-only config space must pass validation."""
        client = FakeClient()
        ops = SessionOperations(client)

        async def fake_api(_req):
            return ("session-ok", "experiment-ok", "run-ok")

        monkeypatch.setattr(client, "_create_traigent_session_via_api", fake_api)

        result = ops.create_session(
            "my_function",
            {"model": ["gpt-4o", "gpt-3.5-turbo"], "top_k": [1, 3]},
            metadata={"max_trials": 5},
        )
        assert result is not None

    def test_single_true_value_raises(self):
        """Even a single True in a list must be caught."""
        client = FakeClient()
        ops = SessionOperations(client)

        with pytest.raises(ValidationException) as exc_info:
            ops.create_session(
                "my_function",
                {"flag": [True]},
                metadata={"max_trials": 5},
            )

        assert "flag" in str(exc_info.value)
        client._create_traigent_session_via_api.assert_not_called()


# ---------------------------------------------------------------------------
# TraigentCloudService tests
# ---------------------------------------------------------------------------


class TestCloudServiceBoolValidation:
    """Boolean validation in TraigentCloudService._validate_request (issue #1488)."""

    @pytest.mark.asyncio
    async def test_boolean_choice_raises_validation_exception(self):
        """Boolean values in the service request must raise before any work."""
        service = TraigentCloudService()
        request = OptimizationRequest(
            function_name="greet",
            dataset=_sample_dataset(),
            configuration_space={"include_schema": [True, False]},
            objectives=["accuracy"],
        )

        with pytest.raises(ValidationException) as exc_info:
            await service.process_optimization_request(request)

        msg = str(exc_info.value)
        assert "include_schema" in msg
        assert "boolean" in msg.lower()

    @pytest.mark.asyncio
    async def test_int_01_values_do_not_raise(self):
        """Integer 0/1 in the service request must NOT trigger the boolean guard."""
        service = TraigentCloudService()
        # This will fail at billing/subset stage (no real dataset), but NOT at
        # boolean validation — that's the only gate we are testing here.
        request = OptimizationRequest(
            function_name="greet",
            dataset=_sample_dataset(),
            configuration_space={"flag": [0, 1], "model": ["gpt-4o"]},
            objectives=["accuracy"],
        )

        # The call will fail past the validation stage (no backend, billing, etc.)
        # but must NOT raise ValidationException about booleans.
        try:
            await service.process_optimization_request(request)
        except ValidationException as exc:
            assert "boolean" not in str(exc).lower(), (
                f"int 0/1 was falsely flagged as boolean: {exc}"
            )
        except Exception:
            # Any other error (billing, subset, etc.) is fine — we only care that
            # the boolean guard was NOT the cause.
            pass
