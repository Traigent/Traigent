"""Native categorical bools survive connected SDK validation and serialization."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.cloud.service import OptimizationRequest, TraigentCloudService
from traigent.cloud.session_operations import SessionOperations
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ValidationError as ValidationException

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online

# ---------------------------------------------------------------------------
# Shared helpers / stubs
# ---------------------------------------------------------------------------


class FakeAuthManager:
    def __init__(self) -> None:
        self.auth = SimpleNamespace(get_headers=AsyncMock(return_value={}))

    def has_api_key(self) -> bool:
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


@pytest.mark.parametrize(
    "value",
    [
        False,
        True,
        [False, True],
        (False, True),
        {"type": "categorical", "choices": [False, True]},
    ],
)
def test_connected_session_preserves_boolean_choices(value):
    from traigent.cloud.api_operations import ApiOperations

    client = FakeClient()
    result = SessionOperations(client).create_session(
        "my_function", {"flag": value}, metadata={"max_trials": 5}
    )
    assert result is not None
    client._create_traigent_session_via_api.assert_awaited_once()
    request = client._create_traigent_session_via_api.call_args.args[0]
    payload = ApiOperations(client)._build_typed_session_payload(request, max_trials=5)
    wire = json.loads(json.dumps(payload))
    choices = wire["configuration_space"]["flag"]["choices"]
    if type(value) is bool:
        expected = [value]
    elif isinstance(value, (list, tuple)):
        expected = list(value)
    else:
        expected = value["choices"]
    assert choices == expected
    assert all(type(choice) is bool for choice in choices)


@pytest.mark.parametrize(
    "value",
    [
        False,
        True,
        [False, True],
        (False, True),
        {"type": "categorical", "choices": [False, True]},
    ],
)
def test_service_validation_accepts_native_booleans(value):
    request = OptimizationRequest(
        function_name="greet",
        dataset=_sample_dataset(),
        configuration_space={"flag": value},
        objectives=["accuracy"],
    )
    TraigentCloudService._validate_request(request)
    assert request.configuration_space["flag"] is value


@pytest.mark.parametrize("value", [[0, 1], [0.1, 0.5], ["cheap", "strong"]])
def test_non_boolean_choices_keep_their_types(value):
    client = FakeClient()
    SessionOperations(client).create_session(
        "my_function", {"flag": value}, metadata={"max_trials": 5}
    )
    request = client._create_traigent_session_via_api.call_args.args[0]
    assert request.configuration_space["flag"] == value
    assert [type(item) for item in request.configuration_space["flag"]] == [
        type(item) for item in value
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("configuration_space", {}),
        ("objectives", []),
        ("max_trials", 0),
        ("max_trials", True),
    ],
)
def test_service_keeps_unrelated_invalid_request_validation(field, value):
    request = OptimizationRequest(
        function_name="greet",
        dataset=_sample_dataset(),
        configuration_space={"flag": [False, True]},
        objectives=["accuracy"],
    )
    setattr(request, field, value)
    with pytest.raises(ValidationException):
        TraigentCloudService._validate_request(request)


def test_offline_grid_delivers_false_and_true_as_native_config(monkeypatch, tmp_path):
    from traigent import get_config, optimize

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    observed = []

    @optimize(
        configuration_space={"flag": [False, True]},
        objectives=["accuracy"],
        eval_dataset=_sample_dataset(),
        offline=True,
        max_trials=2,
    )
    def agent(prompt):
        flag = get_config()["flag"]
        observed.append(flag)
        return "Hi" if flag else "Bye"

    result = agent.optimize_sync(algorithm="grid", max_trials=2, progress_bar=False)
    assert observed == [False, True]
    assert all(type(value) is bool for value in observed)
    assert len(result.trials) == 2
    assert result.best_config["flag"] is True
