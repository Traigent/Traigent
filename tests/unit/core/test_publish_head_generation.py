"""Publishing the winner is refused when the agent head moved during the run.

The head generation is read once, when the optimization step starts, and sent
with the promoting publish. A head that another writer advanced in between makes
the backend answer 409; the SDK raises a typed error and publishes nothing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

import traigent
from traigent.api.types import OptimizationResult
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.best_config_runtime import (
    CloudBestConfigStaleHeadError,
    CloudPublishUnavailable,
    CloudPublishUnavailableReason,
    canonical_json,
    compute_spec_hash,
    sha256_digest,
)
from traigent.core.session_types import SessionCreationResult
from traigent.evaluators.base import Dataset, EvaluationExample

# Opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online

_SPACE = {"temperature": [0.0, 0.5]}
_AGENT = "agent-7"


def _dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"text": "a"}, "ok"),
            EvaluationExample({"text": "b"}, "ok"),
        ],
        name="head_generation_cas",
    )


class _FakeBackend:
    """Tracking client and best-config backend with a real compare-and-set head."""

    def __init__(
        self,
        *,
        agent_id: str | None = _AGENT,
        head: int = 3,
        report_head: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.head = head
        self.report_head = report_head
        self.head_reads = 0
        self.publishes: list[dict] = []
        self.no_egress = False
        self.cloud_egress_intent = False
        self.enable_fallback = False
        self.local_storage = None
        auth = Mock()
        auth.has_api_key = Mock(return_value=True)
        self.auth_manager = auth
        self.auth = auth
        self.submit_result = Mock()
        if not report_head:
            # An older client/backend has no head read at all.
            self.fetch_agent_head_generation_sync = None  # type: ignore[assignment]

    # session tracking
    def create_session(self, function_name, search_space, optimization_goal, **kw):
        return SessionCreationResult.connected(
            session_id="bs-1", agent_id=self.agent_id
        )

    def get_session_mapping(self, session_id):
        return SimpleNamespace(experiment_id="exp-1", experiment_run_id="run-1")

    def request_trial_slot(self, session_id):
        return SimpleNamespace(trial_id=None, optimization_complete=True, reason="done")

    def _submit_trial_result_via_session(self, **kwargs):
        return True

    # head + best-config surface
    def fetch_agent_head_generation_sync(self, agent_id, *, environment=None):
        self.head_reads += 1
        return self.head

    def fetch_best_config_sync(self, config_id, **kwargs):
        return None

    def publish_best_config_sync(
        self,
        spec,
        *,
        environment=None,
        if_match=None,
        agent_id=None,
        expected_head_generation=None,
    ):
        self.publishes.append(
            {"agent_id": agent_id, "expected": expected_head_generation}
        )
        if expected_head_generation is not None:
            if expected_head_generation != self.head:
                raise CloudBestConfigStaleHeadError(
                    current_generation=self.head,
                    expected_generation=expected_head_generation,
                    agent_id=agent_id,
                    environment=environment,
                )
            self.head += 1
        return {
            "config_id": spec["config_id"],
            "version": 1,
            "spec": spec,
            "spec_hash": compute_spec_hash(spec),
            "config_hash": sha256_digest(canonical_json(spec["config"])),
        }


@pytest.fixture
def online(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg-fake-portal-key-head")
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))


def _make_wrapper():
    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space=_SPACE,
        injection_mode="parameter",
    )
    def answer(text: str, config) -> str:
        return "ok"

    return answer


async def _optimize(answer, fake: _FakeBackend):
    with (
        patch.object(
            BackendSessionManager,
            "create_backend_client",
            staticmethod(lambda _c: fake),
        ),
        patch("traigent.optimizers.interactive_optimizer.InteractiveOptimizer"),
        patch("traigent.cloud.client.TraigentCloudClient"),
    ):
        result = await answer.optimize(algorithm="grid")
    assert isinstance(result, OptimizationResult) and result.best_config
    return result


async def _run(fake: _FakeBackend):
    answer = _make_wrapper()
    return answer, await _optimize(answer, fake)


def _publish(answer, fake):
    with patch("traigent.cloud.backend_client.get_backend_client", return_value=fake):
        return answer.publish_best_config()


@pytest.mark.asyncio
async def test_publish_sends_the_generation_read_at_step_start(online):
    fake = _FakeBackend(head=3)
    answer, result = await _run(fake)
    assert result.metadata["promotion_precondition"]["generation"] == 3

    _publish(answer, fake)

    assert fake.publishes == [{"agent_id": _AGENT, "expected": 3}]
    assert fake.head == 4


@pytest.mark.asyncio
async def test_publish_after_another_writer_is_refused_with_captured_generation(online):
    fake = _FakeBackend(head=3)
    answer, result = await _run(fake)
    reads_after_run = fake.head_reads

    fake.head = 4  # another writer promoted while this run was in flight

    with pytest.raises(CloudBestConfigStaleHeadError) as excinfo:
        _publish(answer, fake)

    err = excinfo.value
    # The captured value went out, not a fresh read of the moved head.
    assert fake.publishes == [{"agent_id": _AGENT, "expected": 3}]
    assert fake.head_reads == reads_after_run
    assert (err.expected_generation, err.current_generation) == (3, 4)
    assert err.agent_id == _AGENT
    assert err.reason is CloudPublishUnavailableReason.STALE_HEAD_GENERATION
    assert isinstance(err, CloudPublishUnavailable)
    # The other writer's head is untouched and the run's winner is still usable.
    assert fake.head == 4
    assert err.best_config == result.best_config
    assert answer.get_optimization_results() is result


@pytest.mark.asyncio
async def test_generation_is_captured_at_step_start_not_at_publish_time(online):
    """The head moves after the step starts; a publish-time read would see 4 and pass."""
    fake = _FakeBackend(head=3)
    answer, _ = await _run(fake)
    assert fake.head_reads == 1  # exactly one read, during the run

    fake.head = 4
    with pytest.raises(CloudBestConfigStaleHeadError):
        _publish(answer, fake)

    assert fake.head_reads == 1  # publishing did not read the head again
    assert fake.publishes[0]["expected"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fake_kwargs",
    [{"agent_id": None}, {"report_head": False}],
    ids=["backend-disclosed-no-agent", "client-cannot-read-heads"],
)
async def test_without_a_reported_generation_publish_is_unchanged(online, fake_kwargs):
    fake = _FakeBackend(**fake_kwargs)
    answer, result = await _run(fake)
    assert "promotion_precondition" not in result.metadata

    _publish(answer, fake)

    assert fake.publishes == [{"agent_id": None, "expected": None}]


@pytest.mark.asyncio
async def test_failed_head_read_does_not_fail_the_run(online):
    fake = _FakeBackend()
    fake.fetch_agent_head_generation_sync = Mock(side_effect=CloudServiceError("boom"))
    answer, result = await _run(fake)
    assert "promotion_precondition" not in result.metadata

    _publish(answer, fake)
    assert fake.publishes == [{"agent_id": None, "expected": None}]


@pytest.mark.asyncio
async def test_generation_captured_for_another_environment_is_not_sent(online):
    fake = _FakeBackend()
    answer, _ = await _run(fake)
    answer._csm.best_config_environment = "production"

    _publish(answer, fake)

    assert fake.publishes == [{"agent_id": None, "expected": None}]


# --- the precondition belongs to the config's producing result -----------------


@pytest.mark.asyncio
async def test_applying_an_older_result_publishes_with_its_own_generation(online):
    """Run A captures 3; the head moves to 4; run B captures 4. Publishing A's
    config must send 3 (and be refused), not B's 4 (and wrongly succeed)."""
    fake = _FakeBackend(head=3)
    answer = _make_wrapper()
    result_a = await _optimize(answer, fake)
    fake.head = 4
    result_b = await _optimize(answer, fake)
    assert result_a.metadata["promotion_precondition"]["generation"] == 3
    assert result_b.metadata["promotion_precondition"]["generation"] == 4

    answer.apply_best_config(result_a)
    with pytest.raises(CloudBestConfigStaleHeadError) as excinfo:
        _publish(answer, fake)

    assert fake.publishes == [{"agent_id": _AGENT, "expected": 3}]
    assert fake.head == 4
    assert excinfo.value.best_config == result_a.best_config


@pytest.mark.asyncio
async def test_applying_the_newer_result_publishes_with_its_generation(online):
    fake = _FakeBackend(head=3)
    answer = _make_wrapper()
    result_a = await _optimize(answer, fake)
    fake.head = 4
    result_b = await _optimize(answer, fake)

    answer.apply_best_config(result_a)
    answer.apply_best_config(result_b)
    _publish(answer, fake)

    assert fake.publishes == [{"agent_id": _AGENT, "expected": 4}]


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["set_config", "best_config_setter"])
async def test_manual_or_replaced_config_inherits_no_generation(online, how):
    fake = _FakeBackend(head=3)
    answer, _ = await _run(fake)
    fake.head = 4  # a run's generation would now be stale

    if how == "set_config":
        answer.set_config({"temperature": 0.5})
    else:
        answer._csm.best_config = {"temperature": 0.5}
    _publish(answer, fake)

    assert fake.publishes == [{"agent_id": None, "expected": None}]


@pytest.mark.asyncio
async def test_result_swapped_after_snapshot_does_not_change_what_is_sent(online):
    fake = _FakeBackend(head=3)
    answer = _make_wrapper()
    result_a = await _optimize(answer, fake)
    config_a = dict(result_a.best_config)
    fake.head = 4
    result_b = await _optimize(answer, fake)
    other = 0.5 if config_a["temperature"] == 0.0 else 0.0
    result_b.best_config = {**result_b.best_config, "temperature": other}
    answer.apply_best_config(result_a)

    def swap_during_network_io(config_id, **kwargs):
        answer.apply_best_config(result_b)
        return None

    fake.fetch_best_config_sync = swap_during_network_io

    with pytest.raises(CloudBestConfigStaleHeadError) as excinfo:
        _publish(answer, fake)

    assert fake.publishes == [{"agent_id": _AGENT, "expected": 3}]
    assert excinfo.value.best_config == config_a


# --- backend client wire mapping -------------------------------------------------

_KEY = "tg_" + "x" * 61  # pragma: allowlist secret


def _client() -> BackendIntegratedClient:
    client = BackendIntegratedClient(api_key=_KEY, base_url="https://api.test")
    client.auth_manager.auth.get_headers = AsyncMock(
        return_value={"Authorization": "Bearer test-token"}
    )
    return client


def _resp(status: int, body) -> Mock:
    response = Mock(status_code=status, text=str(body), headers={})
    response.json.return_value = body
    return response


_SPEC = {"config_id": "answerer", "environment": "staging", "config": {"t": 1}}
_STALE_BODY = {
    "success": False,
    "code": "STALE_HEAD_GENERATION",
    "error_code": "STALE_HEAD_GENERATION",
    "current_generation": 5,
    "expected_generation": 3,
    "message": "agent head is at generation 5, not the expected 3",
}


@patch("requests.post")
def test_409_stale_body_maps_to_typed_error(mock_post):
    mock_post.return_value = _resp(409, _STALE_BODY)
    with pytest.raises(CloudBestConfigStaleHeadError) as excinfo:
        _client().publish_best_config_sync(
            _SPEC, environment="staging", agent_id="a1", expected_head_generation=3
        )
    err = excinfo.value
    assert (err.current_generation, err.expected_generation) == (5, 3)
    assert (err.agent_id, err.environment) == ("a1", "staging")
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["X-Traigent-Agent-Id"] == "a1"
    assert headers["X-Traigent-Expected-Head-Generation"] == "3"


@patch("requests.post")
@pytest.mark.parametrize(
    "body",
    [
        {"code": "IDEMPOTENCY_CONFLICT", "message": "x"},
        {"code": "STALE_HEAD_GENERATION"},  # no current_generation
        ["not", "an", "object"],
    ],
)
def test_other_or_malformed_409_stays_a_generic_rejection(mock_post, body):
    mock_post.return_value = _resp(409, body)
    with pytest.raises(CloudServiceError) as excinfo:
        _client().publish_best_config_sync(
            _SPEC, agent_id="a1", expected_head_generation=3
        )
    assert not isinstance(excinfo.value, CloudBestConfigStaleHeadError)


@patch("requests.post")
def test_publish_without_precondition_sends_no_head_headers(mock_post):
    mock_post.return_value = _resp(201, {"success": True, "data": {"ok": True}})
    _client().publish_best_config_sync(_SPEC)
    headers = mock_post.call_args.kwargs["headers"]
    assert "X-Traigent-Agent-Id" not in headers
    assert "X-Traigent-Expected-Head-Generation" not in headers


def test_publish_rejects_half_a_precondition_before_any_request():
    with patch("requests.post") as mock_post:
        with pytest.raises(CloudServiceError):
            _client().publish_best_config_sync(_SPEC, agent_id="a1")
    mock_post.assert_not_called()


@patch("requests.get")
def test_fetch_agent_head_generation_reads_generation(mock_get):
    mock_get.return_value = _resp(200, {"success": True, "data": {"generation": 0}})
    assert _client().fetch_agent_head_generation_sync("a/1", environment="prod") == 0
    call = mock_get.call_args
    assert call.args[0].endswith("/api/v1/best-configs/agent-heads/a%2F1")
    assert call.kwargs["params"] == {"environment": "prod"}


@patch("requests.get")
def test_fetch_agent_head_generation_rejects_missing_generation(mock_get):
    mock_get.return_value = _resp(200, {"success": True, "data": {}})
    with pytest.raises(CloudServiceError):
        _client().fetch_agent_head_generation_sync("a1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata_agent", "expected"),
    [("agent-9", "agent-9"), (None, None), ("  ", None)],
)
async def test_session_create_response_agent_id_is_read_from_metadata(
    metadata_agent, expected
):
    from traigent.cloud.api_operations import ApiOperations

    metadata = {"experiment_id": "e", "experiment_run_id": "r"}
    if metadata_agent is not None:
        metadata["agent_id"] = metadata_agent
    response = AsyncMock()
    response.json = AsyncMock(return_value={"session_id": "s", "metadata": metadata})

    parsed = await ApiOperations(Mock())._parse_session_response(response)

    assert parsed.agent_id == expected
