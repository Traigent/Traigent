"""Tests for :class:`TraigentCloudRemoteGuidanceAdapter`.

These cover the documented public path: the
``InteractiveOptimizer(remote_service=TraigentCloudRemoteGuidanceAdapter(client))``
setup shown in ``docs/user-guide/interactive_optimization.md`` and
``docs/user-guide/choosing_optimization_model.md``.

The adapter has to satisfy the dataclass-based ``RemoteGuidanceService``
protocol that the optimizer calls. ``TraigentCloudClient`` itself exposes
the same backend endpoints but under a different public method signature
(separate keyword args and ``submit_trial_result`` instead of
``submit_result``) — passing the cloud client directly as
``remote_service`` would raise at runtime, which is the bug tracked in
issue #883.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from traigent.cloud import TraigentCloudRemoteGuidanceAdapter
from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import (
    DatasetSubsetIndices,
    NextTrialRequest,
    NextTrialResponse,
    OptimizationFinalizationRequest,
    OptimizationFinalizationResponse,
    OptimizationSessionStatus,
    SessionCreationRequest,
    SessionCreationResponse,
    TrialResultSubmission,
    TrialStatus,
    TrialSuggestion,
)
from traigent.optimizers.interactive_optimizer import (
    InteractiveOptimizer,
    RemoteGuidanceService,
)
from traigent.utils.exceptions import OptimizationError


def _make_cloud_client_mock(
    *,
    session_id: str = "session-abc",
    suggestion_configs: list[dict[str, Any]] | None = None,
) -> AsyncMock:
    """Return a mock ``TraigentCloudClient`` whose method signatures match the real client.

    The mock exposes only the HTTP-facing methods that the adapter is
    expected to call. Using ``spec=TraigentCloudClient`` ensures the test
    fails immediately if the adapter ever calls an attribute the real
    client does not have.
    """
    suggestion_configs = suggestion_configs or [
        {"temperature": 0.3},
        {"temperature": 0.7},
    ]
    client = AsyncMock(spec=TraigentCloudClient)

    client.create_optimization_session.return_value = SessionCreationResponse(
        session_id=session_id,
        status=OptimizationSessionStatus.ACTIVE,
        optimization_strategy={"exploration_ratio": 0.3},
    )

    counter = {"n": 0}

    async def _get_next_trial(
        session_id_arg: str,
        previous_results: list[TrialResultSubmission] | None = None,
    ) -> NextTrialResponse:
        # The cloud client signature is positional session_id + keyword
        # previous_results — assert the adapter respected that shape.
        assert isinstance(session_id_arg, str)
        if counter["n"] >= len(suggestion_configs):
            return NextTrialResponse(
                suggestion=None,
                should_continue=False,
                reason="exhausted",
                session_status=OptimizationSessionStatus.COMPLETED,
            )
        idx = counter["n"]
        counter["n"] += 1
        return NextTrialResponse(
            suggestion=TrialSuggestion(
                trial_id=f"trial-{idx + 1}",
                session_id=session_id_arg,
                trial_number=idx + 1,
                config=suggestion_configs[idx],
                dataset_subset=DatasetSubsetIndices(
                    indices=[0, 1, 2],
                    selection_strategy="diverse_sampling",
                    confidence_level=0.8,
                    estimated_representativeness=0.8,
                ),
                exploration_type="exploration" if idx == 0 else "exploitation",
            ),
            should_continue=True,
            session_status=OptimizationSessionStatus.ACTIVE,
        )

    client.get_next_trial.side_effect = _get_next_trial

    client.submit_trial_result.return_value = None

    client.finalize_optimization.return_value = OptimizationFinalizationResponse(
        session_id=session_id,
        best_config=suggestion_configs[-1],
        best_metrics={"accuracy": 0.91},
        total_trials=len(suggestion_configs),
        successful_trials=len(suggestion_configs),
        total_duration=12.5,
        cost_savings=0.5,
    )

    return client


class TestProtocolConformance:
    """The adapter must implement the protocol the optimizer relies on."""

    def test_adapter_satisfies_remote_guidance_service_protocol(self) -> None:
        client = AsyncMock(spec=TraigentCloudClient)
        adapter = TraigentCloudRemoteGuidanceAdapter(client)

        # Runtime structural check is enough for a Protocol without
        # ``runtime_checkable``: every required coroutine must exist.
        for method_name in (
            "create_session",
            "get_next_trial",
            "submit_result",
            "finalize_session",
        ):
            attr = getattr(adapter, method_name, None)
            assert attr is not None, f"Adapter missing {method_name}"
            assert callable(attr), f"Adapter.{method_name} must be callable"

        # The bare cloud client does *not* satisfy the protocol — this is
        # the bug we're fixing. ``submit_result`` is absent on the real
        # client (it's called ``submit_trial_result``), so wrapping in the
        # adapter is mandatory.
        assert not hasattr(TraigentCloudClient, "submit_result")

    def test_adapter_typed_as_remote_guidance_service(self) -> None:
        client = AsyncMock(spec=TraigentCloudClient)
        adapter: RemoteGuidanceService = TraigentCloudRemoteGuidanceAdapter(client)
        # Sanity: the variable annotation above is the contract — if the
        # adapter ever drops a method, the optimizer setup breaks. We
        # verify the methods are awaitable bound methods.
        assert callable(adapter.create_session)
        assert callable(adapter.get_next_trial)
        assert callable(adapter.submit_result)
        assert callable(adapter.finalize_session)


class TestAdapterDelegation:
    """Each adapter method must forward to the right cloud client method."""

    @pytest.mark.asyncio
    async def test_create_session_forwards_request_object(self) -> None:
        client = _make_cloud_client_mock()
        adapter = TraigentCloudRemoteGuidanceAdapter(client)

        request = SessionCreationRequest(
            function_name="q_and_a",
            configuration_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            dataset_metadata={"size": 100},
            max_trials=10,
        )

        response = await adapter.create_session(request)

        client.create_optimization_session.assert_awaited_once_with(request)
        assert isinstance(response, SessionCreationResponse)
        assert response.session_id == "session-abc"
        assert response.status is OptimizationSessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_next_trial_maps_session_id_and_previous_results(self) -> None:
        client = _make_cloud_client_mock()
        adapter = TraigentCloudRemoteGuidanceAdapter(client)

        previous = [
            TrialResultSubmission(
                session_id="session-abc",
                trial_id="trial-prior",
                metrics={"accuracy": 0.8},
                duration=1.2,
                status=TrialStatus.COMPLETED,
            )
        ]
        request = NextTrialRequest(session_id="session-abc", previous_results=previous)

        response = await adapter.get_next_trial(request)

        client.get_next_trial.assert_awaited_once_with(
            "session-abc", previous_results=previous
        )
        assert isinstance(response, NextTrialResponse)
        assert response.suggestion is not None
        assert response.suggestion.trial_id == "trial-1"

    @pytest.mark.asyncio
    async def test_submit_result_maps_dataclass_to_keyword_arguments(self) -> None:
        client = _make_cloud_client_mock()
        adapter = TraigentCloudRemoteGuidanceAdapter(client)

        result = TrialResultSubmission(
            session_id="session-abc",
            trial_id="trial-7",
            metrics={"accuracy": 0.93, "latency": 0.21},
            duration=2.5,
            status=TrialStatus.COMPLETED,
            outputs_sample=["ok"],
            error_message=None,
            metadata={"note": "n=3"},
        )

        await adapter.submit_result(result)

        # The cloud client exposes the result-submission as
        # ``submit_trial_result`` with status as a string. The adapter
        # must translate both.
        client.submit_trial_result.assert_awaited_once_with(
            session_id="session-abc",
            trial_id="trial-7",
            metrics={"accuracy": 0.93, "latency": 0.21},
            duration=2.5,
            status="completed",
            outputs_sample=["ok"],
            error_message=None,
            metadata={"note": "n=3"},
        )

    @pytest.mark.asyncio
    async def test_submit_result_translates_failed_status(self) -> None:
        client = _make_cloud_client_mock()
        adapter = TraigentCloudRemoteGuidanceAdapter(client)

        result = TrialResultSubmission(
            session_id="session-abc",
            trial_id="trial-fail",
            metrics={},
            duration=0.1,
            status=TrialStatus.FAILED,
            error_message="boom",
        )

        await adapter.submit_result(result)

        kwargs = client.submit_trial_result.await_args.kwargs
        assert kwargs["status"] == "failed"
        assert kwargs["error_message"] == "boom"

    @pytest.mark.asyncio
    async def test_finalize_session_returns_cloud_response(self) -> None:
        client = _make_cloud_client_mock()
        adapter = TraigentCloudRemoteGuidanceAdapter(client)

        request = OptimizationFinalizationRequest(
            session_id="session-abc", include_full_history=True
        )

        response = await adapter.finalize_session(request)

        client.finalize_optimization.assert_awaited_once_with(
            "session-abc", include_full_history=True
        )
        assert isinstance(response, OptimizationFinalizationResponse)
        assert response.session_id == "session-abc"
        assert response.successful_trials == 2

    def test_client_property_exposes_wrapped_instance(self) -> None:
        client = AsyncMock(spec=TraigentCloudClient)
        adapter = TraigentCloudRemoteGuidanceAdapter(client)
        assert adapter.client is client


class TestDocumentedInteractiveOptimizerSetup:
    """End-to-end test of the documented setup path.

    Wires ``InteractiveOptimizer`` with the adapter wrapping a mocked
    ``TraigentCloudClient`` and runs the full workflow shown in
    ``docs/user-guide/interactive_optimization.md``. This is the
    "runnable test that executes the documented InteractiveOptimizer
    setup with the intended remote client or adapter" required by
    issue #883.
    """

    @pytest.mark.asyncio
    async def test_initialize_get_report_finalize_against_cloud_client(self) -> None:
        configs = [
            {"temperature": 0.3, "model": "o4-mini"},
            {"temperature": 0.7, "model": "GPT-4o"},
        ]
        client = _make_cloud_client_mock(suggestion_configs=configs)
        remote_service = TraigentCloudRemoteGuidanceAdapter(client)

        optimizer = InteractiveOptimizer(
            config_space={
                "temperature": (0.0, 1.0),
                "model": ["o4-mini", "GPT-4o"],
            },
            objectives=["accuracy", "latency"],
            remote_service=remote_service,
            dataset_metadata={"size": 100, "type": "qa"},
        )

        session = await optimizer.initialize_session(
            function_name="optimize_qa", max_trials=10
        )

        assert session.session_id == "session-abc"
        # The adapter must hand the optimizer's SessionCreationRequest
        # straight to the cloud client.
        client.create_optimization_session.assert_awaited_once()
        (forwarded_request,), _ = client.create_optimization_session.call_args
        assert isinstance(forwarded_request, SessionCreationRequest)
        assert forwarded_request.function_name == "optimize_qa"
        assert forwarded_request.max_trials == 10

        # Drive the suggestion / report loop.
        observed_trial_ids: list[str] = []
        while True:
            suggestion = await optimizer.get_next_suggestion(dataset_size=100)
            if suggestion is None:
                break
            observed_trial_ids.append(suggestion.trial_id)
            await optimizer.report_results(
                trial_id=suggestion.trial_id,
                metrics={"accuracy": 0.8 + 0.05 * len(observed_trial_ids)},
                duration=0.42,
                status=TrialStatus.COMPLETED,
            )

        assert observed_trial_ids == ["trial-1", "trial-2"]
        assert client.get_next_trial.await_count == 3  # 2 suggestions + 1 stop
        assert client.submit_trial_result.await_count == 2

        # Each submit call should be keyword-based with the string status
        # the cloud client expects on the wire.
        for call in client.submit_trial_result.await_args_list:
            assert call.args == ()
            assert call.kwargs["session_id"] == "session-abc"
            assert call.kwargs["status"] == "completed"

        summary = await optimizer.finalize_optimization(include_full_history=True)
        client.finalize_optimization.assert_awaited_once_with(
            "session-abc", include_full_history=True
        )
        assert summary.best_config == configs[-1]
        assert summary.successful_trials == 2
        assert optimizer.session is not None
        assert optimizer.session.status is OptimizationSessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_raw_cloud_client_without_adapter_raises_clear_error(self) -> None:
        """Drive the documented-broken bare client setup and assert it fails.

        Passing ``TraigentCloudClient`` directly as ``remote_service`` is the
        documented-but-broken setup from issue #883. The cloud client exposes
        ``submit_trial_result`` rather than the protocol's ``submit_result``,
        so the first ``report_results`` call inside ``InteractiveOptimizer``
        explodes. This test wires the bare client into the optimizer and
        asserts the runtime failure surfaces — a regression that accidentally
        lets the bare client "work" without the adapter would be caught.
        """
        client = AsyncMock(spec=TraigentCloudClient)
        # ``spec=TraigentCloudClient`` mirrors the real client's surface, so
        # accessing ``submit_result`` raises ``AttributeError`` — the exact
        # failure users hit at runtime today.
        assert not hasattr(TraigentCloudClient, "submit_result")

        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            remote_service=client,  # type: ignore[arg-type]
            dataset_metadata={"size": 1},
        )
        # Skip initialize_session for this regression pin: the cloud client's
        # legacy ``create_session`` accepts the call (with the wrong return
        # shape), so the cleanest, most diagnosable failure to assert is the
        # missing ``submit_result`` the optimizer hits during reporting.
        optimizer.session_id = "session-abc"

        with pytest.raises(OptimizationError) as exc_info:
            await optimizer.report_results(
                trial_id="trial-1",
                metrics={"accuracy": 0.8},
                duration=1.0,
                status=TrialStatus.COMPLETED,
            )

        assert "Failed to report results" in str(exc_info.value)
        assert "submit_result" in str(exc_info.value)
