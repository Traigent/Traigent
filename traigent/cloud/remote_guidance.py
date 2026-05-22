"""Adapter exposing :class:`TraigentCloudClient` as a remote guidance service.

:class:`~traigent.optimizers.interactive_optimizer.InteractiveOptimizer`
consumes a service whose methods accept the dataclass requests defined in
:mod:`traigent.cloud.models` (``SessionCreationRequest``,
``NextTrialRequest``, ``TrialResultSubmission``,
``OptimizationFinalizationRequest``) and return their matching response
objects. :class:`~traigent.cloud.client.TraigentCloudClient` already speaks
those endpoints, but its public methods use a different signature shape
(positional / keyword arguments and a ``submit_trial_result`` name).

The :class:`TraigentCloudRemoteGuidanceAdapter` wraps a cloud client so the
documented ``InteractiveOptimizer(remote_service=...)`` setup works without
leaking those signature differences into user code.
"""

# Traceability: CONC-Layer-Integration FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

from typing import TYPE_CHECKING

from traigent.cloud.models import (
    NextTrialRequest,
    NextTrialResponse,
    OptimizationFinalizationRequest,
    OptimizationFinalizationResponse,
    SessionCreationRequest,
    SessionCreationResponse,
    TrialResultSubmission,
)

if TYPE_CHECKING:
    from traigent.cloud.client import TraigentCloudClient


class TraigentCloudRemoteGuidanceAdapter:
    """Adapt :class:`TraigentCloudClient` to the ``RemoteGuidanceService`` protocol.

    Pass an instance of this adapter as the ``remote_service`` argument when
    constructing :class:`InteractiveOptimizer` with a
    :class:`TraigentCloudClient` so the optimizer's dataclass-based protocol is
    correctly bridged to the cloud client's HTTP-facing signature.
    """

    def __init__(self, client: TraigentCloudClient) -> None:
        """Store the wrapped cloud client.

        Args:
            client: An initialized :class:`TraigentCloudClient`. The adapter
                does not take ownership of the client lifecycle; callers
                remain responsible for opening and closing the underlying
                HTTP session (e.g. with ``async with`` on the client).
        """
        self._client = client

    @property
    def client(self) -> TraigentCloudClient:
        """Return the wrapped cloud client."""
        return self._client

    async def create_session(
        self, request: SessionCreationRequest
    ) -> SessionCreationResponse:
        """Create a remote optimization session.

        Delegates to :meth:`TraigentCloudClient.create_optimization_session`,
        which already accepts a :class:`SessionCreationRequest` instance.
        """
        return await self._client.create_optimization_session(request)

    async def get_next_trial(self, request: NextTrialRequest) -> NextTrialResponse:
        """Request the next trial suggestion for ``request.session_id``."""
        return await self._client.get_next_trial(
            request.session_id,
            previous_results=request.previous_results,
        )

    async def submit_result(self, result: TrialResultSubmission) -> None:
        """Submit a completed trial result back to the cloud service.

        Maps the dataclass to the keyword-argument form expected by
        :meth:`TraigentCloudClient.submit_trial_result` and converts the
        ``TrialStatus`` enum to its wire-format string.
        """
        status_value = getattr(result.status, "value", str(result.status))
        await self._client.submit_trial_result(
            session_id=result.session_id,
            trial_id=result.trial_id,
            metrics=result.metrics,
            duration=result.duration,
            status=status_value,
            outputs_sample=result.outputs_sample,
            error_message=result.error_message,
            metadata=result.metadata,
        )

    async def finalize_session(
        self, request: OptimizationFinalizationRequest
    ) -> OptimizationFinalizationResponse:
        """Finalize the session and return the cloud client's response."""
        return await self._client.finalize_optimization(
            request.session_id,
            include_full_history=request.include_full_history,
        )


__all__ = ["TraigentCloudRemoteGuidanceAdapter"]
