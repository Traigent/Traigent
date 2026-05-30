"""A GuidancePlanProvider backed by the Traigent backend session API.

Owns the GuidancePlan contract end of the wire: it serializes the content-free
``GuidancePlanRequest``, POSTs it to ``/sessions/{id}/guidance-plan``, and parses
the opaque ``GuidancePlan`` response. Transport + auth are injected (a sync
``post_json`` callable, or an async one via :meth:`from_async_post`) so this stays
testable and does not depend on the backend client's internals.

Fail-closed: a missing/garbled response raises rather than fabricating a plan,
honoring the "require the backend plan" invariant.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from .models import GuidancePlan, GuidancePlanRequest, GuidanceResultSubmission

# (path, json_body) -> response json
SyncPostJson = Callable[[str, dict[str, Any]], dict[str, Any]]
AsyncPostJson = Callable[[str, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


class BackendGuidanceError(RuntimeError):
    """Raised when the backend does not return a usable GuidancePlan."""


class BackendGuidanceProvider:
    """Fetches opaque GuidancePlans from the backend session API."""

    def __init__(self, session_id: str, post_json: SyncPostJson) -> None:
        if not session_id:
            raise BackendGuidanceError("session_id is required")
        self._session_id = session_id
        self._post_json = post_json

    @property
    def _plan_path(self) -> str:
        return f"/api/v1/sessions/{self._session_id}/guidance-plan"

    @property
    def _results_path(self) -> str:
        return f"/api/v1/sessions/{self._session_id}/guidance-results"

    def get_guidance_plan(self, request: GuidancePlanRequest) -> GuidancePlan:
        response = self._post_json(self._plan_path, request.to_dict())
        if not isinstance(response, dict) or "plan_id" not in response:
            raise BackendGuidanceError(
                "backend did not return a GuidancePlan; refusing to fabricate guidance"
            )
        try:
            return GuidancePlan.from_dict(response)
        except (KeyError, ValueError, TypeError) as exc:
            raise BackendGuidanceError(
                f"malformed GuidancePlan from backend: {exc}"
            ) from exc

    def submit_guidance_results(self, submission: GuidanceResultSubmission) -> None:
        """Best-effort, content-free report of what was generated (optional)."""
        self._post_json(self._results_path, submission.to_dict())

    @classmethod
    def from_async_post(
        cls, session_id: str, async_post: AsyncPostJson
    ) -> BackendGuidanceProvider:
        """Bridge an async POST (e.g. from the async backend client) to the sync loop.

        Uses a worker thread when an event loop is already running, mirroring the
        ``optimize_sync`` pattern.
        """

        def sync_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
            import asyncio
            import concurrent.futures

            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None

            if running is not None and running.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(asyncio.run, async_post(path, body)).result()
            return asyncio.run(async_post(path, body))

        return cls(session_id, sync_post)


__all__ = ["BackendGuidanceProvider", "BackendGuidanceError"]
