"""B4 ROUND 5 regression: ``_get_sync_auth_headers`` must fail closed.

Codex's external review identified the last remaining fail-open path in
the SDK auth chain:

* ``BackendIntegratedClient._get_sync_auth_headers()`` previously caught
  every exception from ``auth.get_headers()`` (including the
  ``AuthenticationError`` raised by the round-3 fail-closed check in
  ``AuthManager.get_auth_headers()``) and returned bare default headers
  containing only ``Content-Type`` and ``User-Agent``.
* ``upload_example_features()`` then proceeded to call ``requests.post``
  with those default headers -- shipping an UNAUTHENTICATED request to
  the backend after the key had already been rejected.

After round 5:

* ``_get_sync_auth_headers`` re-raises ``AuthenticationError`` and
  surfaces other failures as ``CloudServiceError`` instead of silently
  returning default headers.
* ``upload_example_features`` catches both, logs, and returns ``False``
  WITHOUT calling ``requests.post`` -- so no unauthenticated POST ever
  reaches the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.cloud.auth import AuthenticationError, AuthManager
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError

_VALID_LOOKING_KEY = "tg_" + "x" * 61


def _force_backend_reject():
    """Force ``AuthManager._validate_api_key_with_backend`` to fail.

    Mirrors the helper used by ``test_backend_client_fail_closed.py``:
    returning a non-None reason from the validation hook causes the
    auth manager to raise ``InvalidCredentialsError`` (a subclass of
    ``AuthenticationError``) the next time headers are requested.
    """

    async def _fail(self, api_key):  # noqa: ARG001
        return "backend-rejected"

    return patch.object(
        AuthManager,
        "_validate_api_key_with_backend",
        new=_fail,
    )


# ---------------------------------------------------------------------------
# Core regression: no unauthenticated POST after backend rejection.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_example_features_does_not_post_when_backend_rejects():
    """KEY ASSERTION: ``requests.post`` must NOT be called.

    Pre-round-5 behavior: ``_get_sync_auth_headers`` swallowed the auth
    rejection and returned ``{"Content-Type": ..., "User-Agent": ...}``,
    after which ``upload_example_features`` happily POSTed to the
    backend with NO authentication. Round 5 closes that hole.
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)
    fake_post = MagicMock()

    try:
        with _force_backend_reject(), patch(
            "requests.post", new=fake_post
        ):
            result = client.upload_example_features(
                experiment_run_id="run-123",
                feature_kind="task_complexity",
                features=[{"example_id": "ex1", "value": 0.5}],
            )

        # The upload must have short-circuited (False), and crucially the
        # POST must NEVER have been issued.
        assert result is False
        fake_post.assert_not_called()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_sync_auth_headers_raises_authentication_error_on_reject():
    """``_get_sync_auth_headers`` itself must propagate the auth failure.

    Direct guard against re-introducing a fail-open catch in the helper.
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)

    try:
        with _force_backend_reject():
            with pytest.raises(AuthenticationError):
                client._get_sync_auth_headers(target="backend")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_sync_auth_headers_does_not_return_default_on_reject():
    """Defensive: confirm the rejection path never produces bare defaults.

    If a future regression catches ``AuthenticationError`` and returns
    ``{"Content-Type": ..., "User-Agent": ...}`` it would slip past the
    assertions above only if the test stopped raising. Pin the
    behavior explicitly: header generation must NOT silently produce
    a 2-key default dict in the rejection path.
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)

    try:
        with _force_backend_reject():
            captured: dict[str, str] | None = None
            try:
                captured = client._get_sync_auth_headers(target="backend")
            except AuthenticationError:
                captured = None

        # Either we raised (captured is None) or, defensively, we did
        # NOT return a bare defaults-only dict.
        if captured is not None:  # pragma: no cover - defensive guard
            assert set(captured.keys()) != {"Content-Type", "User-Agent"}
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Unexpected failures must also fail closed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_sync_auth_headers_raises_cloud_service_error_on_unexpected():
    """Non-auth failures must surface as ``CloudServiceError``.

    If ``auth.get_headers`` raises a generic ``RuntimeError`` (e.g. a
    transient programming bug), the helper must NOT swallow it into
    default headers. It must raise ``CloudServiceError`` so the caller
    cannot mistake "header generation blew up" for "everything's fine".
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)

    async def _boom(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("simulated header failure")

    try:
        with patch.object(
            client.auth_manager.auth, "get_headers", new=_boom
        ):
            with pytest.raises(CloudServiceError):
                client._get_sync_auth_headers(target="backend")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_sync_auth_headers_raises_when_auth_manager_auth_missing():
    """B4 ROUND 6: auth_manager present but ``.auth`` missing must fail closed.

    Codex found that the previous fix still returned bare ``default_headers``
    when ``auth_manager.auth`` was None or lacked ``get_headers``. That path
    is an internal-bug indicator -- not a license to send an unauthenticated
    request. The helper must now raise ``CloudServiceError``.
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)

    try:
        # Case 1: auth_manager.auth is None
        with patch.object(client.auth_manager, "auth", new=None):
            with pytest.raises(CloudServiceError):
                client._get_sync_auth_headers(target="backend")

        # Case 2: auth_manager.auth lacks ``get_headers``
        broken_auth = object()  # plain object, no ``get_headers`` attr
        with patch.object(client.auth_manager, "auth", new=broken_auth):
            with pytest.raises(CloudServiceError):
                client._get_sync_auth_headers(target="backend")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upload_example_features_skips_post_on_unexpected_failure():
    """``upload_example_features`` must skip the POST on ``CloudServiceError`` too.

    Same fail-closed guarantee, different failure mode.
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)
    fake_post = MagicMock()

    async def _boom(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("simulated header failure")

    try:
        with patch.object(
            client.auth_manager.auth, "get_headers", new=_boom
        ), patch("requests.post", new=fake_post):
            result = client.upload_example_features(
                experiment_run_id="run-456",
                feature_kind="task_complexity",
                features=[{"example_id": "ex1", "value": 0.5}],
            )

        assert result is False
        fake_post.assert_not_called()
    finally:
        await client.close()
