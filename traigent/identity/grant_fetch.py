"""Fetch the tenant's content-identity purpose-key grant from the Backend.

Release 1, opt-in (``TraigentConfig(content_identity=True)`` or
``TRAIGENT_CONTENT_IDENTITY=1``). The orchestrator calls
:func:`fetch_content_identity_keys` once per run, before session create, and
installs the result for that run only.

**Fail safe.** Any failure -- transport error, 401/403/404/429/503, a body that
is not exactly ``PurposeKeyGrantV1`` -- returns ``None``: the run proceeds with
no content identity (spec: ``key_status`` unavailable means no block is sent),
exactly as a run without a grant. Nothing here raises.

**Content-free logging.** Only the HTTP status, the envelope's error code or an
exception type name is logged; never the grant, a key, or a response body.
"""

from __future__ import annotations

from typing import Any

from traigent.identity.content_identity import ContentIdentityError
from traigent.identity.keys import ContentIdentityKeys
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["fetch_content_identity_keys"]

_UNAVAILABLE = "Content identity unavailable for this run (%s); no example ids, roots or versions will be emitted."


def fetch_content_identity_keys(client: Any) -> ContentIdentityKeys | None:
    """This tenant's purpose keys from ``client``, or ``None`` on any failure.

    ``client`` is the run's :class:`~traigent.cloud.backend_client.BackendIntegratedClient`
    (anything with ``fetch_content_identity_grant_sync()``).
    """
    try:
        grant = client.fetch_content_identity_grant_sync()
    except Exception as exc:  # noqa: BLE001 - a failed fetch never fails a run
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status != 200:
            reason = f"purpose-key grant fetch failed: HTTP {status}"
        elif status == 200:
            reason = "purpose-key grant fetch failed: malformed grant"
        else:
            reason = f"purpose-key grant fetch failed: {type(exc).__name__}"
        logger.warning(_UNAVAILABLE, reason)
        return None
    try:
        keys = ContentIdentityKeys.from_grant(grant)
    except (ContentIdentityError, TypeError, ValueError) as exc:
        # ContentIdentityError messages name fields, never values; still log
        # only the type so no future message can carry key material.
        logger.warning(
            _UNAVAILABLE,
            f"purpose-key grant fetch failed: malformed grant ({type(exc).__name__})",
        )
        return None
    logger.debug("Content identity purpose-key grant received (kid=%s).", keys.kid)
    return keys
