"""Bind a client evidence root to a certificate build session and co-sign it.

Order (one invocation, SPEC v3 1.7): readiness is checked by the caller before
spend; then issue-session -> link-run -> bind-g1 -> prepare -> finalize. If a
session expires part-way, rerun :func:`bind_g1`; it starts from issue-session.

Only closed catalog fields cross to the Backend: the nonce, the run id, the
manifest root, the commitment scheme and attestor version, the derived client
key reference, the signature algorithm, the declared mode, the SDK identity and
the final co-attestation. Slot documents, blinds, snapshot digests and locators
never leave :class:`~traigent.certification.client_evidence.ClientEvidence`.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import secrets
from dataclasses import dataclass
from typing import Any, Final

from traigent_schema.certification import (
    ClientCoAttestationContext,
    RelyingPartyVerificationError,
    prepare_client_co_attestation,
)

from traigent.certification.client_evidence import COMMITMENT_SCHEME, ClientEvidence
from traigent.certification.client_keys import ALGORITHM, ClientKey, ClientKeyError
from traigent.certification.pins import Pins
from traigent.certification.transport import G1RequestError, G1Transport

__all__ = [
    "ATTESTOR_VERSION",
    "BindError",
    "BindResult",
    "SESSIONS_PATH",
    "abandon_session",
    "bind_g1",
    "check_g1_readiness",
    "resolve_sdk_identity",
]

SESSIONS_PATH: Final = "/api/v1beta/certificate-build-sessions"
READINESS_VERSION: Final = 1
ATTESTOR_VERSION: Final = "0.1.0"
DEFAULT_SESSION_TTL_SECONDS: Final = 3600
_GIT_SHA_RE = re.compile(r"[a-f0-9]{40}")
_SEMVER_PREFIX_RE = re.compile(r"([0-9]{1,4})\.([0-9]{1,4})\.([0-9]{1,4})")
_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}")


class BindError(Exception):
    """Closed failure. ``code`` is a fixed token; ``session_ref`` is the server-issued
    ``bsn:`` reference if one was already issued (so it can be abandoned)."""

    CODES: Final = frozenset(
        {
            "READINESS_UNAVAILABLE",
            "SDK_IDENTITY_UNAVAILABLE",
            "KEY_NOT_PINNED",
            "SESSION_REFUSED",
            "LINK_REFUSED",
            "BIND_REFUSED",
            "BIND_RESPONSE_INVALID",
            "PREPARE_REFUSED",
            "CO_CONTEXT",
            "FINALIZE_REFUSED",
            "ABANDON_REFUSED",
        }
    )

    def __init__(self, code: str, session_ref: str | None = None) -> None:
        if code not in self.CODES:
            raise ValueError("unknown bind error code")
        self.code = code
        self.session_ref = session_ref
        super().__init__(code)


@dataclass(frozen=True)
class BindResult:
    build_session_ref: str
    certificate_ref: str | None
    nonce: str


def check_g1_readiness(transport: G1Transport) -> None:
    """Require the Backend readiness probe to say G1 issuance is available.

    An absent route, an unreadable or unexpected body, or ``unavailable`` all
    refuse closed with ``READINESS_UNAVAILABLE``.
    """
    try:
        response = transport.request("GET", f"{SESSIONS_PATH}/readiness")
    except G1RequestError:
        raise BindError("READINESS_UNAVAILABLE") from None
    body = response.body
    if (
        response.status != 200
        or not isinstance(body, dict)
        or body.get("readiness_version") != READINESS_VERSION
        or body.get("g1_issuance") != "available"
    ):
        raise BindError("READINESS_UNAVAILABLE")


def resolve_sdk_identity(sdk_ref: str | None = None) -> tuple[str, str]:
    """``(40-hex git sha, x.y.z)`` of this SDK build.

    The git sha is taken from ``sdk_ref``, then ``TRAIGENT_SDK_REF``, then the
    VCS commit recorded by pip for a git install; otherwise the identity is
    unavailable and the caller refuses closed.
    """
    candidate = sdk_ref or os.environ.get("TRAIGENT_SDK_REF")
    if candidate is None:
        try:
            direct_url = importlib.metadata.distribution("traigent").read_text(
                "direct_url.json"
            )
            candidate = (
                json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")
            )
        except (importlib.metadata.PackageNotFoundError, ValueError, AttributeError):
            candidate = None
    if type(candidate) is not str or not _GIT_SHA_RE.fullmatch(candidate):
        raise BindError("SDK_IDENTITY_UNAVAILABLE")
    from traigent._version import get_version

    match = _SEMVER_PREFIX_RE.match(get_version())
    if match is None:
        raise BindError("SDK_IDENTITY_UNAVAILABLE")
    return candidate, ".".join(match.groups())


def _post(
    transport: G1Transport,
    path: str,
    body: dict[str, Any] | None,
    code: str,
    session_ref: str | None,
    statuses: tuple[int, ...] = (200, 201),
) -> Any:
    try:
        response = transport.request("POST", path, body)
    except G1RequestError:
        raise BindError(code, session_ref) from None
    if response.status not in statuses or not isinstance(response.body, dict):
        raise BindError(code, session_ref)
    return response.body


def bind_g1(
    transport: G1Transport,
    evidence: ClientEvidence,
    key: ClientKey,
    pins: Pins,
    *,
    run_id: str,
    sdk_ref: str,
    sdk_version: str,
    declared_mode: str = "current_online",
    ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
) -> BindResult:
    """Run issue-session -> link-run -> bind -> prepare -> co-sign -> finalize.

    ``evidence`` must already have passed local reconstruction
    (:func:`~traigent.certification.client_evidence.reconstruct_and_compare`) or
    have been built in this process and durably saved. The Backend's prepare
    response is checked against locally held pins before anything is signed.
    """
    try:
        key_ref = key.pinned_ref(pins.project_ref)
    except ClientKeyError:
        raise BindError("KEY_NOT_PINNED") from None
    nonce = secrets.token_hex(16)
    issued = _post(
        transport,
        SESSIONS_PATH,
        {"nonce": nonce, "ttl_seconds": ttl_seconds},
        "SESSION_REFUSED",
        None,
    )
    session_ref = issued.get("build_session_ref")
    if type(session_ref) is not str:
        raise BindError("SESSION_REFUSED")
    base = f"{SESSIONS_PATH}/{session_ref}"
    _post(
        transport, f"{base}/link-run", {"run_id": run_id}, "LINK_REFUSED", session_ref
    )
    bound = _post(
        transport,
        f"{SESSIONS_PATH}/bind-g1",
        {
            "build_session_ref": session_ref,
            "manifest_root_digest": evidence.root,
            "commitment_scheme": COMMITMENT_SCHEME,
            "client_attestor_version": ATTESTOR_VERSION,
            "client_key_ref": key_ref,
            "client_signature_algorithm": ALGORITHM,
            "declared_mode": declared_mode,
            "sdk_ref": sdk_ref,
            "sdk_version": sdk_version,
        },
        "BIND_REFUSED",
        session_ref,
    )
    commitment_digest = bound.get("session_commitment_digest")
    if (
        type(commitment_digest) is not str
        or not _DIGEST_RE.fullmatch(commitment_digest)
        or bound.get("manifest_root_digest") != evidence.root
    ):
        raise BindError("BIND_RESPONSE_INVALID", session_ref)
    projection = _post(
        transport,
        f"{SESSIONS_PATH}/prepare",
        {"build_session_ref": session_ref, "nonce": nonce},
        "PREPARE_REFUSED",
        session_ref,
    )
    try:
        context = ClientCoAttestationContext(
            expected_project_ref=pins.project_ref,
            expected_build_session_ref=session_ref,
            expected_session_commitment_digest=commitment_digest,
            expected_nonce=nonce,
            expected_privacy_mode=declared_mode,
            expected_sdk_ref=sdk_ref,
            expected_sdk_version=sdk_version,
            expected_disclosure_profile_id=pins.disclosure_profile_id,
            expected_issuer_key_ref=pins.issuer_key_ref,
            expected_trust_ring_ref=pins.trust_ring_ref,
            expected_issuer_algorithm=pins.issuer_algorithm,
            expected_compiler_register_versions=pins.compiler_register_versions,
            expected_g1_verifier_id=pins.g1_verifier_id,
            expected_g1_verifier_version=pins.g1_verifier_version,
            manifest_root_digest=evidence.root,
            commitment_scheme=COMMITMENT_SCHEME,
            client_attestor_version=ATTESTOR_VERSION,
            expected_client_algorithm=ALGORITHM,
            client_public_key=key.public_key,
            issuer_public_key=pins.issuer_public_key,
        )
        prepared = prepare_client_co_attestation(projection, context=context)
    except RelyingPartyVerificationError:
        raise BindError("CO_CONTEXT", session_ref) from None
    finalized = _post(
        transport,
        f"{SESSIONS_PATH}/finalize",
        {
            "build_session_ref": session_ref,
            "nonce": nonce,
            "co_attestation": {
                "algorithm": ALGORITHM,
                "client_key_ref": key_ref,
                "signed_manifest_digest": prepared.signed_manifest_digest,
                "nonce": nonce,
                "signature": key.sign(prepared.signing_bytes),
            },
        },
        "FINALIZE_REFUSED",
        session_ref,
    )
    certificate_ref = finalized.get("certificate_ref")
    return BindResult(
        session_ref,
        certificate_ref if type(certificate_ref) is str else None,
        nonce,
    )


_RUN_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,100}")


def abandon_run(transport: G1Transport, run_id: str) -> None:
    """Abandon a run's G1 intent before any build session exists (no body).

    Backend-private route. Failure is the closed ``ABANDON_REFUSED``; the run id
    is validated locally so it can never alter the request path.
    """
    if _RUN_ID_RE.fullmatch(run_id) is None or run_id in {".", ".."}:
        raise BindError("ABANDON_REFUSED")
    _post(
        transport,
        f"{SESSIONS_PATH}/runs/{run_id}/abandon",
        None,
        "ABANDON_REFUSED",
        None,
    )


def abandon_session(transport: G1Transport, build_session_ref: str) -> None:
    """Abandon a build session (no body). Failure is the closed ``ABANDON_REFUSED``."""
    _post(
        transport,
        f"{SESSIONS_PATH}/{build_session_ref}/abandon",
        None,
        "ABANDON_REFUSED",
        build_session_ref,
    )
