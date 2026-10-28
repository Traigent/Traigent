"""Recording fake Backend for the G1 endpoints (shapes taken from the BE-3b branch)."""

from __future__ import annotations

import base64
import copy
from collections.abc import Mapping
from typing import Any

from cryptography.hazmat.primitives import serialization
from traigent_schema.certification import derive_client_key_ref

from traigent.certification.transport import G1RequestError, G1Response

PROJECT_REF = "prj_" + "ab" * 16
SESSIONS = "/api/v1beta/certificate-build-sessions"


class FakeG1Backend:
    """Implements ``G1Transport``; records every request body it receives."""

    def __init__(self, project_ref: str = PROJECT_REF) -> None:
        self.project_ref = project_ref
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []
        self.readiness = "available"
        self.fail_with: tuple[int | None, str | None] | None = None
        self.enroll_ref_override: str | None = None
        self.fixture: dict[str, Any] | None = None
        self.refuse_step: str | None = None
        self.bind_root_override: str | None = None
        self.finalized: dict[str, Any] | None = None

    def request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> G1Response:
        self.requests.append(
            (method, path, copy.deepcopy(dict(body)) if body is not None else None)
        )
        if self.fail_with is not None:
            raise G1RequestError(*self.fail_with)
        if method == "POST" and path == "/api/v1beta/certificate-client-keys":
            assert body is not None and set(body) == {"algorithm", "public_key_der_b64"}
            key = serialization.load_der_public_key(
                base64.b64decode(body["public_key_der_b64"])
            )
            ref = self.enroll_ref_override or derive_client_key_ref(
                self.project_ref, body["algorithm"], key
            )
            return G1Response(
                201,
                {
                    "id": "ck_1",
                    "tenant_id": "t",
                    "project_id": "p",
                    "client_key_ref": ref,
                    "algorithm": body["algorithm"],
                    "created_at": "2026-10-28T00:00:00Z",
                },
            )
        if method == "GET" and path == f"{SESSIONS}/readiness":
            if self.readiness == "absent":
                raise G1RequestError(404, "not_found")
            return G1Response(
                200, {"readiness_version": 1, "g1_issuance": self.readiness}
            )
        if method == "POST" and path.startswith(f"{SESSIONS}/runs/"):
            # Pre-bind abandonment by run (Backend-private route): no body.
            assert path.endswith("/abandon") and body is None
            if self.refuse_step == "abandon-run":
                raise G1RequestError(404, "not_found")
            return G1Response(200, {"state": "abandoned"})
        if self.fixture is not None:
            return self._g1_flow(method, path, body)
        raise AssertionError(f"unexpected request {method} {path}")

    # Allowed body keys per step, as in the Backend routes of feature/cert-g1-be3b-issuance.
    _BODY_KEYS = {
        "issue": {"nonce", "ttl_seconds"},
        "link-run": {"run_id"},
        "bind-g1": {
            "build_session_ref",
            "manifest_root_digest",
            "commitment_scheme",
            "client_attestor_version",
            "client_key_ref",
            "client_signature_algorithm",
            "declared_mode",
            "sdk_ref",
            "sdk_version",
        },
        "prepare": {"build_session_ref", "nonce"},
        "finalize": {"build_session_ref", "nonce", "co_attestation"},
    }

    def _g1_flow(
        self, method: str, path: str, body: Mapping[str, Any] | None
    ) -> G1Response:
        assert self.fixture is not None and method == "POST"
        session = self.fixture["session"]
        ref = session["build_session_ref"]
        if path == SESSIONS:
            step = "issue"
        elif path == f"{SESSIONS}/{ref}/link-run":
            step = "link-run"
        elif path == f"{SESSIONS}/{ref}/abandon":
            step = "abandon"
        elif path in {
            f"{SESSIONS}/bind-g1",
            f"{SESSIONS}/prepare",
            f"{SESSIONS}/finalize",
        }:
            step = path.rsplit("/", 1)[1]
        else:
            raise AssertionError(f"unexpected request {method} {path}")
        if step == "abandon":
            assert body is None
        else:
            assert body is not None and set(body) == self._BODY_KEYS[step], step
        if self.refuse_step == step:
            raise G1RequestError(400, "invalid_request")
        if step == "issue":
            return G1Response(201, {"build_session_ref": ref, "state": "open"})
        if step == "link-run":
            return G1Response(200, {"linked": True})
        if step == "abandon":
            return G1Response(200, {"state": "abandoned"})
        if step == "bind-g1":
            assert body is not None
            return G1Response(
                200,
                {
                    "binding_id": "bnd_1",
                    "build_session_ref": ref,
                    "session_commitment_digest": session["session_commitment_digest"],
                    "manifest_root_digest": self.bind_root_override
                    or body["manifest_root_digest"],
                    "has_g1_binding": True,
                },
            )
        if step == "prepare":
            return G1Response(200, copy.deepcopy(self.fixture["prepare_response"]))
        assert body is not None
        self.finalized = copy.deepcopy(dict(body))
        return G1Response(201, {"certificate_ref": "cert:" + "c" * 43})
