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


class FakeG1Backend:
    """Implements ``G1Transport``; records every request body it receives."""

    def __init__(self, project_ref: str = PROJECT_REF) -> None:
        self.project_ref = project_ref
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []
        self.readiness = "available"
        self.fail_with: tuple[int | None, str | None] | None = None
        self.enroll_ref_override: str | None = None

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
        raise AssertionError(f"unexpected request {method} {path}")
