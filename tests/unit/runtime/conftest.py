"""Fixtures for the runtime-requirements tests: a real local HTTP fake backend."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

PROJECT = "proj-1"
AGENT = "agent-1"
PREFIX = f"/api/v1beta/projects/{PROJECT}"
FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"

Handler = Callable[[dict[str, Any] | None], tuple[int, dict[str, Any]]]


class FakeBackend:
    """Routes ``(method, path-after-prefix)`` to handlers and records every request body."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Handler] = {}
        self.requests: list[dict[str, Any]] = []
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                path = self.path.split("?")[0]
                outer.requests.append(
                    {
                        "method": self.command,
                        "path": path,
                        "body": body,
                        "raw": raw.decode(),
                    }
                )
                suffix = path[len(PREFIX) :] if path.startswith(PREFIX) else path
                handler = outer.routes.get((self.command, suffix))
                status, payload = (
                    handler(body) if handler else (404, {"error_code": "not_found"})
                )
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _serve  # noqa: N815

            def log_message(self, *args: Any) -> None:  # silence
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def on(
        self, method: str, suffix: str, status: int, payload: dict[str, Any]
    ) -> None:
        self.routes[(method, suffix)] = lambda _body: (status, payload)

    def bodies(self) -> list[dict[str, Any]]:
        return [r["body"] for r in self.requests if r["body"] is not None]


def head(**overrides: Any) -> dict[str, Any]:
    """A ``RequirementResponse``-shaped row (see the Backend's response schema)."""
    row: dict[str, Any] = {
        "head_id": "head-1",
        "revision_id": "rev-1",
        "revision_number": 1,
        "previous_revision_id": None,
        "status": "active",
        "scope": "agent",
        "project_id": PROJECT,
        "agent_id": AGENT,
        "kind": "latency_violation_rate",
        "latency_threshold_ms": 2000,
        "environment": "production",
        "max_violation_rate": 0.05,
        "confidence": 0.95,
        "window_hours": 6,
        "grace_minutes": 30,
        "min_samples": 200,
        "max_unresolved_fraction": 0.05,
        "closure_windows": 2,
        "closure_max_age_hours": 24,
        "false_alarm_budget": 0.05,
        "horizon_days": 30,
        "provenance": "api",
        "declared_by_user_id": "user-1",
        "created_at": "2026-10-21T00:00:00Z",
    }
    row.update(overrides)
    return row


def breach(breach_id: str = "breach-1", **overrides: Any) -> dict[str, Any]:
    """A ``BreachResponse``-shaped row."""
    row: dict[str, Any] = {
        "breach_id": breach_id,
        "agent_id": AGENT,
        "kind": "latency_violation_rate",
        "latency_threshold_ms": 2000,
        "environment": "production",
        "max_violation_rate": 0.05,
        "confidence": 0.95,
        "window_hours": 6,
        "grace_minutes": 30,
        "min_samples": 200,
        "max_unresolved_fraction": 0.05,
        "closure_windows": 2,
        "closure_max_age_hours": 24,
        "false_alarm_budget": 0.05,
        "horizon_days": 30,
        "opened_at": "2026-10-21T01:00:00Z",
        "opening_evaluation_id": "eval-1",
        "last_breached_evaluation_id": "eval-1",
        "outcome": None,
        "withdrawn_reason": None,
        "closing_evaluation_ids": None,
        "closed_at": None,
    }
    row.update(overrides)
    return row


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeBackend]:
    fake = FakeBackend()
    fake.thread.start()
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", fake.url)
    monkeypatch.setenv("TRAIGENT_API_KEY", "test-only-key")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_RUNTIME_STATE_DIR", str(tmp_path / "state"))
    try:
        yield fake
    finally:
        fake.server.shutdown()
        fake.server.server_close()


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


MODULES_DIR = Path(__file__).parent / "temp_modules"

FIX_MODULE = """
import traigent

@traigent.requirement(
    latency_ms=2000, max_violation_rate=0.05, on_breach="fix",
    max_fix_cost_usd=5.0, agent_id="agent-1",
)
@traigent.optimize(
    eval_dataset="data.jsonl", objectives=["accuracy"],
    configuration_space={"t": [0.1, 0.2]}, injection_mode="seamless",
)
def answer(q: str, t: float = 0.1) -> str:
    return q
"""

FRAGMENT_MODULE = """
import traigent

@traigent.requirement(tvl={fragment!r}{extra})
@traigent.optimize(
    eval_dataset="data.jsonl", objectives=["accuracy"],
    configuration_space={{"t": [0.1, 0.2]}}, injection_mode="seamless",
)
def answer(q: str, t: float = 0.1) -> str:
    return q
"""


@pytest.fixture
def user_module() -> Iterator[Callable[[str], Path]]:
    """Write a module inside the workspace (secure_path requires it); removed afterwards."""
    import uuid

    created: list[Path] = []
    MODULES_DIR.mkdir(exist_ok=True)

    def write(source: str) -> Path:
        path = MODULES_DIR / f"mod_{uuid.uuid4().hex}.py"
        path.write_text(source)
        created.append(path)
        return path

    try:
        yield write
    finally:
        for path in created:
            path.unlink(missing_ok=True)
        if MODULES_DIR.exists() and not any(MODULES_DIR.iterdir()):
            MODULES_DIR.rmdir()
