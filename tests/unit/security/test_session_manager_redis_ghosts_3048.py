"""Redis-mode SessionManager: rate-limit fallback and expired-session ghosts (#3048)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from traigent.security.session_manager import SessionManager


class _FakePipeline:
    def __init__(self, fail: bool) -> None:
        self._fail = fail

    def __getattr__(self, name: str):
        return lambda *a, **k: None

    def execute(self) -> list[Any]:
        raise RuntimeError("pipeline down")


class _FakeRedis:
    def __init__(self, fail_pipeline: bool = False) -> None:
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.fail_pipeline = fail_pipeline

    def ping(self) -> bool:
        return True

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.kv[key] = value

    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def delete(self, key: str) -> None:
        self.kv.pop(key, None)

    def sadd(self, key: str, member: str) -> None:
        self.sets.setdefault(key, set()).add(member)

    def srem(self, key: str, member: str) -> None:
        self.sets.get(key, set()).discard(member)

    def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    def expire(self, key: str, ttl: int) -> None:
        return None

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self.fail_pipeline)


def _manager(fake: _FakeRedis, **kw: Any) -> SessionManager:
    with (
        patch("traigent.security.session_manager.REDIS_AVAILABLE", True),
        patch("traigent.security.session_manager.redis") as mod,
    ):
        mod.from_url.return_value = fake
        return SessionManager(redis_url="redis://x/0", **kw)


def test_create_session_survives_redis_rate_limit_failure() -> None:
    mgr = _manager(_FakeRedis(fail_pipeline=True))
    session_id, token = mgr.create_session("user-1")
    assert session_id and token


def test_expired_session_ids_do_not_count_against_the_cap() -> None:
    fake = _FakeRedis()
    mgr = _manager(fake, max_sessions_per_user=3, enable_rate_limiting=False)
    for _ in range(3):
        mgr.create_session("user-1")
    for key in [k for k in fake.kv if k.startswith("session:")]:
        del fake.kv[key]  # TTL expiry: keys vanish, ids stay in the set

    for _ in range(5):
        mgr.create_session("user-1")

    live = [k for k in fake.kv if k.startswith("session:")]
    assert len(live) <= 3
    assert len(fake.sets["user_sessions:user-1"]) <= 3
    assert all(json.loads(fake.kv[k])["user_id"] == "user-1" for k in live)
