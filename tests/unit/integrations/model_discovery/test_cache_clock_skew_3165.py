"""Persisted cache entries cannot outlive the caller's ttl or the clock (Traigent#3165)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from traigent.integrations.model_discovery.cache import CacheEntry, ModelCache


def test_future_timestamp_is_expired() -> None:
    entry = CacheEntry(data=["m"], timestamp=time.time() + 30 * 86400)
    assert entry.is_expired()


def test_small_clock_skew_is_tolerated() -> None:
    entry = CacheEntry(data=["m"], timestamp=time.time() + 1)
    assert not entry.is_expired()


def test_file_ttl_is_capped_at_cache_default(tmp_path: Path) -> None:
    cache = ModelCache(cache_dir=tmp_path, default_ttl=60, enable_file_cache=True)
    cache.set("k", ["m"])
    files = list(tmp_path.rglob("*.json"))
    assert files, "cache file was not written"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    payload["timestamp"] = time.time() - 10 * 86400
    payload["ttl_seconds"] = 10**12
    files[0].write_text(json.dumps(payload), encoding="utf-8")

    fresh = ModelCache(cache_dir=tmp_path, default_ttl=60, enable_file_cache=True)
    assert fresh.get("k") is None
