"""Caching utilities for model discovery.

Provides TTL-based caching with file persistence and manual refresh support.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traigent.utils.secure_path import safe_open, validate_path

logger = logging.getLogger(__name__)

# Default cache directory (user's cache dir or temp)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "traigent" / "models"


@dataclass
class CacheEntry:
    """A single cache entry with TTL support.

    Attributes:
        data: The cached model list.
        timestamp: When the cache was created (Unix timestamp).
        ttl_seconds: Time-to-live in seconds (default 24 hours).
        provider: The provider name for this cache entry.
    """

    data: list[str]
    timestamp: float
    ttl_seconds: int = 86400  # 24 hours default
    provider: str = ""

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return time.time() - self.timestamp > self.ttl_seconds

    def age_seconds(self) -> float:
        """Return the age of this cache entry in seconds."""
        return time.time() - self.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "data": self.data,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            data=data.get("data", []),
            timestamp=data.get("timestamp", 0.0),
            ttl_seconds=data.get("ttl_seconds", 86400),
            provider=data.get("provider", ""),
        )


class ModelCache:
    """Thread-safe cache for model lists with TTL and file persistence.

    This cache supports:
    - In-memory caching with TTL expiration
    - Optional file persistence for cross-session caching
    - Manual refresh/invalidation
    - Thread-safe operations
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        default_ttl: int = 86400,
        enable_file_cache: bool = True,
    ) -> None:
        """Initialize the model cache.

        Args:
            cache_dir: Directory for file-based cache persistence.
            default_ttl: Default TTL in seconds (24 hours).
            enable_file_cache: Whether to persist cache to files.
        """
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._enable_file_cache = enable_file_cache
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR

        if self._enable_file_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, force_refresh: bool = False) -> list[str] | None:
        """Get cached models for a provider.

        Args:
            key: Provider key (e.g., "openai", "anthropic").
            force_refresh: If True, ignore cache and return None.

        Returns:
            List of model names, or None if not cached/expired.
        """
        if force_refresh:
            return None

        with self._lock:
            # Try in-memory cache first
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    logger.debug(
                        f"Cache hit for {key} (age: {entry.age_seconds():.0f}s)"
                    )
                    return entry.data
                else:
                    logger.debug(f"Cache expired for {key}")
                    del self._cache[key]

            # Try file cache
            if self._enable_file_cache:
                file_entry = self._load_from_file(key)
                if file_entry and not file_entry.is_expired():
                    entry = file_entry
                    self._cache[key] = entry
                    logger.debug(
                        f"File cache hit for {key} (age: {entry.age_seconds():.0f}s)"
                    )
                    return entry.data

            return None

    def set(
        self,
        key: str,
        models: list[str],
        ttl: int | None = None,
    ) -> None:
        """Set cached models for a provider.

        Args:
            key: Provider key (e.g., "openai", "anthropic").
            models: List of model names to cache.
            ttl: TTL in seconds (uses default if not specified).
        """
        ttl = ttl or self._default_ttl
        entry = CacheEntry(
            data=models,
            timestamp=time.time(),
            ttl_seconds=ttl,
            provider=key,
        )

        with self._lock:
            self._cache[key] = entry
            if self._enable_file_cache:
                self._save_to_file(key, entry)

        logger.debug(f"Cached {len(models)} models for {key} (TTL: {ttl}s)")

    def invalidate(self, key: str) -> None:
        """Invalidate cache for a provider.

        Args:
            key: Provider key to invalidate.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]

            if self._enable_file_cache:
                cache_file = self._get_cache_file(key)
                # Another worker may remove the same shared cache file between
                # discovery and invalidation; treat that as already invalidated.
                try:
                    cache_file.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass

        logger.debug(f"Invalidated cache for {key}")

    def invalidate_all(self) -> None:
        """Invalidate all cached entries."""
        with self._lock:
            keys = list(self._cache.keys())
            for key in keys:
                self.invalidate(key)

        logger.debug("Invalidated all cache entries")

    def refresh(
        self,
        key: str,
        fetcher: Callable[[], list[str]],
        ttl: int | None = None,
    ) -> list[str]:
        """Refresh cache for a provider using the provided fetcher.

        Args:
            key: Provider key to refresh.
            fetcher: Callable that returns a list of model names.
            ttl: TTL in seconds (uses default if not specified).

        Returns:
            Fresh list of model names.

        Raises:
            Exception: If fetcher fails.
        """
        self.invalidate(key)
        models = fetcher()
        self.set(key, models, ttl)
        return models

    def get_entry(self, key: str) -> CacheEntry | None:
        """Get the raw cache entry for inspection.

        Args:
            key: Provider key.

        Returns:
            CacheEntry or None if not cached.
        """
        with self._lock:
            return self._cache.get(key)

    def _get_cache_file(self, key: str) -> Path:
        """Get the file path for a provider's cache."""
        return self._cache_dir / f"{key}_models.json"

    def _save_to_file(self, key: str, entry: CacheEntry) -> None:
        """Save cache entry to file."""
        try:
            cache_file = self._get_cache_file(key)
            cache_file = validate_path(cache_file, self._cache_dir, must_exist=False)
            with safe_open(
                cache_file, self._cache_dir, mode="w", encoding="utf-8"
            ) as f:
                json.dump(entry.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache to file for {key}: {e}")

    def _load_from_file(self, key: str) -> CacheEntry | None:
        """Load cache entry from file."""
        try:
            cache_file = self._get_cache_file(key)
            if not cache_file.exists():
                return None

            cache_file = validate_path(cache_file, self._cache_dir, must_exist=True)
            with safe_open(
                cache_file, self._cache_dir, mode="r", encoding="utf-8"
            ) as f:
                data = json.load(f)
                return CacheEntry.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache from file for {key}: {e}")
            return None


# Global cache instance (singleton)
_global_cache: ModelCache | None = None
_cache_lock = threading.Lock()


def get_global_cache() -> ModelCache:
    """Get the global model cache instance.

    Returns:
        The singleton ModelCache instance.
    """
    global _global_cache

    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = ModelCache()

    return _global_cache


def reset_global_cache() -> None:
    """Reset the global cache (primarily for testing)."""
    global _global_cache

    with _cache_lock:
        if _global_cache is not None:
            _global_cache.invalidate_all()
        _global_cache = None
