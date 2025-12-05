"""Cache policy handling for configuration deduplication.

Provides a handler for applying cache policies to filter previously-evaluated
configurations during optimization runs.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

import string
from typing import Any

from traigent.config.types import TraigentConfig
from traigent.optimizers.base import BaseOptimizer
from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.function_identity import sanitize_identifier
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class CachePolicyHandler:
    """Handles cache policy application for configuration deduplication.

    Responsibilities:
    1. Filters configurations based on cache policy
    2. Tracks deduplicated configuration counts
    3. Manages storage locking for thread-safe deduplication
    4. Provides cache policy statistics

    Supported Policies:
    - allow_repeats: No filtering, all configurations allowed
    - prefer_new: Filter out previously-seen configurations
    - reuse_cached: Reuse cached results (deferred for v1)
    """

    def __init__(
        self,
        traigent_config: TraigentConfig,
        optimizer: BaseOptimizer,
    ) -> None:
        """Initialize cache policy handler.

        Args:
            traigent_config: Global configuration with storage path
            optimizer: Optimizer instance (for config_space keys)
        """
        self._traigent_config = traigent_config
        self._optimizer = optimizer
        self._storage = LocalStorageManager(traigent_config.get_local_storage_path())

        # Statistics tracking
        self._configs_deduplicated = 0
        self._cache_policy_used: str | None = None

    def apply_policy(
        self,
        configs: list[dict[str, Any]],
        cache_policy: str,
        function_name: str,
        dataset_name: str,
    ) -> list[dict[str, Any]]:
        """Apply cache policy to filter configurations.

        Args:
            configs: List of candidate configurations
            cache_policy: Policy for handling cached configs ('prefer_new', 'reuse_cached', 'allow_repeats')
            function_name: Name of the function being optimized
            dataset_name: Name of the dataset being used

        Returns:
            Filtered list of configurations based on cache policy
        """
        if cache_policy == "allow_repeats":
            self._cache_policy_used = cache_policy
            return configs

        # Get config space keys for consistent hashing
        config_keys = self._get_config_keys()

        filtered_configs = []
        deduplicated_count = 0

        # Use locking for thread-safe deduplication
        safe_function = self._lock_segment(function_name)
        safe_dataset = self._lock_segment(dataset_name)
        lock_name = f"dedup_{safe_function}_{safe_dataset}"
        with self._storage.acquire_lock(lock_name, timeout=5.0):
            for config in configs:
                if not self._storage.is_config_seen(
                    function_name, dataset_name, config, config_keys
                ):
                    filtered_configs.append(config)
                else:
                    deduplicated_count += 1
                # Note: reuse_cached is deferred for v1

        if deduplicated_count > 0:
            self._configs_deduplicated += deduplicated_count
            logger.info("Deduplicated %d previously seen configs", deduplicated_count)

        if not filtered_configs and cache_policy == "prefer_new":
            logger.info(
                "No new configurations to explore - all have been evaluated in previous runs. "
                "Consider using cache_policy='allow_repeats' to re-evaluate configurations."
            )

        # Track cache policy used
        self._cache_policy_used = cache_policy

        return filtered_configs

    def _get_config_keys(self) -> list[str] | None:
        """Get config space keys for consistent hashing.

        Returns:
            List of config space keys, or None if not available
        """
        if hasattr(self._optimizer, "config_space"):
            return list(self._optimizer.config_space.keys())
        return None

    @property
    def configs_deduplicated(self) -> int:
        """Get count of deduplicated configurations.

        Returns:
            Number of configurations filtered out by cache policy
        """
        return self._configs_deduplicated

    @property
    def cache_policy_used(self) -> str | None:
        """Get the cache policy that was used.

        Returns:
            Cache policy name, or None if not yet applied
        """
        return self._cache_policy_used

    def reset_stats(self) -> None:
        """Reset statistics for new optimization run."""
        self._configs_deduplicated = 0
        self._cache_policy_used = None

    @staticmethod
    def _lock_segment(value: str) -> str:
        """Return a readable, filesystem-safe segment without hash suffixes."""

        sanitized: str = sanitize_identifier(value)
        if "_" not in sanitized:
            return sanitized

        base, suffix = sanitized.rsplit("_", 1)
        if len(suffix) == 8 and all(ch in string.hexdigits for ch in suffix):
            return base or sanitized
        return sanitized
