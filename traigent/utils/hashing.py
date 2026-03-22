"""Hash generation utilities for deterministic ID and label creation."""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-ANALYTICS FUNC-STORAGE REQ-ANLY-011 REQ-STOR-007 SYNC-StorageLogging

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from traigent.utils.numpy_compat import convert_numpy_value, is_numpy_type


def _stable_sort_key(value: Any) -> str:
    """Generate a deterministic sort key for sanitized values."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    except TypeError:
        return repr(value)


def _sanitize_json_value(value: Any) -> Any:
    """Convert numpy and other non-serializable types to JSON-friendly values."""

    if is_numpy_type(value):
        return convert_numpy_value(value)
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        sanitized_values = [_sanitize_json_value(v) for v in value]
        return sorted(sanitized_values, key=_stable_sort_key)
    return value


def generate_trial_hash(
    session_id: str, config: dict[str, Any], dataset_name: str = ""
) -> str:
    """Generate deterministic trial ID from configuration.

    Creates a unique, reproducible trial ID based on the session,
    configuration parameters, and optionally the dataset name.
    This ensures that the same configuration in the same session
    always produces the same trial ID, preventing duplicates.

    Args:
        session_id: The optimization session ID
        config: Configuration dictionary with parameter values
        dataset_name: Optional dataset identifier for additional uniqueness

    Returns:
        A deterministic trial ID like "trial_a1b2c3d4e5f6"
    """
    # Sort config keys for consistent hashing
    # This ensures {"a": 1, "b": 2} and {"b": 2, "a": 1} produce same hash
    sanitized_config = _sanitize_json_value(config)
    sorted_config = json.dumps(sanitized_config, sort_keys=True)

    # Create hash input combining all identifying information
    hash_input = f"{session_id}:{sorted_config}:{dataset_name}"

    # Generate SHA256 hash and take first 12 characters for readability
    # 12 hex chars = 48 bits = ~281 trillion unique values (enough for our use)
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    # Create trial_id with readable prefix
    return f"trial_{hash_value}"


def generate_config_hash(config: dict[str, Any]) -> str:
    """Generate a hash from just the configuration parameters.

    This is useful for comparing configurations across different sessions.

    Args:
        config: Configuration dictionary with parameter values

    Returns:
        A hex hash string of the configuration
    """
    sanitized_config = _sanitize_json_value(config)
    sorted_config = json.dumps(sanitized_config, sort_keys=True)
    return hashlib.sha256(sorted_config.encode()).hexdigest()[:16]


def generate_experiment_hash(
    function_name: str,
    configuration_space: dict[str, Any],
    objectives: list[str],
    dataset_characteristics: dict[str, Any] | None = None,
) -> str:
    """Generate deterministic experiment ID from optimization setup.

    Creates a deterministic experiment ID based on the function name,
    configuration space, objectives, and optional dataset characteristics.
    The same optimization setup will produce the same experiment ID.

    Args:
        function_name: Name of the function being optimized
        configuration_space: The search space for optimization parameters
        objectives: List of optimization objectives (e.g., ["accuracy", "cost"])
        dataset_characteristics: Optional dict with dataset properties like
                                size, type, or hash for additional uniqueness

    Returns:
        A deterministic experiment ID like "exp_a1b2c3d4e5f6g7h8"
    """
    # Prepare components for hashing
    components = {
        "function_name": function_name,
        "configuration_space": configuration_space,
        "objectives": sorted(objectives),  # Sort for consistency
    }

    # Add dataset characteristics if provided
    if dataset_characteristics:
        components["dataset"] = dataset_characteristics

    # Create deterministic JSON string with sorted keys
    hash_input = json.dumps(_sanitize_json_value(components), sort_keys=True)

    # Generate SHA256 hash and take first 16 characters for readability
    # 16 hex chars = 64 bits = sufficient uniqueness for experiments
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # Return with readable prefix
    return f"exp_{hash_value}"


def generate_benchmark_hash(
    function_name: str, dataset_info: dict[str, Any] | None = None
) -> str:
    """Generate deterministic benchmark ID from function and dataset info.

    Creates a unique, reproducible benchmark ID based on the function name
    and dataset characteristics. This ensures consistent benchmark IDs
    across runs with the same dataset.

    Args:
        function_name: Name of the function being benchmarked
        dataset_info: Optional dict with dataset properties like
                     example count, types, or content hash

    Returns:
        A deterministic benchmark ID like "bench_a1b2c3d4e5f6"
    """
    # Prepare components for hashing
    components: dict[str, Any] = {"function_name": function_name, "type": "benchmark"}

    # Add dataset info if provided
    if dataset_info:
        components["dataset_info"] = dataset_info

    # Create deterministic JSON string
    hash_input = json.dumps(_sanitize_json_value(components), sort_keys=True)

    # Generate SHA256 hash and take first 12 characters
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    # Return with readable prefix
    return f"bench_{hash_value}"


_RUN_LABEL_MAX_NAME_LEN = 40
_RUN_LABEL_SANITIZE_RE = re.compile(r"[^a-z0-9_]")


def generate_run_label(
    function_name: str,
    optimization_id: str,
    timestamp: datetime,
) -> str:
    """Generate a human-readable label for an optimization run.

    Combines a sanitized function name, UTC timestamp, and a short hash
    of the optimization ID into a compact, unique-enough label for
    human identification (not a primary key).

    Args:
        function_name: Name of the optimized function
        optimization_id: UUID of the optimization run
        timestamp: When the optimization completed (should be UTC)

    Returns:
        Label like ``answer_question_20260315_143022_a3f1b2``
    """
    # Sanitize: lowercase, replace non-alnum/underscore, collapse runs of _
    slug = _RUN_LABEL_SANITIZE_RE.sub("_", function_name.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    slug = slug[:_RUN_LABEL_MAX_NAME_LEN].rstrip("_") or "run"

    ts = timestamp.strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.sha256(optimization_id.encode()).hexdigest()[:6]

    return f"{slug}_{ts}_{short_hash}"
