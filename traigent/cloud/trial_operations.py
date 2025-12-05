"""Trial management operations for TraiGent Cloud Client.

This module handles trial registration, result submission, and metrics tracking
for optimization experiments.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from traigent.cloud.validators import validate_configuration_run_submission
from traigent.config.backend_config import BackendConfig
from traigent.utils.env_config import is_mock_mode
from traigent.utils.logging import get_logger

# Track whether we've warned about weighted score update failures
_warned_weighted_score_failure: bool = False

# Optional aiohttp dependency handling
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient

logger = get_logger(__name__)


class TrialOperations:
    """Handles trial management operations."""

    def __init__(self, client: "BackendIntegratedClient"):
        """Initialize trial operations handler.

        Args:
            client: Parent BackendIntegratedClient instance
        """
        self.client = client

    def _describe_backend(self) -> str:
        """Return sanitized backend connection context for logging."""

        backend_url = (
            self.client.backend_config.backend_base_url
            or BackendConfig.get_backend_url()
        )
        env = os.getenv("TRAIGENT_ENV", "production")
        api_key_preview = None
        auth = getattr(self.client, "auth_manager", None)
        if auth and hasattr(auth, "auth") and hasattr(auth.auth, "get_api_key_preview"):
            try:
                api_key_preview = auth.auth.get_api_key_preview()
            except Exception:  # pragma: no cover - defensive
                api_key_preview = None

        api_key_display = api_key_preview or "not configured"
        return f"backend_url={backend_url}, env={env}, api_key={api_key_display}"

    @staticmethod
    def _summarize_actor(info: dict[str, Any] | None) -> str:
        """Return a concise description of an authenticated actor."""

        if not info:
            return "unknown principal"

        parts: list[str] = []
        if info.get("owner_user_id"):
            parts.append(f"user '{info['owner_user_id']}'")
        if info.get("owner_api_key_id"):
            parts.append(f"api-key '{info['owner_api_key_id']}'")
        if info.get("owner_api_key_preview"):
            parts.append(f"token {info['owner_api_key_preview']}")
        if info.get("credential_source"):
            parts.append(f"source={info['credential_source']}")

        return ", ".join(parts) if parts else "unknown principal"

    def _log_ownership_forbidden(
        self, session_id: str, action: str, status: int, error_msg: str
    ) -> None:
        """Log consistent guidance for ownership enforcement failures."""

        fingerprint: dict[str, Any] | None = None
        auth = getattr(self.client, "auth_manager", None)
        auth_core = getattr(auth, "auth", None) if auth else None
        get_fingerprint = getattr(auth_core, "get_owner_fingerprint", None)
        if callable(get_fingerprint):
            try:
                fingerprint = get_fingerprint() or {}
            except Exception:  # pragma: no cover - defensive logging only
                fingerprint = {}

        summary = self._summarize_actor(fingerprint)
        excerpt = self._first_error_line(error_msg)

        logger.error(
            "❌ %s for session %s denied: HTTP %s Forbidden. Session ownership enforcement is active. "
            "Calling credentials: %s. Re-authenticate with the session owner or an admin-scoped token.",
            action,
            session_id,
            status,
            summary,
        )
        if excerpt:
            logger.error("   Backend response: %s", excerpt)

    @staticmethod
    def _first_error_line(error_text: str | None) -> str:
        """Return a safe, trimmed first line of an error payload for logging."""

        if not error_text:
            return ""

        lines = error_text.strip().splitlines()
        if not lines:
            return ""

        excerpt = lines[0].strip()
        if len(excerpt) > 200:
            return f"{excerpt[:197]}..."
        return excerpt

    @staticmethod
    def _sanitize_for_json(value: Any) -> Any:
        """Recursively convert values so they can be encoded as JSON."""
        from traigent.utils.numpy_compat import convert_numpy_value, is_numpy_type

        if is_numpy_type(value):
            return convert_numpy_value(value)
        if isinstance(value, dict):
            return {
                str(k): TrialOperations._sanitize_for_json(v) for k, v in value.items()
            }
        if isinstance(value, list):
            return [TrialOperations._sanitize_for_json(v) for v in value]
        if isinstance(value, tuple):
            return [TrialOperations._sanitize_for_json(v) for v in value]
        if isinstance(value, set):
            return [TrialOperations._sanitize_for_json(v) for v in value]

        metrics = getattr(value, "metrics", None)
        execution_time = getattr(value, "execution_time", None)
        if metrics is not None or execution_time is not None:
            data = dict(metrics or {})
            if execution_time is not None and "execution_time" not in data:
                data["execution_time"] = execution_time
            return TrialOperations._sanitize_for_json(data)

        return value

    @staticmethod
    def _redact_sensitive_fields(data: Any, depth: int = 0) -> Any:
        """Recursively redact sensitive fields from data for safe logging.

        Redacts: API keys, tokens, passwords, prompts, responses, and other PII.
        """
        if depth > 10:  # Prevent infinite recursion
            return "[MAX_DEPTH]"

        # Sensitive field patterns (case-insensitive matching)
        sensitive_keys = {
            "api_key",
            "apikey",
            "token",
            "secret",
            "password",
            "credential",
            "prompt",
            "response",
            "input",
            "output",
            "content",
            "message",
            "text",
            "query",
            "answer",
            "system_prompt",
            "user_prompt",
        }

        if isinstance(data, dict):
            redacted = {}
            for key, value in data.items():
                key_lower = key.lower().replace("_", "").replace("-", "")
                is_sensitive = any(
                    s.replace("_", "") in key_lower for s in sensitive_keys
                )
                if is_sensitive and isinstance(value, str) and len(value) > 20:
                    redacted[key] = f"[REDACTED:{len(value)} chars]"
                elif is_sensitive and isinstance(value, (list, dict)) and value:
                    redacted[key] = f"[REDACTED:{type(value).__name__}]"
                else:
                    redacted[key] = TrialOperations._redact_sensitive_fields(
                        value, depth + 1
                    )
            return redacted
        elif isinstance(data, list):
            return [
                TrialOperations._redact_sensitive_fields(item, depth + 1)
                for item in data[:5]
            ]
        return data

    async def register_trial_start(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
    ) -> bool:
        """Register a trial start with the backend.

        This creates a configuration run with "running" status before execution begins.

        Args:
            session_id: Session ID
            trial_id: Trial ID (hash-based)
            config: Configuration to be tested

        Returns:
            True if successful, False otherwise
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping trial registration")
            return False

        try:
            # Map status to backend format (use ACTIVE for running state)
            backend_status = self.client._map_to_backend_status("in_progress")

            # Prepare trial registration data
            registration_data = {
                "trial_id": trial_id,
                "config": config,
                "status": backend_status,  # Use mapped status (ACTIVE)
                "metrics": {},  # Empty metrics at start
            }
            registration_payload = self._sanitize_for_json(registration_data)

            # Create connector without SSL for localhost
            connector = None
            backend_url = self.client.backend_config.backend_base_url
            if backend_url and (
                "localhost" in backend_url or "127.0.0.1" in backend_url
            ):
                connector = aiohttp.TCPConnector(ssl=False)

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": "application/json"}
            )

            async with aiohttp.ClientSession(connector=connector) as session:
                # Use the same endpoint but with "running" status
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/sessions/{session_id}/results"
                async with session.post(
                    url,
                    json=registration_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(
                            f"✅ Registered trial start for session {session_id}, trial {trial_id}"
                        )
                        return True
                    elif response.status == 403:
                        error_msg = await response.text()
                        self._log_ownership_forbidden(
                            session_id,
                            "Registering trial start",
                            response.status,
                            error_msg,
                        )
                        return False
                    else:
                        error_msg = await response.text()
                        logger.warning(
                            f"Failed to register trial start: {response.status} - {error_msg[:200]}"
                        )
                        return False

        except Exception:
            logger.exception(
                "Error registering trial start for session %s trial %s (%s)",
                session_id,
                trial_id,
                self._describe_backend(),
            )
            return False

    def register_trial_start_sync(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
    ) -> bool:
        """Synchronous wrapper for register_trial_start.

        Args:
            session_id: Session ID
            trial_id: Trial ID
            config: Configuration to be tested

        Returns:
            True if successful, False otherwise
        """

        async def _register_async() -> bool:
            return await self.register_trial_start(session_id, trial_id, config)

        try:
            # Check if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, use new one
                loop = None

            if loop:
                # We're in an async context - schedule on existing loop
                # Using run_coroutine_threadsafe prevents deadlocks
                future = asyncio.run_coroutine_threadsafe(_register_async(), loop)
                return future.result(timeout=60)
            else:
                # No async context, run directly
                return asyncio.run(_register_async())
        except Exception:
            logger.exception(
                "Failed to register trial start synchronously for session %s trial %s (%s)",
                session_id,
                trial_id,
                self._describe_backend(),
            )
            return False

    def _extract_measures_from_metrics(
        self, metrics: dict[str, float]
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Extract measures and summary_stats from metrics dict.

        Returns:
            tuple of (measures, summary_stats, clean_metrics)
        """
        measures = None
        summary_stats = None
        clean_metrics: dict[str, Any] = {}

        for key, value in metrics.items():
            if key == "measures":
                measures = value
                logger.debug(
                    f"Found measures in metrics dict: "
                    f"{len(measures) if isinstance(measures, list) else 'Not a list'} items"
                )
            elif key == "summary_stats":
                summary_stats = value
                logger.debug("Found summary_stats in metrics dict")
            else:
                clean_metrics[key] = value

        return measures, summary_stats, clean_metrics

    def _build_trial_result_data(
        self,
        trial_id: str,
        config: dict[str, Any],
        clean_metrics: dict[str, Any],
        backend_status: str,
        mode: str,
    ) -> dict[str, Any]:
        """Build the trial result data dict."""
        result_data: dict[str, Any] = {
            "trial_id": trial_id,
            "config": config,
            "metrics": clean_metrics if clean_metrics else {},
            "status": backend_status,
            "metadata": {
                "mode": mode,
                "sdk_version": "2.0.0",
            },
        }
        return result_data

    def _log_measures_debug(self, trial_id: str, measures: Any) -> None:
        """Log debug information for measures."""
        if measures is not None:
            logger.info(f"📊 MEASURES DATA for trial {trial_id}:")
            logger.info(f"  Type: {type(measures)}")
            logger.info(
                f"  Count: {len(measures) if isinstance(measures, list) else 'N/A'}"
            )
            if isinstance(measures, list) and len(measures) > 0:
                first_measure = measures[0]
                logger.info("  🔍 COST DEBUG - First measure costs:")
                logger.info(
                    f"    - input_cost: {first_measure.get('input_cost', 'MISSING')}"
                )
                logger.info(
                    f"    - output_cost: {first_measure.get('output_cost', 'MISSING')}"
                )
                logger.info(
                    f"    - total_cost: {first_measure.get('total_cost', 'MISSING')}"
                )
                measure_json = json.dumps(measures[0], indent=2)[:500]
                logger.debug(f"  Full first measure: {measure_json}")
        else:
            logger.warning(f"⚠️ No measures found for trial {trial_id}")

    def _log_summary_stats_debug(self, trial_id: str, summary_stats: Any) -> None:
        """Log debug information for summary_stats."""
        if summary_stats is not None:
            logger.info(f"📈 SUMMARY_STATS DATA for trial {trial_id}:")
            logger.info(f"  Type: {type(summary_stats)}")
            logger.info(
                f"  Keys: {list(summary_stats.keys()) if isinstance(summary_stats, dict) else 'N/A'}"
            )
            if isinstance(summary_stats, dict):
                stats_metrics = summary_stats.get("metrics", {})
                if isinstance(stats_metrics, dict):
                    logger.info("  🔍 COST DEBUG - Summary stats costs:")
                    for cost_key in ["cost", "input_cost", "output_cost", "total_cost"]:
                        if cost_key in stats_metrics:
                            logger.info(f"    - {cost_key}: {stats_metrics[cost_key]}")
                        else:
                            logger.info(f"    - {cost_key}: MISSING")
        else:
            logger.warning(f"⚠️ No summary_stats found for trial {trial_id}")

    def _create_localhost_connector(self) -> Any:
        """Create connector without SSL for localhost connections."""
        backend_url = self.client.backend_config.backend_base_url
        if backend_url and ("localhost" in backend_url or "127.0.0.1" in backend_url):
            return aiohttp.TCPConnector(ssl=False)
        return None

    async def _handle_trial_success_response(
        self,
        response: Any,
        session_id: str,
        trial_id: str,
        backend_status: str,
        result_data: dict[str, Any],
        clean_metrics: dict[str, Any],
    ) -> bool:
        """Handle successful trial submission response."""
        response_data = await response.json()
        continue_optimization = response_data.get("continue_optimization", True)

        if not continue_optimization:
            logger.info(
                f"✅ Submitted trial result for session {session_id}, trial {trial_id} "
                f"(session auto-finalized by backend)"
            )
        else:
            logger.info(
                f"✅ Submitted trial result for session {session_id}, trial {trial_id}"
            )

        config_run_updated = await self.client._update_config_run_status(
            trial_id, backend_status
        )
        if config_run_updated:
            execution_time = result_data.get("execution_time") or result_data.get(
                "duration"
            )
            if "measures" not in result_data:
                await self.client._update_config_run_measures(
                    trial_id, clean_metrics, execution_time
                )

        return True

    def _handle_trial_error_response(
        self,
        status: int,
        error_msg: str,
        trial_id: str,
        session_id: str,
        url: str,
    ) -> None:
        """Handle error response from trial submission."""
        logger.error(f"❌ Failed to submit trial result: HTTP {status}")
        logger.error(f"   Error message: {error_msg}")
        logger.error(f"   Trial ID: {trial_id}")
        logger.error(f"   Session ID: {session_id}")
        logger.error(f"   URL: {url}")

        try:
            error_json = json.loads(error_msg)
            logger.error(f"   Parsed error: {json.dumps(error_json, indent=2)}")
        except json.JSONDecodeError:
            logger.debug("Received non-JSON error response from backend")

    async def submit_trial_result_via_session(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        status: str,
        error_message: str | None = None,
        execution_mode: str | None = None,
    ) -> bool:
        """Submit trial results via the TraiGent session endpoint.

        Args:
            session_id: Session ID
            trial_id: Trial ID (hash-based)
            config: Configuration that was tested
            metrics: Trial metrics
            status: Trial status
            error_message: Optional error message
            execution_mode: Optional execution mode

        Returns:
            True if successful, False otherwise
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping result submission")
            return False

        try:
            backend_status = self.client._map_to_backend_status(status)
            mode = self.client._normalize_execution_mode(execution_mode)

            # Extract measures and summary_stats from metrics
            measures, summary_stats, clean_metrics = (
                self._extract_measures_from_metrics(metrics)
            )

            # Build result data
            result_data = self._build_trial_result_data(
                trial_id, config, clean_metrics, backend_status, mode
            )

            # Add measures and summary_stats at the top level if present
            if measures is not None:
                result_data["measures"] = measures
            self._log_measures_debug(trial_id, measures)

            if summary_stats is not None:
                result_data["summary_stats"] = summary_stats
            self._log_summary_stats_debug(trial_id, summary_stats)

            # Validate the submission data
            try:
                validate_configuration_run_submission(result_data)
            except ValueError as e:
                logger.error(f"Invalid configuration run submission: {e}")

            if error_message:
                result_data["error"] = self.client._sanitize_error_message(
                    error_message
                )

            connector = self._create_localhost_connector()

            logger.debug(f"📤 Submission data for trial {trial_id}:")
            logger.debug(f"   Has measures: {'measures' in result_data}")
            logger.debug(f"   Has summary_stats: {'summary_stats' in result_data}")
            logger.debug(f"   Top-level keys: {list(result_data.keys())}")

            if logger.isEnabledFor(10):  # DEBUG level
                redacted_data = self._redact_sensitive_fields(result_data)
                payload_json = json.dumps(redacted_data, indent=2, default=str)
                logger.debug(
                    f"🔍 Redacted payload (first 1000 chars): {payload_json[:1000]}"
                )

            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": "application/json"}
            )

            async with aiohttp.ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/sessions/{session_id}/results"

                async with session.post(
                    url,
                    json=self._sanitize_for_json(result_data),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status in [200, 201]:
                        return await self._handle_trial_success_response(
                            response,
                            session_id,
                            trial_id,
                            backend_status,
                            result_data,
                            clean_metrics,
                        )
                    elif response.status == 403:
                        error_msg = await response.text()
                        self._log_ownership_forbidden(
                            session_id,
                            "Submitting trial results",
                            response.status,
                            error_msg,
                        )
                        return False
                    else:
                        error_msg = await response.text()
                        self._handle_trial_error_response(
                            response.status, error_msg, trial_id, session_id, url
                        )
                        return False

        except Exception:
            logger.exception(
                "Error submitting trial result for session %s trial %s (%s)",
                session_id,
                trial_id,
                self._describe_backend(),
            )
            return False

    async def submit_summary_stats(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        summary_stats: dict[str, Any],
        status: str = "completed",
    ) -> bool:
        """Submit summary statistics for privacy-preserving mode.

        Args:
            session_id: Session ID
            trial_id: Trial ID
            config: Configuration that was tested
            summary_stats: pandas.describe()-compatible summary statistics
            status: Trial status (completed/failed)

        Returns:
            True if submission successful, False otherwise
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping summary stats submission")
            return False

        try:
            # Map status to backend format
            backend_status = self.client._map_to_backend_status(status)

            # Prepare metadata - REQUIRED by backend to avoid NoneType errors
            submission_metadata = {
                "mode": "privacy",  # Explicitly set mode for summary stats
                "sdk_version": "2.0.0",
                "aggregation_method": "pandas.describe",
                "execution_mode": "privacy",  # Use "privacy" not "edge_analytics"
                "total_examples": summary_stats.get("total_examples", 0),
                # Include summary_stats metadata
                **summary_stats.get("metadata", {}),
            }

            # IMPORTANT: Backend always expects metrics field
            # For privacy mode, send summary stats as metrics
            submission_data = {
                "trial_id": trial_id,
                "config": config,
                "metrics": summary_stats.get(
                    "metrics", {}
                ),  # Send summary stats metrics here
                "metadata": submission_metadata,  # REQUIRED - backend expects this
                "status": backend_status,
                # Also include summary_stats for backend to detect privacy mode
                "summary_stats": summary_stats,
            }

            # Validate the submission data
            try:
                validate_configuration_run_submission(submission_data)
                logger.debug("✅ Summary stats submission validated successfully")
            except ValueError as e:
                logger.error(f"Invalid summary stats submission: {e}")
                # Still try to send but log the validation error

            # Debug logging to verify structure
            logger.debug(
                f"Submitting summary_stats with structure: {json.dumps(submission_data, indent=2)[:1000]}"
            )

            # Create connector without SSL for localhost
            connector = None
            backend_url = self.client.backend_config.backend_base_url
            if backend_url and (
                "localhost" in backend_url or "127.0.0.1" in backend_url
            ):
                connector = aiohttp.TCPConnector(ssl=False)

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": "application/json"}
            )

            async with aiohttp.ClientSession(connector=connector) as session:
                # Submit to session endpoint with summary_stats
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/sessions/{session_id}/results"
                async with session.post(
                    url,
                    json=self._sanitize_for_json(submission_data),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(
                            f"✅ Submitted summary stats for session {session_id}, trial {trial_id}"
                        )
                        return True
                    elif response.status == 403:
                        error_msg = await response.text()
                        self._log_ownership_forbidden(
                            session_id,
                            "Submitting summary stats",
                            response.status,
                            error_msg,
                        )
                        return False
                    else:
                        error_msg = await response.text()
                        logger.error(
                            f"Failed to submit summary stats: {response.status} - {error_msg[:500]}"
                        )
                        # Log the full submission data for debugging
                        logger.debug(
                            f"Submission data was: {json.dumps(submission_data, indent=2)[:1000]}"
                        )
                        return False

        except Exception:
            logger.exception(
                "Error submitting summary stats for trial %s (%s)",
                trial_id,
                self._describe_backend(),
            )
            return False

    async def update_trial_weighted_scores(
        self,
        trial_id: str,
        weighted_score: float,
        normalization_info: dict[str, Any] | None = None,
        objective_weights: dict[str, float] | None = None,
    ) -> bool:
        """Update configuration run with weighted multi-objective scores.

        This method updates the summary_stats of a configuration run with
        weighted scores calculated post-experiment for multi-objective optimization.

        Args:
            trial_id: Trial ID (will be resolved to configuration_run_id)
            weighted_score: The calculated weighted score
            normalization_info: Information about normalization ranges used
            objective_weights: The weights used for each objective

        Returns:
            bool: True if update successful, False otherwise
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available")
            return False

        try:
            # Prepare summary stats update data
            summary_stats_data = {
                "weighted_score": weighted_score,
                "multi_objective_analysis": {
                    "weighted_score": weighted_score,
                    "normalization_ranges": normalization_info or {},
                    "objective_weights": objective_weights or {},
                    "analysis_timestamp": datetime.now(UTC).isoformat(),
                },
            }

            # Create connector without SSL for localhost
            connector = None
            backend_url = self.client.backend_config.backend_base_url
            if backend_url and (
                "localhost" in backend_url or "127.0.0.1" in backend_url
            ):
                connector = aiohttp.TCPConnector(ssl=False)

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": "application/json"}
            )

            async with aiohttp.ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/configuration-runs/{trial_id}/summary-stats"

                # Prepare request data
                request_data = {"summary_stats": summary_stats_data}

                try:
                    async with session.put(
                        url,
                        json=request_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        if response.status == 200:
                            logger.debug(
                                f"Successfully updated weighted scores for trial {trial_id}"
                            )
                            return True
                        elif response.status == 400:
                            error_msg = await response.text()
                            # Check if it's a "Configuration run not found" error
                            if "Configuration run not found" in error_msg:
                                logger.debug(
                                    f"Configuration run not found for trial {trial_id}. "
                                    "This is expected in standard mode with mock execution. "
                                    "Skipping weighted score update."
                                )
                                # Return True to prevent the error from propagating
                                # The weighted scores won't be updated but the optimization can continue
                                return True
                            else:
                                logger.error(
                                    f"Failed to update weighted scores: {response.status} - {error_msg[:200]}"
                                )
                                return False
                        else:
                            error_msg = await response.text()
                            logger.error(
                                f"Failed to update weighted scores: {response.status} - {error_msg[:100]}"
                            )
                            return False
                except Exception as exc:
                    # Handle connection failures to local or misconfigured backends gracefully
                    if AIOHTTP_AVAILABLE and isinstance(
                        exc, aiohttp.ClientConnectorError
                    ):
                        global _warned_weighted_score_failure
                        # Only warn once and skip in mock mode
                        if not _warned_weighted_score_failure and not is_mock_mode():
                            logger.warning(
                                "Cannot connect to backend while updating weighted score for %s (%s). Skipping.",
                                trial_id,
                                self._describe_backend(),
                            )
                            _warned_weighted_score_failure = True
                        else:
                            logger.debug(
                                "Cannot connect to backend for weighted score update (trial %s)",
                                trial_id,
                            )
                        return False
                    raise

        except Exception:
            logger.exception(
                "Error updating trial weighted scores for %s (%s)",
                trial_id,
                self._describe_backend(),
            )
            return False

    def _generate_trial_id(
        self,
        session_id: str,
        config: dict[str, Any],
    ) -> str:
        """Generate a deterministic trial ID based on session and configuration.

        Args:
            session_id: Session identifier
            config: Trial configuration

        Returns:
            Unique trial ID
        """
        # Create a deterministic hash from session and config
        config_str = json.dumps(config, sort_keys=True)
        hash_input = f"{session_id}:{config_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
