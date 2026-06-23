"""Trial management operations for Traigent Cloud Client.

This module handles trial registration, result submission, and metrics tracking
for optimization experiments.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import asyncio
import concurrent.futures
import hashlib
import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from traigent._version import get_version
from traigent.cloud.client import raise_if_cloud_egress_disabled
from traigent.cloud.dtos import MeasuresDict
from traigent.cloud.validators import validate_configuration_run_submission
from traigent.config.backend_config import BackendConfig
from traigent.utils.env_config import is_backend_offline, resolve_environment_label
from traigent.utils.logging import get_logger

# HTTP Content-Type header constant
_JSON_CONTENT_TYPE_HEADER = {"Content-Type": "application/json"}

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
        self._auth_error_logged: bool = False

    def _describe_backend(self) -> str:
        """Return sanitized backend connection context for logging."""

        backend_url = (
            self.client.backend_config.backend_base_url
            or BackendConfig.get_backend_url()
        )
        env = resolve_environment_label(default="production")
        return f"backend_url={backend_url}, env={env}"

    def _raise_if_backend_egress_disabled(self, operation: str) -> None:
        """Fail closed before any backend HTTP request."""

        raise_if_cloud_egress_disabled(
            operation,
            no_egress=getattr(self.client, "no_egress", False),
        )

    @staticmethod
    def _summarize_actor(info: dict[str, Any] | None) -> str:
        """Return a concise description of an authenticated actor."""

        if not info:
            return "unknown principal"

        parts: list[str] = []
        if info.get("owner_user_id"):
            parts.append(f"user '{info['owner_user_id']}'")
        if info.get("owner_api_key_id") or info.get("owner_api_key_preview"):
            parts.append("api-key configured")
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
        logger.error(
            "❌ %s for session %s denied: HTTP %s Forbidden. Session ownership enforcement is active. "
            "Calling credentials: %s. Re-authenticate with the session owner or an admin-scoped token.",
            action,
            session_id,
            status,
            summary,
        )

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
                if is_sensitive:
                    if isinstance(value, str):
                        redacted[key] = f"[REDACTED:{len(value)} chars]"
                    elif isinstance(value, (list, dict, tuple, set)):
                        redacted[key] = f"[REDACTED:{type(value).__name__}]"
                    else:
                        redacted[key] = "[REDACTED]"
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
    ) -> bool | None:
        """Register a trial start with the backend.

        This creates a configuration run with "running" status before execution begins.

        Args:
            session_id: Session ID
            trial_id: Trial ID (hash-based)
            config: Configuration to be tested

        Returns:
            True if registration succeeded, False if it failed,
            None if the operation was skipped (e.g. offline mode).
        """
        # Skip backend calls in offline mode — return None so callers
        # can distinguish "skipped" from "succeeded" (Rule 2: no fake completion).
        if is_backend_offline():
            logger.debug("Offline mode: skipping trial registration for %s", trial_id)
            return None
        self._raise_if_backend_egress_disabled("register trial start")

        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping trial registration")
            return False

        try:
            # Map status to backend format. This is a configuration-run
            # submission (POST /sessions/{id}/results), so use the config-run
            # wire vocab — a starting trial is RUNNING, not the session-lifecycle
            # ACTIVE the backend would 400-reject (issue #1302).
            backend_status = self.client._map_to_backend_status(
                "in_progress", endpoint="config_run"
            )

            # Prepare trial registration data
            registration_data = {
                "trial_id": trial_id,
                "config": config,
                "status": backend_status,  # Use mapped status (RUNNING)
                "metrics": {},  # Empty metrics at start
            }
            registration_payload = self._sanitize_for_json(registration_data)

            connector = None

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                _JSON_CONTENT_TYPE_HEADER
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
                        logger.warning(
                            "Failed to register trial start: HTTP %s",
                            response.status,
                        )
                        return False

        except Exception as exc:
            if is_backend_offline():
                logger.debug(
                    "Offline mode: trial registration encountered %s; skipping",
                    exc,
                )
                return None
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
    ) -> bool | None:
        """Synchronous wrapper for register_trial_start.

        Args:
            session_id: Session ID
            trial_id: Trial ID
            config: Configuration to be tested

        Returns:
            True if registration succeeded, False if it failed,
            None if the operation was skipped (e.g. offline mode).
        """

        async def _register_async() -> bool | None:
            return await self.register_trial_start(session_id, trial_id, config)

        try:
            # Check if there's a running event loop
            try:
                asyncio.get_running_loop()  # Raises RuntimeError if no loop running
                # Loop is running (e.g., in Jupyter, async frameworks).
                # CRITICAL: run_coroutine_threadsafe().result() DEADLOCKS if called
                # from the same thread as the running loop — the loop cannot process
                # the scheduled coroutine while blocked on .result().
                #
                # Solution: execute in a separate thread with its own event loop.

                def _run_in_new_loop() -> bool | None:
                    """Run the async function in a fresh event loop."""
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(_register_async())
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_in_new_loop)
                    return future.result(timeout=60)

            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                return asyncio.run(_register_async())
        except Exception:
            logger.exception(
                "Failed to register trial start synchronously for session %s trial %s (%s)",
                session_id,
                trial_id,
                self._describe_backend(),
            )
            return False

    async def request_trial_slot(
        self,
        session_id: str,
    ) -> str | None:
        """Acquire a BACKEND-minted trial id (configuration_run) for a session.

        The backend mints configuration_run ids only through its
        ``POST /sessions/{id}/next-trial`` endpoint; a result POST is bound to
        the session by the trial id the backend itself minted, so the SDK MUST
        obtain that id before submitting (a client-hashed id 404s with
        "Trial ... not found in session"). The optimizer still drives the
        config locally — this call is used purely to allocate the trial slot,
        and the backend's own suggested config is irrelevant to the SDK.

        Returns:
            The backend-minted trial id, or ``None`` when no slot could be
            acquired (offline mode, transport unavailable, non-2xx, or the
            backend declined to suggest a further trial). ``None`` NEVER means
            "use a client id" — callers fail closed on it (Rule 2: no fake
            trial acknowledgment).
        """
        # Skip backend calls in offline mode — None lets callers distinguish
        # "skipped" from "acquired" (Rule 2: no fake completion).
        if is_backend_offline():
            logger.debug(
                "Offline mode: skipping trial-slot request for session %s",
                session_id,
            )
            return None
        self._raise_if_backend_egress_disabled("get next trial")

        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping trial-slot request")
            return None

        try:
            headers = await self.client.auth_manager.augment_headers(
                _JSON_CONTENT_TYPE_HEADER
            )
            connector = self._create_localhost_connector()
            async with aiohttp.ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/sessions/{session_id}/next-trial"
                request_body = {"session_id": session_id, "previous_results": []}
                async with session.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status in [200, 201]:
                        data = await response.json()
                        suggestion = (
                            data.get("suggestion") if isinstance(data, dict) else None
                        )
                        trial_id = (
                            suggestion.get("trial_id")
                            if isinstance(suggestion, dict)
                            else None
                        )
                        if isinstance(trial_id, str) and trial_id:
                            logger.debug(
                                "Acquired backend trial slot %s for session %s",
                                trial_id,
                                session_id,
                            )
                            return trial_id
                        logger.info(
                            "Backend next-trial returned no slot for session %s "
                            "(should_continue=%s)",
                            session_id,
                            (
                                data.get("should_continue")
                                if isinstance(data, dict)
                                else None
                            ),
                        )
                        return None
                    if response.status == 403:
                        error_msg = await response.text()
                        self._log_ownership_forbidden(
                            session_id,
                            "Requesting trial slot",
                            response.status,
                            error_msg,
                        )
                        return None
                    logger.warning(
                        "Failed to acquire trial slot for session %s: HTTP %s",
                        session_id,
                        response.status,
                    )
                    return None
        except Exception as exc:
            if is_backend_offline():
                logger.debug(
                    "Offline mode: trial-slot request encountered %s; skipping",
                    exc,
                )
                return None
            logger.exception(
                "Error requesting trial slot for session %s (%s)",
                session_id,
                self._describe_backend(),
            )
            return None

    def _extract_measures_from_metrics(
        self, metrics: dict[str, Any]
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
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the trial result data dict."""
        result_metadata = dict(metadata or {})
        result_metadata.pop("execution_mode", None)
        result_metadata["mode"] = mode
        result_metadata["sdk_version"] = get_version()
        result_data: dict[str, Any] = {
            "trial_id": trial_id,
            "config": config,
            "metrics": clean_metrics if clean_metrics else {},
            "status": backend_status,
            "metadata": result_metadata,
        }
        return result_data

    def _log_measures_debug(self, trial_id: str, measures: Any) -> None:
        """Log debug information for measures."""
        if measures is not None:
            logger.info(f"📊 MEASURES DATA for trial {trial_id}:")
            logger.info(f"  Type: {type(measures)}")
            logger.debug(
                f"  Count: {len(measures) if isinstance(measures, list) else 'N/A'}"
            )
            if isinstance(measures, list) and len(measures) > 0:
                first_measure = measures[0]
                logger.debug("  COST DEBUG - First measure costs:")
                logger.debug(
                    f"    - input_cost: {first_measure.get('input_cost', 'MISSING')}"
                )
                logger.debug(
                    f"    - output_cost: {first_measure.get('output_cost', 'MISSING')}"
                )
                logger.debug(
                    f"    - total_cost: {first_measure.get('total_cost', 'MISSING')}"
                )
                measure_json = json.dumps(measures[0], indent=2)[:500]
                logger.debug(f"  Full first measure: {measure_json}")
        else:
            # measures are optional metadata; don't surface as a warning to users.
            logger.debug("No measures provided for trial %s (optional)", trial_id)

    def _log_summary_stats_debug(self, trial_id: str, summary_stats: Any) -> None:
        """Log debug information for summary_stats."""
        if summary_stats is not None:
            logger.debug(f"SUMMARY_STATS DATA for trial {trial_id}:")
            logger.debug(f"  Type: {type(summary_stats)}")
            logger.debug(
                f"  Keys: {list(summary_stats.keys()) if isinstance(summary_stats, dict) else 'N/A'}"
            )
            if isinstance(summary_stats, dict):
                stats_metrics = summary_stats.get("metrics", {})
                if isinstance(stats_metrics, dict):
                    logger.debug("  COST DEBUG - Summary stats costs:")
                    for cost_key in ["cost", "input_cost", "output_cost", "total_cost"]:
                        if cost_key in stats_metrics:
                            logger.debug(f"    - {cost_key}: {stats_metrics[cost_key]}")
                        else:
                            logger.debug(f"    - {cost_key}: MISSING")
        else:
            # summary_stats is optional metadata; don't surface as a warning to users.
            logger.debug("No summary_stats provided for trial %s (optional)", trial_id)

    def _create_localhost_connector(self) -> Any:
        """Create connector for localhost connections."""
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

    @staticmethod
    def _is_session_not_found_400(status: int, error_text: str) -> bool:
        """Return True only for the specific transient per-worker session-not-found case.

        A 400 is treated as transient only when:
        - status is exactly 400 (not 404, not 5xx), AND
        - the response body signals the *session* (not a trial, config-run, or
          unrelated object) was not found.

        Matches both plain-text patterns and structured JSON error codes so the
        check stays valid if the backend changes its message wording.
        """
        if status != 400:
            return False

        body_lower = error_text.lower()

        # Structured check: parse JSON and look for an error code that unambiguously
        # names a missing session.
        try:
            parsed = json.loads(error_text)
            if isinstance(parsed, dict):
                code = str(parsed.get("code") or parsed.get("error_code") or "").lower()
                if code in {"session_not_found", "session-not-found"}:
                    return True
        except (ValueError, KeyError):
            pass

        # Plain-text heuristic: "session" must be the *subject* of the not-found
        # message, not merely appear somewhere in the text.  Patterns checked:
        #   "session <id> not found"
        #   "session does not exist"
        #   "unknown session"
        # Excluded intentionally: "Trial ... not found in session" or
        # "Configuration run not found" which contain "session" as a prepositional
        # object — these are different error classes and must not be swallowed.
        session_subject_patterns = (
            r"\bsession\b[^.]*\bnot found\b",
            r"\bsession\b[^.]*\bdoes not exist\b",
            r"\bunknown session\b",
        )
        return any(re.search(pat, body_lower) for pat in session_subject_patterns)

    def _handle_trial_error_response(
        self,
        status: int,
        trial_id: str,
        session_id: str,
        url: str,
        error_text: str = "",
    ) -> bool:
        """Handle error response from trial submission.

        Logs the HTTP status and response body so callers can diagnose backend
        rejections without inspecting raw network traffic.  When the backend
        returns 400 and the body indicates that the *session* was not found, the
        error is downgraded to INFO with a user-friendly recovery note (see BE
        #1194 — per-worker session storage race).

        Returns:
            True if the error is a transient session-not-found 400 that the
            caller should treat as a skipped upload rather than a hard failure.
            False for all other errors (caller should degrade backend tracking).
        """
        # Extract a concise user-facing message from the response body when
        # the backend returned JSON with an "error" or "message" field.
        detail: str = error_text.strip()
        if detail:
            try:
                parsed = json.loads(detail)
                if isinstance(parsed, dict):
                    detail = parsed.get("error") or parsed.get("message") or detail
            except (ValueError, KeyError):
                pass

        if self._is_session_not_found_400(status, error_text):
            # Log the backend detail so the raw body is still visible even
            # though this is a benign transient condition.
            logger.info(
                "Session %s not found on backend — trial %s could not be "
                "uploaded (likely transient per-worker storage; run "
                "`traigent sync` after optimization to upload offline). "
                "Backend detail: %s",
                session_id,
                trial_id,
                detail or error_text or "(no body)",
            )
            return True

        # A non-transient 4xx is a PERMANENT rejection -- commonly an invalid or
        # uncomputable optimization objective/metric name (e.g. "cost_usd" instead
        # of "cost"). Label it explicitly so it is not mistaken for a transient
        # backend outage; the run continues but is tracked LOCALLY ONLY.
        if isinstance(status, int) and 400 <= status < 500:
            logger.error(
                "\u274c Trial submission REJECTED by the backend: HTTP %s \u2014 %s. "
                "This is a PERMANENT error (commonly an invalid optimization "
                "objective/metric name, e.g. 'cost_usd' instead of 'cost'), NOT a "
                "transient outage. The run will be tracked LOCALLY ONLY "
                "(source='local_fallback'); fix the request to track it on the "
                "backend.  Trial %s  Session %s  URL %s",
                status,
                detail or "(no response body)",
                trial_id,
                session_id,
                url,
            )
            return False
        logger.warning(
            "\u274c Failed to submit trial result: HTTP %s \u2014 %s",
            status,
            detail or "(no response body)",
        )
        logger.warning(
            "   Trial ID: %s  Session ID: %s  URL: %s", trial_id, session_id, url
        )
        return False

    async def submit_trial_result_via_session(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        metrics: dict[str, Any],
        status: str,
        error_message: str | None = None,
        execution_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool | None:
        """Submit trial results via the Traigent session endpoint.

        Args:
            session_id: Session ID
            trial_id: Trial ID (hash-based)
            config: Configuration that was tested
            metrics: Trial metrics
            status: Trial status
            error_message: Optional error message
            execution_mode: Optional execution mode
            metadata: Optional additional metadata to merge into the result payload

        Returns:
            True if submission succeeded, False if it failed,
            None if the operation was skipped (e.g. offline mode).
        """
        # Skip backend calls in offline mode — return None so callers
        # can distinguish "skipped" from "succeeded" (Rule 2: no fake completion).
        if is_backend_offline():
            logger.debug(
                "Offline mode: skipping trial result submission for %s",
                trial_id,
            )
            return None
        self._raise_if_backend_egress_disabled("submit trial result")

        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping result submission")
            return False

        try:
            # Configuration-run submission path -> config-run wire vocab
            # (preserves PRUNED for early-stopped trials; issue #1302).
            backend_status = self.client._map_to_backend_status(
                status, endpoint="config_run"
            )
            mode = self.client._normalize_execution_mode(execution_mode)

            # Extract transport-only fields before validating trial metrics.
            measures, summary_stats, clean_metrics = (
                self._extract_measures_from_metrics(metrics)
            )

            # Validate key naming and cardinality for numeric metrics while
            # preserving backward compatibility with existing payload shapes.
            validated_metrics: dict[str, Any] = clean_metrics
            try:
                validated_metrics = dict(MeasuresDict(clean_metrics))
            except (TypeError, ValueError) as validation_error:
                logger.warning(
                    "Metrics validation warning for trial %s: %s. "
                    "Submitting unvalidated numeric metrics for backward compatibility.",
                    trial_id,
                    validation_error,
                )

            # Build result data
            result_data = self._build_trial_result_data(
                trial_id,
                config,
                validated_metrics,
                backend_status,
                mode,
                metadata=metadata,
            )

            # Add measures and summary_stats at the top level if present
            if measures is not None:
                result_data["measures"] = measures
            self._log_measures_debug(trial_id, measures)

            if summary_stats is not None:
                result_data["summary_stats"] = summary_stats
            self._log_summary_stats_debug(trial_id, summary_stats)

            # Validate the submission data. Schema validation is authoritative;
            # invalid payloads must not be posted to the backend.
            try:
                validate_configuration_run_submission(result_data)
            except ValueError as e:
                logger.error(f"Invalid configuration run submission: {e}")
                return False

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
                _JSON_CONTENT_TYPE_HEADER
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
                        error_text = await response.text()
                        is_transient = self._handle_trial_error_response(
                            response.status, trial_id, session_id, url, error_text
                        )
                        # Return None for transient session-not-found (400 +
                        # "not found") so _log_trial_to_backend treats this as
                        # a skipped upload
                        # rather than a hard backend failure and does not flag
                        # the backend as degraded (BE #1194 per-worker storage).
                        return None if is_transient else False

        except Exception as exc:
            if is_backend_offline():
                logger.debug(
                    "Offline mode: trial result submission encountered %s; skipping",
                    exc,
                )
                return None
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
    ) -> bool | None:
        """Submit summary statistics for privacy-preserving mode.

        Args:
            session_id: Session ID
            trial_id: Trial ID
            config: Configuration that was tested
            summary_stats: pandas.describe()-compatible summary statistics
            status: Trial status (completed/failed)

        Returns:
            True if submission succeeded, False if it failed,
            None if the operation was skipped (e.g. offline mode).
        """
        # Skip backend calls in offline mode — return None so callers
        # can distinguish "skipped" from "succeeded" (Rule 2: no fake completion).
        if is_backend_offline():
            logger.debug(
                "Offline mode: skipping summary stats submission for %s",
                trial_id,
            )
            return None
        self._raise_if_backend_egress_disabled("submit summary stats")

        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping summary stats submission")
            return False

        try:
            # Configuration-run submission path -> config-run wire vocab
            # (issue #1302).
            backend_status = self.client._map_to_backend_status(
                status, endpoint="config_run"
            )

            # Prepare metadata - REQUIRED by backend to avoid NoneType errors
            submission_metadata = {
                **summary_stats.get("metadata", {}),
                "mode": "privacy",  # Explicitly set mode for summary stats
                "sdk_version": get_version(),
                "aggregation_method": "pandas.describe",
                "total_examples": summary_stats.get("total_examples", 0),
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

            # Validate the submission data. Schema validation is authoritative;
            # invalid payloads must not be posted to the backend.
            try:
                validate_configuration_run_submission(submission_data)
                logger.debug("✅ Summary stats submission validated successfully")
            except ValueError as e:
                logger.error(f"Invalid summary stats submission: {e}")
                return False

            logger.debug(
                "Submitting summary_stats with keys: %s",
                sorted(submission_data.keys()),
            )

            connector = self._create_localhost_connector()

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                _JSON_CONTENT_TYPE_HEADER
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
                        logger.error(
                            "Failed to submit summary stats: HTTP %s",
                            response.status,
                        )
                        return False

        except Exception as exc:
            if is_backend_offline():
                logger.debug(
                    "Offline mode: summary stats submission encountered %s; skipping",
                    exc,
                )
                return None
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
    ) -> bool | None:
        """Update configuration run with weighted multi-objective scores.

        This method updates the summary_stats of a configuration run with
        weighted scores calculated post-experiment for multi-objective optimization.

        Args:
            trial_id: Trial ID (will be resolved to configuration_run_id)
            weighted_score: The calculated weighted score
            normalization_info: Information about normalization ranges used
            objective_weights: The weights used for each objective

        Returns:
            True if update succeeded, False if it failed,
            None if the operation was skipped (e.g. offline mode).
        """
        # Skip backend calls in offline mode — return None so callers
        # can distinguish "skipped" from "succeeded" (Rule 2: no fake completion).
        if is_backend_offline():
            logger.debug(
                "Offline mode: skipping weighted score update for trial %s", trial_id
            )
            return None
        self._raise_if_backend_egress_disabled("update trial weighted scores")

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

            connector = None

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                _JSON_CONTENT_TYPE_HEADER
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
                                    "Skipping weighted score update without counting it as applied."
                                )
                                return False
                            else:
                                # Auth failures: instance-scoped dedup at DEBUG
                                if response.status in (401, 403):
                                    if not self._auth_error_logged:
                                        logger.debug(
                                            "Backend auth rejected weighted score update (%s)",
                                            response.status,
                                        )
                                        self._auth_error_logged = True
                                    return False
                                logger.error(
                                    "Failed to update weighted scores: HTTP %s",
                                    response.status,
                                )
                                return False
                        else:
                            # Auth failures: instance-scoped dedup at DEBUG
                            if response.status in (401, 403):
                                if not self._auth_error_logged:
                                    logger.debug(
                                        "Backend auth rejected weighted score update (%s)",
                                        response.status,
                                    )
                                    self._auth_error_logged = True
                                return False
                            logger.error(
                                "Failed to update weighted scores: HTTP %s",
                                response.status,
                            )
                            return False
                except Exception as exc:
                    # Handle connection failures gracefully
                    if AIOHTTP_AVAILABLE and isinstance(
                        exc, aiohttp.ClientConnectorError
                    ):
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
        # Create a deterministic hash from session and config.
        # Sanitize first so numpy scalars (e.g. np.int64 from Optuna suggest_int)
        # don't crash json.dumps — same coercion as _sanitize_for_json.
        config_str = json.dumps(self._sanitize_for_json(config), sort_keys=True)
        hash_input = f"{session_id}:{config_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
