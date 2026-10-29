"""Authentication CLI commands for Traigent SDK.

Modern authentication management following patterns from GitHub CLI and AWS CLI.
"""

# Traceability: CONC-Layer-API CONC-Quality-Security CONC-Quality-Usability FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-CLOUD-009 REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import click
from rich.console import Console
from rich.panel import Panel

from traigent.core.backend_session_manager import (
    EDGE_BLOCK_SIGNALS as _SESSION_EDGE_BLOCK_SIGNALS,
)
from traigent.labels.transport_gate import gate_session

# Try to import aiohttp for exception handling
try:
    import aiohttp

    # Network errors to catch: OSError covers most, aiohttp.ClientError for async HTTP
    NETWORK_ERRORS: tuple[type[Exception], ...] = (OSError, aiohttp.ClientError)
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    NETWORK_ERRORS = (OSError,)

from rich.prompt import Prompt
from rich.table import Table

from traigent.cloud.auth import (
    AuthenticationError,
    AuthManager,
    InvalidCredentialsError,
)
from traigent.cloud.credential_manager import CredentialManager
from traigent.cloud.url_security import validate_cloud_base_url
from traigent.config.backend_config import (
    SIGNUP_URL,
    BackendConfig,
    EndpointNotConfiguredError,
    _endpoint_required_error,
    get_signup_url,
)
from traigent.config.project import PROJECT_ENV_VAR, read_optional_project_env
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME, read_optional_env
from traigent.security.credentials import (
    CredentialType,
    SecurityError,
    get_secure_credential_store,
)

# Imported for its side effect: loads .env with the #1830 guards (see auth()).
from traigent.utils import env_config as _env_config  # noqa: F401
from traigent.utils.logging import get_logger

console = Console()
RED_DEVICE_LOGIN_EXPIRED_BEFORE_AUTHORIZATION_COMPLETED = (
    "[red]Device login expired before authorization completed.[/red]"
)
logger = get_logger(__name__)

# Constants
TRAIGENT_CONFIG_DIR = Path.home() / ".traigent"
CREDENTIALS_FILE = TRAIGENT_CONFIG_DIR / "credentials.json"
BACKEND_RESPONSE_HEADER = "\n[red]--- Backend Response ---[/red]"


def _secure_credential_store_path() -> Path:
    """Return the on-disk path of the encrypted secure credential store.

    Mirrors ``EnhancedCredentialStore``'s default storage location so logout
    can distinguish "no secure credential exists" from "the secure store
    cannot be opened" without constructing the store — construction raises
    ``SecurityError`` whenever no master secret is configured, even when no
    encrypted credential file exists (Traigent#1721).
    """
    return Path.home() / ".traigent" / "secure_credentials.enc"


# Storage location identifier
STORAGE_FILE = "file"
STORAGE_SECURE = "secure"
SECURE_CLI_CREDENTIAL_NAME = "cli_credentials"
SECURE_CLI_PREFLIGHT_NAME = "__cli_device_login_preflight__"

# Common user-facing messages (avoid duplication per SonarCloud S1192)
MSG_CHECK_NETWORK = "Please check your network connection and try again.\n"
MSG_RUN_LOGIN_AGAIN = "Please run [cyan]traigent auth login[/cyan] again.\n"

_DEVICE_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{43,}$")
_USER_CODE_RE = re.compile(
    r"^[BCDFGHJKMNPQRSTVWXYZ23456789]{4}-[BCDFGHJKMNPQRSTVWXYZ23456789]{4}$"
)
_DEVICE_ERROR_NAMES = {
    "authorization_pending",
    "slow_down",
    "access_denied",
    "expired_token",
}
DEVICE_LOGIN_CLIENT_ID = "traigent-sdk-cli"
DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
_ENV_FILE_KEYS = (
    "TRAIGENT_API_KEY",
    TENANT_ENV_VAR,
    PROJECT_ENV_VAR,
    "TRAIGENT_BACKEND_URL",
)
_ENV_ASSIGNMENT_RE = re.compile(
    r"^(?P<leading>\s*)(?P<export>export\s+)?" r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*="
)
MAX_DEVICE_POLL_TIMEOUT_SECONDS = 30.0
MAX_DEVICE_POLL_TRANSPORT_ERRORS = 3
MAX_DEVICE_POLL_RATE_LIMITS_WITHOUT_RETRY_AFTER = 10


async def _read_json_body(response: Any) -> dict[str, Any]:
    """Read a JSON object response body, failing closed on malformed data."""
    try:
        response_text = await response.text()
    except Exception as exc:
        raise AuthenticationError("Unable to read backend response body") from exc

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise AuthenticationError("Invalid JSON response from backend") from exc

    if not isinstance(data, dict):
        raise AuthenticationError("Backend response must be a JSON object")
    return data


def _format_backend_error_message(
    prefix: str, status_code: int, data: dict[str, Any]
) -> str:
    """Format a backend error envelope without losing HTTP status context."""
    parts: list[str] = []
    for field in ("message", "error", "error_code"):
        value = data.get(field)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized and normalized not in parts:
                parts.append(normalized)

    suffix = f": {' - '.join(parts)}" if parts else ""
    return f"{prefix} (HTTP {status_code}){suffix}"


def _retry_after_seconds(headers: Any) -> int | None:
    """Return a positive Retry-After delay in seconds, if one was supplied."""
    getter = getattr(headers, "get", None)
    if getter is None:
        return None

    raw_value = getter("Retry-After")
    if raw_value is None:
        raw_value = getter("retry-after")
    if raw_value is None:
        return None

    try:
        seconds = int(str(raw_value).strip())
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _resolve_api_base_url(base_url: str) -> str:
    """Return an API base URL with a version path, without duplicating it."""
    origin, path = BackendConfig.split_api_url(base_url)
    if not origin:
        return cast(str, BackendConfig.build_api_base(base_url))
    if path:
        return f"{origin}{path.rstrip('/')}"
    return cast(str, BackendConfig.build_api_base(origin))


def _compose_api_url(base_url: str, path: str) -> str:
    return f"{_resolve_api_base_url(base_url)}/{path.lstrip('/')}"


def _require_non_empty_str(
    data: dict[str, Any], field: str, *, context: str = "device auth"
) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AuthenticationError(f"Malformed {context} response: missing {field}")
    return value.strip()


def _require_positive_int(data: dict[str, Any], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or value <= 0:
        raise AuthenticationError(f"Malformed device auth response: invalid {field}")
    return value


def _require_success_envelope(data: dict[str, Any], *, context: str) -> dict[str, Any]:
    """Return the data payload from a Traigent standard success envelope."""
    if data.get("success") is not True:
        raise AuthenticationError(f"Malformed {context} response: success missing")

    _require_non_empty_str(data, "message", context=context)
    payload = data.get("data")
    if not isinstance(payload, dict):
        raise AuthenticationError(f"Malformed {context} response: data missing")
    return payload


def _format_env_value(value: str) -> str:
    """Format a value for a dotenv assignment."""
    if re.fullmatch(r"[A-Za-z0-9_./:@+-]+", value):
        return value
    return json.dumps(value)


def _env_assignment_key(line: str) -> str | None:
    match = _ENV_ASSIGNMENT_RE.match(line)
    return match.group("key") if match else None


def _format_env_assignment_from_existing(line: str, key: str, value: str) -> str:
    match = _ENV_ASSIGNMENT_RE.match(line)
    leading = match.group("leading") if match else ""
    export_prefix = match.group("export") if match else None
    return f"{leading}{export_prefix or ''}{key}={_format_env_value(value)}"


def _parse_expires_at(raw: Any) -> str | None:
    """Return ``raw`` when it is a timezone-aware ISO-8601 timestamp, else None.

    The backend may omit ``expires_at`` or send null; an absent or malformed
    value is never replaced by an invented one (SDK #1635).
    """
    from datetime import datetime

    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip())
    except ValueError:  # silent-ok: malformed expires_at is deliberately treated as absent (value-free, #1635)
        return None
    if parsed.tzinfo is None:
        return None
    return raw.strip()


_AUTH_CLI_REQUEST = "auth CLI request"
_APPLICATION_JSON = "application/json"
_TRAIGENT_SDK_CLI_1_0 = "Traigent-SDK-CLI/1.0"
_DEVICE_TOKEN = "device token"
_BACKEND_NOT_CONFIGURED = "(not configured)"


class KeyCheckState(Enum):
    """Outcome of asking the backend whether an API key is valid (#2832).

    Only ``REJECTED`` is evidence that the key is bad. ``INDETERMINATE`` means
    the question was not answered (rate limit, server error, timeout, no
    network, unreadable reply) and must never cause stored credentials to be
    deleted.
    """

    VALID = "valid"
    REJECTED = "rejected"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class KeyCheckResult:
    state: KeyCheckState
    metadata: dict[str, Any] | None = None
    #: Short, non-sensitive cause for INDETERMINATE ("HTTP 503", "timeout").
    reason: str = ""


@dataclass
class _DevicePollState:
    """Mutable state shared by the device-token polling helpers."""

    interval: int
    now: Callable[[], float]
    deadline: float
    device_code: str
    consecutive_transport_errors: int = 0
    consecutive_rate_limits_without_retry_after: int = 0


def _device_poll_terminal_result(
    status: str, credentials: dict[str, Any] | None
) -> tuple[dict[str, Any] | None] | None:
    """Terminal poll outcome as a 1-tuple, or None when polling continues."""
    if status == "success":
        return (credentials,)
    if status == "access_denied":
        console.print("[red]Device login was denied in the browser.[/red]")
        return (None,)
    if status == "expired_token":
        console.print("[red]Device login expired. Please run login again.[/red]")
        return (None,)
    return None


def _append_missing_env_keys(
    next_lines: list[str], seen: set[str], updates: dict[str, str]
) -> None:
    if next_lines and next_lines[-1] != "":
        next_lines.append("")
    for key in _ENV_FILE_KEYS:
        if key not in seen:
            next_lines.append(f"{key}={_format_env_value(updates[key])}")


def _merge_env_lines(existing_lines: list[str], updates: dict[str, str]) -> list[str]:
    """Merge ``updates`` into the lines of an existing ``.env`` file."""
    seen: set[str] = set()
    next_lines: list[str] = []

    for line in existing_lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            next_lines.append(line)
            continue

        key = _env_assignment_key(line)
        if key is not None and key in updates:
            if key not in seen:
                next_lines.append(
                    _format_env_assignment_from_existing(line, key, updates[key])
                )
                seen.add(key)
        else:
            next_lines.append(line)

    _append_missing_env_keys(next_lines, seen, updates)
    return next_lines


def _interpret_api_key_response(
    status: int, response_text: str
) -> tuple[str | None, str | None]:
    """Parse the API-key creation response into ``(api_key, expires_at)``."""
    api_key = None
    api_key_expires_at: str | None = None
    if status == 201:
        try:
            api_key_data = json.loads(response_text)
            api_key = api_key_data["data"]["key"]
            api_key_expires_at = _parse_expires_at(
                api_key_data["data"].get("expires_at")
            )
            console.print("[green]✓ API key created[/green]")
        except (json.JSONDecodeError, KeyError) as e:
            console.print("\n[yellow]--- API Key Response ---[/yellow]")
            console.print(f"[yellow]Status: {status}[/yellow]")
            console.print("[yellow]Body: <suppressed>[/yellow]")
            console.print(f"[yellow]⚠️ Could not parse API key: {e}[/yellow]")
    elif status == 409:
        # Key name conflict - shouldn't happen with unique names, but handle it
        console.print("[yellow]⚠️ API key name conflict (409). Try again.[/yellow]")
    else:
        console.print("\n[yellow]--- API Key Response ---[/yellow]")
        console.print(f"[yellow]Status Code: {status}[/yellow]")
        console.print("[yellow]Body: <suppressed>[/yellow]")
        console.print(
            "[yellow]⚠️ API key creation failed, using JWT token only[/yellow]"
        )
    return api_key, api_key_expires_at


class TraigentAuthCLI:
    """Modern CLI authentication manager for Traigent SDK."""

    def __init__(self, backend_url_override: str | None = None) -> None:
        """Initialize the authentication CLI.

        Args:
            backend_url_override: If provided, use this backend URL instead of
                the default. Useful for targeting specific environments (e.g. dev).
        """
        self.config_dir = TRAIGENT_CONFIG_DIR
        self.credentials_file = CREDENTIALS_FILE
        self.auth_manager = AuthManager()
        self._backend_url: str | None = None
        self._backend_api_url: str | None = None
        self._endpoint_error: EndpointNotConfiguredError | None = None
        if backend_url_override:
            origin, path = BackendConfig.split_api_url(backend_url_override)
            normalized = origin or BackendConfig.normalize_backend_origin(
                backend_url_override
            )
            backend_origin = normalized or backend_url_override
            self._backend_url = validate_cloud_base_url(
                backend_origin.rstrip("/"), purpose=_AUTH_CLI_REQUEST
            )
            api_base_candidate = (
                f"{self._backend_url}{path or BackendConfig.get_default_api_path()}"
            )
            self._backend_api_url = validate_cloud_base_url(
                api_base_candidate.rstrip("/"), purpose=_AUTH_CLI_REQUEST
            )
        else:
            # Resolved lazily: in strict-endpoint (self-hosted) mode there may be
            # no endpoint yet, and `auth configure`, `auth status` and
            # `auth logout` must still work locally. Network paths read the
            # `backend_url` / `backend_api_url` properties, which raise.
            try:
                self._backend_url = validate_cloud_base_url(
                    BackendConfig.get_cloud_backend_url().rstrip("/"),
                    purpose=_AUTH_CLI_REQUEST,
                )
                self._backend_api_url = validate_cloud_base_url(
                    BackendConfig.get_cloud_api_url().rstrip("/"),
                    purpose=_AUTH_CLI_REQUEST,
                )
            except EndpointNotConfiguredError as exc:
                self._endpoint_error = exc

        # Ensure config directory exists
        self.config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    @property
    def backend_url(self) -> str:
        """Backend origin; raises EndpointNotConfiguredError in strict mode."""
        if self._backend_url is None:
            raise self._endpoint_error or _endpoint_required_error()
        return self._backend_url

    @backend_url.setter
    def backend_url(self, value: str) -> None:
        self._backend_url = value

    @property
    def backend_api_url(self) -> str:
        """Backend API base; raises EndpointNotConfiguredError in strict mode."""
        if self._backend_api_url is None:
            raise self._endpoint_error or _endpoint_required_error()
        return self._backend_api_url

    @backend_api_url.setter
    def backend_api_url(self, value: str) -> None:
        self._backend_api_url = value

    @property
    def device_login_target_url(self) -> str:
        """URL displayed for device login; it matches the HTTP API base.

        Device-login endpoint precedence follows the SDK cloud-client path:
        an explicit CLI backend override wins; otherwise the configured API
        base is used, so TRAIGENT_API_URL controls the called API host/path
        when set and TRAIGENT_BACKEND_URL is only the backend origin fallback.
        """
        return _resolve_api_base_url(self.backend_api_url)

    def _api_url(self, path: str) -> str:
        """Compose an API endpoint URL from the resolved API base."""
        return _compose_api_url(self.backend_api_url, path)

    def _load_stored_credentials(self) -> dict[str, Any] | None:
        """Load stored credentials from local file.

        Returns:
            Stored credentials or None if not found
        """
        secure_credentials = self._load_secure_credentials()
        if secure_credentials is not None:
            return secure_credentials

        if self.credentials_file.exists():
            if os.getenv("TRAIGENT_ALLOW_PLAINTEXT_CREDENTIALS", "").lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                logger.warning(
                    "Ignoring legacy plaintext credentials at %s. Configure "
                    "TRAIGENT_MASTER_PASSWORD and run 'traigent auth login' again, "
                    "or set TRAIGENT_ALLOW_PLAINTEXT_CREDENTIALS=true only for "
                    "one-time migration.",
                    self.credentials_file,
                )
                return None
            try:
                with open(self.credentials_file) as f:
                    data = json.load(f)
                    return dict(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Failed to load credentials file: {e}")
        elif self.config_dir.exists():
            # Config dir exists but no credentials file - user may have
            # authenticated when keyring was the primary store.
            logger.info(
                "No credentials file found. If you previously authenticated, "
                "please run 'traigent auth login' again."
            )

        return None

    @staticmethod
    def _secure_store_locked() -> bool:
        """True when an encrypted credential file exists but cannot be opened.

        That is the "present but locked" state (no or wrong
        ``TRAIGENT_MASTER_PASSWORD``): ``_load_stored_credentials`` returns None
        for it just as for "nothing stored", and callers must not conflate the
        two (#2833).
        """
        if not _secure_credential_store_path().exists():
            return False
        try:
            get_secure_credential_store()
        except SecurityError:
            return True
        return False

    def _load_secure_credentials(self) -> dict[str, Any] | None:
        """Load encrypted CLI credentials from the secure credential store."""
        try:
            store = get_secure_credential_store()
            serialized = store.get(SECURE_CLI_CREDENTIAL_NAME, check_env=False)
        except SecurityError as exc:
            logger.debug("Secure credential store unavailable: %s", exc)
            return None

        if not serialized:
            return None

        try:
            decoded = json.loads(serialized)
        except (TypeError, ValueError) as exc:
            logger.warning("Secure CLI credential payload is invalid JSON: %s", exc)
            return None

        if not isinstance(decoded, dict):
            logger.warning("Secure CLI credential payload must be a JSON object")
            return None

        return dict(decoded)

    def _save_credentials(self, credentials: dict[str, Any]) -> str | None:
        """Save credentials to encrypted local storage.

        Args:
            credentials: Credentials to save

        Returns:
            Storage location string if saved successfully, None if failed
        """
        try:
            store = get_secure_credential_store()
            store.set(
                SECURE_CLI_CREDENTIAL_NAME,
                json.dumps(credentials, separators=(",", ":")),
                CredentialType.TOKEN,
                metadata={"source": "traigent-cli"},
            )
            logger.debug("Credentials saved to secure credential store")
            return STORAGE_SECURE
        except OSError as e:
            logger.error(
                "Failed to save credentials securely: %s. Set "
                "TRAIGENT_MASTER_PASSWORD before running auth commands.",
                e,
            )
            return None
        except SecurityError as e:
            message = str(e).lower()
            if "weak" in message or "placeholder" in message:
                logger.error("Failed to save credentials securely: %s.", e)
            else:
                logger.error(
                    "Failed to save credentials securely: %s. Set "
                    "TRAIGENT_MASTER_PASSWORD before running auth commands.",
                    e,
                )
            return None

    def _preflight_secure_credential_store(self) -> bool:
        """Verify secure CLI credential storage is writable before auth starts."""
        try:
            store = get_secure_credential_store()
            probe = json.dumps(
                {"token": secrets.token_urlsafe(32)}, separators=(",", ":")
            )
            store.set(
                SECURE_CLI_PREFLIGHT_NAME,
                probe,
                CredentialType.TOKEN,
                metadata={
                    "source": "traigent-cli",
                    "purpose": "device-login-preflight",
                },
            )
            store.delete_secure(SECURE_CLI_PREFLIGHT_NAME)
            return True
        except (OSError, SecurityError, ValueError) as exc:
            logger.error(
                "Secure credential store preflight failed: %s. Set "
                "TRAIGENT_MASTER_PASSWORD before running device login.",
                exc,
            )
            console.print(
                "[red]❌ Secure credential storage is not ready.[/red]\n"
                "[yellow]Set TRAIGENT_MASTER_PASSWORD and run "
                "`traigent auth device-login` again.[/yellow]"
            )
            return False

    @staticmethod
    def _resolve_env_file_path(path: str | Path = ".env") -> Path:
        """Resolve a local .env path without allowing writes outside cwd."""
        candidate = Path(path).expanduser()
        if candidate.name != ".env":
            raise ValueError("Credentials export path must point to a .env file")

        cwd = Path.cwd().resolve()
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(cwd):
            raise ValueError(
                "Credentials export path must remain within the current directory"
            )
        return resolved

    def _save_api_key_to_env_file(
        self,
        api_key: str,
        *,
        tenant_id: str,
        project_id: str,
        backend_url: str,
        path: str | Path = ".env",
    ) -> Path:
        """Persist auth environment values to a local .env file with 0600 perms."""
        env_path = self._resolve_env_file_path(path)
        existing_lines: list[str] = []
        if env_path.exists():
            existing_lines = env_path.read_text(encoding="utf-8").splitlines()

        updates = {
            "TRAIGENT_API_KEY": api_key,
            TENANT_ENV_VAR: tenant_id,
            PROJECT_ENV_VAR: project_id,
            "TRAIGENT_BACKEND_URL": backend_url,
        }
        next_lines = _merge_env_lines(existing_lines, updates)

        env_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            env_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(next_lines).rstrip() + "\n")
        env_path.chmod(0o600)
        return env_path

    async def _validate_api_key(
        self, api_key: str, verbose: bool = False
    ) -> dict[str, Any] | None:
        """Validate an API key against the backend.

        Compatibility wrapper over :meth:`_check_api_key_state`.

        Returns:
            Key metadata dict if valid, None otherwise (rejected OR could not
            be determined; use ``_check_api_key_state`` to tell them apart).
        """
        result = await self._check_api_key_state(api_key, verbose=verbose)
        return result.metadata if result.state is KeyCheckState.VALID else None

    async def _check_api_key_state(
        self, api_key: str, verbose: bool = False
    ) -> KeyCheckResult:
        """Ask the backend whether ``api_key`` is valid, as a tri-state.

        VALID: 200 with ``valid: true``. REJECTED: 401, an authorization 403,
        or 200 with ``valid: false``. INDETERMINATE: everything else (429, 5xx,
        edge-block 403, timeout, network error, unreadable 200).
        """
        import aiohttp

        url = self._api_url("keys/validate")
        headers = {"X-API-Key": api_key}

        if verbose:
            console.print(f"[dim]POST {url}[/dim]")
            console.print("[dim]X-API-Key: <redacted>[/dim]")

        try:
            async with gate_session(
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15),
                    trust_env=True,
                )
            ) as session:
                async with session.post(url, headers=headers) as response:
                    response_text = await response.text()

                    if verbose:
                        console.print(f"[dim]Response status: {response.status}[/dim]")

                    return _key_check_from_response(
                        response.status, response_text, response.headers
                    )
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.debug(f"API key validation failed: {e}")
            if verbose:
                console.print(f"[dim]Connection error: {e}[/dim]")
            reason = "timeout" if isinstance(e, TimeoutError) else "network error"
            return KeyCheckResult(KeyCheckState.INDETERMINATE, reason=reason)

    def _validate_device_authorize_payload(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate the device authorization response contract."""
        payload = _require_success_envelope(data, context="device auth")

        device_code = _require_non_empty_str(payload, "device_code")
        if not _DEVICE_CODE_RE.fullmatch(device_code):
            raise AuthenticationError(
                "Malformed device auth response: invalid device_code"
            )

        user_code = _require_non_empty_str(payload, "user_code")
        if not _USER_CODE_RE.fullmatch(user_code):
            raise AuthenticationError(
                "Malformed device auth response: invalid user_code"
            )

        verification_uri = _require_non_empty_str(payload, "verification_uri")
        verification_uri_complete = _require_non_empty_str(
            payload, "verification_uri_complete"
        )
        expires_in = _require_positive_int(payload, "expires_in")
        interval = _require_positive_int(payload, "interval")

        return {
            "device_code": device_code,
            "user_code": user_code,
            "verification_uri": verification_uri,
            "verification_uri_complete": verification_uri_complete,
            "expires_in": expires_in,
            "interval": interval,
        }

    async def _start_device_authorization(self, session: Any) -> dict[str, Any]:
        """Start the RFC 8628 device authorization flow."""
        url = self._api_url("auth/device/authorize")
        headers = {
            "Accept": _APPLICATION_JSON,
            "Content-Type": _APPLICATION_JSON,
            "User-Agent": _TRAIGENT_SDK_CLI_1_0,
        }
        async with session.post(
            url,
            json={"client_id": DEVICE_LOGIN_CLIENT_ID},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30) if aiohttp else None,
        ) as response:
            data = await _read_json_body(response)
            if response.status != 200:
                raise AuthenticationError(
                    _format_backend_error_message(
                        "Device authorization failed", response.status, data
                    )
                )
            return self._validate_device_authorize_payload(data)

    def _validate_device_token_success(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize a successful device token envelope."""
        token_data = _require_success_envelope(data, context=_DEVICE_TOKEN)

        api_key = _require_non_empty_str(token_data, "api_key", context=_DEVICE_TOKEN)
        if not api_key.startswith("sk_"):
            raise AuthenticationError(
                "Malformed device token response: invalid api_key"
            )

        tenant_id = _require_non_empty_str(
            token_data, "tenant_id", context=_DEVICE_TOKEN
        )
        project_id = _require_non_empty_str(
            token_data, "project_id", context=_DEVICE_TOKEN
        )
        subscription_tier = _require_non_empty_str(
            token_data, "subscription_tier", context=_DEVICE_TOKEN
        )

        user = token_data.get("user")
        if not isinstance(user, dict):
            raise AuthenticationError("Malformed device token response: user missing")
        normalized_user = {
            "id": _require_non_empty_str(user, "id", context=_DEVICE_TOKEN),
            "email": _require_non_empty_str(user, "email", context=_DEVICE_TOKEN),
        }

        quota = token_data.get("quota")
        if not isinstance(quota, dict):
            raise AuthenticationError("Malformed device token response: quota missing")
        trial_limit = quota.get("trial_limit")
        api_call_limit = quota.get("api_call_limit")
        if not isinstance(trial_limit, int) or not isinstance(api_call_limit, int):
            raise AuthenticationError("Malformed device token response: invalid quota")

        credentials: dict[str, Any] = {
            "api_key": api_key,
            "tenant_id": tenant_id,
            "project_id": project_id,
            "user": normalized_user,
            "subscription_tier": subscription_tier,
            "quota": {
                "trial_limit": trial_limit,
                "api_call_limit": api_call_limit,
            },
            "backend_url": self.backend_url,
        }
        expires_at = _parse_expires_at(token_data.get("expires_at"))
        if expires_at is not None:
            credentials["expires_at"] = expires_at
        return credentials

    @staticmethod
    def _device_error_from_envelope(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Return a known RFC 8628 error name and details from an error envelope."""
        if data.get("success") is not False:
            raise AuthenticationError("Unexpected device token response from backend")
        _require_non_empty_str(data, "message", context="device token error")

        raw_error = data.get("error")
        raw_error_code = data.get("error_code")
        if raw_error and raw_error_code and raw_error != raw_error_code:
            raise AuthenticationError(
                "Malformed device token error response: error mismatch"
            )

        error_name = raw_error_code or raw_error
        if not isinstance(error_name, str) or error_name not in _DEVICE_ERROR_NAMES:
            raise AuthenticationError("Unexpected device token response from backend")

        details = data.get("details")
        return error_name, details if isinstance(details, dict) else {}

    async def _poll_device_token_once(
        self, session: Any, device_code: str, request_timeout: float
    ) -> tuple[str, dict[str, Any] | None, int | None]:
        """Poll the token endpoint once."""
        url = self._api_url("auth/device/token")
        headers = {
            "Accept": _APPLICATION_JSON,
            "Content-Type": _APPLICATION_JSON,
            "User-Agent": _TRAIGENT_SDK_CLI_1_0,
        }
        async with session.post(
            url,
            json={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "device_code": device_code,
                "client_id": DEVICE_LOGIN_CLIENT_ID,
            },
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=request_timeout) if aiohttp else None,
        ) as response:
            status_code = response.status
            if status_code == 429:
                return "rate_limited", None, _retry_after_seconds(response.headers)
            data = await _read_json_body(response)

        if data.get("success") is True:
            if status_code != 200:
                raise AuthenticationError(
                    f"Device token request failed (HTTP {status_code})"
                )
            return "success", self._validate_device_token_success(data), None

        try:
            error_name, details = self._device_error_from_envelope(data)
        except AuthenticationError as exc:
            if status_code != 200:
                raise AuthenticationError(
                    _format_backend_error_message(
                        "Device token request failed", status_code, data
                    )
                ) from exc
            raise
        if error_name == "slow_down":
            new_interval = details.get("interval")
            if not isinstance(new_interval, int) or new_interval < 6:
                raise AuthenticationError(
                    "Malformed slow_down response: details.interval must be >= 6"
                )
            return error_name, None, new_interval
        return error_name, None, None

    async def _poll_device_token(
        self,
        session: Any,
        authorization: dict[str, Any],
        sleep: Any,
        clock: Callable[[], float] | None = None,
    ) -> dict[str, Any] | None:
        """Poll the token endpoint until success, denial, or expiry."""
        now = clock or time.monotonic
        state = _DevicePollState(
            interval=int(authorization["interval"]),
            now=now,
            deadline=now() + int(authorization["expires_in"]),
            device_code=str(authorization["device_code"]),
        )

        while True:
            finished, result = await self._poll_device_token_step(session, state, sleep)
            if finished:
                return result

    async def _poll_device_token_step(
        self, session: Any, state: _DevicePollState, sleep: Any
    ) -> tuple[bool, dict[str, Any] | None]:
        """Run one poll iteration; returns ``(finished, credentials)``."""
        remaining = state.deadline - state.now()
        if remaining <= 0:
            console.print(RED_DEVICE_LOGIN_EXPIRED_BEFORE_AUTHORIZATION_COMPLETED)
            return True, None

        try:
            status, credentials, new_interval = await self._poll_device_token_once(
                session,
                state.device_code,
                min(MAX_DEVICE_POLL_TIMEOUT_SECONDS, remaining),
            )
            state.consecutive_transport_errors = 0
        except NETWORK_ERRORS as exc:
            return await self._backoff_after_transport_error(exc, state, sleep)

        if status == "rate_limited":
            return await self._backoff_after_rate_limit(new_interval, state, sleep)

        state.consecutive_rate_limits_without_retry_after = 0

        terminal = _device_poll_terminal_result(status, credentials)
        if terminal is not None:
            return True, terminal[0]
        if status == "slow_down":
            if new_interval is None:
                raise AuthenticationError(
                    "Malformed slow_down response: missing interval"
                )
            state.interval = new_interval
            console.print(
                f"[yellow]Backend requested slower polling; waiting {state.interval}s.[/yellow]"
            )

        slept = await self._sleep_within_deadline(state, state.interval, sleep)
        return (not slept), None

    @staticmethod
    async def _sleep_within_deadline(
        state: _DevicePollState,
        delay: float,
        sleep: Any,
        announce: Callable[[float], str] | None = None,
    ) -> bool:
        """Sleep ``delay`` capped by the deadline; False if the login expired."""
        sleep_seconds = min(float(delay), max(0.0, state.deadline - state.now()))
        if sleep_seconds <= 0:
            console.print(RED_DEVICE_LOGIN_EXPIRED_BEFORE_AUTHORIZATION_COMPLETED)
            return False
        if announce is not None:
            console.print(announce(sleep_seconds))
        await sleep(sleep_seconds)
        return True

    async def _backoff_after_transport_error(
        self, exc: BaseException, state: _DevicePollState, sleep: Any
    ) -> tuple[bool, dict[str, Any] | None]:
        state.consecutive_transport_errors += 1
        state.consecutive_rate_limits_without_retry_after = 0
        if state.consecutive_transport_errors >= MAX_DEVICE_POLL_TRANSPORT_ERRORS:
            console.print(
                "[red]Device token polling failed after "
                f"{state.consecutive_transport_errors} consecutive transport "
                "errors. Please check your network and run login again.[/red]"
            )
            return True, None
        slept = await self._sleep_within_deadline(
            state,
            state.interval,
            sleep,
            lambda sleep_seconds: (
                "[yellow]Device token polling transport error "
                f"({type(exc).__name__}); retrying in {sleep_seconds:g}s.[/yellow]"
            ),
        )
        return (not slept), None

    async def _backoff_after_rate_limit(
        self, new_interval: int | None, state: _DevicePollState, sleep: Any
    ) -> tuple[bool, dict[str, Any] | None]:
        if new_interval is None:
            state.consecutive_rate_limits_without_retry_after += 1
            if (
                state.consecutive_rate_limits_without_retry_after
                >= MAX_DEVICE_POLL_RATE_LIMITS_WITHOUT_RETRY_AFTER
            ):
                console.print(
                    "[red]Device token polling is still rate-limited after "
                    f"{state.consecutive_rate_limits_without_retry_after} "
                    "consecutive 429 responses without Retry-After. "
                    "Please run login again later.[/red]"
                )
                return True, None
            retry_after = state.interval + 5
            reason = "without Retry-After"
        else:
            state.consecutive_rate_limits_without_retry_after = 0
            retry_after = new_interval
            reason = "per Retry-After"

        slept = await self._sleep_within_deadline(
            state,
            retry_after,
            sleep,
            lambda sleep_seconds: (
                "[yellow]Device token polling rate-limited "
                f"({reason}); retrying in {sleep_seconds:g}s.[/yellow]"
            ),
        )
        return (not slept), None

    def _display_device_authorization(self, authorization: dict[str, Any]) -> None:
        """Print the browser URL and user code prominently."""
        body = (
            f"Open this URL:\n[bold cyan]{authorization['verification_uri_complete']}[/bold cyan]\n\n"
            f"Code:\n[bold yellow]{authorization['user_code']}[/bold yellow]"
        )
        console.print(
            Panel.fit(body, title="Traigent Device Login", border_style="blue")
        )

    def _persist_device_credentials(
        self,
        credentials: dict[str, Any],
        *,
        save_env_file: bool,
        env_file: str | Path,
        write_env_explicit: bool = False,
    ) -> str | None:
        """Persist device credentials through existing credential machinery."""
        storage_location = self._save_credentials(credentials)
        if save_env_file:
            if storage_location != STORAGE_SECURE and not write_env_explicit:
                console.print(
                    "[yellow]Skipping .env write because secure credential storage "
                    "failed.[/yellow]"
                )
            else:
                if storage_location != STORAGE_SECURE:
                    console.print(
                        "[yellow]Secure credential storage failed; writing .env "
                        "because --write-env was explicitly passed.[/yellow]"
                    )
                try:
                    saved_path = self._save_api_key_to_env_file(
                        str(credentials["api_key"]),
                        tenant_id=str(credentials["tenant_id"]),
                        project_id=str(credentials["project_id"]),
                        backend_url=str(credentials["backend_url"]),
                        path=env_file,
                    )
                    console.print(
                        f"[green]✅ Project auth environment updated:[/green] {saved_path}"
                    )
                except (OSError, ValueError) as exc:
                    console.print(
                        "[yellow]Could not update local .env file. Reason: "
                        f"{exc}[/yellow]"
                    )
        return storage_location

    async def device_login(
        self,
        *,
        save_env_file: bool = False,
        env_file: str | Path = ".env",
        write_env_explicit: bool = False,
        sleep: Any | None = None,
        clock: Callable[[], float] | None = None,
    ) -> bool:
        """Authenticate with the browser-based device-code flow."""
        import aiohttp

        poll_sleep = sleep or asyncio.sleep
        console.print("\n[bold blue]Traigent Device Authentication[/bold blue]")
        console.print(
            f"Authenticating with: [cyan]{self.device_login_target_url}[/cyan]\n"
        )

        if not self._preflight_secure_credential_store():
            return False

        try:
            async with gate_session(aiohttp.ClientSession(trust_env=True)) as session:
                authorization = await self._start_device_authorization(session)
                self._display_device_authorization(authorization)
                credentials = await self._poll_device_token(
                    session, authorization, poll_sleep, clock=clock
                )
        except AuthenticationError as exc:
            console.print(f"[red]❌ Device authentication failed: {exc}[/red]")
            return False
        except TimeoutError as exc:
            console.print(f"[red]❌ Device authentication timed out: {exc}[/red]")
            console.print(MSG_CHECK_NETWORK)
            return False
        except NETWORK_ERRORS as exc:
            error_type = type(exc).__name__
            console.print(f"[red]❌ Connection error ({error_type}): {exc}[/red]")
            console.print(MSG_CHECK_NETWORK)
            return False

        if not credentials:
            return False

        storage_location = self._persist_device_credentials(
            credentials,
            save_env_file=save_env_file,
            env_file=env_file,
            write_env_explicit=write_env_explicit,
        )
        if not storage_location:
            self._display_storage_location(storage_location)
            return False

        user = credentials.get("user", {})
        console.print(
            f"[green]✅ Authenticated as {user.get('email', 'Traigent user')}[/green]"
        )
        console.print("[green]✅ API key received and stored securely[/green]")
        self._display_storage_location(storage_location)
        return True

    def _clear_credentials(self) -> bool:
        """Clear stored credentials.

        Returns:
            True if cleared successfully
        """
        success = True

        if self.credentials_file.exists():
            try:
                self.credentials_file.unlink()
                logger.debug("Cleared credentials file")
            except OSError as e:
                logger.error(f"Failed to delete credentials file: {e}")
                success = False

        secure_delete_failed = False
        try:
            store = get_secure_credential_store()
            deleted = store.delete_secure(SECURE_CLI_CREDENTIAL_NAME)
            if deleted:
                logger.debug("Cleared secure CLI credentials")
        except SecurityError as e:
            # The secure store could not be opened (typically because no
            # master secret is configured). Treat this as a real logout
            # failure ONLY when an encrypted credential actually exists on
            # disk that we would now be leaving behind; a store that is simply
            # not configured has nothing to delete (Traigent#1721). This
            # mirrors how _load_secure_credentials treats an unopenable store
            # as "unavailable" rather than an error.
            if _secure_credential_store_path().exists():
                logger.error("Could not confirm secure CLI credential deletion: %s", e)
                secure_delete_failed = True
            else:
                logger.debug(
                    "Secure credential store not configured during logout: %s", e
                )
        except OSError as e:
            # The store opened but deletion failed at the filesystem level.
            logger.error("Could not confirm secure CLI credential deletion: %s", e)
            secure_delete_failed = True

        if secure_delete_failed:
            console.print(
                "[yellow]⚠️  Could not confirm secure-storage credential "
                "deletion; it may still be present. Set "
                "TRAIGENT_MASTER_PASSWORD and try again if this persists.[/yellow]"
            )
            success = False

        return success

    async def _authenticate_with_backend(
        self, email: str, password: str
    ) -> dict[str, Any]:
        """Authenticate with the backend and get tokens.

        Args:
            email: User email
            password: User password

        Returns:
            Authentication tokens

        Raises:
            AuthenticationError: If authentication fails due to HTTP/protocol errors
            InvalidCredentialsError: If credentials are rejected by the backend
        """
        import aiohttp

        # Step 1: Direct login call for better error visibility
        login_url = f"{self.backend_api_url}/auth/login"
        tenant_id = read_optional_env(TENANT_ENV_VAR)
        project_id = read_optional_project_env()
        if not tenant_id:
            message = (
                f"{TENANT_ENV_VAR} is required for org-bound SDK authentication. "
                "Set it to the organization id before running `traigent auth login`."
            )
            if project_id:
                message += f" {PROJECT_ENV_VAR} is set, but project-bound keys also require a tenant."
            raise AuthenticationError(message)
        login_headers = {
            "Content-Type": _APPLICATION_JSON,
            "User-Agent": _TRAIGENT_SDK_CLI_1_0,
            TENANT_HEADER_NAME: tenant_id,
        }

        console.print(f"[dim]POST {login_url}[/dim]")

        async with gate_session(aiohttp.ClientSession(trust_env=True)) as session:
            async with session.post(
                login_url,
                json={"email": email, "password": password},
                headers=login_headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response_text = await response.text()

                # Show response details on failure
                if response.status != 200:
                    console.print(BACKEND_RESPONSE_HEADER)
                    console.print(f"[red]Status Code: {response.status}[/red]")
                    safe_headers = {
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() in ("content-type", "x-request-id", "x-trace-id")
                    }
                    console.print(f"[red]Headers: {safe_headers}[/red]")
                    console.print("[red]Body: <suppressed>[/red]")
                    raise AuthenticationError(
                        f"Authentication failed (HTTP {response.status})"
                    ) from None

                try:
                    login_data = json.loads(response_text)
                except json.JSONDecodeError:
                    console.print(BACKEND_RESPONSE_HEADER)
                    console.print(f"[red]Status Code: {response.status}[/red]")
                    console.print("[red]Body (not JSON): <suppressed>[/red]")
                    raise AuthenticationError(
                        "Invalid JSON response from backend"
                    ) from None

                if not login_data.get("success"):
                    console.print(BACKEND_RESPONSE_HEADER)
                    error_msg = login_data.get("error", "Unknown error")
                    console.print(f"[red]Error: {error_msg}[/red]")
                    raise InvalidCredentialsError(
                        f"Authentication failed: {error_msg}"
                    ) from None

                token_data = login_data.get("data", {})
                jwt_token = token_data.get("access_token")
                refresh_token = token_data.get("refresh_token")

                if not jwt_token:
                    console.print(BACKEND_RESPONSE_HEADER)
                    raise AuthenticationError("No access_token in response") from None

                # Show truncated token
                console.print("[green]✓ JWT Token received[/green]")

        # Step 2: Create API key using JWT token
        api_key = None
        api_key_expires_at: str | None = None
        api_key_url = f"{self.backend_api_url}/keys"

        # Generate a unique key name to avoid conflicts
        import platform
        from datetime import datetime

        hostname = platform.node() or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        key_name = f"Traigent SDK CLI ({hostname[:20]} {timestamp})"
        key_payload = {
            "key_name": key_name,
            "permissions": [
                "read",
                "write",
                "experiment.read",
                "experiment.write",
                "session.read",
                "session.write",
            ],
        }
        if project_id:
            key_payload["project_id"] = project_id
        key_headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": _APPLICATION_JSON,
            "User-Agent": _TRAIGENT_SDK_CLI_1_0,
            TENANT_HEADER_NAME: tenant_id,
        }

        console.print(f"\n[dim]POST {api_key_url}[/dim]")
        console.print(f"[dim]Creating API key: {key_name}[/dim]")

        async with gate_session(aiohttp.ClientSession(trust_env=True)) as session:
            async with session.post(
                api_key_url,
                json=key_payload,
                headers=key_headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response_text = await response.text()

                api_key, api_key_expires_at = _interpret_api_key_response(
                    response.status, response_text
                )

        # Get user info from stored token data (if available)
        # Note: SecureAuthManager doesn't expose user info directly
        # We'll need to extract it from the initial auth response
        user_info: dict[str, str] = {}

        # regression fix: do NOT fabricate `refresh_token = jwt_token`. The
        # backend's auth response did not return a real refresh token,
        # so storing the JWT under that key was misleading — downstream
        # callers (e.g. `auth_commands.py:_perform_token_refresh`) would
        # try to use the JWT as a refresh token at the auth server,
        # which fails. The honest behavior: omit the field entirely.
        # Refresh-needing flows already gracefully degrade to "please
        # re-login" when no refresh_token is present.
        credentials = {
            "jwt_token": jwt_token,
            "api_key": api_key,
            "user": user_info,
            "backend_url": self.backend_url,
        }
        if refresh_token:
            credentials["refresh_token"] = refresh_token
        if api_key and api_key_expires_at is not None:
            credentials["expires_at"] = api_key_expires_at

        return credentials

    async def _check_stored_api_key(self) -> bool:
        """Check if stored API key is valid.

        Returns:
            True if valid stored credentials found (already authenticated)
        """
        existing_creds = self._load_stored_credentials()
        if not existing_creds or not existing_creds.get("api_key"):
            return False

        console.print("[dim]Checking stored API key...[/dim]")
        check = await self._check_api_key_state(existing_creds["api_key"], verbose=True)
        user_info = check.metadata

        if check.state is KeyCheckState.VALID and user_info is not None:
            console.print("[green]✅ Already authenticated with valid API key[/green]")
            email_display = user_info.get("email") or existing_creds.get(
                "user", {}
            ).get("email")
            if email_display:
                console.print(f"User: [cyan]{email_display}[/cyan]")
            console.print(
                "\nTo re-authenticate, first run [cyan]traigent auth logout[/cyan]\n"
            )
            return True

        if check.state is KeyCheckState.INDETERMINATE:
            # A rate limit, outage or network blip says nothing about the key:
            # keep the stored credentials (#2832).
            console.print(
                f"[yellow]⚠️ Could not validate the stored API key ({check.reason}); "
                "stored credentials kept, proceeding with login...[/yellow]\n"
            )
            return False

        # The backend rejected the key: clear it
        console.print(
            "[yellow]⚠️ Stored API key is no longer valid, proceeding with login...[/yellow]\n"
        )
        self._clear_credentials()
        return False

    async def _check_env_api_key(self) -> bool:
        """Check if TRAIGENT_API_KEY environment variable is valid.

        Returns:
            True if valid env API key found (already authenticated)
        """
        env_api_key = os.environ.get("TRAIGENT_API_KEY")
        if not env_api_key:
            return False

        console.print("[dim]Found TRAIGENT_API_KEY in environment, validating...[/dim]")
        check = await self._check_api_key_state(env_api_key, verbose=True)
        user_info = check.metadata

        if check.state is KeyCheckState.VALID and user_info is not None:
            console.print(
                "[green]✅ Valid API key found in TRAIGENT_API_KEY environment variable[/green]"
            )
            email_display = user_info.get("email")
            if email_display:
                console.print(f"User: [cyan]{email_display}[/cyan]")
            console.print(
                "\nYou're already authenticated via environment variable.\n"
                "To use a different account, unset TRAIGENT_API_KEY and run login again.\n"
            )
            return True

        if check.state is KeyCheckState.INDETERMINATE:
            console.print(
                f"[yellow]⚠️ Could not validate TRAIGENT_API_KEY in environment "
                f"({check.reason}), proceeding with login...[/yellow]\n"
            )
            return False
        console.print(
            "[yellow]⚠️ TRAIGENT_API_KEY in environment is invalid, proceeding with login...[/yellow]\n"
        )
        return False

    def _get_user_credentials(
        self, email: str | None, non_interactive: bool
    ) -> tuple[str, str] | None:
        """Get email and password from user.

        Returns:
            Tuple of (email, password) or None if failed in non-interactive mode
        """
        # Get email from arg, env var, or prompt
        if not email:
            email = os.environ.get("TRAIGENT_AUTH_EMAIL")
        if not email:
            if non_interactive:
                console.print(
                    "[red]Email required. Use --email or set TRAIGENT_AUTH_EMAIL[/red]"
                )
                return None
            email = Prompt.ask("Email")
        else:
            console.print(f"Using email: [cyan]{email}[/cyan]")

        # Get password from env var or prompt
        password = os.environ.get("TRAIGENT_AUTH_PASSWORD")
        if not password:
            if non_interactive:
                console.print(
                    "[red]Password required. Set TRAIGENT_AUTH_PASSWORD[/red]"
                )
                return None
            from getpass import getpass

            password = getpass("Password: ")
        else:
            console.print("Using password from: [cyan]TRAIGENT_AUTH_PASSWORD[/cyan]")

        if email is None:
            raise ValueError("Email must be provided")
        return (email, password)

    def _display_storage_location(self, storage_location: str | None) -> None:
        """Display where credentials are stored."""
        if storage_location == STORAGE_SECURE:
            console.print(
                "\n[bold]Credentials stored in encrypted local storage[/bold]"
            )
        elif storage_location == STORAGE_FILE:
            console.print(
                f"\n[yellow]Legacy plaintext credentials are present at:[/yellow] "
                f"[cyan]{self.credentials_file}[/cyan]"
            )
        else:
            console.print(
                "\n[yellow]Could not save credentials securely. Set "
                "TRAIGENT_MASTER_PASSWORD and rerun the command.[/yellow]"
            )

    async def _enforce_password_login_rate_limit(self) -> None:
        """Delay password login attempts using PasswordAuthHandler state."""
        try:
            should_rate_limit = self.auth_manager._should_rate_limit_login()
            wait_time = (
                self.auth_manager._get_rate_limit_wait() if should_rate_limit else 0.0
            )
        except Exception as exc:
            logger.error("Unable to verify password login rate-limit state: %s", exc)
            raise AuthenticationError(
                "Unable to verify password login rate-limit state"
            ) from exc

        if not should_rate_limit:
            return

        logger.warning(
            "CLI password login rate limited after repeated failures; waiting %.2fs",
            wait_time,
        )
        try:
            await asyncio.sleep(wait_time)
        except Exception as exc:
            logger.error("Unable to enforce password login rate-limit delay: %s", exc)
            raise AuthenticationError(
                "Unable to enforce password login rate-limit delay"
            ) from exc

    def _record_password_login_failure(self) -> None:
        """Record a failed CLI password login through PasswordAuthHandler state."""
        try:
            self.auth_manager._record_failure()
        except Exception as exc:
            logger.error("Unable to record password login failure: %s", exc)
            raise AuthenticationError(
                "Unable to record password login failure"
            ) from exc

    def _reset_password_login_failures(self) -> None:
        """Reset CLI password login failure tracking after a successful login."""
        self.auth_manager._reset_failure_tracking()

    async def login(
        self, email: str | None = None, non_interactive: bool = False
    ) -> bool:
        """Login to Traigent backend.

        Args:
            email: Email address (optional, will use env var or prompt)
            non_interactive: If True, fail instead of prompting

        Returns:
            True if login successful
        """
        console.print("\n[bold blue]🔐 Traigent Authentication[/bold blue]")
        console.print(f"Authenticating with: [cyan]{self.backend_url}[/cyan]\n")

        # Check existing authentication methods
        if await self._check_stored_api_key():
            return True
        if await self._check_env_api_key():
            return True

        # Get user credentials
        creds = self._get_user_credentials(email, non_interactive)
        if creds is None:
            return False
        email, password = creds

        try:
            # Authenticate with backend
            console.print("\n[yellow]Authenticating...[/yellow]")
            await self._enforce_password_login_rate_limit()
            credentials = await self._authenticate_with_backend(email, password)
            self._reset_password_login_failures()

            # Store and display results
            storage_location = self._save_credentials(credentials)
            self._display_login_success(credentials, email, storage_location)

            if storage_location is None:
                # Authentication succeeded but nothing was persisted: the user
                # is not actually logged in for future commands. Report this
                # as a failure rather than a clean success (Traigent#1721).
                console.print(
                    "\n[red]❌ Login failed: credentials could not be saved.[/red]"
                )
                return False

            console.print(
                "\nYou can now use Traigent SDK with backend tracking enabled.\n"
            )
            return True

        except AuthenticationError as e:
            self._record_password_login_failure()
            console.print(f"\n[red]❌ Authentication failed: {e}[/red]")
            console.print("\nPlease check your credentials and try again.")
            if get_signup_url():
                console.print(f"If you don't have an account, visit: {SIGNUP_URL}\n")
            return False
        except TimeoutError as e:
            self._record_password_login_failure()
            console.print(f"\n[red]❌ Connection timed out: {e}[/red]")
            console.print("\nThe backend server took too long to respond.")
            console.print(MSG_CHECK_NETWORK)
            return False
        except NETWORK_ERRORS as e:
            self._record_password_login_failure()
            error_type = type(e).__name__
            console.print(f"\n[red]❌ Connection error ({error_type}): {e}[/red]")
            console.print("\nCould not connect to the authentication server.")
            console.print(MSG_CHECK_NETWORK)
            return False

    def _display_login_success(
        self,
        credentials: dict[str, Any],
        email: str,
        storage_location: str | None,
    ) -> None:
        """Display success message after login."""
        user = credentials.get("user", {})
        console.print(
            f"\n[green]✅ Successfully authenticated as {user.get('email', email)}[/green]"
        )

        if credentials.get("api_key"):
            console.print("[green]✅ API key generated and stored securely[/green]")
        else:
            console.print(
                "[yellow]⚠️  Using JWT token (API key generation not available)[/yellow]"
            )

        self._display_storage_location(storage_location)

    async def logout(self) -> bool:
        """Logout and clear stored credentials.

        Returns:
            True if logout successful
        """
        console.print("\n[bold blue]🔓 Logging out[/bold blue]")

        # Check if we have stored credentials
        creds = self._load_stored_credentials()
        locked = not creds and self._secure_store_locked()
        # An ignored legacy plaintext file is still stored credentials to clear.
        has_plaintext_file = self.credentials_file.exists()
        if not creds and not locked and not has_plaintext_file:
            console.print("[yellow]No stored credentials found[/yellow]")
            return True

        # Optional: Revoke API key on backend
        if creds and creds.get("api_key"):
            try:
                import aiohttp

                async with gate_session(aiohttp.ClientSession(trust_env=True)):
                    # Backend may not expose a revoke endpoint yet; placeholder for future call.
                    pass  # Silently continue if revocation fails
            except Exception as e:
                logger.debug(
                    f"Could not revoke API key from backend (may not be supported): {e}"
                )

        # Clear local credentials
        if self._clear_credentials():
            console.print("[green]✅ Successfully logged out[/green]")
            console.print("Local credentials have been cleared.\n")
            return True
        else:
            console.print("[red]❌ Failed to clear some credentials[/red]")
            if locked:
                console.print(
                    f"The encrypted credential file {_secure_credential_store_path()} "
                    "is still present and locked. Set TRAIGENT_MASTER_PASSWORD to "
                    "the password it was created with and run "
                    "[cyan]traigent auth logout[/cyan] again.\n"
                )
            return False

    def _api_key_expiry_row(self, creds: dict[str, Any]) -> str | None:
        """Describe the stored API key's expiry/rotation state (SDK #1635).

        Only a recorded ``expires_at`` is reported; an unknown expiry is never
        presented as a measured one. Returns None when nothing is known.
        """
        from datetime import UTC, datetime

        raw = creds.get("expires_at")
        expires_at: datetime | None = None
        if isinstance(raw, str):
            try:
                expires_at = datetime.fromisoformat(raw)
            except ValueError:
                expires_at = None
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            try:
                expires_at = datetime.fromtimestamp(raw, tz=UTC)
            except (OverflowError, OSError, ValueError):
                expires_at = None
        if expires_at is None:
            return None

        manager = getattr(self, "auth_manager", None) or AuthManager()
        manager.set_api_key(
            creds["api_key"], expires_at=expires_at, source="stored credentials"
        )
        status = manager.get_api_key_status()
        state = status.get("state")
        days = status.get("days_remaining")
        if state == "expired":
            return "[red]expired - run traigent auth login to rotate[/red]"
        if not isinstance(days, (int, float)):
            return None
        if state == "critical":
            return (
                f"[red]expires in {days:.1f} days - rotate now "
                "(traigent auth login)[/red]"
            )
        if state == "warning":
            return (
                f"[yellow]expires in {days:.0f} days - plan a rotation "
                "(traigent auth login)[/yellow]"
            )
        if state == "ok":
            return f"expires in {days:.0f} days"
        return None

    def _print_env_api_key_status(self, creds: dict[str, Any] | None) -> None:
        env_table = Table(show_header=False, box=None)
        env_table.add_column("Field", style="cyan")
        env_table.add_column("Value")
        env_table.add_row(
            "Status", "[green]✅ Authenticated (environment variable)[/green]"
        )
        env_table.add_row("TRAIGENT_API_KEY", "configured (environment variable)")
        env_table.add_row("Backend", self._backend_url or _BACKEND_NOT_CONFIGURED)
        if creds and creds.get("api_key"):
            env_table.add_row(
                "Stored login",
                "present but not used (environment variable takes precedence)",
            )
        console.print(env_table)
        console.print()

    @staticmethod
    def _print_jwt_only_notice(creds: dict[str, Any]) -> None:
        console.print("[yellow]ℹ️  Using JWT token authentication[/yellow]")
        if creds.get("refresh_token"):
            console.print("JWT tokens expire and need periodic refresh.\n")
        else:
            console.print(
                "No refresh token is available. Run "
                "[cyan]traigent auth login[/cyan] again before the JWT expires, "
                "or create an API key for long-lived authentication.\n"
            )

    def status(self) -> bool:
        """Show current authentication status.

        Returns:
            True if authenticated
        """
        console.print("\n[bold blue]🔍 Authentication Status[/bold blue]\n")

        # Check for stored credentials
        creds = self._load_stored_credentials()

        # Environment first (SDK #2227): TRAIGENT_API_KEY beats stored login,
        # the same order CredentialManager uses for every request.
        env_api_key = os.environ.get("TRAIGENT_API_KEY")
        if env_api_key:
            self._print_env_api_key_status(creds)
            return True

        if not creds and self._secure_store_locked():
            console.print(
                "[yellow]🔒 Credentials exist but the secure store is locked[/yellow]"
            )
            console.print(
                f"\nThe encrypted file {_secure_credential_store_path()} cannot be "
                "opened. Set TRAIGENT_MASTER_PASSWORD to the password it was "
                "created with, then run [cyan]traigent auth status[/cyan] again.\n"
            )
            return False

        if not creds:
            console.print("[yellow]❌ Not authenticated[/yellow]")
            console.print("\nRun [cyan]traigent auth login[/cyan] to authenticate.\n")
            return False

        # Display status
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        user = creds.get("user", {})
        table.add_row("Status", "[green]✅ Authenticated[/green]")
        table.add_row("Email", user.get("email", "Unknown"))
        table.add_row("User ID", str(user.get("id", "Unknown")))
        table.add_row(
            "Backend",
            creds.get("backend_url") or self._backend_url or _BACKEND_NOT_CONFIGURED,
        )

        if creds.get("api_key"):
            table.add_row("API Key", "configured")
            expiry_row = self._api_key_expiry_row(creds)
            if expiry_row:
                table.add_row("API Key expiry", expiry_row)
        else:
            table.add_row("Auth Type", "JWT Token")

        console.print(table)
        console.print()

        # Check if token needs refresh
        if creds.get("jwt_token") and not creds.get("api_key"):
            self._print_jwt_only_notice(creds)

        return True

    def _display_refresh_failure(self, refresh_result: Any) -> None:
        """Display detailed refresh failure information."""
        console.print("[red]❌ Refresh failed[/red]")
        if refresh_result.error_message:
            console.print(f"[yellow]Reason:[/yellow] {refresh_result.error_message}")
        if refresh_result.status:
            console.print(f"[yellow]Status:[/yellow] {refresh_result.status.value}")
        if refresh_result.retry_after:
            console.print(
                f"[yellow]Retry after:[/yellow] {refresh_result.retry_after:.1f}s"
            )
        console.print(
            "\nPlease run [cyan]traigent auth login[/cyan] to re-authenticate.\n"
        )

    async def _perform_token_refresh(self, creds: dict[str, Any]) -> bool:
        """Perform the actual token refresh operation.

        Returns:
            True if refresh successful
        """
        refresh_result = await self.auth_manager.refresh_authentication()

        if not refresh_result:
            self._display_refresh_failure(refresh_result)
            return False

        # Get the new token from auth manager
        auth_headers = await self.auth_manager.get_auth_headers()

        if "Authorization" not in auth_headers:
            console.print("[red]❌ No token received after refresh[/red]")
            console.print(
                "[dim]Auth headers returned without Authorization header[/dim]"
            )
            return False

        # Extract and store new JWT token
        jwt_token = auth_headers["Authorization"].replace("Bearer ", "")
        creds["jwt_token"] = jwt_token
        storage_location = self._save_credentials(creds)
        if storage_location is None:
            # Refreshed token but could not persist it: the on-disk/secure
            # credentials are now stale. Report failure instead of a clean
            # success (Traigent#1721).
            console.print(
                "[red]❌ Refresh failed: refreshed credentials could not be "
                "saved.[/red]"
            )
            console.print(
                "Set TRAIGENT_MASTER_PASSWORD and rerun, or run "
                "[cyan]traigent auth login[/cyan] again.\n"
            )
            return False

        console.print("[green]✅ Authentication refreshed successfully[/green]\n")
        return True

    async def refresh(self) -> bool:
        """Refresh authentication tokens.

        Returns:
            True if refresh successful
        """
        console.print("\n[bold blue]🔄 Refreshing Authentication[/bold blue]")

        # Load and validate current credentials
        creds = self._load_stored_credentials()
        if not creds:
            console.print("[red]❌ No credentials to refresh[/red]")
            console.print("Run [cyan]traigent auth login[/cyan] first.\n")
            return False

        # API keys don't need refresh
        if creds.get("api_key"):
            console.print("[green]✅ Using API key (no refresh needed)[/green]")
            return True

        # Check for refresh token
        if not creds.get("refresh_token"):
            console.print("[red]❌ No refresh token available[/red]")
            console.print(
                "This login only stored a short-lived JWT access token. "
                "Run [cyan]traigent auth login[/cyan] again or create an API key "
                "for long-lived authentication.\n"
            )
            return False

        try:
            return await self._perform_token_refresh(creds)
        except AuthenticationError as e:
            console.print(f"[red]❌ Refresh failed: {e}[/red]")
            console.print(MSG_RUN_LOGIN_AGAIN)
            return False
        except ValueError as e:
            console.print(f"[red]❌ Refresh failed (invalid data): {e}[/red]")
            console.print(MSG_RUN_LOGIN_AGAIN)
            return False
        except TimeoutError as e:
            console.print(f"[red]❌ Connection timed out: {e}[/red]")
            console.print("\nThe backend server took too long to respond.")
            console.print(MSG_CHECK_NETWORK)
            return False
        except NETWORK_ERRORS as e:
            error_type = type(e).__name__
            console.print(f"[red]❌ Refresh failed ({error_type}): {e}[/red]")
            console.print(MSG_RUN_LOGIN_AGAIN)
            return False

    def _configure_backend_url(self) -> tuple[bool, str | None]:
        """Prompt for a backend URL; returns ``(proceed, current_backend)``."""
        try:
            current_backend: str | None = BackendConfig.get_cloud_backend_url()
        except EndpointNotConfiguredError:
            current_backend = None

        if current_backend is None:
            # Strict-endpoint (self-hosted) mode with nothing configured yet:
            # there is no vendor default to keep, so the endpoint is required.
            console.print(
                "[yellow]No backend configured (strict-endpoint mode).[/yellow]"
            )
            change_backend = "y"
        else:
            console.print(f"Current backend: [cyan]{current_backend}[/cyan]")
            change_backend = Prompt.ask(
                "Change backend URL?", choices=["y", "n"], default="n"
            )

        if change_backend.lower() == "y":
            if current_backend is None:
                new_backend = Prompt.ask("Backend URL").strip()
                try:
                    new_backend = validate_cloud_base_url(
                        new_backend.rstrip("/"), purpose=_AUTH_CLI_REQUEST
                    )
                except ValueError as exc:
                    console.print(f"[red]❌ Invalid backend URL: {exc}[/red]\n")
                    return False, current_backend
                current_backend = new_backend
            else:
                new_backend = Prompt.ask("Backend URL", default=current_backend)

            console.print("\nTo use this backend, set the environment variable:")
            console.print(f"[cyan]export TRAIGENT_BACKEND_URL={new_backend}[/cyan]")
            console.print("\nOr add to your .env file:")
            console.print(f"[cyan]TRAIGENT_BACKEND_URL={new_backend}[/cyan]\n")
        return True, current_backend

    def _configure_auth_method(
        self, auth_method: str, current_backend: str | None
    ) -> None:
        if auth_method == "1":
            console.print("\nRun [cyan]traigent auth login[/cyan] to authenticate.\n")
        elif auth_method == "2":
            api_key = Prompt.ask("Enter API Key", password=True)
            credentials = {"api_key": api_key, "backend_url": current_backend}
            storage_location = self._save_credentials(credentials)
            if storage_location:
                console.print("[green]✅ API key stored securely[/green]")
                console.print("[dim]Location: encrypted local credential store[/dim]\n")
            else:
                console.print("[red]❌ Failed to store API key[/red]\n")
        else:
            console.print("\nSet the environment variable:")
            console.print("[cyan]export TRAIGENT_API_KEY=your_api_key_here[/cyan]\n")

    def configure(self) -> bool:
        """Interactive configuration wizard.

        Returns:
            True if configuration successful
        """
        console.print("\n[bold blue]⚙️  Traigent Configuration[/bold blue]")
        console.print("Configure your Traigent SDK settings.\n")

        # Backend URL configuration
        proceed, current_backend = self._configure_backend_url()
        if not proceed:
            return False

        # Authentication method
        console.print("\n[bold]Authentication Methods:[/bold]")
        console.print("1. Email/Password (recommended) - Get long-lived API key")
        console.print("2. API Key - Use existing API key")
        console.print(
            "3. Environment Variable - Use TRAIGENT_API_KEY from environment\n"
        )

        auth_method = Prompt.ask(
            "Select authentication method", choices=["1", "2", "3"], default="1"
        )

        self._configure_auth_method(auth_method, current_backend)

        return True


@click.group()
def auth() -> None:
    """Manage Traigent authentication.

    Modern authentication management for Traigent SDK, following patterns
    from GitHub CLI and AWS CLI.

    Examples:
        traigent auth login              # Interactive login
        traigent auth status            # Check authentication status
        traigent auth logout            # Logout and clear credentials
        traigent auth refresh           # Refresh authentication tokens
        traigent auth configure         # Configure authentication settings
    """
    # No load_dotenv() here: this module imports traigent.utils.env_config,
    # whose import runs _load_dotenv_files(). That loader honours
    # TRAIGENT_SKIP_DOTENV and stops at the project boundary (#1830). A bare
    # load_dotenv() searches upward to `/` (from this file's directory when run
    # as a script or console entry point, from cwd in a REPL or under a tracer
    # such as a debugger or coverage), so it
    # bypassed both and could load an unrelated ancestor's `.env`.


@auth.command()
@click.option("--email", "-e", help="Email address")
@click.option("--non-interactive", is_flag=True, help="Non-interactive mode")
@click.option(
    "--backend-url",
    default=None,
    help="Backend URL to authenticate against",
)
def login(
    email: str | None,
    non_interactive: bool,
    backend_url: str | None,
) -> None:
    """Authenticate with Traigent backend.

    This command will:
    1. Prompt for email and password
    2. Authenticate with the backend
    3. Generate a long-lived API key
    4. Store credentials securely

    Examples:
        traigent auth login
        traigent auth login --email user@example.com
        traigent auth login --backend-url https://your-backend.example.com
    """
    cli = TraigentAuthCLI(backend_url_override=backend_url)
    success = asyncio.run(cli.login(email, non_interactive))
    sys.exit(0 if success else 1)


@auth.command("device-login")
@click.option(
    "--backend-url",
    default=None,
    help="Backend URL to authenticate against",
)
@click.option(
    "--env-file",
    default=".env",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Local .env file to update when --write-env is set.",
)
@click.option(
    "--write-env",
    is_flag=True,
    help="Also write project-scoped credentials to a local .env file.",
)
@click.option(
    "--no-env-file",
    is_flag=True,
    hidden=True,
)
def device_login(
    backend_url: str | None,
    env_file: Path,
    write_env: bool,
    no_env_file: bool,
) -> None:
    """Authenticate through browser device-code login.

    The command prints a browser URL and one-time code, then polls until
    the backend returns a project-scoped API key or the request expires.
    """
    save_env_file = False
    write_env_explicit = False
    if write_env:
        save_env_file = True
        write_env_explicit = True
    elif not no_env_file and sys.stdin.isatty():
        save_env_file = click.confirm(
            "Write project credentials to this directory's .env file?",
            default=False,
        )

    cli = TraigentAuthCLI(backend_url_override=backend_url)
    success = asyncio.run(
        cli.device_login(
            save_env_file=save_env_file,
            env_file=env_file,
            write_env_explicit=write_env_explicit,
        )
    )
    sys.exit(0 if success else 1)


@auth.command()
def logout() -> None:
    """Logout and clear stored credentials.

    This command will:
    1. Clear locally stored credentials
    2. Optionally revoke API keys on the backend

    Examples:
        traigent auth logout
    """
    cli = TraigentAuthCLI()
    success = asyncio.run(cli.logout())
    sys.exit(0 if success else 1)


@auth.command()
def status() -> None:
    """Show current authentication status.

    This command will display:
    - Authentication status
    - User information
    - Backend URL
    - Token type (API key or JWT)

    Examples:
        traigent auth status
    """
    cli = TraigentAuthCLI()
    authenticated = cli.status()
    sys.exit(0 if authenticated else 1)


@auth.command()
def refresh() -> None:
    """Refresh authentication tokens.

    This command will:
    1. Check if refresh is needed
    2. Refresh JWT tokens if expired
    3. Update stored credentials

    Note: API keys don't need refresh.

    Examples:
        traigent auth refresh
    """
    cli = TraigentAuthCLI()
    success = asyncio.run(cli.refresh())
    sys.exit(0 if success else 1)


@auth.command()
def configure() -> None:
    """Configure Traigent authentication settings.

    Interactive configuration wizard for:
    - Backend URL
    - Authentication method
    - Credential storage

    Examples:
        traigent auth configure
    """
    cli = TraigentAuthCLI()
    success = cli.configure()
    sys.exit(0 if success else 1)


@auth.command()
@click.argument("key", required=False)
def whoami(key: str | None) -> None:
    """Validate an API key and show its associated information.

    Without KEY, validates the API key the SDK resolves: TRAIGENT_API_KEY
    first, then an API key stored by 'traigent auth login', then
    TRAIGENT_DEV_API_KEY only when development mode is enabled. Prints where
    the key came from, never the key itself.

    Passing KEY on the command line still works, but exposes the key in
    the process list and shell history.

    Examples:
        traigent auth whoami
    """
    console.print("\n[bold blue]🔍 API Key Information[/bold blue]\n")

    source: str | None
    if key:
        console.print(
            "[yellow]Warning: passing an API key as a command-line argument "
            "exposes it in the process list and shell history. Run "
            "'traigent auth whoami' without KEY instead.[/yellow]",
            soft_wrap=True,
        )
        source = "command-line argument"
    else:
        key, source = CredentialManager.get_api_key_with_source()
    if not key:
        console.print("[red]❌ Missing API key[/red]")
        console.print(
            "Set TRAIGENT_API_KEY in your environment, or run "
            "'traigent auth login' to store credentials. If you already logged "
            "in, set TRAIGENT_MASTER_PASSWORD so the stored credentials can be "
            "unlocked.\n",
            soft_wrap=True,
        )
        sys.exit(1)
    # Before the format check, so a malformed key still says where it came from.
    console.print(f"[dim]API key source: {source}[/dim]")

    # Validate format
    valid_prefixes = ("tg_", "uk_", "sk_", "ak_", "tk_")
    if not any(key.startswith(prefix) for prefix in valid_prefixes):
        console.print("[red]❌ Invalid API key format[/red]")
        console.print(
            "API keys should start with 'tg_', 'uk_', 'sk_', 'ak_', or 'tk_'\n"
        )
        sys.exit(1)

    # Check with backend
    backend_api_url = BackendConfig.get_cloud_api_url()
    console.print(f"[dim]Backend: {backend_api_url}[/dim]")

    success = asyncio.run(_check_api_key(backend_api_url, key))
    sys.exit(0 if success else 1)


_REDACTED_KEY_MARKER = "<redacted API key>"


def _redact_key(text: str, api_key: str | None) -> str:
    """Replace every occurrence of ``api_key`` in ``text`` with a fixed marker.

    whoami prints backend text (error bodies, exception messages, key metadata)
    that can echo the key it was sent, e.g. a 422 validation error's ``input``.
    """
    if not api_key:
        return text
    representations = {
        api_key,
        json.dumps(api_key, ensure_ascii=True)[1:-1],
        json.dumps(api_key, ensure_ascii=False)[1:-1],
    }
    for representation in sorted(representations, key=len, reverse=True):
        text = text.replace(representation, _REDACTED_KEY_MARKER)
    return text


def _print_valid_key_info(
    key_data: dict[str, Any], *, api_key: str | None = None
) -> None:
    """Print a table with valid API key metadata."""

    def _shown(value: Any) -> Any:
        return _redact_key(value, api_key) if isinstance(value, str) else value

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Status", "[green]✅ Valid[/green]")
    table.add_row("Category", "authenticated")
    table.add_row("HTTP", "200")
    table.add_row("Key Name", _shown(key_data.get("key_name", "Unknown")))
    table.add_row("User ID", _shown(str(key_data.get("user_id", "Unknown"))))
    table.add_row("Created", _shown(key_data.get("created_at", "Unknown")))

    console.print(table)
    console.print()


#: Signals that a 403 came from the EDGE (Cloudflare/WAF) rather than from Traigent.
#: Same vocabulary the session path classifies on (core/backend_session_manager,
#: SessionCreationFailureClassification.EDGE_BLOCKED), kept in step deliberately --
#: two different answers for one HTTP response is the defect #1754 fixed there and
#: #1775 reports here.
#:
#: IMPORTED, not re-declared. The previous hand-synced copy meant `whoami` and a
#: session-creation failure could give two different answers for one HTTP response
#: -- the defect #1754 fixed on the session path and #1775 reports here. One list,
#: so they cannot drift.
#:
#: These signals are now a best-effort IMPROVEMENT rather than a correctness
#: dependency: anything unrecognised falls through to "indeterminate", which
#: asserts no remediation. The original list was Cloudflare-specific AND
#: load-bearing, so every other vendor fell through to a confident
#: "grant the scope rather than rotating the key".
_EDGE_BLOCK_SIGNALS: tuple[str, ...] = _SESSION_EDGE_BLOCK_SIGNALS

_ERROR_STATUS_MAP: dict[int | str, tuple[str, str]] = {
    # 401 and 403 used to share one message. #1754 (PR #1762) split them on the
    # session path; #1775 is the same collapse still live on the CLI. A 403 is NOT a
    # bad key -- it is a good key lacking a scope, or an edge block -- and telling the
    # user to rotate a perfectly valid credential is the wrong remediation.
    401: ("[red]❌ Invalid or expired API key[/red]", "authentication"),
    403: ("[red]❌ API key lacks the required scope[/red]", "authorization"),
    404: ("[red]❌ Backend endpoint mismatch[/red]", "backend_endpoint_mismatch"),
    408: ("[red]❌ Backend request timed out[/red]", "timeout"),
    409: ("[red]❌ Backend reported a request conflict[/red]", "backend_conflict"),
    429: ("[red]❌ Backend rate limit exceeded[/red]", "rate_limited"),
    "5xx": ("[red]❌ Backend server error[/red]", "server_error"),
}


#: Response HEADERS that identify a Cloudflare-served response. These were
#: originally searched for in the body, where they essentially never appear --
#: `cf-ray` and `cf-mitigated` are headers, not page text.
_EDGE_HEADER_NAMES: tuple[str, ...] = ("cf-ray", "cf-mitigated", "x-amzn-waf-action")

#: A Traigent 403 is JSON from our own API. Rather than trying to enumerate every
#: edge vendor on earth -- Cloudflare, AWS WAF, Akamai, API Gateway, Fastly, the
#: next one -- classify on the POSITIVE signal we control and treat everything else
#: as unknown. Enumerating the negative fails open onto a confident wrong answer.
_TRAIGENT_ERROR_KEYS: frozenset[str] = frozenset(
    {"detail", "error", "error_code", "message", "required_scope", "scopes"}
)


def _classify_403(body: str, headers: Mapping[str, str] | None = None) -> str:
    """Classify a 403 as ``edge_blocked``, ``authorization`` or ``indeterminate``.

    Three outcomes, not two. The two-outcome version defaulted everything it did not
    recognise to "authorization", whose remediation is "grant the scope rather than
    rotating the key" -- a *confidently wrong* instruction for an AWS WAF, Akamai or
    API Gateway block, and strictly worse than the ambiguous message it replaced.
    An honest "could not tell" is better than a confident wrong turn.

    Order matters: a Cloudflare response header is decisive, because only the edge
    can set it. Body text is weaker evidence -- a genuine Traigent 403 can *mention*
    cloudflare (a link to /troubleshooting/cloudflare) or contain "waf" inside a
    request id -- so a body that parses as a Traigent API error wins over body-text
    signals.
    """
    try:
        present = {str(key).lower() for key in (headers or {})}
    except TypeError:
        # A caller passed something that is not iterable. Diagnostics must never be
        # the reason a command dies, so fall through to body classification.
        present = set()
    if present.intersection(_EDGE_HEADER_NAMES):
        return "edge_blocked"

    if _is_traigent_error_body(body):
        return "authorization"

    lowered = (body or "").lower()
    if any(_edge_signal_present(signal, lowered) for signal in _EDGE_BLOCK_SIGNALS):
        return "edge_blocked"

    # Reached Traigent? Unknown. Say so rather than inventing a remediation.
    return "indeterminate"


def _edge_signal_present(signal: str, lowered_body: str) -> bool:
    """Substring match, except for short tokens which need word boundaries.

    ``"waf" in body`` is a 3-character substring with no boundary, so a request id
    that happens to contain ``waf`` -- e.g. ``{"request_id": "Zx9wAFq1kP"}`` --
    flipped a real scope problem into "check your proxy".
    """
    if len(signal) <= 4 and signal.isalnum():
        return re.search(rf"\b{re.escape(signal)}\b", lowered_body) is not None
    return signal in lowered_body


def _body_says_key_invalid(body: str) -> bool:
    """The /keys/validate verdict shape: ``{"valid": false, ...}``."""
    try:
        parsed = json.loads((body or "").strip())
    except (ValueError, TypeError):  # silent-ok: not JSON means no verdict here
        return False
    return isinstance(parsed, dict) and parsed.get("valid") is False


def _is_traigent_error_body(body: str) -> bool:
    """True when the body parses as a JSON object shaped like our own API error.

    Deliberately narrow. A bare edge payload such as API Gateway's
    ``{"message": "Forbidden"}`` is JSON with one of our keys, so a key match alone
    is not enough -- it must also NOT read as a generic one-word denial.
    """
    try:
        parsed = json.loads((body or "").strip())
    except (ValueError, TypeError):  # silent-ok: a non-JSON body is not a
        # Traigent error body -- that IS the answer this predicate returns, not a
        # failure to report. Edge blocks are HTML precisely because they never
        # reached our API.
        return False
    if not isinstance(parsed, dict):
        return False
    if not _TRAIGENT_ERROR_KEYS.intersection(parsed):
        return False

    # API Gateway/ALB emit {"message": "Forbidden"} and nothing else. That is an
    # edge denial wearing JSON, not a Traigent error.
    if set(parsed) <= {"message"} and str(
        parsed.get("message", "")
    ).strip().lower() in {
        "forbidden",
        "access denied",
        "unauthorized",
    }:
        return False
    return True


def _looks_edge_blocked(
    body_preview: str, headers: Mapping[str, str] | None = None
) -> bool:
    """Back-compatible boolean wrapper over :func:`_classify_403`."""
    return _classify_403(body_preview, headers) == "edge_blocked"


def _key_check_from_response(
    status: int, body: str, headers: Mapping[str, str] | None = None
) -> KeyCheckResult:
    """Map a ``/keys/validate`` reply onto the tri-state (#2832)."""
    if status == 200:
        try:
            data = json.loads(body)
        except ValueError:
            return KeyCheckResult(
                KeyCheckState.INDETERMINATE, reason="unreadable reply"
            )
        if isinstance(data, dict):
            if data.get("valid") is True:
                payload = data.get("data")
                return KeyCheckResult(
                    KeyCheckState.VALID,
                    metadata=dict(payload) if isinstance(payload, dict) else {},
                )
            if data.get("valid") is False:
                return KeyCheckResult(KeyCheckState.REJECTED)
        return KeyCheckResult(KeyCheckState.INDETERMINATE, reason="unexpected reply")
    if status == 401:
        # Deleting stored credentials needs evidence that Traigent itself said
        # no: a proxy/SSO/captive-portal 401 (HTML, edge headers) says nothing
        # about the key.
        # Body evidence only: the production host sits behind Cloudflare, which
        # stamps edge headers on every response, so a header cannot outrank a
        # Traigent-shaped body.
        if _is_traigent_error_body(body) and _body_says_key_invalid(body):
            return KeyCheckResult(KeyCheckState.REJECTED)
        return KeyCheckResult(KeyCheckState.INDETERMINATE, reason="HTTP 401")
    if status == 403 and _classify_403(body, headers) == "authorization":
        return KeyCheckResult(KeyCheckState.REJECTED)
    # 429, 5xx, edge-blocked/indeterminate 403, 400/404 ...: not a verdict on the key.
    return KeyCheckResult(KeyCheckState.INDETERMINATE, reason=f"HTTP {status}")


def _print_error_status(
    status: int,
    body_preview: str,
    headers: Mapping[str, str] | None = None,
    full_body: str | None = None,
) -> None:
    """Print error details for a non-200 validation response.

    ``full_body`` is classified on; ``body_preview`` is only displayed. They differ
    because the preview is truncated to 220 characters, and an edge block's giveaway
    (a Cloudflare ray id, an Akamai reference) routinely sits past that cut.
    """
    verdict = None
    if status == 403:
        classified_body = full_body if full_body is not None else body_preview
        verdict = _classify_403(classified_body, headers)
    if verdict == "edge_blocked":
        # A Cloudflare/WAF 403 is neither a bad key nor a missing scope, and
        # "request the missing scope" would send the user somewhere useless.
        msg, category = (
            "[red]❌ Blocked at the network edge before reaching Traigent[/red]",
            "edge_blocked",
        )
    elif verdict == "indeterminate":
        # Neither signature matched. Do NOT fall through to "authorization" -- its
        # remediation asserts the key is fine and the scope is missing, which is a
        # confident wrong turn for an unrecognised edge vendor.
        msg, category = (
            "[red]❌ Forbidden (403) — could not tell whether this came from "
            "Traigent or from something in front of it[/red]",
            "forbidden_indeterminate",
        )
    elif status in _ERROR_STATUS_MAP:
        msg, category = _ERROR_STATUS_MAP[status]
    elif 500 <= status <= 599:
        msg, category = _ERROR_STATUS_MAP["5xx"]
    else:
        msg = "[red]❌ Unable to validate API key (unexpected backend response)[/red]"
        category = "backend_response_error"

    console.print(msg)
    console.print(f"[yellow]Category:[/yellow] {category}")
    if status == 404:
        console.print(
            "[yellow]Hint:[/yellow] Check TRAIGENT_BACKEND_URL / TRAIGENT_API_URL"
        )
    elif category == "authorization":
        console.print(
            "[yellow]Hint:[/yellow] The key is valid but lacks a required scope. "
            "Grant the scope rather than rotating the key."
        )
    elif category == "edge_blocked":
        console.print(
            "[yellow]Hint:[/yellow] The request did not reach Traigent. Check for a "
            "proxy, VPN or WAF between you and the backend; the key is unaffected."
        )
    elif category == "forbidden_indeterminate":
        console.print(
            "[yellow]Hint:[/yellow] Both are possible: the key may lack a required "
            "scope, or a proxy/VPN/WAF may have blocked the request before it "
            "reached Traigent. The response preview below is the best evidence — "
            "do not rotate the key on the strength of this alone."
        )
    console.print(f"[yellow]HTTP status:[/yellow] {status}")
    if body_preview:
        console.print(f"[dim]Response preview:[/dim] {body_preview}")
    console.print()


async def _check_api_key(backend_api_url: str, key: str) -> bool:
    """Validate an API key against the backend."""
    try:
        import aiohttp
    except ImportError:
        console.print(
            "[red]aiohttp dependency not installed; cannot validate API key.[/red]"
        )
        return False

    backend_api_url = validate_cloud_base_url(
        _resolve_api_base_url(backend_api_url).rstrip("/"),
        purpose=_AUTH_CLI_REQUEST,
    )
    url = _compose_api_url(backend_api_url, "keys/validate")
    headers = {
        "Accept": _APPLICATION_JSON,
        "Content-Type": _APPLICATION_JSON,
        "X-API-Key": key,
    }

    try:
        async with gate_session(
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                trust_env=True,
            )
        ) as session:
            async with session.post(
                url,
                headers=headers,
                json={"api_key": key},
            ) as response:
                if response.status == 200:
                    return _handle_200_response(
                        await _safe_parse_json(response), api_key=key
                    )

                # Classify on the WHOLE body, display only the preview: an edge
                # block's giveaway often sits past the 220-character cut. Redact
                # before cutting, so a key straddling the cut cannot show a prefix.
                full_body = (await response.text()).strip().replace("\n", " ")
                body_preview = _redact_key(full_body, key)[:220]
                _print_error_status(
                    response.status,
                    body_preview,
                    headers=response.headers,
                    full_body=full_body,
                )
                return False
    except (TimeoutError, aiohttp.ClientError) as exc:
        console.print("[red]❌ Cannot reach backend to validate API key[/red]")
        console.print("[yellow]Category:[/yellow] connectivity_error")
        console.print(
            f"[yellow]Error:[/yellow] {type(exc).__name__}: "
            f"{_redact_key(str(exc), key)}"
        )
        console.print()
        return False


async def _safe_parse_json(response: Any) -> dict[str, Any]:
    """Parse JSON response body, returning empty dict on failure."""
    try:
        data = await response.json(content_type=None)
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def _handle_200_response(data: dict[str, Any], *, api_key: str | None = None) -> bool:
    """Handle a 200 response from the key validation endpoint."""
    if data.get("valid"):
        _print_valid_key_info(data.get("data", {}), api_key=api_key)
        return True

    console.print("[red]❌ Invalid API key[/red]")
    console.print("[yellow]Category:[/yellow] authentication")
    console.print()
    return False
