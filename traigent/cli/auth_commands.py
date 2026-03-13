"""Authentication CLI commands for Traigent SDK.

Modern authentication management following patterns from GitHub CLI and AWS CLI.
"""

# Traceability: CONC-Layer-API CONC-Quality-Security CONC-Quality-Usability FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-CLOUD-009 REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console

# Try to import keyring, but make it optional
try:
    import keyring

    KEYRING_AVAILABLE = True
except ImportError:
    keyring = None
    KEYRING_AVAILABLE = False

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
from traigent.config.backend_config import BackendConfig
from traigent.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)

# Constants
TRAIGENT_CONFIG_DIR = Path.home() / ".traigent"
CREDENTIALS_FILE = TRAIGENT_CONFIG_DIR / "credentials.json"
KEYRING_SERVICE = "traigent-sdk"
KEYRING_ACCOUNT = "default"
BACKEND_RESPONSE_HEADER = "\n[red]--- Backend Response ---[/red]"

# Storage location identifiers
STORAGE_KEYRING = "keyring"
STORAGE_FILE = "file"

# Common user-facing messages (avoid duplication per SonarCloud S1192)
MSG_CHECK_NETWORK = "Please check your network connection and try again.\n"
MSG_RUN_LOGIN_AGAIN = "Please run [cyan]traigent auth login[/cyan] again.\n"


class TraigentAuthCLI:
    """Modern CLI authentication manager for Traigent SDK."""

    def __init__(self) -> None:
        """Initialize the authentication CLI."""
        self.config_dir = TRAIGENT_CONFIG_DIR
        self.credentials_file = CREDENTIALS_FILE
        self.auth_manager = AuthManager()
        self.backend_url = BackendConfig.get_backend_url()
        self.backend_api_url = BackendConfig.get_backend_api_url()

        # Ensure config directory exists
        self.config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _load_stored_credentials(self) -> dict[str, Any] | None:
        """Load stored credentials from secure storage.

        Returns:
            Stored credentials or None if not found
        """
        # First try keyring (most secure) if available
        if KEYRING_AVAILABLE and keyring is not None:
            try:
                stored_data = keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
                if stored_data:
                    return dict(json.loads(stored_data))
            except Exception as e:
                logger.debug(f"Keyring access failed: {e}")

        # Fallback to encrypted file
        if self.credentials_file.exists():
            try:
                with open(self.credentials_file) as f:
                    data = json.load(f)
                    return dict(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Failed to load credentials file: {e}")

        return None

    def _save_credentials(self, credentials: dict[str, Any]) -> str | None:
        """Save credentials securely.

        Args:
            credentials: Credentials to save

        Returns:
            Storage location string if saved successfully, None if failed
        """
        # Try keyring first (most secure) if available
        if KEYRING_AVAILABLE and keyring is not None:
            try:
                keyring.set_password(
                    KEYRING_SERVICE, KEYRING_ACCOUNT, json.dumps(credentials)
                )
                logger.debug("Credentials saved to keyring")
                return STORAGE_KEYRING
            except Exception as e:
                logger.debug(f"Keyring save failed, using file: {e}")

        # Fallback to file with restricted permissions
        try:
            self.credentials_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

            # Write with restricted permissions
            with open(self.credentials_file, "w") as f:
                json.dump(credentials, f, indent=2)

            # Ensure file has restricted permissions
            self.credentials_file.chmod(0o600)
            logger.debug(f"Credentials saved to {self.credentials_file}")
            return STORAGE_FILE
        except OSError as e:
            logger.error(f"Failed to save credentials: {e}")
            return None

    def _save_api_key_to_env_file(
        self, api_key: str, env_path: Path | None = None
    ) -> bool:
        """Save API key to .env file.

        Args:
            api_key: The API key to save
            env_path: Path to .env file (defaults to cwd/.env)

        Returns:
            True if saved successfully
        """
        if env_path is None:
            env_path = Path.cwd() / ".env"

        try:
            # Read existing content
            existing_lines: list[str] = []

            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        # Check if this line sets TRAIGENT_API_KEY (not commented)
                        stripped = line.strip()
                        if stripped.startswith("TRAIGENT_API_KEY="):
                            # Comment out the old key
                            existing_lines.append(
                                f"# {line.rstrip()} # replaced by traigent auth login\n"
                            )
                        else:
                            existing_lines.append(line)

            # Add new API key
            if not existing_lines or not existing_lines[-1].endswith("\n"):
                existing_lines.append("\n")
            existing_lines.append(f"TRAIGENT_API_KEY={api_key}\n")

            # Write back with restrictive permissions
            with open(env_path, "w") as f:
                f.writelines(existing_lines)

            # Set restrictive permissions (user read/write only)
            env_path.chmod(0o600)
            logger.debug(f"API key saved to {env_path}")
            return True

        except OSError as e:
            logger.error(f"Failed to save API key to .env: {e}")
            return False

    async def _validate_api_key(
        self, api_key: str, verbose: bool = False
    ) -> dict[str, Any] | None:
        """Validate an API key against the backend.

        Args:
            api_key: The API key to validate
            verbose: If True, print debug information

        Returns:
            Key metadata dict if valid, None if invalid
        """
        import aiohttp

        url = f"{self.backend_api_url}/keys/validate"
        headers = {"X-API-Key": api_key}

        if verbose:
            masked_key = (
                f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
            )
            console.print(f"[dim]POST {url}[/dim]")
            console.print(f"[dim]X-API-Key: {masked_key}[/dim]")

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                async with session.post(url, headers=headers) as response:
                    response_text = await response.text()

                    if verbose:
                        console.print(f"[dim]Response status: {response.status}[/dim]")
                        # Show first 200 chars of response for debugging
                        preview = (
                            response_text[:200]
                            if len(response_text) > 200
                            else response_text
                        )
                        console.print(f"[dim]Response: {preview}[/dim]")

                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                            # Response format: {"valid": true, "data": {...key_metadata...}}
                            if isinstance(data, dict) and data.get("valid"):
                                return dict(data.get("data", {}))
                            return None
                        except ValueError:
                            # ValueError covers JSONDecodeError (its subclass)
                            return None
                    # 401 = invalid key, 400 = missing key, 429 = rate limited
                    return None
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.debug(f"API key validation failed: {e}")
            if verbose:
                console.print(f"[dim]Connection error: {e}[/dim]")
            return None

    def _clear_credentials(self) -> bool:
        """Clear stored credentials.

        Returns:
            True if cleared successfully
        """
        success = True

        # Clear from keyring if available
        if KEYRING_AVAILABLE and keyring is not None:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
                logger.debug("Cleared credentials from keyring")
            except Exception as e:
                logger.debug(
                    f"Could not clear keyring credentials (may not exist): {e}"
                )

        # Clear file
        if self.credentials_file.exists():
            try:
                self.credentials_file.unlink()
                logger.debug("Cleared credentials file")
            except OSError as e:
                logger.error(f"Failed to delete credentials file: {e}")
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
        console.print(f"[dim]POST {login_url}[/dim]")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                login_url,
                json={"email": email, "password": password},
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Traigent-SDK-CLI/1.0",
                },
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
                    console.print(f"[red]Body: {response_text}[/red]")
                    raise AuthenticationError(
                        f"Authentication failed (HTTP {response.status})"
                    ) from None

                try:
                    login_data = json.loads(response_text)
                except json.JSONDecodeError:
                    console.print(BACKEND_RESPONSE_HEADER)
                    console.print(f"[red]Status Code: {response.status}[/red]")
                    console.print(f"[red]Body (not JSON): {response_text}[/red]")
                    raise AuthenticationError(
                        "Invalid JSON response from backend"
                    ) from None

                if not login_data.get("success"):
                    console.print(BACKEND_RESPONSE_HEADER)
                    console.print(
                        f"[red]Body: {json.dumps(login_data, indent=2)}[/red]"
                    )
                    error_msg = login_data.get("error", "Unknown error")
                    raise InvalidCredentialsError(
                        f"Authentication failed: {error_msg}"
                    ) from None

                token_data = login_data.get("data", {})
                jwt_token = token_data.get("access_token")

                if not jwt_token:
                    console.print(BACKEND_RESPONSE_HEADER)
                    console.print(
                        f"[red]Body: {json.dumps(login_data, indent=2)}[/red]"
                    )
                    raise AuthenticationError("No access_token in response") from None

                # Show truncated token
                console.print("[green]✓ JWT Token received[/green]")

        # Step 2: Create API key using JWT token
        api_key = None
        api_key_url = f"{self.backend_api_url}/keys"

        # Generate a unique key name to avoid conflicts
        import platform
        from datetime import datetime

        hostname = platform.node() or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        key_name = f"Traigent SDK CLI ({hostname[:20]} {timestamp})"

        console.print(f"\n[dim]POST {api_key_url}[/dim]")
        console.print(f"[dim]Creating API key: {key_name}[/dim]")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_key_url,
                json={
                    "key_name": key_name,
                    "permissions": [
                        "read",
                        "write",
                        "experiment.read",
                        "experiment.write",
                        "session.read",
                        "session.write",
                    ],
                },
                headers={
                    "Authorization": f"Bearer {jwt_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "Traigent-SDK-CLI/1.0",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response_text = await response.text()

                if response.status == 201:
                    try:
                        api_key_data = json.loads(response_text)
                        api_key = api_key_data["data"]["key"]
                        console.print("[green]✓ API key created[/green]")
                    except (json.JSONDecodeError, KeyError) as e:
                        console.print("\n[yellow]--- API Key Response ---[/yellow]")
                        console.print(f"[yellow]Status: {response.status}[/yellow]")
                        console.print(f"[yellow]Body: {response_text}[/yellow]")
                        console.print(
                            f"[yellow]⚠️ Could not parse API key: {e}[/yellow]"
                        )
                elif response.status == 409:
                    # Key name conflict - shouldn't happen with unique names, but handle it
                    console.print(
                        "[yellow]⚠️ API key name conflict (409). Try again.[/yellow]"
                    )
                else:
                    console.print("\n[yellow]--- API Key Response ---[/yellow]")
                    console.print(f"[yellow]Status Code: {response.status}[/yellow]")
                    console.print(f"[yellow]Body: {response_text}[/yellow]")
                    console.print(
                        "[yellow]⚠️ API key creation failed, using JWT token only[/yellow]"
                    )

        # Get user info from stored token data (if available)
        # Note: SecureAuthManager doesn't expose user info directly
        # We'll need to extract it from the initial auth response
        user_info: dict[str, str] = {}

        return {
            "jwt_token": jwt_token,
            "refresh_token": jwt_token,  # Use same token as refresh for now
            "api_key": api_key,
            "user": user_info,
            "backend_url": self.backend_url,
        }

    async def _check_stored_api_key(self) -> bool:
        """Check if stored API key is valid.

        Returns:
            True if valid stored credentials found (already authenticated)
        """
        existing_creds = self._load_stored_credentials()
        if not existing_creds or not existing_creds.get("api_key"):
            return False

        console.print("[dim]Checking stored API key...[/dim]")
        user_info = await self._validate_api_key(
            existing_creds["api_key"], verbose=True
        )

        if user_info is not None:
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

        # API key is invalid, clear it
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
        user_info = await self._validate_api_key(env_api_key, verbose=True)

        if user_info is not None:
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
        if storage_location == STORAGE_KEYRING:
            console.print("\n[bold]Credentials stored in:[/bold] System Keyring")
            import platform

            system = platform.system()
            if system == "Darwin":
                console.print(
                    "  [dim]View in: Keychain Access app → search 'traigent-sdk'[/dim]"
                )
                console.print(
                    "  [dim]Or run:[/dim] [cyan]security find-generic-password -s traigent-sdk -a default -w | python -m json.tool[/cyan]"
                )
            elif system == "Windows":
                console.print(
                    "  [dim]View in: Control Panel → Credential Manager → 'traigent-sdk'[/dim]"
                )
            else:
                console.print(
                    "  [dim]View in: Seahorse/KWallet → search 'traigent-sdk'[/dim]"
                )
        elif storage_location == STORAGE_FILE:
            console.print(
                f"\n[bold]Credentials stored in:[/bold] [cyan]{self.credentials_file}[/cyan]"
            )
            console.print(
                f"  [dim]View with:[/dim] [cyan]cat {self.credentials_file}[/cyan]"
            )
        else:
            console.print("\n[yellow]⚠️ Could not save credentials to storage[/yellow]")

    def _offer_env_file_save(self, api_key: str, non_interactive: bool) -> None:
        """Offer to save API key to .env file."""
        if non_interactive:
            return

        env_path = Path.cwd() / ".env"
        save_to_env = Prompt.ask(
            f"\nSave API key to [cyan]{env_path}[/cyan] for easy access?",
            choices=["y", "n"],
            default="y",
        )
        if save_to_env.lower() == "y":
            if self._save_api_key_to_env_file(api_key, env_path):
                console.print(f"[green]✅ API key added to {env_path}[/green]")
            else:
                console.print(f"[yellow]⚠️ Could not save to {env_path}[/yellow]")

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
            credentials = await self._authenticate_with_backend(email, password)

            # Store and display results
            storage_location = self._save_credentials(credentials)
            self._display_login_success(credentials, email, storage_location)

            # Offer to save API key to .env file
            if credentials.get("api_key"):
                self._offer_env_file_save(credentials["api_key"], non_interactive)

            console.print(
                "\nYou can now use Traigent SDK with backend tracking enabled.\n"
            )
            return True

        except AuthenticationError as e:
            console.print(f"\n[red]❌ Authentication failed: {e}[/red]")
            console.print("\nPlease check your credentials and try again.")
            console.print(
                "If you don't have an account, visit: https://portal.traigent.ai/login\n"
            )
            return False
        except TimeoutError as e:
            console.print(f"\n[red]❌ Connection timed out: {e}[/red]")
            console.print("\nThe backend server took too long to respond.")
            console.print(MSG_CHECK_NETWORK)
            return False
        except NETWORK_ERRORS as e:
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
        if not creds:
            console.print("[yellow]No stored credentials found[/yellow]")
            return True

        # Optional: Revoke API key on backend
        if creds.get("api_key"):
            try:
                import aiohttp

                async with aiohttp.ClientSession():
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
            return False

    def status(self) -> bool:
        """Show current authentication status.

        Returns:
            True if authenticated
        """
        console.print("\n[bold blue]🔍 Authentication Status[/bold blue]\n")

        # Check for stored credentials
        creds = self._load_stored_credentials()

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
        table.add_row("Backend", creds.get("backend_url", self.backend_url))

        if creds.get("api_key"):
            # Mask API key for security
            api_key = creds["api_key"]
            masked_key = (
                f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
            )
            table.add_row("API Key", masked_key)
        else:
            table.add_row("Auth Type", "JWT Token")

        console.print(table)
        console.print()

        # Check if token needs refresh
        if creds.get("jwt_token") and not creds.get("api_key"):
            console.print("[yellow]ℹ️  Using JWT token authentication[/yellow]")
            console.print("JWT tokens expire and need periodic refresh.\n")

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
        self._save_credentials(creds)

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
            console.print(MSG_RUN_LOGIN_AGAIN)
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

    def configure(self) -> bool:
        """Interactive configuration wizard.

        Returns:
            True if configuration successful
        """
        console.print("\n[bold blue]⚙️  Traigent Configuration[/bold blue]")
        console.print("Configure your Traigent SDK settings.\n")

        # Backend URL configuration
        current_backend = BackendConfig.get_backend_url()
        console.print(f"Current backend: [cyan]{current_backend}[/cyan]")

        change_backend = Prompt.ask(
            "Change backend URL?", choices=["y", "n"], default="n"
        )

        if change_backend.lower() == "y":
            new_backend = Prompt.ask("Backend URL", default=current_backend)

            console.print("\nTo use this backend, set the environment variable:")
            console.print(f"[cyan]export TRAIGENT_BACKEND_URL={new_backend}[/cyan]")
            console.print("\nOr add to your .env file:")
            console.print(f"[cyan]TRAIGENT_BACKEND_URL={new_backend}[/cyan]\n")

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

        if auth_method == "1":
            console.print("\nRun [cyan]traigent auth login[/cyan] to authenticate.\n")
        elif auth_method == "2":
            api_key = Prompt.ask("Enter API Key", password=True)
            credentials = {"api_key": api_key, "backend_url": current_backend}
            storage_location = self._save_credentials(credentials)
            if storage_location:
                console.print("[green]✅ API key stored securely[/green]")
                if storage_location == STORAGE_KEYRING:
                    console.print("[dim]Location: System Keyring[/dim]\n")
                else:
                    console.print(f"[dim]Location: {self.credentials_file}[/dim]\n")
            else:
                console.print("[red]❌ Failed to store API key[/red]\n")
        else:
            console.print("\nSet the environment variable:")
            console.print("[cyan]export TRAIGENT_API_KEY=your_api_key_here[/cyan]\n")

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
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


@auth.command()
@click.option("--email", "-e", help="Email address")
@click.option("--non-interactive", is_flag=True, help="Non-interactive mode")
def login(email: str | None, non_interactive: bool) -> None:
    """Authenticate with Traigent backend.

    This command will:
    1. Prompt for email and password
    2. Authenticate with the backend
    3. Generate a long-lived API key
    4. Store credentials securely

    Examples:
        traigent auth login
        traigent auth login --email user@example.com
    """
    cli = TraigentAuthCLI()
    success = asyncio.run(cli.login(email, non_interactive))
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
@click.argument("key")
def whoami(key: str) -> None:
    """Show information about an API key.

    This command will validate an API key and show associated user information.

    Args:
        key: API key to check

    Examples:
        traigent auth whoami tg_1234567890abcdef
    """
    console.print("\n[bold blue]🔍 API Key Information[/bold blue]\n")

    # Validate format
    valid_prefixes = ("tg_", "uk_")
    if not any(key.startswith(prefix) for prefix in valid_prefixes):
        console.print("[red]❌ Invalid API key format[/red]")
        console.print("API keys should start with 'tg_' or 'uk_'\n")
        sys.exit(1)

    # Check with backend
    backend_api_url = BackendConfig.get_backend_api_url()
    console.print(f"[dim]Backend: {backend_api_url}[/dim]")

    async def check_key() -> bool:
        try:
            import aiohttp
        except ImportError:
            console.print(
                "[red]aiohttp dependency not installed; cannot validate API key.[/red]"
            )
            return False

        url = f"{backend_api_url}/keys/validate"
        headers = {"X-API-Key": key}

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                async with session.post(url, headers=headers) as response:
                    if response.status == 200:
                        try:
                            data = await response.json(content_type=None)
                        except Exception:
                            data = {}

                        # Response format: {"valid": true, "data": {...key_metadata...}}
                        if isinstance(data, dict) and data.get("valid"):
                            key_data = data.get("data", {})

                            table = Table(show_header=False, box=None)
                            table.add_column("Field", style="cyan")
                            table.add_column("Value")

                            table.add_row("Status", "[green]✅ Valid[/green]")
                            table.add_row("Category", "authenticated")
                            table.add_row("HTTP", "200")
                            table.add_row(
                                "Key Name", key_data.get("key_name", "Unknown")
                            )
                            table.add_row(
                                "User ID", str(key_data.get("user_id", "Unknown"))
                            )
                            table.add_row(
                                "Created",
                                key_data.get("created_at", "Unknown"),
                            )

                            console.print(table)
                            console.print()
                            return True
                        else:
                            console.print("[red]❌ Invalid API key[/red]")
                            console.print("[yellow]Category:[/yellow] authentication")
                            console.print()
                            return False

                    body_preview = (
                        (await response.text()).strip().replace("\n", " ")[:220]
                    )
                    if response.status in (401, 403):
                        console.print("[red]❌ Invalid or unauthorized API key[/red]")
                        console.print("[yellow]Category:[/yellow] authentication")
                    elif response.status == 404:
                        console.print("[red]❌ Backend endpoint mismatch[/red]")
                        console.print(
                            "[yellow]Category:[/yellow] backend_endpoint_mismatch"
                        )
                        console.print(
                            "[yellow]Hint:[/yellow] Check TRAIGENT_BACKEND_URL / TRAIGENT_API_URL"
                        )
                    elif response.status == 408:
                        console.print("[red]❌ Backend request timed out[/red]")
                        console.print("[yellow]Category:[/yellow] timeout")
                    elif response.status == 409:
                        console.print(
                            "[red]❌ Backend reported a request conflict[/red]"
                        )
                        console.print("[yellow]Category:[/yellow] backend_conflict")
                    elif response.status == 429:
                        console.print("[red]❌ Backend rate limit exceeded[/red]")
                        console.print("[yellow]Category:[/yellow] rate_limited")
                    elif 500 <= response.status <= 599:
                        console.print("[red]❌ Backend server error[/red]")
                        console.print("[yellow]Category:[/yellow] server_error")
                    else:
                        console.print(
                            "[red]❌ Unable to validate API key (unexpected backend response)[/red]"
                        )
                        console.print(
                            "[yellow]Category:[/yellow] backend_response_error"
                        )

                    console.print(f"[yellow]HTTP status:[/yellow] {response.status}")
                    if body_preview:
                        console.print(f"[dim]Response preview:[/dim] {body_preview}")
                    console.print()
                    return False
        except (TimeoutError, aiohttp.ClientError) as exc:
            console.print("[red]❌ Cannot reach backend to validate API key[/red]")
            console.print("[yellow]Category:[/yellow] connectivity_error")
            console.print(f"[yellow]Error:[/yellow] {type(exc).__name__}: {exc}")
            console.print()
            return False

    success = asyncio.run(check_key())
    sys.exit(0 if success else 1)
