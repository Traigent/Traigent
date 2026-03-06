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
from rich.prompt import Prompt
from rich.table import Table

from traigent.cloud.auth import AuthManager
from traigent.config.backend_config import BackendConfig
from traigent.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)

# Constants
TRAIGENT_CONFIG_DIR = Path.home() / ".traigent"
CREDENTIALS_FILE = TRAIGENT_CONFIG_DIR / "credentials.json"
KEYRING_SERVICE = "traigent-sdk"
KEYRING_ACCOUNT = "default"


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
            except Exception as e:
                logger.debug(f"Failed to load credentials file: {e}")

        return None

    def _save_credentials(self, credentials: dict[str, Any]) -> bool:
        """Save credentials securely.

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully
        """
        # Try keyring first (most secure) if available
        if KEYRING_AVAILABLE and keyring is not None:
            try:
                keyring.set_password(
                    KEYRING_SERVICE, KEYRING_ACCOUNT, json.dumps(credentials)
                )
                logger.debug("Credentials saved to keyring")
                return True
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
            return True
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            return False

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
            except Exception as e:
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
            Exception: If authentication fails
        """
        import aiohttp

        # Step 1: Direct login call for better error visibility
        login_url = f"{self.backend_api_url}/auth/login"
        console.print(f"[dim]POST {login_url}[/dim]")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                login_url,
                json={"email": email, "password": password},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response_text = await response.text()

                # Show response details on failure
                if response.status != 200:
                    console.print("\n[red]--- Backend Response ---[/red]")
                    console.print(f"[red]Status Code: {response.status}[/red]")
                    safe_headers = {
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() in ("content-type", "x-request-id", "x-trace-id")
                    }
                    console.print(f"[red]Headers: {safe_headers}[/red]")
                    console.print(f"[red]Body: {response_text}[/red]")
                    raise Exception(
                        f"Authentication failed (HTTP {response.status})"
                    ) from None

                try:
                    login_data = json.loads(response_text)
                except json.JSONDecodeError:
                    console.print("\n[red]--- Backend Response ---[/red]")
                    console.print(f"[red]Status Code: {response.status}[/red]")
                    console.print(f"[red]Body (not JSON): {response_text}[/red]")
                    raise Exception("Invalid JSON response from backend") from None

                if not login_data.get("success"):
                    console.print("\n[red]--- Backend Response ---[/red]")
                    console.print(
                        f"[red]Body: {json.dumps(login_data, indent=2)}[/red]"
                    )
                    error_msg = login_data.get("error", "Unknown error")
                    raise Exception(f"Authentication failed: {error_msg}") from None

                token_data = login_data.get("data", {})
                jwt_token = token_data.get("access_token")

                if not jwt_token:
                    console.print("\n[red]--- Backend Response ---[/red]")
                    console.print(
                        f"[red]Body: {json.dumps(login_data, indent=2)}[/red]"
                    )
                    raise Exception("No access_token in response") from None

                # Show truncated token
                console.print(f"[green]✓ JWT Token received[/green]")

        # Step 2: Create API key using JWT token
        api_key = None
        api_key_url = f"{self.backend_api_url}/api-keys"
        console.print(f"\n[dim]POST {api_key_url}[/dim]")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_key_url,
                json={
                    "name": "Traigent SDK CLI",
                    "description": "Generated by traigent auth login",
                },
                headers={
                    "Authorization": f"Bearer {jwt_token}",
                    "Content-Type": "application/json",
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

        # Get email from arg, env var, or prompt
        if not email:
            email = os.environ.get("TRAIGENT_AUTH_EMAIL")
        if not email:
            if non_interactive:
                console.print(
                    "[red]Email required. Use --email or set TRAIGENT_AUTH_EMAIL[/red]"
                )
                return False
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
                return False
            from getpass import getpass

            password = getpass("Password: ")
        else:
            console.print("Using password from: [cyan]TRAIGENT_AUTH_PASSWORD[/cyan]")

        try:
            # Authenticate
            console.print("\n[yellow]Authenticating...[/yellow]")
            credentials = await self._authenticate_with_backend(
                email, password  # type: ignore[arg-type]
            )

            # Store credentials
            self._save_credentials(credentials)

            # Success message
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

            console.print(f"\nCredentials stored in: [cyan]{self.config_dir}[/cyan]")
            console.print(
                "You can now use Traigent SDK with backend tracking enabled.\n"
            )

            return True

        except Exception as e:
            console.print(f"\n[red]❌ Authentication failed: {e}[/red]")
            console.print("\nPlease check your credentials and try again.")
            console.print(
                "If you don't have an account, visit: https://traigent.ai/signup\n"
            )
            return False

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

    async def refresh(self) -> bool:
        """Refresh authentication tokens.

        Returns:
            True if refresh successful
        """
        console.print("\n[bold blue]🔄 Refreshing Authentication[/bold blue]")

        # Load current credentials
        creds = self._load_stored_credentials()
        if not creds:
            console.print("[red]❌ No credentials to refresh[/red]")
            console.print("Run [cyan]traigent auth login[/cyan] first.\n")
            return False

        # If using API key, no refresh needed
        if creds.get("api_key"):
            console.print("[green]✅ Using API key (no refresh needed)[/green]")
            return True

        # Refresh JWT token
        if not creds.get("refresh_token"):
            console.print("[red]❌ No refresh token available[/red]")
            console.print("Please run [cyan]traigent auth login[/cyan] again.\n")
            return False

        try:
            # Use SecureAuthManager for refresh with resilient client
            refresh_success = await self.auth_manager.refresh_authentication()

            if refresh_success:
                # Get the new token from auth manager
                auth_headers = await self.auth_manager.get_auth_headers()

                if "Authorization" in auth_headers:
                    # Extract new JWT token
                    jwt_token = auth_headers["Authorization"].replace("Bearer ", "")

                    # Update stored credentials
                    creds["jwt_token"] = jwt_token
                    self._save_credentials(creds)

                    console.print(
                        "[green]✅ Authentication refreshed successfully[/green]\n"
                    )
                    return True
                else:
                    console.print("[red]❌ No token received after refresh[/red]")
                    return False
            else:
                console.print("[red]❌ Refresh failed[/red]")
                return False

        except Exception as e:
            console.print(f"[red]❌ Refresh failed: {e}[/red]")
            console.print("Please run [cyan]traigent auth login[/cyan] again.\n")
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
            if self._save_credentials(credentials):
                console.print("[green]✅ API key stored securely[/green]\n")
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

    async def check_key() -> bool:
        try:
            import aiohttp
        except ImportError:
            console.print(
                "[red]aiohttp dependency not installed; cannot validate API key.[/red]"
            )
            return False

        async with aiohttp.ClientSession() as session:
            # Try to use the API key
            url = f"{backend_api_url}/user/me"
            headers = {"X-API-Key": key}

            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    user = data.get("data", {})

                    table = Table(show_header=False, box=None)
                    table.add_column("Field", style="cyan")
                    table.add_column("Value")

                    table.add_row("Status", "[green]✅ Valid[/green]")
                    table.add_row("Email", user.get("email", "Unknown"))
                    table.add_row("Name", user.get("name", "Unknown"))
                    table.add_row("Organization", user.get("organization", "N/A"))

                    console.print(table)
                    console.print()
                    return True
                else:
                    console.print("[red]❌ Invalid or expired API key[/red]\n")
                    return False

    success = asyncio.run(check_key())
    sys.exit(0 if success else 1)
