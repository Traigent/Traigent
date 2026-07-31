#!/usr/bin/env python3
"""Test the new CLI authentication system for Traigent SDK."""

import asyncio
import sys
from pathlib import Path

# Add the repo root to the path so `traigent` is importable. This must NOT be
# `tests/`: putting `tests/` on sys.path makes its subpackages importable as
# top-level modules, so `tests/mcp/` shadows the installed `mcp` distribution
# for every later import in the same session (breaking, e.g.,
# tests/unit/mcp/test_local_server.py's `from mcp import ClientSession`).
sys.path.insert(0, str(Path(__file__).parents[2]))

from traigent.cli.auth_commands import TraigentAuthCLI
from traigent.cloud.auth import get_auth_headers
from traigent.cloud.credential_manager import CredentialManager
from traigent.utils.logging import setup_logging

# Set up logging
setup_logging("INFO")


async def test_auth_flow():
    """Test the complete authentication flow."""
    print("=" * 60)
    print("Testing Traigent SDK Authentication System")
    print("=" * 60)

    # Step 1: Check initial status
    print("\n1. Checking initial authentication status...")
    cli = TraigentAuthCLI()
    is_authenticated = cli.status()
    print(f"   Initial authentication: {is_authenticated}")

    # Step 2: Check credential manager
    print("\n2. Testing credential manager...")
    if CredentialManager.is_authenticated():
        print("   Found API key: configured")
    else:
        print("   No API key found")

    # Step 3: Get auth headers
    print("\n3. Getting authentication headers...")
    headers = get_auth_headers()
    if headers:
        print("   Authentication headers: configured")
    else:
        print("   No authentication headers available")

    # Step 4: Check credentials details
    print("\n4. Getting credential details...")
    creds = CredentialManager.get_credentials()
    if creds:
        print("   Credentials: configured")
    else:
        print("   No credentials found")

    # Step 5: Test auth manager integration
    print("\n5. Testing AuthManager integration...")
    from traigent.cloud.auth import AuthManager

    auth_manager = AuthManager()
    if auth_manager.has_api_key():
        print("   AuthManager has API key: Yes")
    else:
        print("   AuthManager has no API key")

    print("\n" + "=" * 60)
    print("Authentication system test complete!")
    print("=" * 60)

    # Instructions for users
    if not is_authenticated:
        print("\nTo authenticate, run:")
        print("  traigent auth login")
        print("\nOr set environment variable:")
        print("  export TRAIGENT_API_KEY=your_api_key_here")
        print("\nFor more information:")
        print("  traigent auth --help")

    # Verify that CLI and manager objects were created successfully
    assert cli is not None, "TraigentAuthCLI should be created"
    assert auth_manager is not None, "AuthManager should be created"


if __name__ == "__main__":
    asyncio.run(test_auth_flow())
