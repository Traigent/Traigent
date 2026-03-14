#!/usr/bin/env python3
"""Generate Traigent API key from email/password.

Usage:
    # Via command line args:
    python scripts/auth/get_api_key.py --email user@example.com --password yourpass

    # Via environment variables:
    export TRAIGENT_AUTH_EMAIL=user@example.com
    export TRAIGENT_AUTH_PASSWORD=yourpass
    python scripts/auth/get_api_key.py

    # Mixed (CLI args take precedence):
    export TRAIGENT_AUTH_PASSWORD=yourpass
    python scripts/auth/get_api_key.py --email user@example.com
"""

import argparse
import ipaddress
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

# Auto-load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    # Look for .env in the repo root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
except ImportError:
    # python-dotenv not installed, rely on shell environment
    pass


def normalize_backend_url(backend_url: str) -> str:
    """Validate and normalize the backend URL before issuing HTTP requests."""
    candidate = backend_url.strip()
    parsed = urlparse(candidate)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Backend URL must start with http:// or https://")
    if not parsed.netloc:
        raise ValueError("Backend URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("Backend URL must not include embedded credentials")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError("Backend URL must not include params, query strings, or fragments")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Backend URL must include a hostname")

    normalized_host = hostname.lower()
    is_localhost = normalized_host in {"localhost", "127.0.0.1", "::1"}

    if not is_localhost and parsed.scheme != "https":
        raise ValueError("Non-local backend URLs must use https")

    if normalized_host.endswith(".local"):
        raise ValueError("Backend URL must not target .local hosts")

    try:
        ip_addr = ipaddress.ip_address(normalized_host)
    except ValueError:
        ip_addr = None

    if ip_addr is not None:
        if ip_addr.is_private or ip_addr.is_loopback or ip_addr.is_link_local:
            if not is_localhost:
                raise ValueError("Backend URL must not target private or loopback IPs")
        if ip_addr.is_multicast or ip_addr.is_reserved or ip_addr.is_unspecified:
            raise ValueError("Backend URL must not target multicast, reserved, or unspecified IPs")
    elif not is_localhost:
        try:
            addr_infos = socket.getaddrinfo(normalized_host, None)
        except socket.gaierror:
            addr_infos = []
        for _family, _socktype, _proto, _canon, sockaddr in addr_infos:
            try:
                resolved_ip = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                continue
            if (
                resolved_ip.is_private
                or resolved_ip.is_loopback
                or resolved_ip.is_link_local
                or resolved_ip.is_multicast
                or resolved_ip.is_reserved
                or resolved_ip.is_unspecified
            ):
                raise ValueError("Backend URL must not resolve to private or unsafe IPs")

    normalized_path = parsed.path.rstrip("/")
    return urlunparse(
        parsed._replace(
            path=normalized_path,
            params="",
            query="",
            fragment="",
        )
    )


def get_api_key(email: str, password: str, backend_url: str, verbose: bool = True) -> str:
    """Authenticate and create an API key.

    Args:
        email: User email
        password: User password
        backend_url: Backend API URL
        verbose: Whether to print detailed response info

    Returns:
        The generated API key

    Raises:
        Exception: If authentication or API key creation fails
    """
    backend_url = normalize_backend_url(backend_url)

    # Step 1: Login to get JWT token
    login_url = f"{backend_url}/api/v1/auth/login"
    print(f"\n{'='*60}")
    print("STEP 1: LOGIN")
    print(f"{'='*60}")
    print(f"URL: {login_url}")
    print(f"Email: {email}")
    print("Sending request...")

    login_resp = requests.post(
        login_url,
        json={"email": email, "password": password},
        timeout=30,
        allow_redirects=False,
    )

    print(f"\n--- Response ---")
    print(f"Status Code: {login_resp.status_code}")
    print(f"Headers: {dict(login_resp.headers)}")

    try:
        login_data = login_resp.json()
        if verbose:
            import json
            print(f"Body (JSON):\n{json.dumps(login_data, indent=2)}")
    except Exception:
        print(f"Body (raw): {login_resp.text}")
        raise Exception(f"Login failed: {login_resp.status_code} - {login_resp.text}")

    if login_resp.status_code != 200:
        raise Exception(f"Login failed: {login_resp.status_code} - {login_resp.text}")

    if not login_data.get("success"):
        raise Exception(f"Login failed: {login_data.get('error', 'Unknown error')}")

    token = login_data["data"]["access_token"]
    # Show truncated token for security
    token_preview = f"{token[:20]}...{token[-10:]}" if len(token) > 30 else token
    print(f"\n✅ Login successful!")
    print(f"JWT Token: {token_preview}")

    # Step 2: Create API key
    api_key_url = f"{backend_url}/api/v1/keys"
    print(f"\n{'='*60}")
    print("STEP 2: CREATE API KEY")
    print(f"{'='*60}")
    print(f"URL: {api_key_url}")
    print("Sending request...")

    key_resp = requests.post(
        api_key_url,
        headers={"Authorization": f"Bearer {token}"},
        json={"key_name": "CLI Generated Key"},
        timeout=30,
        allow_redirects=False,
    )

    print(f"\n--- Response ---")
    print(f"Status Code: {key_resp.status_code}")
    print(f"Headers: {dict(key_resp.headers)}")

    try:
        key_data = key_resp.json()
        if verbose:
            import json
            print(f"Body (JSON):\n{json.dumps(key_data, indent=2)}")
    except Exception:
        print(f"Body (raw): {key_resp.text}")
        raise Exception(f"API key creation failed: {key_resp.status_code} - {key_resp.text}")

    if key_resp.status_code != 201:
        raise Exception(
            f"API key creation failed: {key_resp.status_code} - {key_resp.text}"
        )

    api_key = key_data["data"]["key"]
    print(f"\n✅ API key created successfully!")

    return api_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Traigent API key from email/password"
    )
    parser.add_argument(
        "--email",
        "-e",
        help="Email address (or set TRAIGENT_AUTH_EMAIL env var)",
    )
    parser.add_argument(
        "--password",
        "-p",
        help="Password (or set TRAIGENT_AUTH_PASSWORD env var)",
    )
    parser.add_argument(
        "--backend-url",
        "-b",
        help="Backend URL (or set TRAIGENT_BACKEND_URL env var)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Show full response bodies (default: True)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show the API key (minimal output)",
    )

    args = parser.parse_args()

    # Get values from args or env vars (args take precedence)
    email = args.email or os.environ.get("TRAIGENT_AUTH_EMAIL")
    password = args.password or os.environ.get("TRAIGENT_AUTH_PASSWORD")
    backend_url = (
        args.backend_url
        or os.environ.get("TRAIGENT_BACKEND_URL")
        or "https://api.traigent.ai"
    )

    # Validate required fields
    if not email:
        print("Error: Email required. Use --email or set TRAIGENT_AUTH_EMAIL")
        sys.exit(1)

    if not password:
        print("Error: Password required. Use --password or set TRAIGENT_AUTH_PASSWORD")
        sys.exit(1)

    verbose = not args.quiet

    try:
        api_key = get_api_key(email, password, backend_url, verbose=verbose)

        if args.quiet:
            print(api_key)
        else:
            print(f"\n{'=' * 60}")
            print("SUCCESS!")
            print(f"{'=' * 60}")
            print(f"API Key: {api_key}")
            print(f"\nTo use this key, add to your .env file:")
            print(f"TRAIGENT_API_KEY={api_key}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
