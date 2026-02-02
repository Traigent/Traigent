#!/usr/bin/env python3
"""Run the complete Traigent Hybrid Mode demo.

This script:
1. Starts the Flask server in the background
2. Waits for it to be ready
3. Runs the test client to verify all endpoints
4. Stops the server

Usage:
    python run_demo.py
"""

import subprocess
import sys
import time
from pathlib import Path

import requests

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.absolute()
SERVER_URL = "http://localhost:8080"
STARTUP_TIMEOUT = 10  # seconds


def wait_for_server(url: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Wait for server to become ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/traigent/v1/health", timeout=1)
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def main():
    """Run the complete demo."""
    print("=" * 60)
    print("  Traigent Hybrid Mode Demo")
    print("=" * 60)
    print()

    # Start the server
    print("Starting Flask server...")
    server_process = subprocess.Popen(
        [sys.executable, str(SCRIPT_DIR / "app.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Wait for server to be ready
        print(f"Waiting for server at {SERVER_URL}...")
        if not wait_for_server(SERVER_URL):
            print("\nERROR: Server failed to start within timeout")
            print("\nServer output:")
            server_process.terminate()
            stdout, _ = server_process.communicate(timeout=2)
            print(stdout)
            sys.exit(1)

        print("Server is ready!")
        print()

        # Run the test client
        print("=" * 60)
        print("  Running Tests")
        print("=" * 60)

        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "test_client.py"), SERVER_URL],
            capture_output=False,
        )

        print()
        if result.returncode == 0:
            print("=" * 60)
            print("  Demo completed successfully!")
            print("=" * 60)
            print()
            print("Your Flask server correctly implements the Traigent Hybrid API.")
            print()
            print("Next steps:")
            print("  1. Review the code in app.py to understand the implementation")
            print("  2. Customize the tunables, execute, and evaluate functions")
            print("  3. Integrate with Traigent using hybrid_api execution mode")
            print()
        else:
            print("=" * 60)
            print("  Demo failed!")
            print("=" * 60)
            sys.exit(1)

    finally:
        # Stop the server
        print("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
