"""Regression tests for Codex security review fixes.

These tests ensure the 5 security issues identified in the Codex review
remain fixed and don't regress:

1. CRITICAL: Deterministic fallback key in crypto_utils.py
2. CRITICAL: Sensitive trial payloads logged in plaintext
3. HIGH: ES256 tokens rejected due to RSA key size check
4. HIGH: create_session blocks inside running event loop
5. HIGH: Summary stats submission uses closed aiohttp session

Tests are designed to catch regressions by verifying the specific
fixes are in place.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
from unittest.mock import MagicMock, patch

import pytest


class TestFix1DeterministicFallbackKey:
    """Tests for Fix #1: Deterministic fallback key vulnerability.

    The original bug used uid/home path to derive a fallback encryption key,
    which was deterministic and guessable.

    The fix:
    - Fails fast in production if no key is provided
    - Uses secrets.token_hex(32) for random ephemeral key in development
    """

    def test_production_requires_explicit_key(self):
        """Production mode must fail if no encryption key is provided."""
        from traigent.security.crypto_utils import SecureCredentialStorage

        # Ensure we clear all environment variables that might affect the mode
        env_overrides = {
            "ENVIRONMENT": "production",
            "TRAIGENT_ENVIRONMENT": "",
            "TRAIGENT_ENV": "",
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            # Remove any existing key
            env_copy = os.environ.copy()
            env_copy.pop("TRAIGENT_ENCRYPTION_KEY", None)

            with patch.dict(os.environ, env_copy, clear=True):
                with pytest.raises(RuntimeError, match="required in production"):
                    SecureCredentialStorage()

    def test_production_with_prod_variant_requires_key(self):
        """'prod' environment variant must also require explicit key."""
        from traigent.security.crypto_utils import SecureCredentialStorage

        env_overrides = {
            "ENVIRONMENT": "prod",
            "TRAIGENT_ENVIRONMENT": "",
            "TRAIGENT_ENV": "",
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            env_copy = os.environ.copy()
            env_copy.pop("TRAIGENT_ENCRYPTION_KEY", None)

            with patch.dict(os.environ, env_copy, clear=True):
                with pytest.raises(RuntimeError, match="required in production"):
                    SecureCredentialStorage()

    def test_development_uses_random_ephemeral_key(self):
        """Development mode must use random (not deterministic) ephemeral key."""
        from traigent.security.crypto_utils import SecureCredentialStorage

        env_overrides = {
            "ENVIRONMENT": "development",
            "TRAIGENT_ENVIRONMENT": "",
            "TRAIGENT_ENV": "",
        }

        with patch.dict(
            os.environ,
            env_overrides,
            clear=False,
        ):
            env_copy = os.environ.copy()
            env_copy.pop("TRAIGENT_ENCRYPTION_KEY", None)

            with patch.dict(os.environ, env_copy, clear=True):
                with pytest.warns(UserWarning, match="random ephemeral"):
                    mgr1 = SecureCredentialStorage()

            with patch.dict(os.environ, env_copy, clear=True):
                with pytest.warns(UserWarning, match="random ephemeral"):
                    mgr2 = SecureCredentialStorage()

            # Keys must be different (random, not deterministic)
            assert mgr1.password != mgr2.password

    def test_no_uid_or_home_path_in_key_derivation(self):
        """Verify the key derivation doesn't use uid or home path."""
        import inspect

        from traigent.security.crypto_utils import SecureCredentialStorage

        # Get the source code of __init__
        source = inspect.getsource(SecureCredentialStorage.__init__)

        # These patterns indicate the old vulnerable code
        vulnerable_patterns = [
            r"os\.getuid",
            r"os\.path\.expanduser",
            r"getpass\.getuser",
            r"_generate_default_key",
        ]

        for pattern in vulnerable_patterns:
            assert not re.search(
                pattern, source
            ), f"Found vulnerable pattern '{pattern}' in SecureCredentialManager.__init__"


class TestFix2SensitivePayloadLogging:
    """Tests for Fix #2: Sensitive trial payloads logged in plaintext.

    The original bug logged full trial payloads at INFO level, exposing
    prompts, responses, and API keys.

    The fix:
    - Changed INFO logging to DEBUG for payload details
    - Added _redact_sensitive_fields() to redact sensitive data
    """

    def test_redact_sensitive_fields_method_exists(self):
        """Verify _redact_sensitive_fields method exists."""
        from traigent.cloud.trial_operations import TrialOperations

        assert hasattr(TrialOperations, "_redact_sensitive_fields")
        assert callable(TrialOperations._redact_sensitive_fields)

    def test_redact_sensitive_fields_redacts_api_keys(self):
        """API keys must be redacted."""
        from traigent.cloud.trial_operations import TrialOperations

        data = {"api_key": "placeholder_key"}
        redacted = TrialOperations._redact_sensitive_fields(data)

        assert "placeholder_key" not in str(redacted)
        assert "[REDACTED:" in str(redacted)

    def test_redact_sensitive_fields_redacts_prompts(self):
        """Prompts must be redacted."""
        from traigent.cloud.trial_operations import TrialOperations

        data = {"prompt": "This is a secret prompt with sensitive information" * 3}
        redacted = TrialOperations._redact_sensitive_fields(data)

        assert "secret prompt" not in str(redacted)
        assert "[REDACTED:" in str(redacted)

    def test_redact_sensitive_fields_redacts_responses(self):
        """Response content must be redacted."""
        from traigent.cloud.trial_operations import TrialOperations

        data = {"response": "This is a long response with sensitive data" * 3}
        redacted = TrialOperations._redact_sensitive_fields(data)

        assert "sensitive data" not in str(redacted)

    def test_redact_sensitive_fields_handles_nested_data(self):
        """Nested sensitive data must be redacted."""
        from traigent.cloud.trial_operations import TrialOperations

        data = {"outer": {"inner": {"api_key": "placeholder_key"}}}
        redacted = TrialOperations._redact_sensitive_fields(data)

        assert "placeholder_key" not in str(redacted)

    def test_redact_sensitive_fields_prevents_recursion(self):
        """Deep nesting must not cause recursion errors."""
        from traigent.cloud.trial_operations import TrialOperations

        # Create deeply nested structure
        data: dict = {"level": 0}
        current = data
        for i in range(20):  # More than max depth
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        # Should not raise RecursionError
        redacted = TrialOperations._redact_sensitive_fields(data)
        assert "[MAX_DEPTH]" in str(redacted)

    def test_payload_logging_uses_debug_level(self):
        """Verify payload details are logged at DEBUG, not INFO."""
        import inspect

        from traigent.cloud.trial_operations import TrialOperations

        source = inspect.getsource(TrialOperations)

        # Look for patterns that indicate INFO-level logging of payloads
        # The fix should use DEBUG for detailed payload logging
        # Allow info logging for status messages, but not for payload contents
        info_payload_patterns = [
            r"logger\.info\([^)]*payload[^)]*\)",
            r"logger\.info\([^)]*submission_data[^)]*\)",
        ]

        for pattern in info_payload_patterns:
            matches = re.findall(pattern, source, re.IGNORECASE)
            # Filter out false positives (like "payload submitted successfully")
            suspicious = [
                m for m in matches if "submit" not in m.lower() or "data" in m.lower()
            ]
            assert len(suspicious) == 0, f"Found INFO logging of payloads: {suspicious}"


class TestFix3ES256KeySizeValidation:
    """Tests for Fix #3: ES256 tokens rejected due to RSA key size check.

    The original bug applied RSA's 2048-bit minimum to ES256, which uses
    256-bit ECDSA keys (P-256 curve).

    The fix:
    - Added MIN_KEY_SIZES dictionary with algorithm-specific minimums
    - Made key size validation algorithm-aware
    """

    def test_min_key_sizes_dictionary_exists(self):
        """Verify MIN_KEY_SIZES dictionary exists with correct values."""
        from traigent.security.jwt_validator import SecureJWTValidator

        assert hasattr(SecureJWTValidator, "MIN_KEY_SIZES")

        min_sizes = SecureJWTValidator.MIN_KEY_SIZES

        # RSA algorithms should require 2048 bits
        assert min_sizes.get("RS256") == 2048
        assert min_sizes.get("RS384") == 2048
        assert min_sizes.get("RS512") == 2048

        # ECDSA algorithms should have appropriate curve-based sizes
        assert min_sizes.get("ES256") == 256  # P-256 curve
        assert min_sizes.get("ES384") == 384  # P-384 curve
        assert min_sizes.get("ES512") == 521  # P-521 curve (521 bits, not 512)

    def test_no_hardcoded_min_key_length_in_validation(self):
        """Verify MIN_KEY_LENGTH is not used in production validation."""
        import inspect

        from traigent.security.jwt_validator import SecureJWTValidator

        source = inspect.getsource(SecureJWTValidator._validate_production)

        # The old code used self.MIN_KEY_LENGTH directly
        # The new code should use self.MIN_KEY_SIZES.get(algorithm)
        assert (
            "MIN_KEY_LENGTH" not in source
        ), "_validate_production still uses MIN_KEY_LENGTH instead of algorithm-aware MIN_KEY_SIZES"

    def test_algorithm_aware_key_validation_pattern(self):
        """Verify the key validation uses algorithm from signing_key."""
        import inspect

        from traigent.security.jwt_validator import SecureJWTValidator

        source = inspect.getsource(SecureJWTValidator._validate_production)

        # Should get algorithm from signing_key
        assert "algorithm" in source.lower()
        assert "MIN_KEY_SIZES" in source


class TestFix4CreateSessionDeadlock:
    """Tests for Fix #4: create_session blocks inside running event loop.

    The original bug used run_coroutine_threadsafe().result() which deadlocks
    when called from the same thread as the running event loop.

    The fix:
    - Uses ThreadPoolExecutor to run async code in a separate thread
    - Creates a new event loop in that thread
    """

    def test_create_session_uses_thread_pool_executor(self):
        """Verify create_session uses ThreadPoolExecutor for running loop case."""
        import inspect

        from traigent.cloud.session_operations import SessionOperations

        source = inspect.getsource(SessionOperations.create_session)

        # The fix should use ThreadPoolExecutor (either directly or via helper)
        assert (
            "ThreadPoolExecutor" in source or "_get_session_executor" in source
        ), "create_session should use ThreadPoolExecutor to avoid deadlock"

    def test_create_session_creates_new_loop_in_thread(self):
        """Verify a new event loop is created in the thread."""
        import inspect

        from traigent.cloud.session_operations import SessionOperations

        source = inspect.getsource(SessionOperations.create_session)

        # The fix should create a new event loop
        assert (
            "new_event_loop" in source
        ), "create_session should create new_event_loop in thread"

    def test_no_run_coroutine_threadsafe_result_pattern(self):
        """Verify the deadlock-prone pattern is removed."""
        import inspect

        from traigent.cloud.session_operations import SessionOperations

        source = inspect.getsource(SessionOperations.create_session)

        # The old deadlock-prone pattern was:
        # run_coroutine_threadsafe(..., loop).result()
        # This should NOT be present anymore
        deadlock_pattern = r"run_coroutine_threadsafe\([^)]+\)\.result\("
        matches = re.findall(deadlock_pattern, source)
        assert len(matches) == 0, f"Found deadlock-prone pattern: {matches}"

    def test_create_session_does_not_deadlock_with_running_loop(self):
        """Integration test: create_session should not deadlock when called from async context."""
        from unittest.mock import AsyncMock

        from traigent.cloud.session_operations import SessionOperations

        # Create a mock client
        mock_client = MagicMock()
        mock_client.backend_config = MagicMock()
        mock_client.backend_config.backend_base_url = "http://localhost:5000"
        mock_client.local_storage = None
        mock_client._register_security_session = MagicMock()

        # Mock the async method to return quickly
        mock_client._create_traigent_session_via_api = AsyncMock(
            side_effect=Exception("Mock error - expected")
        )
        mock_client.session_bridge = MagicMock()

        ops = SessionOperations(mock_client)

        # This should complete (with fallback) without deadlocking
        # We use a timeout to detect deadlock
        result = [None]
        error = [None]

        def run_in_async_context():
            async def async_caller():
                # We're now inside a running event loop
                # The sync create_session should NOT deadlock
                try:
                    return ops.create_session(
                        function_name="test_function",
                        search_space={"param": [1, 2, 3]},
                        optimization_goal="maximize",
                        metadata={"max_trials": 5},
                    )
                except Exception as e:
                    error[0] = e
                    # Fallback path returns a local session ID
                    return "local_session_expected"

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result[0] = loop.run_until_complete(async_caller())
            finally:
                loop.close()

        # Run with timeout to detect deadlock
        thread = threading.Thread(target=run_in_async_context)
        thread.start()
        thread.join(timeout=10)  # 10 second timeout

        if thread.is_alive():
            # If thread is still alive, it deadlocked. We can't easily kill it,
            # but we can fail the test.
            pass

        assert not thread.is_alive(), "create_session deadlocked!"
        # Should have returned a fallback session ID
        assert result[0] is not None or error[0] is not None


class TestFix5ClosedAiohttpSession:
    """Tests for Fix #5: Summary stats submission uses closed aiohttp session.

    The original bug had an indentation issue where session.post was outside
    the ClientSession context manager, using a closed session.

    The fix:
    - Corrected indentation to keep session.post inside the context manager
    """

    def test_session_post_inside_context_manager(self):
        """Verify session.post is properly indented inside ClientSession context."""
        import ast
        import inspect
        import textwrap

        from traigent.cloud.trial_operations import TrialOperations

        source = textwrap.dedent(
            inspect.getsource(TrialOperations.submit_summary_stats)
        )

        # Parse the AST to verify proper nesting
        # The session.post should be inside the async with ClientSession block
        tree = ast.parse(source)

        # Find the async with statement for ClientSession
        found_proper_nesting = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncWith):
                # Check if this is the ClientSession context
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        # Check if it's aiohttp.ClientSession
                        func = item.context_expr.func
                        if (
                            isinstance(func, ast.Attribute)
                            and func.attr == "ClientSession"
                        ):
                            # Now check if there's an async with session.post inside
                            for inner_node in ast.walk(node):
                                if isinstance(inner_node, ast.AsyncWith):
                                    for inner_item in inner_node.items:
                                        if isinstance(
                                            inner_item.context_expr, ast.Call
                                        ):
                                            inner_func = inner_item.context_expr.func
                                            if (
                                                isinstance(inner_func, ast.Attribute)
                                                and inner_func.attr == "post"
                                            ):
                                                found_proper_nesting = True
                                                break

        assert (
            found_proper_nesting
        ), "session.post is not properly nested inside ClientSession context manager"

    def test_no_session_usage_after_context_exit(self):
        """Verify there's no session usage after the context manager exits."""
        import inspect

        from traigent.cloud.trial_operations import TrialOperations

        source = inspect.getsource(TrialOperations.submit_summary_stats)

        # The pattern we want to avoid is session.post AFTER the context manager closes
        # Looking for the structure:
        # async with aiohttp.ClientSession(...) as session:
        #     ...  (should contain session.post)
        # session.post(...)  # BAD - this would use closed session

        lines = source.split("\n")

        # Track indentation levels
        in_client_session = False
        client_session_indent = 0
        found_post_outside = False

        for line in lines:
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)

            if "aiohttp.ClientSession" in line and "async with" in line:
                in_client_session = True
                client_session_indent = current_indent
            elif in_client_session:
                # Check if we've exited the context (less or equal indentation)
                if (
                    stripped
                    and current_indent <= client_session_indent
                    and "async with" not in line
                ):
                    in_client_session = False
                    # Check if there's a session.post after exiting
                    if "session.post" in stripped:
                        found_post_outside = True
                        break

            # Also check for session.post at the same level as ClientSession
            if not in_client_session and "session.post" in stripped:
                found_post_outside = True
                break

        assert (
            not found_post_outside
        ), "Found session.post outside of ClientSession context manager"


class TestRegressionSummary:
    """Summary test to verify all fixes are in place."""

    def test_all_five_fixes_present(self):
        """Meta-test ensuring all fix test classes exist and have tests."""
        test_classes = [
            TestFix1DeterministicFallbackKey,
            TestFix2SensitivePayloadLogging,
            TestFix3ES256KeySizeValidation,
            TestFix4CreateSessionDeadlock,
            TestFix5ClosedAiohttpSession,
        ]

        for test_class in test_classes:
            # Each class should have at least one test method
            test_methods = [m for m in dir(test_class) if m.startswith("test_")]
            assert len(test_methods) > 0, f"{test_class.__name__} has no test methods"
