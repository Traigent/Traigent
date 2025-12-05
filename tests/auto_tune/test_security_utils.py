#!/usr/bin/env python3
"""
Unit tests for security utilities.
Tests validation, sanitization, rate limiting, and credential management.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import TestCase

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from auto_tune.security_utils import (
    AuditLogger,
    CostController,
    RateLimiter,
    SecureCredentialManager,
    retry_with_backoff,
    sanitize_input,
    validate_json_schema,
    validate_path,
)


class TestPathValidation(TestCase):
    """Test path validation functions."""

    def test_valid_path_within_repo(self):
        """Test valid path within repository."""
        # Current directory should be valid
        self.assertTrue(validate_path(Path.cwd()))

    def test_invalid_path_traversal(self):
        """Test path traversal attack prevention."""
        self.assertFalse(validate_path(Path("../../etc/passwd")))

    def test_absolute_path_outside_repo(self):
        """Test absolute path outside repo."""
        self.assertFalse(validate_path(Path("/etc/passwd")))


class TestInputSanitization(TestCase):
    """Test input sanitization."""

    def test_sanitize_normal_input(self):
        """Test normal input sanitization."""
        result = sanitize_input("normal text")
        self.assertEqual(result, "normal text")

    def test_sanitize_command_injection(self):
        """Test command injection prevention."""
        result = sanitize_input("text; rm -rf /")
        self.assertNotIn(";", result)

    def test_sanitize_null_bytes(self):
        """Test null byte removal."""
        result = sanitize_input("text\x00with\x00nulls")
        self.assertNotIn("\x00", result)

    def test_sanitize_max_length(self):
        """Test max length enforcement."""
        long_input = "a" * 2000
        result = sanitize_input(long_input, max_length=1000)
        self.assertEqual(len(result), 1000)

    def test_sanitize_none_input(self):
        """Test None input handling."""
        result = sanitize_input(None)
        self.assertEqual(result, "")


class TestCredentialManager(TestCase):
    """Test secure credential management."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cred_file = Path(self.temp_dir) / ".credentials"

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_valid_credentials(self):
        """Test loading valid credentials."""
        creds = {"ANTHROPIC_API_KEY": "test-key", "OPENAI_API_KEY": "test-key-2"}
        self.cred_file.write_text(json.dumps(creds))

        manager = SecureCredentialManager(str(self.cred_file))
        loaded = manager.get_credential("ANTHROPIC_API_KEY")
        self.assertEqual(loaded, "test-key")

    def test_invalid_credential_format(self):
        """Test invalid credential format handling."""
        self.cred_file.write_text("not json")

        manager = SecureCredentialManager(str(self.cred_file))
        result = manager.get_credential("ANY_KEY")
        self.assertIsNone(result)

    def test_missing_credentials(self):
        """Test missing credentials file."""
        manager = SecureCredentialManager("/nonexistent/path")
        result = manager.get_credential("ANY_KEY")
        self.assertIsNone(result)

    def test_aws_credential_validation(self):
        """Test AWS credential format validation."""
        manager = SecureCredentialManager()

        # Valid format
        valid_key = (
            "A" * 20
        )  # synthetic value satisfying length requirement without resembling a real key
        self.assertTrue(manager.validate_credential("AWS_ACCESS_KEY_ID", valid_key))

        # Invalid format
        self.assertFalse(manager.validate_credential("AWS_ACCESS_KEY_ID", "invalid"))


class TestRateLimiter(TestCase):
    """Test rate limiting functionality."""

    def test_rate_limit_allows_initial_calls(self):
        """Test rate limiter allows calls within limit."""
        limiter = RateLimiter(max_calls=3, window_seconds=1)

        self.assertTrue(limiter.check_rate_limit())
        self.assertTrue(limiter.check_rate_limit())
        self.assertTrue(limiter.check_rate_limit())

    def test_rate_limit_blocks_excess_calls(self):
        """Test rate limiter blocks calls over limit."""
        limiter = RateLimiter(max_calls=2, window_seconds=1)

        self.assertTrue(limiter.check_rate_limit())
        self.assertTrue(limiter.check_rate_limit())
        self.assertFalse(limiter.check_rate_limit())

    def test_rate_limit_window_reset(self):
        """Test rate limit window resets."""
        limiter = RateLimiter(max_calls=1, window_seconds=0.1)

        self.assertTrue(limiter.check_rate_limit())
        self.assertFalse(limiter.check_rate_limit())

        time.sleep(0.15)
        self.assertTrue(limiter.check_rate_limit())

    def test_wait_if_needed(self):
        """Test wait functionality."""
        limiter = RateLimiter(max_calls=1, window_seconds=0.1)
        limiter.check_rate_limit()

        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start

        # Should wait approximately 0.1 seconds (with small buffer)
        self.assertTrue(0.05 < elapsed < 0.2)


class TestCostController(TestCase):
    """Test cost control functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.old_cwd)
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_budget_check_within_limit(self):
        """Test budget check within limit."""
        controller = CostController(max_budget=10.0)
        self.assertTrue(controller.check_budget(5.0))

    def test_budget_check_exceeds_limit(self):
        """Test budget check exceeding limit."""
        controller = CostController(max_budget=10.0)
        controller.spent = 9.0
        self.assertFalse(controller.check_budget(2.0))

    def test_budget_warning_at_80_percent(self):
        """Test warning at 80% budget usage."""
        controller = CostController(max_budget=10.0)
        controller.spent = 7.5

        with self.assertLogs(level="WARNING"):
            controller.check_budget(1.0)

    def test_track_spending(self):
        """Test spending tracking."""
        controller = CostController(max_budget=10.0)
        controller.track_spending(2.5)
        self.assertEqual(controller.spent, 2.5)

    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        controller = CostController(max_budget=10.0)
        controller.spent = 3.0
        self.assertEqual(controller.remaining_budget(), 7.0)


class TestAuditLogger(TestCase):
    """Test audit logging functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".jsonl"
        )
        self.temp_file.close()
        self.logger = AuditLogger(self.temp_file.name)

    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.temp_file.name)

    def test_log_event_success(self):
        """Test logging successful event."""
        self.logger.log_event("test_action", {"key": "value"})

        with open(self.temp_file.name) as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0])
        self.assertEqual(event["event_type"], "test_action")
        self.assertTrue(event["success"])

    def test_log_event_failure(self):
        """Test logging failed event."""
        self.logger.log_event("test_action", {"key": "value"}, success=False)

        with open(self.temp_file.name) as f:
            event = json.loads(f.readline())

        self.assertFalse(event["success"])

    def test_multiple_events(self):
        """Test logging multiple events."""
        self.logger.log_event("event1", {})
        self.logger.log_event("event2", {})

        with open(self.temp_file.name) as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)


class TestJSONValidation(TestCase):
    """Test JSON schema validation."""

    def test_valid_schema(self):
        """Test valid JSON against schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "required": ["name"],
        }

        data = {"name": "test", "age": 25}
        self.assertTrue(validate_json_schema(data, schema))

    def test_missing_required_key(self):
        """Test missing required key."""
        schema = {"type": "object", "required": ["name"]}

        data = {"other": "value"}
        self.assertFalse(validate_json_schema(data, schema))

    def test_extra_keys_allowed(self):
        """Test extra keys are allowed by default."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        data = {"name": "test", "extra": "value"}
        self.assertTrue(validate_json_schema(data, schema))


class TestRetryDecorator(TestCase):
    """Test retry decorator functionality."""

    def test_successful_on_first_attempt(self):
        """Test function succeeds on first attempt."""
        call_count = [0]

        @retry_with_backoff(max_attempts=3)
        def success_func():
            call_count[0] += 1
            return "success"

        result = success_func()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 1)

    def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = [0]

        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return "success"

        result = fail_then_succeed()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)

    def test_max_attempts_exceeded(self):
        """Test max attempts exceeded."""
        call_count = [0]

        @retry_with_backoff(max_attempts=2, initial_delay=0.01)
        def always_fails():
            call_count[0] += 1
            raise RuntimeError("Always fails")

        with self.assertRaises(RuntimeError):
            always_fails()

        self.assertEqual(call_count[0], 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
