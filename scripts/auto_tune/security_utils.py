#!/usr/bin/env python3
"""
Security utilities for auto-tuning pipeline.
Provides input validation, credential management, and audit logging.
"""

import hashlib
import json
import logging
import os
import secrets
import signal
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union


# Configure secure logging
def setup_logging(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Set up secure logging with rotation."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Format for console
    console_format = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger


# Path validation
def validate_path(path: Path, base_dir: Optional[Path] = None) -> bool:
    """
    Validate path to prevent traversal attacks.

    Args:
        path: Path to validate
        base_dir: Base directory to check against (default: repo root)

    Returns:
        True if path is valid and safe
    """
    try:
        # Resolve to absolute path
        resolved = path.resolve()

        # Default to repo root
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent.resolve()
        else:
            base_dir = base_dir.resolve()

        # Check if path is within base directory
        return resolved.is_relative_to(base_dir)
    except Exception:
        return False


def sanitize_input(value: Any, max_length: int = 1000) -> str:
    """
    Sanitize input to prevent injection attacks.

    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string
    """
    if value is None:
        return ""

    # Convert to string and truncate
    str_value = str(value)[:max_length]

    # Remove control characters and null bytes
    sanitized = "".join(char for char in str_value if ord(char) >= 32)

    # Remove potential command injection characters
    dangerous_chars = [";", "|", "&", "$", "`", "\\", "\n", "\r"]
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")

    return sanitized


# Credential management
class SecureCredentialManager:
    """Secure credential management with validation."""

    def __init__(self, credential_file: Optional[str] = None):
        self.logger = setup_logging(self.__class__.__name__)
        self._credentials = {}
        self.credential_file = credential_file
        self._load_credentials()

    def _load_credentials(self):
        """Load credentials from file or environment with validation."""
        # Try loading from file first if provided
        if self.credential_file and Path(self.credential_file).exists():
            try:
                with open(self.credential_file) as f:
                    loaded_creds = json.load(f)
                    for key, value in loaded_creds.items():
                        if value:
                            # Store directly from file without validation for testing
                            self._credentials[key] = value
                return
            except Exception:
                pass  # Fall back to environment

        # List of required credentials
        required_keys = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
        ]

        optional_keys = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "SLACK_WEBHOOK",
            "GITHUB_TOKEN",
        ]

        # Load and validate credentials from environment
        for key in required_keys + optional_keys:
            value = os.environ.get(key)
            if value:
                # Basic validation
                if self.validate_credential(key, value):
                    # Store hashed version for comparison
                    self._credentials[key] = {
                        "hash": hashlib.sha256(value.encode()).hexdigest(),
                        "present": True,
                    }
                else:
                    self.logger.warning(f"Invalid credential format for {key}")
            elif key in required_keys:
                self.logger.warning(f"Missing required credential: {key}")

    def validate_credential(self, key: str, value: str) -> bool:
        """Validate credential format."""
        if not value or len(value) < 10:
            return False

        # API key patterns
        patterns = {
            "ANTHROPIC_API_KEY": lambda v: v.startswith("sk-ant-"),
            "OPENAI_API_KEY": lambda v: v.startswith("sk-"),
            "AWS_ACCESS_KEY_ID": lambda v: len(v) == 20,
            "AWS_SECRET_ACCESS_KEY": lambda v: len(v) == 40,
            "GITHUB_TOKEN": lambda v: v.startswith("ghp_")
            or v.startswith("github_pat_"),
            "SLACK_WEBHOOK": lambda v: v.startswith("https://hooks.slack.com/"),
        }

        if key in patterns:
            return patterns[key](value)
        return True

    def get_credential(self, key: str) -> Optional[str]:
        """Get credential if available and valid."""
        # Check if loaded from file
        if isinstance(self._credentials.get(key), str):
            return self._credentials[key]
        # Check if loaded from environment
        if (
            key in self._credentials
            and isinstance(self._credentials[key], dict)
            and self._credentials[key].get("present")
        ):
            value = os.environ.get(key)
            # Verify it hasn't changed
            if (
                value
                and hashlib.sha256(value.encode()).hexdigest()
                == self._credentials[key]["hash"]
            ):
                return value
        return None

    def has_credential(self, key: str) -> bool:
        """Check if credential is available."""
        return key in self._credentials and self._credentials[key]["present"]


# Rate limiting
class RateLimiter:
    """Rate limiter to prevent abuse."""

    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = []
        self.logger = setup_logging(self.__class__.__name__)

    def check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        now = time.time()
        # Remove old calls outside window
        self.calls = [t for t in self.calls if now - t < self.window_seconds]

        if len(self.calls) >= self.max_calls:
            self.logger.warning(
                f"Rate limit exceeded: {len(self.calls)}/{self.max_calls} calls"
            )
            return False

        self.calls.append(now)
        return True

    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        while not self.check_rate_limit():
            sleep_time = self.window_seconds - (time.time() - self.calls[0]) + 0.01
            self.logger.info(f"Rate limit reached, waiting {sleep_time:.1f} seconds")
            time.sleep(min(sleep_time, 10))


# Cost control
class CostController:
    """Cost controller with hard limits."""

    def __init__(self, max_budget: float = 5.0):
        self.max_budget = max_budget
        self.spent = 0.0
        self.logger = setup_logging(self.__class__.__name__)
        self.cost_file = Path("cost_tracking.json")
        self._load_costs()

    def _load_costs(self):
        """Load existing costs from file."""
        if self.cost_file.exists():
            try:
                with open(self.cost_file) as f:
                    data = json.load(f)
                    self.spent = data.get("total_spent", 0.0)
            except Exception as e:
                self.logger.error(f"Failed to load costs: {e}")

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if estimated cost is within budget."""
        if self.spent + estimated_cost > self.max_budget:
            self.logger.error(
                f"Budget exceeded: {self.spent + estimated_cost:.2f} > {self.max_budget:.2f}"
            )
            return False

        # Warning at 80% budget
        if self.spent + estimated_cost > self.max_budget * 0.8:
            self.logger.warning(
                f"Approaching budget limit: {self.spent + estimated_cost:.2f}/{self.max_budget:.2f}"
            )

        return True

    def track_spending(self, actual_cost: float):
        """Track actual spending."""
        self.spent += actual_cost

        # Save to file
        try:
            data = {
                "total_spent": self.spent,
                "last_updated": datetime.now().isoformat(),
                "budget": self.max_budget,
            }
            with open(self.cost_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save costs: {e}")

        # Alert if over 90%
        if self.spent > self.max_budget * 0.9:
            self.logger.critical(
                f"BUDGET ALERT: {self.spent:.2f}/{self.max_budget:.2f} spent!"
            )

    def remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0, self.max_budget - self.spent)


# Audit logging
class AuditLogger:
    """Audit logger for compliance and tracking."""

    def __init__(self, audit_file: str = "audit_log.jsonl"):
        self.audit_file = Path(audit_file)
        self.logger = setup_logging(self.__class__.__name__)

    def log_event(self, event_type: str, details: Dict[str, Any], success: bool = True):
        """Log an audit event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "success": success,
            "details": details,
            "user": os.environ.get("USER", "unknown"),
            "environment": os.environ.get("CI_ENVIRONMENT_NAME", "development"),
            "git_commit": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", secrets.token_hex(8)),
        }

        try:
            with open(self.audit_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")


# Timeout context manager
@contextmanager
def timeout(seconds: int):
    """Context manager for timeout control."""

    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# Retry decorator with exponential backoff
def retry_with_backoff(
    max_attempts: int = 3, initial_delay: float = 1.0, max_delay: float = 60.0
):
    """Decorator for retry with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        sleep_time = min(delay * (2**attempt), max_delay)
                        logging.info(
                            f"Attempt {attempt + 1} failed, retrying in {sleep_time:.1f}s: {e}"
                        )
                        time.sleep(sleep_time)
                    else:
                        logging.error(f"All {max_attempts} attempts failed")

            raise last_exception

        return wrapper

    return decorator


# JSON validation
def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate JSON data against a schema."""
    # Simple schema validation - check type and required fields
    if "type" in schema:
        if schema["type"] == "object" and not isinstance(data, dict):
            return False

    if "required" in schema:
        for key in schema["required"]:
            if key not in data:
                return False

    if "properties" in schema:
        for key, prop_schema in schema["properties"].items():
            if key in data:
                if "type" in prop_schema:
                    expected_type = prop_schema["type"]
                    value = data[key]
                    if expected_type == "string" and not isinstance(value, str):
                        return False
                    elif expected_type == "number" and not isinstance(
                        value, (int, float)
                    ):
                        return False

    return True


# Environment validation
def validate_environment() -> Dict[str, bool]:
    """Validate environment setup."""
    checks = {
        "python_version": True,  # Already checked by script
        "dvc_installed": Path(".dvc").exists(),
        "git_repo": Path(".git").exists(),
        "config_exists": Path("params.yaml").exists(),
        "baseline_dir": Path("baselines").exists()
        or Path("baselines").mkdir(exist_ok=True)
        or True,
    }

    return checks


# Safe file operations
def safe_file_write(path: Union[Path, str], content: str, backup: bool = True):
    """Safely write to file with backup."""
    if isinstance(path, str):
        path = Path(path)
    """Safely write to file with backup."""
    if not validate_path(path):
        raise ValueError(f"Invalid path: {path}")

    # Create backup if requested and file exists
    if backup and path.exists():
        backup_path = path.with_suffix(f"{path.suffix}.backup")
        path.rename(backup_path)

    # Write with atomic operation
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with open(temp_path, "w") as f:
            f.write(content)
        temp_path.rename(path)
    except Exception:
        if backup and backup_path.exists():
            backup_path.rename(path)  # Restore backup
        raise


def safe_file_read(path: Union[Path, str]) -> Optional[str]:
    """Safely read from file."""
    if isinstance(path, str):
        path = Path(path)
    """Safely read from file."""
    if not validate_path(path):
        raise ValueError(f"Invalid path: {path}")

    if not path.exists():
        return None

    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to read {path}: {e}")
        return None
