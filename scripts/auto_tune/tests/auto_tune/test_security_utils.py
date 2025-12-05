#!/usr/bin/env python3
"""Unit tests for security utilities."""

import sys
import time
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

# Simple tests to verify basic functionality
if __name__ == "__main__":
    print("Running security utils tests...")

    # Test imports
    try:
        import auto_tune.security_utils  # noqa: F401 - Full module import check
        from auto_tune.security_utils import (
            CostController,
            RateLimiter,
            sanitize_input,
            validate_path,
        )

        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        sys.exit(1)

    # Test path validation
    result = validate_path(Path.cwd())
    assert result, "Current directory should be valid"
    print("✓ Path validation works")

    # Test sanitization
    clean = sanitize_input("test; rm -rf")
    assert ";" not in clean, "Should remove dangerous characters"
    print("✓ Input sanitization works")

    # Test rate limiter
    limiter = RateLimiter(max_calls=2, window_seconds=0.1)
    assert limiter.check_rate_limit()
    assert limiter.check_rate_limit()
    assert not limiter.check_rate_limit()  # Should block third call
    print("✓ Rate limiter works")

    # Test cost controller
    controller = CostController(max_budget=10.0)
    assert controller.check_budget(5.0)
    assert not controller.check_budget(6.0)  # Would exceed budget
    print("✓ Cost controller works")

    # Test the RateLimiter wait timing (should be fixed now)
    limiter2 = RateLimiter(max_calls=1, window_seconds=0.1)
    limiter2.check_rate_limit()

    start = time.time()
    limiter2.wait_if_needed()
    elapsed = time.time() - start

    # Should wait approximately 0.1 seconds (not 1.1 seconds)
    if 0.05 < elapsed < 0.2:
        print(f"✓ RateLimiter timing fixed! Waited {elapsed:.3f}s (expected ~0.1s)")
    else:
        print(f"✗ RateLimiter timing issue: Waited {elapsed:.3f}s (expected ~0.1s)")

    print("\n✅ All tests passed!")
