#!/usr/bin/env python3
"""Run customer support agent example."""

import os
import sys
from pathlib import Path

# Ensure TraiGent SDK is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Initialize in mock mode for examples
os.environ["TRAIGENT_MOCK_MODE"] = "true"

from customer_support import demonstrate_customer_support

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_customer_support()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nTo run without mock mode:")
    print("  export TRAIGENT_MOCK_MODE=false")
    print("  python run.py")
