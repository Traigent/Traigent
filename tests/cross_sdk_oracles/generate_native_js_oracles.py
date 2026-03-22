#!/usr/bin/env python3
"""Emit the canonical Python-owned oracle payload for JS cross-SDK parity tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    payload_path = Path(__file__).with_name("native_js_oracles.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
