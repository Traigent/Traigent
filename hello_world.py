"""Convenience alias - runs the bundled quickstart example.

Equivalent to: python -m traigent.examples.quickstart
"""

import runpy

runpy.run_module("traigent.examples.quickstart", run_name="__main__")
