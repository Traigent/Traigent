"""Tests for the no-execution evaluation-compatibility contract (#1979).

Every test in this package is fully offline: no network, no LLM, no cost. The
contract is proven to be execution-free (agents / metrics / custom evaluators
are never called, no configuration context is ever entered) and to stay in
lock-step with the real ``LocalEvaluator`` production path.
"""
