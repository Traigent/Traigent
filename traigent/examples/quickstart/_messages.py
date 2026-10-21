"""User-facing completion text for the keyless quickstart.

Kept in a side-effect-free module so tests can assert the wording without
importing the demo (which enables mock mode and imports litellm).
"""

from __future__ import annotations


def completion_message(completed_trials: int | None = None) -> str:
    """Return the honest end-of-run message for the keyless demo.

    The keyless demo scores trials from a hardcoded table, not from model
    output, so it must never claim a winner or a best configuration.
    """
    count = (
        f" ({completed_trials} trials ran)" if isinstance(completed_trials, int) else ""
    )
    return (
        f"[traigent] Quickstart complete{count}: plumbing check passed. "
        "This run used a mock LLM and simulated scores, so it is NOT a model "
        "comparison and no model or configuration was found to be best. "
        "To run a real optimization, set a provider API key and see "
        "walkthrough/real/01_tuning_qa.py."
    )
