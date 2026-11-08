"""Per-example metric names follow the ASCII wire contract (not str.isidentifier)."""

import pytest

from traigent.core.metadata_helpers import _validate_metric_entry


@pytest.mark.parametrize(
    "key", ["précision", "café_metric", "a٣", "score\n", "bad-key", ""]
)
def test_non_ascii_or_malformed_metric_key_rejected_at_example_check(key: str) -> None:
    with pytest.raises(ValueError, match="Invalid metric key"):
        _validate_metric_entry(key, 1.0, 0)


@pytest.mark.parametrize("key", ["accuracy", "_x", "f1_score2", "A"])
def test_ascii_identifier_metric_key_accepted(key: str) -> None:
    _validate_metric_entry(key, 1.0, 0)
