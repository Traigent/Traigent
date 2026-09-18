"""Tests for MeasuresDict validation.

This module tests the MeasuresDict class to ensure it properly validates
measures and enforces cardinality limits.

Covers:
- Key pattern validation (Python identifier syntax)
- Numeric-only enforcement on every mutation path.

The numeric-only rule is a PRIVACY boundary, not a typing nicety: measures cross
to the Traigent backend on every trial submission, so a string measure (an LLM
judge's rationale, a captured response, an error excerpt) would put content on
the wire. These tests previously asserted the opposite -- that a non-numeric
value is accepted with a warning -- and so held the leak open. They now pin the
rejection so it cannot be reintroduced.
"""

import pytest

from traigent.cloud.dtos import MeasuresDict

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online


class TestMeasuresDictValidation:
    """Tests for MeasuresDict type and cardinality validation."""

    def test_accepts_numeric_and_none_values(self):
        """Should accept the numeric types the wire contract allows."""
        measures = MeasuresDict(
            {
                "int_val": 42,
                "float_val": 3.14,
                "none_val": None,
            }
        )
        assert len(measures) == 3
        assert measures["int_val"] == 42
        assert measures["float_val"] == 3.14
        assert measures["none_val"] is None

    def test_rejects_string_value(self):
        """A string measure is content on the wire and must not be accepted."""
        with pytest.raises(TypeError, match="must be numeric"):
            MeasuresDict({"rationale": "the model explained itself"})

    def test_rejects_list_type(self):
        """Non-numeric values are rejected, never warned about and sent."""
        with pytest.raises(TypeError, match="must be numeric"):
            MeasuresDict({"list_val": [1, 2, 3]})

    def test_rejects_dict_type(self):
        """A nested dict can hide arbitrary text; reject it."""
        with pytest.raises(TypeError, match="must be numeric"):
            MeasuresDict({"dict_val": {"nested": "dict"}})

    def test_rejects_tuple_type(self):
        """Sequences are not measures, even when they hold numbers."""
        with pytest.raises(TypeError, match="must be numeric"):
            MeasuresDict({"tuple_val": (1, 2, 3)})

    def test_rejects_object_type(self):
        """An arbitrary object serializes to whatever ``str()`` gives; reject it."""

        class CustomObject:
            pass

        with pytest.raises(TypeError, match="must be numeric"):
            MeasuresDict({"object_val": CustomObject()})

    def test_rejects_non_string_keys(self):
        """Should reject non-string keys."""
        measures = MeasuresDict()
        with pytest.raises(TypeError, match="Key must be string"):
            measures[123] = "value"

    def test_enforces_cardinality_limit_on_init(self):
        """Should reject excessive keys during initialization."""
        # Create dict with MAX_KEYS + 1 items
        too_many = {f"metric_{i}": i for i in range(MeasuresDict.MAX_KEYS + 1)}

        with pytest.raises(ValueError, match="cannot exceed.*keys"):
            MeasuresDict(too_many)

    def test_enforces_cardinality_limit_on_assignment(self):
        """Should reject excessive keys during assignment."""
        measures = MeasuresDict()

        # Add up to limit
        for i in range(MeasuresDict.MAX_KEYS):
            measures[f"metric_{i}"] = i

        # Exceeding limit should raise
        with pytest.raises(ValueError, match="cannot exceed"):
            measures["too_many"] = 999

    def test_allows_updating_existing_keys_at_limit(self):
        """Should allow updating existing keys even at cardinality limit."""
        measures = MeasuresDict()

        # Add up to limit
        for i in range(MeasuresDict.MAX_KEYS):
            measures[f"metric_{i}"] = i

        # Updating existing key should work
        measures["metric_0"] = 999
        assert measures["metric_0"] == 999

    def test_empty_initialization(self):
        """Should allow empty initialization."""
        measures = MeasuresDict()
        assert len(measures) == 0

    def test_none_initialization(self):
        """Should allow None initialization (creates empty dict)."""
        measures = MeasuresDict(None)
        assert len(measures) == 0

    def test_assignment_rejects_non_numeric(self):
        """__setitem__ is a mutation path and must enforce the same rule."""
        measures = MeasuresDict()

        measures["valid"] = 42
        assert measures["valid"] == 42

        with pytest.raises(TypeError, match="must be numeric"):
            measures["invalid"] = [1, 2, 3]
        assert "invalid" not in measures

    def test_update_method_rejects_non_numeric(self):
        """update() must not be a way around the guard."""
        measures = MeasuresDict({"existing": 1})

        measures.update({"new_key": 2})
        assert measures["new_key"] == 2

        with pytest.raises(TypeError, match="must be numeric"):
            measures.update({"bad_key": [1, 2, 3]})
        assert "bad_key" not in measures

    def test_or_operator_rejects_non_numeric(self):
        """|= must not be a way around the guard."""
        measures = MeasuresDict({"existing": 1})

        measures |= {"new_key": 2}
        assert measures["new_key"] == 2

        with pytest.raises(TypeError, match="must be numeric"):
            measures |= {"bad_key": {"nested": "dict"}}
        assert "bad_key" not in measures

    def test_or_operator_accepts_measuresdict(self):
        """Should accept MeasuresDict as operand for |= operator."""
        measures1 = MeasuresDict({"key1": 1.0, "key2": 2.0})
        measures2 = MeasuresDict({"key3": 3.0, "key4": 4.0})

        # Store original id to verify it returns self
        original_id = id(measures1)

        # Merge two MeasuresDict instances
        measures1 |= measures2

        # Verify all keys are present
        assert measures1["key1"] == 1.0
        assert measures1["key2"] == 2.0
        assert measures1["key3"] == 3.0
        assert measures1["key4"] == 4.0

        # Verify the operation returns self (id unchanged)
        assert id(measures1) == original_id

    def test_setdefault_rejects_non_numeric(self):
        """setdefault() must not be a way around the guard."""
        measures = MeasuresDict()

        result = measures.setdefault("key1", 42)
        assert result == 42
        assert measures["key1"] == 42

        with pytest.raises(TypeError, match="must be numeric"):
            measures.setdefault("key2", [1, 2, 3])
        assert "key2" not in measures

    def test_dict_access_operations(self):
        """Should support standard dict operations."""
        measures = MeasuresDict({"key1": 1, "key2": 2})

        # get()
        assert measures.get("key1") == 1
        assert measures.get("missing", 999) == 999

        # keys()
        assert set(measures.keys()) == {"key1", "key2"}

        # values()
        assert set(measures.values()) == {1, 2}

        # items()
        assert set(measures.items()) == {("key1", 1), ("key2", 2)}

        # __contains__
        assert "key1" in measures
        assert "missing" not in measures

        # __len__
        assert len(measures) == 2

    def test_deletion_operations(self):
        """Should support deletion operations."""
        measures = MeasuresDict({"key1": 1, "key2": 2, "key3": 3})

        # del
        del measures["key1"]
        assert "key1" not in measures

        # pop()
        val = measures.pop("key2")
        assert val == 2
        assert "key2" not in measures

        # popitem()
        key, val = measures.popitem()
        assert key == "key3"
        assert val == 3
        assert len(measures) == 0

    def test_clear_operation(self):
        """Should support clear() operation."""
        measures = MeasuresDict({"key1": 1, "key2": 2})
        measures.clear()
        assert len(measures) == 0

    def test_iteration(self):
        """Should support iteration."""
        measures = MeasuresDict({"a": 1, "b": 2, "c": 3})

        # Iterate over keys
        keys = list(measures)
        assert set(keys) == {"a", "b", "c"}

        # Iterate over items
        items = [(k, v) for k, v in measures.items()]
        assert set(items) == {("a", 1), ("b", 2), ("c", 3)}

    def test_equality(self):
        """Should support equality comparison."""
        m1 = MeasuresDict({"a": 1, "b": 2})
        m2 = MeasuresDict({"a": 1, "b": 2})
        m3 = MeasuresDict({"a": 1, "b": 3})

        assert m1 == m2
        assert m1 != m3

        # Should be equal to regular dict with same content
        assert m1 == {"a": 1, "b": 2}

    def test_cardinality_limit_value(self):
        """Verify the cardinality limit is set correctly."""
        assert MeasuresDict.MAX_KEYS == 50

    def test_measures_at_exact_limit(self):
        """Should accept exactly MAX_KEYS items."""
        measures_data = {f"metric_{i}": i for i in range(MeasuresDict.MAX_KEYS)}
        measures = MeasuresDict(measures_data)
        assert len(measures) == MeasuresDict.MAX_KEYS

    def test_error_message_includes_count(self):
        """Error message should include actual count."""
        too_many = {f"metric_{i}": i for i in range(MeasuresDict.MAX_KEYS + 5)}

        with pytest.raises(ValueError, match=f"got {MeasuresDict.MAX_KEYS + 5}"):
            MeasuresDict(too_many)

    def test_mixed_valid_types(self):
        """Should accept the full range of numeric values in a single dict."""
        measures = MeasuresDict(
            {
                "int_metric": 42,
                "float_metric": 3.14,
                "none_metric": None,
                "negative_int": -10,
                "negative_float": -2.5,
                "zero": 0,
            }
        )
        assert len(measures) == 6
        assert measures["int_metric"] == 42
        assert measures["float_metric"] == 3.14
        assert measures["none_metric"] is None
        assert measures["negative_int"] == -10
        assert measures["negative_float"] == -2.5
        assert measures["zero"] == 0

    def test_one_string_rejects_the_whole_dict(self):
        """A single non-numeric value fails the whole submission, not silently.

        Fail closed: the caller must fix the evaluator, not discover later that
        one measure was quietly dropped (or quietly sent).
        """
        with pytest.raises(TypeError, match="must be numeric"):
            MeasuresDict({"accuracy": 0.9, "empty_string": ""})


class TestMeasuresDictKeyPattern:
    """Test key pattern validation (Python identifier syntax) - Phase 0."""

    def test_accepts_valid_python_identifier_keys(self):
        """Should accept keys matching Python identifier pattern."""
        valid_keys = {
            "accuracy": 0.95,
            "cost": 0.001,
            "latency": 1.5,
            "_private_metric": 0.8,
            "metric_123": 0.7,
            "metric123": 0.6,
            "ALLCAPS": 0.5,
            "CamelCase": 0.4,
        }
        measures = MeasuresDict(valid_keys)
        assert len(measures) == 8
        assert measures["accuracy"] == 0.95

    def test_rejects_hyphenated_keys(self):
        """Should reject keys with hyphens (not Python identifiers)."""
        with pytest.raises(
            ValueError, match=r"Measure key 'my-metric' must match pattern"
        ):
            MeasuresDict({"my-metric": 0.95})

    def test_rejects_keys_starting_with_digit(self):
        """Should reject keys starting with digits."""
        with pytest.raises(
            ValueError, match=r"Measure key '123abc' must match pattern"
        ):
            MeasuresDict({"123abc": 0.95})

    def test_rejects_keys_with_spaces(self):
        """Should reject keys with spaces."""
        with pytest.raises(
            ValueError, match=r"Measure key 'my metric' must match pattern"
        ):
            MeasuresDict({"my metric": 0.95})

    def test_rejects_keys_with_special_chars(self):
        """Should reject keys with special characters."""
        with pytest.raises(
            ValueError, match=r"Measure key 'my@metric' must match pattern"
        ):
            MeasuresDict({"my@metric": 0.95})

    def test_error_message_provides_examples(self):
        """Error message should show valid and invalid examples."""
        with pytest.raises(ValueError) as exc_info:
            MeasuresDict({"my-metric": 0.95})

        error_msg = str(exc_info.value)
        assert "Python identifier syntax" in error_msg
        assert "my-metric" in error_msg  # Shows the invalid key
        assert "my_metric" in error_msg  # Shows valid alternative

    def test_setitem_validates_key_pattern(self):
        """Assignment should also validate key pattern."""
        measures = MeasuresDict()

        # Valid assignment works
        measures["valid_key"] = 0.95
        assert measures["valid_key"] == 0.95

        # Invalid assignment raises
        with pytest.raises(ValueError, match=r"must match pattern"):
            measures["invalid-key"] = 0.8


class TestMeasuresDictNumericEnforcement:
    """Test numeric type enforcement (Phase 0: warn, Phase 2: enforce)."""

    def test_accepts_numeric_types(self):
        """Should accept int, float, None."""
        measures = MeasuresDict(
            {
                "int_metric": 42,
                "float_metric": 3.14,
                "none_metric": None,
            }
        )
        assert measures["int_metric"] == 42
        assert measures["float_metric"] == 3.14
        assert measures["none_metric"] is None

    def test_rejects_non_numeric_string(self):
        """A string measure must be rejected, not warned about and submitted.

        This test asserted the opposite until the privacy-egress fix: a warn-and-
        pass guard let an evaluator returning text put that text on the wire.
        """
        with pytest.raises(TypeError, match="must be numeric"):
            MeasuresDict({"model_name": "gpt-4o-mini"})

    def test_rejects_boolean_metric(self):
        """Booleans must be rejected even though bool is a subclass of int."""
        with pytest.raises(TypeError, match="got bool"):
            MeasuresDict({"is_valid": True})

    def test_rejects_non_numeric_list(self):
        """Lists are rejected too -- a list can carry strings."""
        with pytest.raises(TypeError, match="must be numeric"):
            MeasuresDict({"scores": [0.9, 0.8, 0.7]})

    def test_error_message_says_why_and_what_to_do(self):
        """The error must name the key and type, the reason, and the remedy."""
        with pytest.raises(TypeError) as excinfo:
            MeasuresDict({"model_name": "gpt-4o"})

        message = str(excinfo.value)
        assert "model_name" in message
        assert "str" in message
        # WHY: the numeric-only contract is what keeps content off the wire.
        assert "Traigent backend" in message
        assert "no text content ever leaves" in message
        # WHAT TO DO INSTEAD.
        assert "Return numbers only from evaluators" in message
        assert "local" in message

    def test_setitem_rejects_non_numeric(self):
        """Assignment of non-numeric values is rejected on the same terms."""
        measures = MeasuresDict()

        with pytest.raises(TypeError, match="must be numeric"):
            measures["tag"] = "production"
        assert "tag" not in measures
