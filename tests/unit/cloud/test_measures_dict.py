"""Tests for MeasuresDict validation.

This module tests the MeasuresDict class to ensure it properly validates
measures and enforces cardinality limits.

Phase 0 tests added for:
- Key pattern validation (Python identifier syntax)
- Non-numeric value warnings (gradual migration to numeric-only)
"""

import logging

import pytest

from traigent.cloud.dtos import MeasuresDict


class TestMeasuresDictValidation:
    """Tests for MeasuresDict type and cardinality validation."""

    def test_accepts_valid_primitive_types(self):
        """Should accept all primitive types."""
        measures = MeasuresDict(
            {
                "int_val": 42,
                "float_val": 3.14,
                "str_val": "test",
                "none_val": None,
            }
        )
        assert len(measures) == 4
        assert measures["int_val"] == 42
        assert measures["float_val"] == 3.14
        assert measures["str_val"] == "test"
        assert measures["none_val"] is None

    def test_accepts_list_type_with_warning(self, caplog):
        """Should accept list values with warning (Phase 0)."""
        with caplog.at_level(logging.WARNING):
            measures = MeasuresDict({"list_val": [1, 2, 3]})

        # Phase 0: Accept but warn
        assert measures["list_val"] == [1, 2, 3]

        # Check warning was logged
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)

    def test_accepts_dict_type_with_warning(self, caplog):
        """Should accept dict values with warning (Phase 0)."""
        with caplog.at_level(logging.WARNING):
            measures = MeasuresDict({"dict_val": {"nested": "dict"}})

        # Phase 0: Accept but warn
        assert measures["dict_val"] == {"nested": "dict"}

        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)

    def test_accepts_tuple_type_with_warning(self, caplog):
        """Should accept tuple values with warning (Phase 0)."""
        with caplog.at_level(logging.WARNING):
            measures = MeasuresDict({"tuple_val": (1, 2, 3)})

        # Phase 0: Accept but warn
        assert measures["tuple_val"] == (1, 2, 3)

        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)

    def test_accepts_object_type_with_warning(self, caplog):
        """Should accept object values with warning (Phase 0)."""

        class CustomObject:
            pass

        obj = CustomObject()
        with caplog.at_level(logging.WARNING):
            measures = MeasuresDict({"object_val": obj})

        # Phase 0: Accept but warn
        assert measures["object_val"] is obj

        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)

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

    def test_assignment_warns_on_non_numeric(self, caplog):
        """Should warn on non-numeric assignment (Phase 0)."""
        measures = MeasuresDict()

        # Valid numeric assignment
        measures["valid"] = 42
        assert measures["valid"] == 42

        # Non-numeric value triggers warning
        with caplog.at_level(logging.WARNING):
            measures["invalid"] = [1, 2, 3]

        # Phase 0: Accept but warn
        assert measures["invalid"] == [1, 2, 3]

        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)

    def test_update_method_warns_on_non_numeric(self, caplog):
        """Should warn on non-numeric update() (Phase 0)."""
        measures = MeasuresDict({"existing": 1})

        # Valid update
        measures.update({"new_key": 2})
        assert measures["new_key"] == 2

        # Non-numeric value triggers warning
        with caplog.at_level(logging.WARNING):
            measures.update({"bad_key": [1, 2, 3]})

        # Phase 0: Accept but warn
        assert measures["bad_key"] == [1, 2, 3]

        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)

    def test_or_operator_warns_on_non_numeric(self, caplog):
        """Should warn on non-numeric |= operator (Phase 0)."""
        measures = MeasuresDict({"existing": 1})

        # Valid union
        measures |= {"new_key": 2}
        assert measures["new_key"] == 2

        # Non-numeric value triggers warning
        with caplog.at_level(logging.WARNING):
            measures |= {"bad_key": {"nested": "dict"}}

        # Phase 0: Accept but warn
        assert measures["bad_key"] == {"nested": "dict"}

        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)

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

    def test_setdefault_warns_on_non_numeric(self, caplog):
        """Should warn on non-numeric setdefault() (Phase 0)."""
        measures = MeasuresDict()

        # Valid setdefault
        result = measures.setdefault("key1", 42)
        assert result == 42
        assert measures["key1"] == 42

        # Non-numeric value triggers warning
        with caplog.at_level(logging.WARNING):
            result = measures.setdefault("key2", [1, 2, 3])

        # Phase 0: Accept but warn
        assert result == [1, 2, 3]
        assert measures["key2"] == [1, 2, 3]

        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)

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
        keys = [k for k in measures]
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
        """Should accept mixed primitive types in single dict."""
        measures = MeasuresDict(
            {
                "int_metric": 42,
                "float_metric": 3.14,
                "str_metric": "success",
                "none_metric": None,
                "negative_int": -10,
                "negative_float": -2.5,
                "zero": 0,
                "empty_string": "",
            }
        )
        assert len(measures) == 8
        assert measures["int_metric"] == 42
        assert measures["float_metric"] == 3.14
        assert measures["str_metric"] == "success"
        assert measures["none_metric"] is None
        assert measures["negative_int"] == -10
        assert measures["negative_float"] == -2.5
        assert measures["zero"] == 0
        assert measures["empty_string"] == ""


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

    def test_warns_on_non_numeric_string(self, caplog):
        """Should warn (not reject) for string values in Phase 0."""
        with caplog.at_level(logging.WARNING):
            measures = MeasuresDict({"model_name": "gpt-4o-mini"})

        # Phase 0: Accept but warn
        assert measures["model_name"] == "gpt-4o-mini"

        # Check warning was logged
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)
        assert any("v2.0" in msg for msg in warning_messages)

    def test_rejects_boolean_metric(self):
        """Booleans must be rejected even though bool is a subclass of int."""
        with pytest.raises(TypeError, match="got bool"):
            MeasuresDict({"is_valid": True})

    def test_warns_on_non_numeric_list(self, caplog):
        """Should warn for list values in Phase 0."""
        with caplog.at_level(logging.WARNING):
            measures = MeasuresDict({"scores": [0.9, 0.8, 0.7]})

        # Phase 0: Accept but warn
        assert measures["scores"] == [0.9, 0.8, 0.7]

    def test_warning_includes_helpful_hint(self, caplog):
        """Warning should suggest using metadata for non-numeric data."""
        with caplog.at_level(logging.WARNING):
            MeasuresDict({"model_name": "gpt-4o"})

        warning_messages = " ".join(
            record.message for record in caplog.records if record.levelname == "WARNING"
        )
        assert "metadata" in warning_messages.lower()

    def test_setitem_warns_on_non_numeric(self, caplog):
        """Assignment of non-numeric values should also warn."""
        measures = MeasuresDict()

        with caplog.at_level(logging.WARNING):
            measures["tag"] = "production"

        # Phase 0: Accept but warn
        assert measures["tag"] == "production"

        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("non-numeric" in msg.lower() for msg in warning_messages)
