"""Tests for MeasuresDict validation.

This module tests the MeasuresDict class to ensure it properly validates
measures and enforces cardinality limits.
"""

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
                "bool_val": True,
                "none_val": None,
            }
        )
        assert len(measures) == 5
        assert measures["int_val"] == 42
        assert measures["float_val"] == 3.14
        assert measures["str_val"] == "test"
        assert measures["bool_val"] is True
        assert measures["none_val"] is None

    def test_rejects_list_type(self):
        """Should reject list values."""
        with pytest.raises(TypeError, match="must be primitive type"):
            MeasuresDict({"list_val": [1, 2, 3]})

    def test_rejects_dict_type(self):
        """Should reject dict values."""
        with pytest.raises(TypeError, match="must be primitive type"):
            MeasuresDict({"dict_val": {"nested": "dict"}})

    def test_rejects_tuple_type(self):
        """Should reject tuple values."""
        with pytest.raises(TypeError, match="must be primitive type"):
            MeasuresDict({"tuple_val": (1, 2, 3)})

    def test_rejects_object_type(self):
        """Should reject object values."""

        class CustomObject:
            pass

        with pytest.raises(TypeError, match="must be primitive type"):
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

    def test_assignment_validates_types(self):
        """Should validate types on assignment."""
        measures = MeasuresDict()

        # Valid assignment
        measures["valid"] = 42
        assert measures["valid"] == 42

        # Invalid value type
        with pytest.raises(TypeError, match="Value must be primitive"):
            measures["invalid"] = [1, 2, 3]

    def test_update_method_validates(self):
        """Should validate on update() operation."""
        measures = MeasuresDict({"existing": 1})

        # Valid update
        measures.update({"new_key": 2})
        assert measures["new_key"] == 2

        # Invalid type in update should raise during __setitem__
        with pytest.raises(TypeError, match="Value must be primitive"):
            measures.update({"bad_key": [1, 2, 3]})

    def test_or_operator_validates(self):
        """Should validate on |= operator (dict union)."""
        measures = MeasuresDict({"existing": 1})

        # Valid union
        measures |= {"new_key": 2}
        assert measures["new_key"] == 2

        # Invalid type in union should raise
        with pytest.raises(TypeError, match="Value must be primitive"):
            measures |= {"bad_key": {"nested": "dict"}}

    def test_setdefault_validates(self):
        """Should validate on setdefault() operation."""
        measures = MeasuresDict()

        # Valid setdefault
        result = measures.setdefault("key1", 42)
        assert result == 42
        assert measures["key1"] == 42

        # Invalid type in setdefault should raise
        with pytest.raises(TypeError, match="Value must be primitive"):
            measures.setdefault("key2", [1, 2, 3])

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
                "bool_metric": True,
                "none_metric": None,
                "negative_int": -10,
                "negative_float": -2.5,
                "zero": 0,
                "empty_string": "",
            }
        )
        assert len(measures) == 9
        assert measures["int_metric"] == 42
        assert measures["float_metric"] == 3.14
        assert measures["str_metric"] == "success"
        assert measures["bool_metric"] is True
        assert measures["none_metric"] is None
        assert measures["negative_int"] == -10
        assert measures["negative_float"] == -2.5
        assert measures["zero"] == 0
        assert measures["empty_string"] == ""
