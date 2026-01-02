"""Comprehensive tests for traigent.utils.function_identity module.

Tests cover FunctionDescriptor, resolve_function_descriptor, sanitize_identifier,
and all internal helper functions with focus on edge cases and real-world scenarios.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from traigent.utils.function_identity import (
    FunctionDescriptor,
    resolve_function_descriptor,
    sanitize_identifier,
)


# Test functions for descriptor resolution
def simple_function():
    """A simple test function."""
    return "simple"


def function_with_args(x, y):
    """Function with arguments."""
    return x + y


class TestClass:
    """Test class for nested function testing."""

    def instance_method(self):
        """Instance method."""
        return "instance"

    @classmethod
    def class_method(cls):
        """Class method."""
        return "class"

    @staticmethod
    def static_method():
        """Static method."""
        return "static"

    class NestedClass:
        """Nested class."""

        def nested_method(self):
            """Nested method."""
            return "nested"


class TestFunctionDescriptor:
    """Test FunctionDescriptor dataclass."""

    def test_create_descriptor(self):
        """Test creating FunctionDescriptor."""
        desc = FunctionDescriptor(
            identifier="path_module_function",
            display_name="my_function",
            module="my_module",
            relative_path="src/my_file.py",
            slug="path_module_function_abcd1234",
        )

        assert desc.identifier == "path_module_function"
        assert desc.display_name == "my_function"
        assert desc.module == "my_module"
        assert desc.relative_path == "src/my_file.py"
        assert desc.slug == "path_module_function_abcd1234"

    def test_descriptor_is_frozen(self):
        """Test that FunctionDescriptor is immutable."""
        desc = FunctionDescriptor(
            identifier="test",
            display_name="test",
            module="test",
            relative_path="test.py",
            slug="test_abc123",
        )

        with pytest.raises(AttributeError):
            desc.identifier = "new_value"

    def test_descriptor_equality(self):
        """Test descriptor equality comparison."""
        desc1 = FunctionDescriptor(
            identifier="test",
            display_name="test",
            module="test",
            relative_path="test.py",
            slug="test_abc",
        )

        desc2 = FunctionDescriptor(
            identifier="test",
            display_name="test",
            module="test",
            relative_path="test.py",
            slug="test_abc",
        )

        assert desc1 == desc2

    def test_descriptor_hash(self):
        """Test that descriptor is hashable."""
        desc = FunctionDescriptor(
            identifier="test",
            display_name="test",
            module="test",
            relative_path="test.py",
            slug="test_abc",
        )

        # Should be hashable
        desc_set = {desc}
        assert desc in desc_set


class TestSanitizeIdentifier:
    """Test sanitize_identifier function."""

    def test_sanitize_simple_identifier(self):
        """Test sanitizing simple identifier."""
        result = sanitize_identifier("my_function")

        assert result.startswith("my_function_")
        assert len(result) > len("my_function")
        assert result.islower()

    def test_sanitize_with_special_characters(self):
        """Test sanitizing identifier with special characters."""
        result = sanitize_identifier("my-function.test@123")

        # Should replace special chars with underscores
        assert "_" in result
        assert "-" not in result
        assert "." not in result
        assert "@" not in result

    def test_sanitize_empty_string(self):
        """Test sanitizing empty string."""
        result = sanitize_identifier("")

        assert result == "unnamed"

    def test_sanitize_whitespace_only(self):
        """Test sanitizing whitespace-only string."""
        result = sanitize_identifier("   ")

        # Whitespace gets stripped, resulting in empty -> "identifier"
        assert result.startswith("identifier")

    def test_sanitize_long_identifier(self):
        """Test sanitizing very long identifier."""
        long_id = "a" * 200
        result = sanitize_identifier(long_id)

        # Should truncate to max_length (120) + hash
        assert len(result) < 150
        assert result.startswith("a")

    def test_sanitize_with_custom_max_length(self):
        """Test sanitizing with custom max length."""
        long_id = "test" * 50
        result = sanitize_identifier(long_id, max_length=20)

        # Should respect custom max_length
        prefix = result.rsplit("_", 1)[0]
        assert len(prefix) <= 20

    def test_sanitize_unicode_characters(self):
        """Test sanitizing Unicode characters."""
        result = sanitize_identifier("функция_测试")

        # Should handle Unicode gracefully
        assert "_" in result
        assert len(result) > 0

    def test_sanitize_path_like_string(self):
        """Test sanitizing path-like strings."""
        result = sanitize_identifier("src/utils/my_function.py")

        assert "_" in result
        assert "/" not in result
        assert result.islower()

    def test_sanitize_special_chars_only(self):
        """Test sanitizing string with only special characters."""
        result = sanitize_identifier("!@#$%^&*()")

        # Special chars get replaced and stripped, resulting in "identifier"
        assert result.startswith("identifier")

    def test_sanitize_numbers_only(self):
        """Test sanitizing numeric identifier."""
        result = sanitize_identifier("12345")

        assert result.startswith("12345_")

    def test_sanitize_mixed_case(self):
        """Test that sanitize converts to lowercase."""
        result = sanitize_identifier("MyFunction_Test")

        assert result.islower()
        assert "myfunction" in result

    def test_sanitize_deterministic(self):
        """Test that sanitization is deterministic."""
        input_str = "test_function"

        result1 = sanitize_identifier(input_str)
        result2 = sanitize_identifier(input_str)

        assert result1 == result2

    def test_sanitize_hash_suffix(self):
        """Test that hash suffix is included."""
        result = sanitize_identifier("test")

        # Should have format: text_hash
        parts = result.split("_")
        assert len(parts) >= 2
        # Last part should be 8-char hash
        assert len(parts[-1]) == 8


class TestResolveFunctionDescriptor:
    """Test resolve_function_descriptor function."""

    def test_resolve_simple_function(self):
        """Test resolving simple function descriptor."""
        desc = resolve_function_descriptor(simple_function)

        assert desc.display_name == "simple_function"
        assert "simple_function" in desc.identifier
        assert desc.module is not None
        assert desc.slug is not None

    def test_resolve_function_with_args(self):
        """Test resolving function with arguments."""
        desc = resolve_function_descriptor(function_with_args)

        assert desc.display_name == "function_with_args"
        assert "function_with_args" in desc.identifier

    def test_resolve_instance_method(self):
        """Test resolving instance method."""
        obj = TestClass()
        desc = resolve_function_descriptor(obj.instance_method)

        assert "instance_method" in desc.display_name
        assert "TestClass" in desc.identifier

    def test_resolve_class_method(self):
        """Test resolving class method."""
        desc = resolve_function_descriptor(TestClass.class_method)

        assert "class_method" in desc.display_name

    def test_resolve_static_method(self):
        """Test resolving static method."""
        desc = resolve_function_descriptor(TestClass.static_method)

        assert "static_method" in desc.display_name

    def test_resolve_nested_method(self):
        """Test resolving nested class method."""
        obj = TestClass.NestedClass()
        desc = resolve_function_descriptor(obj.nested_method)

        assert "nested_method" in desc.display_name
        assert "NestedClass" in desc.identifier

    def test_resolve_lambda(self):
        """Test resolving lambda function."""
        lambda_func = lambda x: x + 1  # noqa: E731
        desc = resolve_function_descriptor(lambda_func)

        # Display name includes full qualname for lambdas
        assert "<lambda>" in desc.display_name
        assert desc.identifier is not None

    def test_resolve_with_base_dir(self):
        """Test resolving with custom base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            desc = resolve_function_descriptor(simple_function, base_dir=tmpdir)

            assert desc.identifier is not None
            assert desc.relative_path is not None

    def test_resolve_decorated_function(self):
        """Test resolving decorated function."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        @decorator
        def decorated_func():
            return "decorated"

        desc = resolve_function_descriptor(decorated_func)

        # Should unwrap to original function
        assert "decorated_func" in desc.display_name

    def test_resolve_builtin_function(self):
        """Test resolving builtin function."""
        desc = resolve_function_descriptor(len)

        # Builtins have different behavior
        assert desc.display_name == "len"
        assert desc.module is not None

    def test_descriptor_fields_not_none(self):
        """Test that all descriptor fields are populated."""
        desc = resolve_function_descriptor(simple_function)

        assert desc.identifier is not None
        assert desc.display_name is not None
        assert desc.module is not None
        assert desc.relative_path is not None
        assert desc.slug is not None

    def test_descriptor_slug_filesystem_safe(self):
        """Test that slug is filesystem-safe."""
        desc = resolve_function_descriptor(simple_function)

        # Should not contain problematic characters
        assert "/" not in desc.slug
        assert "\\" not in desc.slug
        assert ":" not in desc.slug
        assert "*" not in desc.slug

    def test_descriptor_relative_path_uses_forward_slashes(self):
        """Test that relative_path uses forward slashes."""
        desc = resolve_function_descriptor(simple_function)

        # Should use POSIX-style forward slashes
        assert (
            "\\" not in desc.relative_path
        ), "Relative path should use forward slashes"


class TestFunctionIdentityEdgeCases:
    """Test edge cases and special scenarios."""

    def test_function_without_module(self):
        """Test function without __module__ attribute."""

        # Create function-like object without __module__
        class FakeFunction:
            def __call__(self):
                return "fake"

        fake_func = FakeFunction()
        desc = resolve_function_descriptor(fake_func)

        # Should handle gracefully
        assert desc.identifier is not None
        assert "unknown" in desc.module.lower() or desc.module is not None

    def test_function_without_qualname(self):
        """Test function without __qualname__ attribute."""

        # Can't actually remove __qualname__ from function objects in Python 3
        # Just test that resolution works with functions that have it
        def test_func():
            return "test"

        desc = resolve_function_descriptor(test_func)

        # Should have display name from __qualname__ or __name__
        assert desc.display_name is not None
        assert "test_func" in desc.display_name

    def test_function_in_main_module(self):
        """Test function in __main__ module."""

        def test_func():
            return "test"

        # Simulate __main__ module
        test_func.__module__ = "__main__"

        desc = resolve_function_descriptor(test_func)

        assert desc.module is not None

    def test_very_long_function_name(self):
        """Test function with very long name."""
        # Create function with long name
        long_name = "a" * 500

        def test_func():
            return "test"

        test_func.__qualname__ = long_name

        desc = resolve_function_descriptor(test_func)

        # Slug should be truncated but valid
        assert len(desc.slug) < 500

    def test_function_with_special_chars_in_name(self):
        """Test function with special characters in qualname."""

        def test_func():
            return "test"

        test_func.__qualname__ = "Test<lambda>.function"

        desc = resolve_function_descriptor(test_func)

        # Should sanitize special characters
        assert desc.identifier is not None

    def test_descriptor_uniqueness(self):
        """Test that different functions get different descriptors."""

        def func1():
            pass

        def func2():
            pass

        desc1 = resolve_function_descriptor(func1)
        desc2 = resolve_function_descriptor(func2)

        # Different functions should have different identifiers
        assert desc1.identifier != desc2.identifier

    def test_descriptor_consistency(self):
        """Test that same function gets consistent descriptor."""
        desc1 = resolve_function_descriptor(simple_function)
        desc2 = resolve_function_descriptor(simple_function)

        assert desc1.identifier == desc2.identifier
        assert desc1.slug == desc2.slug

    def test_nested_function(self):
        """Test resolving nested function."""

        def outer():
            def inner():
                return "inner"

            return inner

        inner_func = outer()
        desc = resolve_function_descriptor(inner_func)

        assert "inner" in desc.display_name
        assert "outer" in desc.identifier

    def test_closure_function(self):
        """Test resolving closure function."""

        def make_adder(n):
            def adder(x):
                return x + n

            return adder

        add_five = make_adder(5)
        desc = resolve_function_descriptor(add_five)

        # Display name includes full qualname
        assert "adder" in desc.display_name

    def test_partial_function(self):
        """Test resolving functools.partial function."""
        from functools import partial

        def add(x, y):
            return x + y

        add_five = partial(add, 5)
        desc = resolve_function_descriptor(add_five)

        # Should unwrap partial
        assert desc.identifier is not None

    def test_class_as_callable(self):
        """Test resolving class used as callable."""

        class CallableClass:
            def __call__(self):
                return "called"

        instance = CallableClass()
        desc = resolve_function_descriptor(instance)

        assert desc.identifier is not None


class TestSanitizeIdentifierRobustness:
    """Test robustness of sanitize_identifier."""

    def test_sanitize_with_null_bytes(self):
        """Test sanitizing string with null bytes."""
        result = sanitize_identifier("test\x00func")

        assert "\x00" not in result

    def test_sanitize_with_newlines(self):
        """Test sanitizing string with newlines."""
        result = sanitize_identifier("test\nfunc\r\nmore")

        assert "\n" not in result
        assert "\r" not in result

    def test_sanitize_with_tabs(self):
        """Test sanitizing string with tabs."""
        result = sanitize_identifier("test\tfunc")

        assert "\t" not in result

    def test_sanitize_consecutive_special_chars(self):
        """Test sanitizing consecutive special characters."""
        result = sanitize_identifier("test---func___more")

        # Should collapse consecutive underscores
        assert result.startswith("test")

    def test_sanitize_leading_numbers(self):
        """Test sanitizing identifier starting with numbers."""
        result = sanitize_identifier("123_function")

        assert result.startswith("123")

    def test_sanitize_trailing_underscores(self):
        """Test sanitizing removes trailing underscores before hash."""
        result = sanitize_identifier("test___")

        # Should not have trailing underscores before hash
        parts = result.split("_")
        assert parts[-2] != ""

    def test_sanitize_emoji(self):
        """Test sanitizing string with emoji."""
        result = sanitize_identifier("test_🚀_function")

        # Should handle emoji
        assert result.startswith("test")
        assert "🚀" not in result


class TestResolveDescriptorPathHandling:
    """Test path handling in resolve_function_descriptor."""

    def test_resolve_with_pathlib_base_dir(self):
        """Test resolving with Path object as base_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            desc = resolve_function_descriptor(simple_function, base_dir=base_path)

            assert desc.identifier is not None

    def test_resolve_with_string_base_dir(self):
        """Test resolving with string as base_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            desc = resolve_function_descriptor(simple_function, base_dir=tmpdir)

            assert desc.identifier is not None

    def test_resolve_with_invalid_base_dir(self):
        """Test resolving with invalid base directory."""
        # Should handle gracefully
        desc = resolve_function_descriptor(
            simple_function, base_dir="/nonexistent/path"
        )

        assert desc.identifier is not None

    def test_resolve_without_base_dir(self):
        """Test resolving without base_dir uses cwd."""
        desc = resolve_function_descriptor(simple_function, base_dir=None)

        assert desc.identifier is not None
        assert desc.relative_path is not None
