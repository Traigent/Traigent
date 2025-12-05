"""AST transformer for safe seamless configuration injection.

This module provides safe AST transformation capabilities to replace
variable assignments without using exec(), maintaining the same
functionality while eliminating security risks.
"""

# Traceability: CONC-ConfigInjection CONC-Invocation FUNC-API-ENTRY FUNC-INVOKERS REQ-INJ-002 SYNC-OptimizationFlow CONC-Layer-Core

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConfigTransformer(ast.NodeTransformer):
    # Dangerous dictionary keys to reject
    DANGEROUS_KEYS = {
        "__proto__",
        "__constructor__",
        "constructor",
        "prototype",
        "__defineGetter__",
        "__defineSetter__",
        "__lookupGetter__",
        "__lookupSetter__",
    }
    """AST transformer that safely replaces variable assignments with config values.

    This transformer walks the AST and replaces assignments to variables
    that match configuration keys with the configured values.

    Example:
        >>> config = {"model": "gpt-4", "temperature": 0.2}
        >>> transformer = ConfigTransformer(config)
        >>> tree = ast.parse("model = 'gpt-3.5'\\ntemperature = 0.7")
        >>> modified_tree = transformer.visit(tree)
        >>> # model assignment now has value 'gpt-4'
        >>> # temperature assignment now has value 0.2
    """

    def __init__(self, config: dict[str, Any], max_depth: int = 10) -> None:
        """Initialize transformer with configuration to inject.

        Args:
            config: Dictionary of variable names to values to inject
            max_depth: Maximum recursion depth for nested structures
        """
        # Validate config keys to prevent injection
        self._validate_config_keys(config)
        self.config = config
        self.modified_vars: set[str] = set()
        self.in_function = False
        self.function_level = 0
        self.max_depth = max_depth
        self.current_depth = 0

    def _validate_config_keys(self, config: dict[str, Any]) -> None:
        """Validate configuration keys to prevent injection attacks.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If config contains dangerous keys
        """
        dangerous_keys = {
            "__import__",
            "__builtins__",
            "__code__",
            "__class__",
            "__globals__",
            "__dict__",
            "exec",
            "eval",
            "compile",
            "open",
            "input",
            "__file__",
            "__name__",
            "__loader__",
        }

        for key in config:
            if not isinstance(key, str):
                raise ValueError(f"Config key must be string, got {type(key)}")

            # Check for dangerous keys
            if key in dangerous_keys or key.startswith("__"):
                raise ValueError(f"Dangerous config key not allowed: {key}")

            # Check for valid Python identifier
            if not key.isidentifier():
                raise ValueError(f"Config key must be valid Python identifier: {key}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track when we enter/exit function definitions."""
        self.function_level += 1
        was_in_function = self.in_function
        self.in_function = True

        # Visit child nodes
        self.generic_visit(node)

        self.function_level -= 1
        if self.function_level == 0:
            self.in_function = was_in_function

        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        """Track when we enter/exit async function definitions."""
        self.function_level += 1
        was_in_function = self.in_function
        self.in_function = True

        # Visit child nodes
        self.generic_visit(node)

        self.function_level -= 1
        if self.function_level == 0:
            self.in_function = was_in_function

        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Replace assignments to configured variables with config values.

        Only replaces simple assignments like:
            model = "gpt-3.5"
            temperature = 0.7

        Does not replace:
            - Tuple unpacking: a, b = 1, 2
            - Attribute assignments: self.model = "gpt-3.5"
            - Subscript assignments: config["model"] = "gpt-3.5"
        """
        # Only process assignments inside functions
        if not self.in_function:
            return node

        # Only handle simple single-target assignments
        if len(node.targets) != 1:
            return node

        target = node.targets[0]

        # Only handle Name targets (simple variables)
        if not isinstance(target, ast.Name):
            return node

        var_name = target.id

        # Check if this variable is in our config
        if var_name not in self.config:
            return node

        # Create new value node based on config value
        config_value = self.config[var_name]
        new_value = self._create_value_node(config_value)

        if new_value is None:
            # If we can't create a safe value node, keep original
            return node

        # Track that we modified this variable
        self.modified_vars.add(var_name)

        # Create new assignment with the config value
        new_node = ast.Assign(
            targets=[target], value=new_value, type_comment=node.type_comment
        )

        # Copy location information
        ast.copy_location(new_node, node)

        return new_node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        """Handle annotated assignments like: model: str = "gpt-3.5" """
        # Only process assignments inside functions
        if not self.in_function:
            return node

        # Only handle Name targets (simple variables)
        if not isinstance(node.target, ast.Name):
            return node

        var_name = node.target.id

        # Check if this variable is in our config
        if var_name not in self.config:
            return node

        # Create new value node based on config value
        config_value = self.config[var_name]
        new_value = self._create_value_node(config_value)

        if new_value is None:
            # If we can't create a safe value node, keep original
            return node

        # Track that we modified this variable
        self.modified_vars.add(var_name)

        # Create new annotated assignment with the config value
        new_node = ast.AnnAssign(
            target=node.target,
            annotation=node.annotation,
            value=new_value,
            simple=node.simple,
        )

        # Copy location information
        ast.copy_location(new_node, node)

        return new_node

    def _create_value_node(self, value: Any, depth: int = 0) -> ast.expr | None:
        """Create an AST node for a configuration value.

        Only creates nodes for safe, simple types to prevent injection attacks.

        Args:
            value: The configuration value to convert to an AST node
            depth: Current recursion depth

        Returns:
            AST expression node or None if value type is not supported
        """
        # Prevent deep recursion attacks
        if depth > self.max_depth:
            logger.warning(f"Max depth {self.max_depth} exceeded in value creation")
            return None
        if value is None:
            return ast.Constant(value=None)
        elif isinstance(value, bool):
            # bool must come before int since bool is a subclass of int
            return ast.Constant(value=value)
        elif isinstance(value, (int, float)):
            return ast.Constant(value=value)
        elif isinstance(value, str):
            return ast.Constant(value=value)
        elif isinstance(value, list):
            # Limit list size to prevent memory exhaustion
            if len(value) > 1000:
                logger.warning(f"List too large ({len(value)} items), rejecting")
                return None

            # Only allow lists of simple values
            elements = []
            for item in value:
                item_node = self._create_value_node(item, depth + 1)
                if item_node is None:
                    # If any item can't be safely converted, reject the whole list
                    return None
                elements.append(item_node)
            return ast.List(elts=elements, ctx=ast.Load())
        elif isinstance(value, tuple):
            # Limit tuple size to prevent memory exhaustion
            if len(value) > 1000:
                logger.warning(f"Tuple too large ({len(value)} items), rejecting")
                return None

            # Only allow tuples of simple values
            elements = []
            for item in value:
                item_node = self._create_value_node(item, depth + 1)
                if item_node is None:
                    # If any item can't be safely converted, reject the whole tuple
                    return None
                elements.append(item_node)
            return ast.Tuple(elts=elements, ctx=ast.Load())
        elif isinstance(value, dict):
            # Limit dict size to prevent memory exhaustion
            if len(value) > 1000:
                logger.warning(f"Dict too large ({len(value)} items), rejecting")
                return None

            # Only allow dicts with string keys and simple values
            keys: list[ast.expr | None] = []
            values: list[ast.expr] = []
            for k, v in value.items():
                if not isinstance(k, str):
                    # Only string keys allowed
                    return None

                # Validate key doesn't contain dangerous patterns
                if "__" in k or k in self.DANGEROUS_KEYS:
                    logger.warning(f"Dangerous dict key rejected: {k}")
                    return None

                key_node: ast.expr = ast.Constant(value=k)
                value_node = self._create_value_node(v, depth + 1)
                if value_node is None:
                    # If any value can't be safely converted, reject the whole dict
                    return None
                keys.append(key_node)
                values.append(value_node)
            return ast.Dict(keys=keys, values=values)
        else:
            # Unsupported type - don't convert to prevent injection
            return None

    def get_modified_variables(self) -> set[str]:
        """Return the set of variables that were modified.

        Returns:
            Set of variable names that were replaced with config values
        """
        return self.modified_vars.copy()


class SafeASTCompiler:
    """Safe compiler for transformed AST trees.

    This class provides safe compilation of AST trees with validation
    to ensure no dangerous operations are present.
    """

    # Dangerous AST node types that should not be present
    DANGEROUS_NODES = {
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        # Note: We allow ast.Call since functions need to call other functions
        # but we validate specific dangerous calls
    }

    # Dangerous function names that should not be called
    DANGEROUS_CALLS = {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "file",
        "input",
        "raw_input",
        "execfile",
        "reload",
        "vars",
        "locals",
        "globals",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "__builtins__",
        "__dict__",
        "__class__",
        "type",
        "isinstance",
        "issubclass",
        "callable",
        "classmethod",
        "staticmethod",
        "property",
        "super",
        "object",
        "id",
        "help",
    }

    # Dangerous attribute patterns
    DANGEROUS_ATTRIBUTES = {
        "__code__",
        "__globals__",
        "__builtins__",
        "__import__",
        "__loader__",
        "__package__",
        "__spec__",
        "__cached__",
        "__file__",
        "__mro__",
        "__bases__",
        "__subclasses__",
    }

    @classmethod
    def validate_ast(cls, tree: ast.AST) -> bool:
        """Validate that an AST tree is safe to compile.

        Args:
            tree: The AST tree to validate

        Returns:
            True if the tree is safe, False otherwise
        """
        for node in ast.walk(tree):
            # Check for dangerous node types
            if type(node) in cls.DANGEROUS_NODES:
                return False

            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.DANGEROUS_CALLS:
                        return False
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in cls.DANGEROUS_CALLS:
                        return False

            # Check for access to dangerous attributes
            if isinstance(node, ast.Attribute):
                # Check against dangerous attributes list
                if node.attr in cls.DANGEROUS_ATTRIBUTES:
                    return False

                if node.attr.startswith("__") and node.attr.endswith("__"):
                    # Allow common magic methods
                    allowed_magic = {
                        "__init__",
                        "__str__",
                        "__repr__",
                        "__len__",
                        "__getitem__",
                        "__setitem__",
                        "__delitem__",
                        "__iter__",
                        "__next__",
                        "__contains__",
                        "__eq__",
                        "__ne__",
                        "__lt__",
                        "__le__",
                        "__gt__",
                        "__ge__",
                        "__add__",
                        "__sub__",
                        "__mul__",
                        "__div__",
                        "__mod__",
                        "__and__",
                        "__or__",
                        "__xor__",
                        "__not__",
                        "__enter__",
                        "__exit__",
                        "__call__",
                        "__hash__",
                        "__bool__",
                        "__bytes__",
                        "__format__",
                    }
                    if node.attr not in allowed_magic:
                        return False

        return True

    @classmethod
    def compile_safe(cls, source: str, filename: str = "<config>") -> Any | None:
        """Safely compile source code after validation.

        Args:
            source: Source code to compile
            filename: Filename for error reporting

        Returns:
            Compiled code object or None if validation fails
        """
        try:
            # Parse the source
            tree = ast.parse(source, filename)

            # Validate the AST
            if not cls.validate_ast(tree):
                logger.error("AST validation failed - dangerous operations detected")
                return None

            # Compile the safe AST
            return compile(tree, filename, "exec")

        except SyntaxError as e:
            logger.error(f"Syntax error in source: {e}")
            return None
        except Exception as e:
            logger.error(f"Error compiling source: {e}")
            return None

    @classmethod
    def compile_ast_safe(cls, tree: ast.AST, filename: str = "<ast>") -> Any | None:
        """Safely compile an AST tree to a code object.

        Args:
            tree: The AST tree to compile
            filename: The filename to use in the code object

        Returns:
            Compiled code object

        Raises:
            ValueError: If the AST tree contains dangerous operations
            SyntaxError: If the AST tree has syntax errors
        """
        # Validate the tree is safe
        if not cls.validate_ast(tree):
            raise ValueError("AST tree contains dangerous operations") from None

        # Fix missing location information
        ast.fix_missing_locations(tree)

        # Ensure tree is a Module for compilation
        if not isinstance(tree, ast.Module):
            raise ValueError("AST tree must be a Module for exec compilation")

        # Compile the tree
        try:
            return compile(tree, filename, "exec")
        except SyntaxError as e:
            raise SyntaxError(f"Failed to compile transformed AST: {e}") from None
