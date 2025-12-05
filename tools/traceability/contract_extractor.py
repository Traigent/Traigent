#!/usr/bin/env python3
"""Contract Extractor Tool - Extract and verify abstract method contracts.

Traceability: CONC-Layer-Infra CONC-Quality-Maintainability REQ-TRACE-GAP-004

This tool extracts contracts (abstract methods) from base classes and verifies
that all implementations properly implement the required methods.

Usage:
    python contract_extractor.py --base traigent/optimizers/base.py --impl traigent/optimizers/
    python contract_extractor.py --scan traigent/ --output contracts.yml
    python contract_extractor.py --verify traigent/ --report gaps.yml
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MethodContract:
    """Represents an abstract method contract."""

    name: str
    is_async: bool
    parameters: list[str]
    return_annotation: str | None
    docstring: str | None
    line_number: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_async": self.is_async,
            "parameters": self.parameters,
            "return_annotation": self.return_annotation,
            "docstring": self.docstring,
            "line_number": self.line_number,
        }


@dataclass
class ClassContract:
    """Represents a base class contract."""

    name: str
    file_path: str
    line_number: int
    abstract_methods: list[MethodContract]
    parent_classes: list[str]
    docstring: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "abstract_methods": [m.to_dict() for m in self.abstract_methods],
            "parent_classes": self.parent_classes,
            "docstring": self.docstring,
        }


@dataclass
class Implementation:
    """Represents a class implementing a contract."""

    name: str
    file_path: str
    line_number: int
    base_classes: list[str]
    implemented_methods: list[str]
    missing_methods: list[str] = field(default_factory=list)
    is_compliant: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "base_classes": self.base_classes,
            "implemented_methods": self.implemented_methods,
            "missing_methods": self.missing_methods,
            "is_compliant": self.is_compliant,
        }


class ContractExtractor:
    """Extracts contracts from Python source files."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.contracts: dict[str, ClassContract] = {}
        self.implementations: list[Implementation] = []

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[DEBUG] {message}")

    def extract_contract(self, file_path: Path) -> list[ClassContract]:
        """Extract contracts from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of ClassContract objects
        """
        try:
            src = file_path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError) as e:
            self.log(f"Failed to parse {file_path}: {e}")
            return []

        contracts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                contract = self._extract_class_contract(node, file_path)
                if contract and contract.abstract_methods:
                    contracts.append(contract)
                    self.contracts[contract.name] = contract

        return contracts

    def _extract_class_contract(
        self, node: ast.ClassDef, file_path: Path
    ) -> ClassContract | None:
        """Extract contract from a class definition."""
        # Check if class inherits from ABC
        is_abc = self._inherits_from_abc(node)
        if not is_abc:
            return None

        abstract_methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_abstract_method(item):
                    method = self._extract_method_contract(item)
                    abstract_methods.append(method)

        parent_classes = [self._get_base_name(base) for base in node.bases]

        docstring = ast.get_docstring(node)

        return ClassContract(
            name=node.name,
            file_path=str(file_path),
            line_number=node.lineno,
            abstract_methods=abstract_methods,
            parent_classes=parent_classes,
            docstring=docstring,
        )

    def _inherits_from_abc(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from ABC."""
        for base in node.bases:
            base_name = self._get_base_name(base)
            if base_name in ("ABC", "abc.ABC"):
                return True
        return False

    def _get_base_name(self, base: ast.expr) -> str:
        """Get string representation of base class."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self._get_base_name(base.value)}.{base.attr}"
        elif isinstance(base, ast.Subscript):
            return self._get_base_name(base.value)
        return "Unknown"

    def _is_abstract_method(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if method has @abstractmethod decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                return True
            elif (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "abstractmethod"
            ):
                return True
        return False

    def _extract_method_contract(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> MethodContract:
        """Extract contract from a method definition."""
        params = []
        for arg in node.args.args:
            if arg.arg != "self":
                params.append(arg.arg)

        return_annotation = None
        if node.returns:
            # ast.unparse requires Python 3.9+
            if hasattr(ast, "unparse"):
                return_annotation = ast.unparse(node.returns)
            else:
                # Fallback for Python 3.8
                return_annotation = str(type(node.returns).__name__)

        docstring = ast.get_docstring(node)

        return MethodContract(
            name=node.name,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            parameters=params,
            return_annotation=return_annotation,
            docstring=docstring,
            line_number=node.lineno,
        )

    def find_implementations(
        self, search_path: Path, base_class_name: str
    ) -> list[Implementation]:
        """Find all implementations of a base class.

        Args:
            search_path: Directory to search
            base_class_name: Name of base class

        Returns:
            List of Implementation objects
        """
        implementations = []

        # First pass: collect all class definitions and their methods
        all_classes: dict[str, tuple[list[str], list[str]]] = (
            {}
        )  # name -> (bases, methods)

        for py_file in search_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                src = py_file.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = [self._get_base_name(base) for base in node.bases]
                    methods = []
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.append(item.name)
                    all_classes[node.name] = (bases, methods)

        # Store for inheritance chain resolution
        self._all_classes = all_classes

        # Second pass: find implementations
        for py_file in search_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                src = py_file.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    impl = self._check_implementation(node, py_file, base_class_name)
                    if impl:
                        implementations.append(impl)
                        self.implementations.append(impl)

        return implementations

    def _get_inherited_methods(
        self, class_name: str, visited: set[str] | None = None
    ) -> set[str]:
        """Get all methods inherited from parent classes.

        Args:
            class_name: Name of class to check
            visited: Set of already visited classes (to prevent infinite loops)

        Returns:
            Set of inherited method names
        """
        if visited is None:
            visited = set()

        if class_name in visited:
            return set()

        visited.add(class_name)

        if not hasattr(self, "_all_classes") or class_name not in self._all_classes:
            return set()

        bases, methods = self._all_classes[class_name]
        inherited = set(methods)

        for base in bases:
            inherited |= self._get_inherited_methods(base, visited)

        return inherited

    def _check_implementation(
        self, node: ast.ClassDef, file_path: Path, base_class_name: str
    ) -> Implementation | None:
        """Check if a class implements the specified base class."""
        base_classes = [self._get_base_name(base) for base in node.bases]

        # Check if class inherits from base (directly or indirectly via name match)
        if not any(base_class_name in base for base in base_classes):
            return None

        # Skip abstract classes (they don't need to implement all methods)
        if self._inherits_from_abc(node):
            return None

        # Get all methods defined in this class AND inherited from parents
        own_methods: list[str] = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                own_methods.append(item.name)

        # Also include methods inherited from intermediate classes
        inherited_methods = self._get_inherited_methods(node.name)
        all_implemented = list(set(own_methods) | inherited_methods)

        return Implementation(
            name=node.name,
            file_path=str(file_path),
            line_number=node.lineno,
            base_classes=base_classes,
            implemented_methods=all_implemented,
        )

    def verify_compliance(self, base_class_name: str) -> list[Implementation]:
        """Verify that all implementations satisfy the contract.

        Args:
            base_class_name: Name of base class to check

        Returns:
            List of implementations with compliance status
        """
        if base_class_name not in self.contracts:
            raise ValueError(f"Contract for {base_class_name} not found")

        contract = self.contracts[base_class_name]
        required_methods = {m.name for m in contract.abstract_methods}

        for impl in self.implementations:
            if any(base_class_name in base for base in impl.base_classes):
                implemented = set(impl.implemented_methods)
                missing = required_methods - implemented
                impl.missing_methods = sorted(missing)
                impl.is_compliant = len(missing) == 0

        return self.implementations

    def scan_directory(self, directory: Path) -> dict[str, Any]:
        """Scan directory for all contracts and implementations.

        Args:
            directory: Directory to scan

        Returns:
            Dictionary with contracts and compliance info
        """
        # First pass: find all contracts (ABC classes)
        for py_file in directory.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            self.extract_contract(py_file)

        self.log(f"Found {len(self.contracts)} contracts")

        # Second pass: find implementations and verify
        for contract_name in self.contracts:
            self.find_implementations(directory, contract_name)

        # Verify compliance
        for contract_name in self.contracts:
            self.verify_compliance(contract_name)

        # Build report
        report = {
            "contracts": {},
            "implementations": [],
            "summary": {
                "total_contracts": len(self.contracts),
                "total_implementations": len(self.implementations),
                "compliant": 0,
                "non_compliant": 0,
            },
        }

        for name, contract in self.contracts.items():
            report["contracts"][name] = contract.to_dict()

        for impl in self.implementations:
            report["implementations"].append(impl.to_dict())
            if impl.is_compliant:
                report["summary"]["compliant"] += 1
            else:
                report["summary"]["non_compliant"] += 1

        return report


def output_yaml(data: dict[str, Any], output_path: Path | None = None) -> str:
    """Output data as YAML-like format.

    Args:
        data: Dictionary to output
        output_path: Optional file path to write to

    Returns:
        YAML-formatted string
    """
    lines = []

    def _format_value(value: Any, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        if isinstance(value, dict):
            result = []
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    result.append(f"{prefix}{k}:")
                    result.extend(_format_value(v, indent + 1))
                else:
                    result.append(
                        f"{prefix}{k}: {json.dumps(v) if isinstance(v, str) else v}"
                    )
            return result
        elif isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    result.append(f"{prefix}-")
                    for k, v in item.items():
                        if isinstance(v, (dict, list)):
                            result.append(f"{prefix}  {k}:")
                            result.extend(_format_value(v, indent + 2))
                        else:
                            result.append(
                                f"{prefix}  {k}: {json.dumps(v) if isinstance(v, str) else v}"
                            )
                else:
                    result.append(
                        f"{prefix}- {json.dumps(item) if isinstance(item, str) else item}"
                    )
            return result
        return [f"{prefix}{value}"]

    lines.extend(_format_value(data))
    yaml_str = "\n".join(lines)

    if output_path:
        output_path.write_text(yaml_str, encoding="utf-8")
        print(f"Report written to {output_path}")

    return yaml_str


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and verify abstract method contracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --base traigent/optimizers/base.py --impl traigent/optimizers/
  %(prog)s --scan traigent/ --output contracts.yml
  %(prog)s --verify traigent/optimizers/ --base-class BaseOptimizer
        """,
    )

    parser.add_argument(
        "--base",
        type=Path,
        help="Path to base class file to extract contracts from",
    )
    parser.add_argument(
        "--impl",
        type=Path,
        help="Directory to search for implementations",
    )
    parser.add_argument(
        "--scan",
        type=Path,
        help="Directory to scan for all contracts and implementations",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Directory to verify implementations",
    )
    parser.add_argument(
        "--base-class",
        help="Base class name to verify against",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (YAML format)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of YAML",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args(argv)

    extractor = ContractExtractor(verbose=args.verbose)

    if args.scan:
        # Full directory scan
        if not args.scan.is_dir():
            print(f"Error: {args.scan} is not a directory", file=sys.stderr)
            return 1

        report = extractor.scan_directory(args.scan)

        if args.json:
            output = json.dumps(report, indent=2)
            if args.output:
                args.output.write_text(output, encoding="utf-8")
                print(f"Report written to {args.output}")
            else:
                print(output)
        else:
            output_yaml(report, args.output)
            if not args.output:
                print(output_yaml(report))

        # Print summary
        print(f"\nSummary:")
        print(f"  Contracts found: {report['summary']['total_contracts']}")
        print(f"  Implementations found: {report['summary']['total_implementations']}")
        print(f"  Compliant: {report['summary']['compliant']}")
        print(f"  Non-compliant: {report['summary']['non_compliant']}")

        return 0 if report["summary"]["non_compliant"] == 0 else 1

    elif args.base and args.impl:
        # Extract contract from base file and verify implementations
        if not args.base.is_file():
            print(f"Error: {args.base} is not a file", file=sys.stderr)
            return 1

        if not args.impl.is_dir():
            print(f"Error: {args.impl} is not a directory", file=sys.stderr)
            return 1

        contracts = extractor.extract_contract(args.base)
        if not contracts:
            print(f"No contracts found in {args.base}")
            return 0

        print(f"Contracts found in {args.base}:")
        for contract in contracts:
            print(f"\n  {contract.name}:")
            print(f"    Abstract methods:")
            for method in contract.abstract_methods:
                async_marker = "async " if method.is_async else ""
                print(
                    f"      - {async_marker}{method.name}({', '.join(method.parameters)})"
                )

            # Find implementations
            implementations = extractor.find_implementations(args.impl, contract.name)
            extractor.verify_compliance(contract.name)

            print(f"\n    Implementations:")
            for impl in implementations:
                status = "✅" if impl.is_compliant else "❌"
                print(
                    f"      {status} {impl.name} ({impl.file_path}:{impl.line_number})"
                )
                if impl.missing_methods:
                    print(f"         Missing: {', '.join(impl.missing_methods)}")

        return 0

    elif args.verify and args.base_class:
        # Verify implementations against specific base class
        if not args.verify.is_dir():
            print(f"Error: {args.verify} is not a directory", file=sys.stderr)
            return 1

        # First find the contract
        for py_file in args.verify.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            extractor.extract_contract(py_file)

        if args.base_class not in extractor.contracts:
            print(f"Error: Contract for {args.base_class} not found", file=sys.stderr)
            return 1

        implementations = extractor.find_implementations(args.verify, args.base_class)
        extractor.verify_compliance(args.base_class)

        non_compliant = [impl for impl in implementations if not impl.is_compliant]

        if non_compliant:
            print(f"Non-compliant implementations of {args.base_class}:")
            for impl in non_compliant:
                print(f"  ❌ {impl.name} ({impl.file_path}:{impl.line_number})")
                print(f"     Missing methods: {', '.join(impl.missing_methods)}")
            return 1
        else:
            print(
                f"✅ All {len(implementations)} implementations of {args.base_class} are compliant"
            )
            return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
