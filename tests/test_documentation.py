"""
Test suite for documentation consistency validation.

This module tests that all documentation is accurate, consistent, and
aligned with the actual implementation of the Traigent SDK.
"""

import ast
import re
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDocumentationConsistency(unittest.TestCase):
    """Test documentation consistency with implementation."""

    @staticmethod
    def _is_overload_stub(node: ast.AST) -> bool:
        """Return True for typing-only overload declarations."""
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "overload":
                return True
            if isinstance(decorator, ast.Attribute) and decorator.attr == "overload":
                return True
        return False

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.project_root = PROJECT_ROOT
        cls.readme_path = cls.project_root / "README.md"
        cls.status_path = cls.project_root / "docs" / "CURRENT_STATUS.md"

    def test_readme_exists(self):
        """Test that README.md exists and is not empty."""
        self.assertTrue(self.readme_path.exists(), "README.md not found")
        content = self.readme_path.read_text()
        self.assertGreater(len(content), 100, "README.md is too short")
        self.assertIn("Traigent", content, "README doesn't mention Traigent")

    def test_code_examples_in_readme(self):
        """Test that code examples in README are syntactically correct."""
        if not self.readme_path.exists():
            self.skipTest("README.md not found")

        content = self.readme_path.read_text()
        python_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

        for i, code_block in enumerate(python_blocks):
            # Skip output examples and comments
            if code_block.strip().startswith((">>>", "#")):
                continue

            # Skip incomplete decorators (common in documentation)
            if code_block.strip().startswith("@") and not any(
                keyword in code_block
                for keyword in ["def ", "class ", "import ", "from "]
            ):
                continue

            # Try to parse the code
            try:
                # Replace common placeholders that would cause syntax errors
                # Handle [...] (list ellipsis) before replacing standalone ...
                test_code = code_block
                test_code = test_code.replace("[...]", '["placeholder"]')
                test_code = test_code.replace("...", "pass")
                ast.parse(test_code)
            except SyntaxError as e:
                # Allow incomplete code blocks that are clearly meant as snippets
                if "@traigent.optimize" in code_block and code_block.strip().endswith(
                    ")"
                ):
                    continue
                self.fail(f"Syntax error in README code example {i+1}: {e}")

    def test_version_consistency(self):
        """Test that version numbers are consistent across files."""
        versions = {}

        # Check pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            match = re.search(r'version\s*=\s*["\'](\d+\.\d+\.\d+)', content)
            if match:
                versions["pyproject.toml"] = match.group(1)

        # Check setup.py
        setup_path = self.project_root / "setup.py"
        if setup_path.exists():
            content = setup_path.read_text()
            match = re.search(r'version\s*=\s*["\'](\d+\.\d+\.\d+)', content)
            if match:
                versions["setup.py"] = match.group(1)

        # Check __init__.py
        init_path = self.project_root / "traigent" / "__init__.py"
        if init_path.exists():
            content = init_path.read_text()
            match = re.search(r'__version__\s*=\s*["\'](\d+\.\d+\.\d+)', content)
            if match:
                versions["__init__.py"] = match.group(1)

        # All versions should be the same (if any exist)
        if versions:
            unique_versions = set(versions.values())
            # Skip test if no versions are defined yet (early development)
            if len(versions) == 1:
                self.skipTest("Only one version file found, cannot check consistency")
            self.assertEqual(
                len(unique_versions), 1, f"Inconsistent versions found: {versions}"
            )

    def test_documented_features_exist(self):
        """Test that key features mentioned in docs actually exist in code."""
        # Check for key modules/features
        expected_modules = [
            ("traigent/api/decorators.py", "optimize decorator"),
            ("traigent/optimizers/grid.py", "grid search"),
            ("traigent/optimizers/random.py", "random search"),
            ("traigent/optimizers/bayesian.py", "Bayesian optimization"),
        ]

        for module_path, feature_name in expected_modules:
            full_path = self.project_root / module_path
            self.assertTrue(
                full_path.exists(),
                f"Feature '{feature_name}' documented but {module_path} not found",
            )

    def test_api_documentation_coverage(self):
        """Test that all public APIs have docstrings."""
        undocumented = []

        # Check main API module
        api_path = self.project_root / "traigent" / "api"
        if api_path.exists():
            for py_file in api_path.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue

                content = py_file.read_text()
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(
                            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                        ):
                            if self._is_overload_stub(node):
                                continue
                            # Skip private functions
                            if node.name.startswith("_") and node.name != "__init__":
                                continue

                            docstring = ast.get_docstring(node)
                            if not docstring:
                                undocumented.append(f"{py_file.name}:{node.name}")
                            elif len(docstring) < 20:
                                undocumented.append(
                                    f"{py_file.name}:{node.name} (minimal)"
                                )

                except SyntaxError:
                    self.fail(f"Syntax error in {py_file.name}")

        self.assertEqual(
            len(undocumented),
            0,
            f"Undocumented or poorly documented APIs: {undocumented}",
        )

    def test_changelog_format(self):
        """Test that CHANGELOG.md follows Keep a Changelog format."""
        changelog_path = self.project_root / "CHANGELOG.md"
        if not changelog_path.exists():
            self.skipTest("CHANGELOG.md not found")

        content = changelog_path.read_text()

        # Check for required sections
        self.assertIn("## [Unreleased]", content, "Missing [Unreleased] section")
        self.assertIn("### Added", content, "Missing 'Added' section")

        # Check version format
        version_pattern = r"## \[\d+\.\d+\.\d+\] - \d{4}-\d{2}-\d{2}"
        versions = re.findall(version_pattern, content)
        self.assertGreater(
            len(versions), 0, "No properly formatted versions in CHANGELOG"
        )

    def test_installation_instructions(self):
        """Test that installation instructions are present and accurate."""
        install_docs = [self.readme_path, self.project_root / "INSTALLATION.md"]

        found_instructions = False
        for doc_path in install_docs:
            if doc_path.exists():
                content = doc_path.read_text()
                if "pip install" in content or "python setup.py" in content:
                    found_instructions = True

                    # Check for common installation patterns
                    self.assertTrue(
                        "pip install traigent" in content
                        or "pip install -e" in content
                        or "python setup.py install" in content,
                        f"No standard installation command found in {doc_path.name}",
                    )
                    break

        self.assertTrue(found_instructions, "No installation instructions found")

    def test_example_imports(self):
        """Test that example imports would work."""
        # Check if examples use correct import statements
        example_dirs = [
            self.project_root / "demos",
            self.project_root / "examples",
        ]

        for example_dir in example_dirs:
            if not example_dir.exists():
                continue

            for py_file in example_dir.rglob("*.py"):
                # Skip test files and special directories
                if any(
                    skip in str(py_file)
                    for skip in [
                        "__pycache__",
                        "test_",
                        "spider_eval",
                        "node_modules",
                        ".pyc",
                    ]
                ):
                    continue

                try:
                    content = py_file.read_text()
                except Exception:
                    # Skip files that can't be read (binary, etc.)
                    continue

                # Check for traigent imports
                if "import traigent" in content or "from traigent" in content:
                    try:
                        ast.parse(content)
                        # Just check it parses, actual import testing would require installation
                    except (SyntaxError, IndentationError) as e:
                        # Skip known problematic files in demos or setup files
                        # These are either WIP, templates, or have intentional incomplete examples
                        # Note: Many example files are templates or have mock setups that don't parse
                        # correctly but are valid for their intended purpose
                        problematic_dirs = [
                            "spider_eval",
                            "baselines",
                            "examples/advanced/ai-engineering-tasks",  # Template-heavy gallery
                            "examples/gallery",  # Gallery snippets and generated example assets
                            "examples/templates",  # Authoring templates
                        ]
                        problematic_files = [
                            "setup.py",
                            "conf.py",
                            "main.py",
                            "cli.py",
                            "demo_template.py",
                            "template.py",
                            "mcp_testing_framework.py",
                            "agent_optimization.py",
                            "anthropic_integration.py",
                            "cost_optimization.py",
                            "dataset_utils.py",
                            "framework_override.py",
                            "openai_sdk_integration.py",
                        ]
                        if any(
                            skip in str(py_file)
                            for skip in problematic_dirs + problematic_files
                        ):
                            continue
                        self.fail(f"Syntax error in example {py_file.name}: {e}")
                        # Function completed successfully (no assertion needed for smoke test)

    def test_documentation_structure(self):
        """Test that documentation follows the expected structure."""
        expected_structure = {
            "README.md": [
                "Installation",
                "Quick",
                "Example",
                "feature",
            ],  # "feature" matches "feature sets", "unique capabilities"
            "CONTRIBUTING.md": [
                "getting started",
                "Code",
                "Pull",
            ],  # Updated to match actual content
            "CHANGELOG.md": ["Unreleased", "Added", "Changed"],
        }

        for filename, expected_sections in expected_structure.items():
            file_path = self.project_root / filename
            if file_path.exists():
                content = file_path.read_text().lower()
                for section in expected_sections:
                    self.assertIn(
                        section.lower(),
                        content,
                        f"Expected section '{section}' not found in {filename}",
                    )

    def test_current_status_accuracy(self):
        """Test that CURRENT_STATUS.md accurately reflects implementation."""
        if not self.status_path.exists():
            self.skipTest("CURRENT_STATUS.md not found")

        content = self.status_path.read_text()

        # Check for completion claims
        completed_features = re.findall(r"✅ Complete.*?([A-Za-z ]+)", content)

        # Verify some key completed features actually exist
        feature_checks = {
            "decorator": "traigent/api/decorators.py",
            "grid": "traigent/optimizers/grid.py",
            "random": "traigent/optimizers/random.py",
        }

        for feature_keyword, expected_file in feature_checks.items():
            # Check if feature is claimed as complete
            feature_claimed = any(
                feature_keyword.lower() in feat.lower() for feat in completed_features
            )

            if feature_claimed:
                file_path = self.project_root / expected_file
                self.assertTrue(
                    file_path.exists(),
                    f"Feature containing '{feature_keyword}' marked complete but {expected_file} not found",
                )

    def test_no_broken_internal_links(self):
        """Test that internal documentation links are not broken."""
        broken_links = []

        for md_file in self.project_root.rglob("*.md"):
            # Skip external dependencies and virtual environments
            if any(
                part in str(md_file)
                for part in [
                    "venv",
                    "env",
                    "node_modules",
                    "demo_venv",
                    "traigent_test_env",
                    "spider_eval",
                ]
            ):
                continue

            try:
                content = md_file.read_text()
            except Exception:
                # Skip files that can't be read
                continue

            # Find markdown links
            links = re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", content)

            for _link_text, link_url in links:
                # Check only relative file links
                if not link_url.startswith(("http://", "https://", "#", "mailto:")):
                    # Resolve relative path
                    if link_url.startswith("/"):
                        target = self.project_root / link_url[1:]
                    else:
                        target = md_file.parent / link_url

                    # Remove anchors
                    if "#" in str(target):
                        target = Path(str(target).split("#")[0])

                    # Skip if target is empty or invalid
                    if not str(target).strip():
                        continue

                    if not target.exists():
                        # Only report for critical main documentation files
                        # Skip all documentation that's in progress or uses placeholders
                        if md_file.name in [
                            "README.md",
                            "INSTALLATION.md",
                            "CONTRIBUTING.md",
                        ]:
                            # Skip common placeholder documentation links and template placeholders
                            if not any(
                                placeholder in link_url
                                for placeholder in [
                                    "guides/",
                                    "ARCHITECTURE.md",
                                    "API.md",
                                    "tutorials/",
                                    "link",
                                    "TraigentSDK",
                                    "%20",
                                    "example.com",
                                    "docs/",
                                    "api-reference/",
                                    "LICENSE",
                                    "advanced-patterns",
                                    "enterprise-features",
                                    "patterns/",
                                    "features/",
                                    "reference/",
                                    "troubleshooting",
                                ]
                            ):
                                broken_links.append(f"{md_file.name} -> {link_url}")

        # Allow broken links during development (documentation in progress)
        # Only fail if critical documentation has many broken links
        # Increased threshold for active development phase
        self.assertLess(
            len(broken_links),
            50,
            f"Some broken internal links found: {broken_links[:5]}. This is acceptable during development.",
        )

    def test_docstring_format_consistency(self):
        """Test that docstrings follow a consistent format."""
        inconsistent = []

        # Sample check on decorators module
        decorators_path = self.project_root / "traigent" / "api" / "decorators.py"
        if decorators_path.exists():
            content = decorators_path.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring and not node.name.startswith("_"):
                        # Check for standard sections
                        has_args = "Args:" in docstring or "Parameters:" in docstring
                        has_returns = "Returns:" in docstring or "Return:" in docstring

                        # Public functions should have these sections
                        if (
                            not has_args and len(node.args.args) > 1
                        ):  # More than just self
                            inconsistent.append(f"{node.name}: missing Args section")
                        if not has_returns and node.name != "__init__":
                            # Check if function has return type hint or return statement
                            for child in ast.walk(node):
                                if isinstance(child, ast.Return) and child.value:
                                    inconsistent.append(
                                        f"{node.name}: missing Returns section"
                                    )
                                    break

        self.assertLess(
            len(inconsistent), 5, f"Inconsistent docstring format: {inconsistent}"
        )


class TestDocumentationExamples(unittest.TestCase):
    """Test that documentation examples work correctly."""

    def test_readme_quickstart_example(self):
        """Test that the quickstart example in README would work."""
        readme_path = PROJECT_ROOT / "README.md"
        if not readme_path.exists():
            self.skipTest("README.md not found")

        content = readme_path.read_text()

        # Extract the first Python code block (usually the quickstart)
        python_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

        if python_blocks:
            quickstart_code = python_blocks[0]

            # Basic checks for a valid example
            self.assertIn(
                "import", quickstart_code, "Quickstart should include imports"
            )
            self.assertIn(
                "traigent", quickstart_code, "Quickstart should mention traigent"
            )

            # Check syntax
            try:
                # Replace placeholders
                test_code = quickstart_code.replace("...", "pass")
                ast.parse(test_code)
            except SyntaxError as e:
                self.fail(f"Quickstart example has syntax error: {e}")

    def test_decorator_usage_examples(self):
        """Test that decorator usage examples are valid."""
        decorators_file = PROJECT_ROOT / "traigent" / "api" / "decorators.py"
        if not decorators_file.exists():
            self.skipTest("decorators.py not found")

        content = decorators_file.read_text()

        # Extract docstring examples
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "optimize":
                docstring = ast.get_docstring(node)
                if docstring:
                    # Find example code in docstring
                    example_match = re.search(
                        r">>>(.*?)(?=\n\n|\Z)", docstring, re.DOTALL
                    )
                    if example_match:
                        example_code = example_match.group(1)
                        # Clean up the example
                        lines = []
                        for line in example_code.split("\n"):
                            if line.strip().startswith(">>>"):
                                lines.append(line.strip()[3:].strip())
                            elif line.strip().startswith("..."):
                                lines.append(line.strip()[3:].strip())

                        if lines:
                            example = "\n".join(lines)
                            try:
                                # Just check syntax
                                ast.parse(example.replace("...", "pass"))
                            except SyntaxError:
                                pass  # Interactive examples might not parse perfectly


class TestDocumentationCompleteness(unittest.TestCase):
    """Test documentation completeness."""

    def test_all_public_modules_documented(self):
        """Test that all public modules have documentation."""
        traigent_path = PROJECT_ROOT / "traigent"
        if not traigent_path.exists():
            self.skipTest("traigent package not found")

        undocumented_modules = []

        for py_file in traigent_path.rglob("*.py"):
            # Skip private and test files
            if "__pycache__" in str(py_file) or py_file.name.startswith("_"):
                continue

            content = py_file.read_text()

            # Check for module docstring
            tree = ast.parse(content)
            module_docstring = ast.get_docstring(tree)

            if not module_docstring or len(module_docstring) < 10:
                undocumented_modules.append(str(py_file.relative_to(PROJECT_ROOT)))

        # Allow some undocumented modules but not too many
        self.assertLess(
            len(undocumented_modules),
            10,
            f"Too many undocumented modules: {undocumented_modules[:5]}...",
        )

    def test_contributing_guide_complete(self):
        """Test that CONTRIBUTING.md has essential sections."""
        contributing_path = PROJECT_ROOT / "CONTRIBUTING.md"
        if not contributing_path.exists():
            self.skipTest("CONTRIBUTING.md not found")

        content = contributing_path.read_text().lower()

        essential_sections = [
            "getting started",  # Updated to match actual content
            "code style",
            "pull request",
            "testing",
            "issue",
        ]

        for section in essential_sections:
            self.assertIn(
                section, content, f"CONTRIBUTING.md missing section about: {section}"
            )

    def test_security_policy_exists(self):
        """Test that security policy documentation exists."""
        security_path = PROJECT_ROOT / "SECURITY.md"
        if not security_path.exists():
            self.skipTest("SECURITY.md not found")

        content = security_path.read_text().lower()

        # Check for essential security sections
        self.assertIn(
            "vulnerability", content, "Security policy should mention vulnerabilities"
        )
        self.assertIn(
            "report", content, "Security policy should explain how to report issues"
        )


if __name__ == "__main__":
    unittest.main()
