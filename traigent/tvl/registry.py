"""Default registry resolver implementations for TVL specs.

This module provides simple file-based registry resolvers that can be used
with TVL specs that have registry domains. Registry domains allow TVARs to
reference external registries (e.g., model catalogs, scorer registries).

Example usage:
    ```python
    from traigent.tvl.registry import FileRegistryResolver
    from traigent.tvl.spec_loader import load_tvl_spec

    # Create resolver pointing to registry directory
    resolver = FileRegistryResolver("./registries")

    # Load spec with resolver
    artifact = load_tvl_spec(
        spec_path="my_spec.tvl.yml",
        registry_resolver=resolver,
    )
    ```

Registry file format (YAML):
    ```yaml
    # registries/models.yaml
    items:
      - id: gpt-4o
        version: "2024-08"
        provider: openai
      - id: claude-3-opus
        version: "2024-02"
        provider: anthropic
    ```

Concept: CONC-TVLSpec
Implements: FUNC-TVLSPEC
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Optional YAML support
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class FileRegistryResolver:
    """File-based registry resolver for TVL specs.

    Resolves registry domains by reading from YAML or JSON files in a
    specified directory. Each registry is expected to be a file named
    `{registry_id}.yaml` or `{registry_id}.json`.

    Attributes:
        registry_dir: Path to the directory containing registry files.

    Example:
        ```python
        resolver = FileRegistryResolver("/path/to/registries")

        # Resolves from /path/to/registries/models.yaml
        models = resolver.resolve("models")

        # With filter
        openai_models = resolver.resolve(
            "models",
            filter_expr="provider == 'openai'"
        )
        ```
    """

    def __init__(self, registry_dir: Path | str) -> None:
        """Initialize the file registry resolver.

        Args:
            registry_dir: Path to the directory containing registry files.
        """
        self.registry_dir = Path(registry_dir)

    def resolve(
        self,
        registry_id: str,
        filter_expr: str | None = None,
        version: str | None = None,
    ) -> list[Any]:
        """Resolve a registry reference to concrete values.

        Args:
            registry_id: Identifier of the registry to query (e.g., "scorers",
                "models", "embeddings"). Maps to a file named
                `{registry_id}.yaml` or `{registry_id}.json`.
            filter_expr: Optional filter expression. Supports simple equality
                expressions like "provider == 'openai'" or "version >= '2024'".
            version: Optional version constraint for filtering items by their
                "version" field.

        Returns:
            List of resolved values (typically IDs or names) that can be used
            as the domain for a TVAR.

        Raises:
            ValueError: If the registry_id is invalid, the registry file is
                not found, or the filter_expr contains unsupported syntax.
        """
        # Find registry file
        registry_path = self._find_registry_file(registry_id)
        if registry_path is None:
            raise ValueError(
                f"Registry '{registry_id}' not found in {self.registry_dir}. "
                f"Expected file: {registry_id}.yaml or {registry_id}.json"
            )

        # Load registry data
        data = self._load_registry(registry_path)

        # Extract items
        items = data.get("items", [])
        if not isinstance(items, list):
            raise ValueError(
                f"Registry '{registry_id}' must have an 'items' list, "
                f"got {type(items).__name__}"
            )

        # Apply version filter
        if version:
            items = [i for i in items if i.get("version") == version]

        # Apply filter expression
        if filter_expr:
            items = self._apply_filter(items, filter_expr)

        # Extract IDs (or names, or the item itself if no id field)
        return self._extract_values(items)

    def _find_registry_file(self, registry_id: str) -> Path | None:
        """Find the registry file for a given ID.

        Args:
            registry_id: The registry identifier.

        Returns:
            Path to the registry file, or None if not found.

        Raises:
            ValueError: If registry_id contains path traversal characters.
        """
        # Sanitize registry_id to prevent path traversal attacks
        if "/" in registry_id or "\\" in registry_id or ".." in registry_id:
            raise ValueError(
                f"Invalid registry_id '{registry_id}': "
                "must not contain path separators or '..' sequences"
            )

        # Try YAML first
        yaml_path = self.registry_dir / f"{registry_id}.yaml"
        if yaml_path.exists():
            return yaml_path

        yml_path = self.registry_dir / f"{registry_id}.yml"
        if yml_path.exists():
            return yml_path

        # Try JSON
        json_path = self.registry_dir / f"{registry_id}.json"
        if json_path.exists():
            return json_path

        return None

    def _load_registry(self, path: Path) -> dict[str, Any]:
        """Load registry data from a file.

        Args:
            path: Path to the registry file.

        Returns:
            Registry data as a dictionary.

        Raises:
            ValueError: If the file cannot be parsed.
        """
        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ValueError(
                    f"Cannot load YAML registry '{path}': PyYAML not installed. "
                    "Install with: pip install pyyaml"
                )
            return yaml.safe_load(content) or {}

        if path.suffix == ".json":
            return json.loads(content)

        raise ValueError(f"Unsupported registry file format: {path.suffix}")

    def _apply_filter(
        self, items: list[dict[str, Any]], filter_expr: str
    ) -> list[dict[str, Any]]:
        """Apply a filter expression to items.

        Supports simple expressions:
        - Equality: "field == 'value'" or "field = 'value'"
        - Inequality: "field != 'value'"
        - Comparison: "field >= 'value'", "field <= 'value'", etc.
        - Contains: "field in ['a', 'b']"

        Args:
            items: List of registry items.
            filter_expr: Filter expression string.

        Returns:
            Filtered list of items.

        Raises:
            ValueError: If the filter expression is invalid.
        """
        # Parse simple equality: field == 'value' or field = 'value'
        equality_match = re.match(
            r"^\s*(\w+)\s*(==|=|!=)\s*['\"]([^'\"]+)['\"]\s*$", filter_expr
        )
        if equality_match:
            field, op, value = equality_match.groups()
            if op in ("==", "="):
                return [i for i in items if i.get(field) == value]
            else:  # !=
                return [i for i in items if i.get(field) != value]

        # Parse comparison: field >= 'value'
        comparison_match = re.match(
            r"^\s*(\w+)\s*(>=|<=|>|<)\s*['\"]?([^'\"]+)['\"]?\s*$", filter_expr
        )
        if comparison_match:
            field, op, value = comparison_match.groups()
            return self._apply_comparison(items, field, op, value)

        # Parse 'in' expression: field in ['a', 'b']
        in_match = re.match(
            r"^\s*(\w+)\s+in\s+\[([^\]]+)\]\s*$", filter_expr, re.IGNORECASE
        )
        if in_match:
            field, values_str = in_match.groups()
            # Parse values list
            values = [
                v.strip().strip("'\"") for v in values_str.split(",") if v.strip()
            ]
            return [i for i in items if i.get(field) in values]

        raise ValueError(
            f"Unsupported filter expression: '{filter_expr}'. "
            "Supported: 'field == \"value\"', 'field != \"value\"', "
            '\'field >= "value"\', \'field in ["a", "b"]\''
        )

    def _apply_comparison(
        self,
        items: list[dict[str, Any]],
        field: str,
        op: str,
        value: str,
    ) -> list[dict[str, Any]]:
        """Apply a comparison operation.

        Args:
            items: List of items.
            field: Field name to compare.
            op: Comparison operator (>=, <=, >, <).
            value: Value to compare against.

        Returns:
            Filtered list of items.
        """
        result = []
        for item in items:
            item_value = item.get(field)
            if item_value is None:
                continue

            # String comparison (works for semver-like versions)
            try:
                if op == ">=":
                    if str(item_value) >= value:
                        result.append(item)
                elif op == "<=":
                    if str(item_value) <= value:
                        result.append(item)
                elif op == ">":
                    if str(item_value) > value:
                        result.append(item)
                elif op == "<":
                    if str(item_value) < value:
                        result.append(item)
            except (TypeError, ValueError):
                continue

        return result

    def _extract_values(self, items: list[dict[str, Any]]) -> list[Any]:
        """Extract values from registry items.

        Prefers 'id' field, falls back to 'name', then the whole item.

        Args:
            items: List of registry items.

        Returns:
            List of extracted values.
        """
        values = []
        for item in items:
            if isinstance(item, dict):
                # Prefer id, then name
                if "id" in item:
                    values.append(item["id"])
                elif "name" in item:
                    values.append(item["name"])
                else:
                    values.append(item)
            else:
                values.append(item)
        return values


class DictRegistryResolver:
    """In-memory registry resolver using dictionaries.

    Useful for testing and programmatic registry configuration.

    Example:
        ```python
        resolver = DictRegistryResolver({
            "models": [
                {"id": "gpt-4o", "provider": "openai"},
                {"id": "claude-3", "provider": "anthropic"},
            ],
            "scorers": [
                {"id": "accuracy_v1"},
                {"id": "accuracy_v2"},
            ],
        })

        models = resolver.resolve("models", filter_expr="provider == 'openai'")
        # Returns: ["gpt-4o"]
        ```
    """

    def __init__(self, registries: dict[str, list[dict[str, Any]]]) -> None:
        """Initialize with registry data.

        Args:
            registries: Dict mapping registry IDs to lists of items.
        """
        self.registries = registries

    def resolve(
        self,
        registry_id: str,
        filter_expr: str | None = None,
        version: str | None = None,
    ) -> list[Any]:
        """Resolve a registry reference to concrete values.

        Args:
            registry_id: Identifier of the registry.
            filter_expr: Optional filter expression (same syntax as FileRegistryResolver).
            version: Optional version constraint.

        Returns:
            List of resolved values.

        Raises:
            ValueError: If registry not found.
        """
        if registry_id not in self.registries:
            raise ValueError(
                f"Registry '{registry_id}' not found. "
                f"Available: {list(self.registries.keys())}"
            )

        items = list(self.registries[registry_id])

        # Apply version filter
        if version:
            items = [i for i in items if i.get("version") == version]

        # Apply filter expression (reuse FileRegistryResolver logic)
        if filter_expr:
            file_resolver = FileRegistryResolver(".")
            items = file_resolver._apply_filter(items, filter_expr)

        # Extract values
        return [
            item.get("id", item.get("name", item)) if isinstance(item, dict) else item
            for item in items
        ]
