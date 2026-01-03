"""Secure path validation utilities for Traigent SDK.

This module provides utilities to prevent path traversal attacks and ensure
file operations stay within allowed directories.

Usage:
    from traigent.utils.secure_path import safe_open, validate_path, SafePath

    # Validate a path stays within allowed directory
    safe_path = validate_path(user_input, allowed_base="/app/data")

    # Use context manager for safe file operations
    with safe_open(filename, base_dir="/app/data", mode="r") as f:
        content = f.read()

    # Or use SafePath for path operations
    sp = SafePath("/app/data")
    resolved = sp.resolve(user_input)  # Raises if path escapes base_dir
"""

# Traceability: CONC-Security CONC-Quality-Security FUNC-SECURITY REQ-SEC-010

from __future__ import annotations

from pathlib import Path
from typing import IO, Any


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""


class SafePath:
    """Secure path resolver that prevents directory traversal attacks.

    This class ensures that resolved paths stay within a specified base directory,
    preventing path traversal attacks via sequences like '../' or symbolic links.

    Example:
        sp = SafePath("/app/data")
        safe = sp.resolve("../etc/passwd")  # Raises PathTraversalError
        safe = sp.resolve("subdir/file.txt")  # Returns /app/data/subdir/file.txt
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Initialize SafePath with a base directory.

        Args:
            base_dir: The base directory that all paths must stay within.
                      Will be resolved to an absolute path.
        """
        self.base_dir = Path(base_dir).resolve()
        if self.base_dir.exists() and not self.base_dir.is_dir():
            raise ValueError(f"Base path is not a directory: {self.base_dir}")

    def resolve(self, path: str | Path) -> Path:
        """Safely resolve a path within the base directory.

        Args:
            path: The path to resolve (can be relative or absolute)

        Returns:
            Resolved absolute path that is guaranteed to be within base_dir

        Raises:
            PathTraversalError: If the resolved path escapes base_dir
        """
        # Convert to Path and resolve
        target = Path(path)

        # If path is relative, join with base_dir
        if not target.is_absolute():
            target = self.base_dir / target

        # Resolve to absolute path (follows symlinks, normalizes ..)
        resolved = target.resolve()

        # Security check: ensure resolved path is within base_dir
        try:
            resolved.relative_to(self.base_dir)
        except ValueError as e:
            raise PathTraversalError(
                f"Path traversal detected: '{path}' resolves outside base directory"
            ) from e

        return resolved

    def is_safe(self, path: str | Path) -> bool:
        """Check if a path is safe (within base_dir) without raising.

        Args:
            path: The path to check

        Returns:
            True if path is safe, False otherwise
        """
        try:
            self.resolve(path)
            return True
        except PathTraversalError:
            return False


def _resolve_base_dir(base_dir: str | Path, must_exist: bool = True) -> Path:
    """Resolve and validate the base directory for safe file operations."""
    resolved = Path(base_dir).expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"Base path is not a directory: {resolved}")
    elif must_exist:
        raise FileNotFoundError(f"Base directory does not exist: {resolved}")
    return resolved


def _resolve_path_in_base(
    path: str | Path,
    base_dir: Path,
    must_exist: bool = False,
) -> Path:
    """Resolve a target path and ensure it stays within the base directory."""
    target = Path(path)
    if not target.is_absolute():
        target = base_dir / target
    resolved = target.resolve()
    try:
        resolved.relative_to(base_dir)
    except ValueError as e:
        raise PathTraversalError(
            f"Path traversal detected: '{path}' resolves outside base directory"
        ) from e
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def validate_path(
    path: str | Path,
    allowed_base: str | Path,
    must_exist: bool = False,
) -> Path:
    """Validate that a path stays within an allowed base directory.

    Args:
        path: The path to validate
        allowed_base: The base directory the path must stay within
        must_exist: If True, also check that the path exists

    Returns:
        The resolved, validated path

    Raises:
        PathTraversalError: If path escapes allowed_base
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    sp = SafePath(allowed_base)
    resolved = sp.resolve(path)

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved


def safe_open(
    path: str | Path,
    base_dir: str | Path,
    mode: str = "r",
    **kwargs: Any,
) -> IO[Any]:
    """Safely open a file, ensuring it's within the allowed base directory.

    This is a drop-in replacement for open() that validates the path first.

    Args:
        path: Path to the file (relative to base_dir or absolute)
        base_dir: The base directory the file must be within
        mode: File open mode (same as builtin open())
        **kwargs: Additional arguments passed to open()

    Returns:
        File handle

    Raises:
        PathTraversalError: If path escapes base_dir
        Other exceptions from open() as normal
    """
    resolved_base = _resolve_base_dir(base_dir, must_exist=False)
    validated_path = _resolve_path_in_base(path, resolved_base)
    return open(validated_path, mode, **kwargs)


def safe_read_text(
    path: str | Path,
    base_dir: str | Path,
    encoding: str = "utf-8",
) -> str:
    """Safely read text from a file within allowed base directory.

    Args:
        path: Path to the file
        base_dir: The base directory the file must be within
        encoding: Text encoding

    Returns:
        File contents as string

    Raises:
        PathTraversalError: If path escapes base_dir
    """
    resolved_base = _resolve_base_dir(base_dir, must_exist=False)
    validated_path = _resolve_path_in_base(path, resolved_base, must_exist=True)
    return validated_path.read_text(encoding=encoding)


def safe_read_bytes(
    path: str | Path,
    base_dir: str | Path,
) -> bytes:
    """Safely read bytes from a file within allowed base directory.

    Args:
        path: Path to the file
        base_dir: The base directory the file must be within

    Returns:
        File contents as bytes

    Raises:
        PathTraversalError: If path escapes base_dir
    """
    resolved_base = _resolve_base_dir(base_dir)
    validated_path = _resolve_path_in_base(path, resolved_base, must_exist=True)
    return validated_path.read_bytes()


def safe_write_text(
    path: str | Path,
    content: str,
    base_dir: str | Path,
    encoding: str = "utf-8",
) -> None:
    """Safely write text to a file within allowed base directory.

    Args:
        path: Path to the file
        content: Text content to write
        base_dir: The base directory the file must be within
        encoding: Text encoding

    Raises:
        PathTraversalError: If path escapes base_dir
    """
    resolved_base = _resolve_base_dir(base_dir)
    validated_path = _resolve_path_in_base(path, resolved_base)
    # Ensure parent directory exists
    validated_path.parent.mkdir(parents=True, exist_ok=True)
    validated_path.write_text(content, encoding=encoding)


def safe_write_bytes(
    path: str | Path,
    content: bytes,
    base_dir: str | Path,
) -> None:
    """Safely write bytes to a file within allowed base directory.

    Args:
        path: Path to the file
        content: Bytes content to write
        base_dir: The base directory the file must be within

    Raises:
        PathTraversalError: If path escapes base_dir
    """
    resolved_base = _resolve_base_dir(base_dir)
    validated_path = _resolve_path_in_base(path, resolved_base)
    # Ensure parent directory exists
    validated_path.parent.mkdir(parents=True, exist_ok=True)
    validated_path.write_bytes(content)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename to remove dangerous characters.

    Removes or replaces characters that could be used for path traversal
    or cause issues on different filesystems.

    Args:
        filename: The filename to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized filename safe for use in file operations
    """
    # Remove null bytes
    sanitized = filename.replace("\x00", "")

    # Remove path separators
    sanitized = sanitized.replace("/", "_").replace("\\", "_")

    # Remove other dangerous characters
    dangerous_chars = ["<", ">", ":", '"', "|", "?", "*"]
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "_")

    # Remove leading/trailing dots and spaces (Windows issues)
    sanitized = sanitized.strip(". ")

    # Handle empty result
    if not sanitized:
        sanitized = "unnamed"

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized
