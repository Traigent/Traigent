"""CLI entry point for TVL spec validation (G-2).

This module allows running TVL validation from the command line:
    python -m traigent.tvl [options] [files...]

Used by pre-commit hooks to validate TVL specification files.

Traceability: CONC-Quality-Reliability REQ-TVLSPEC-013
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from traigent.tvl.spec_loader import (
    TVLValidationError,
    load_tvl_spec,
    validate_tvl_schema,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def validate_tvl_files(
    files: list[Path],
    *,
    strict: bool = False,
    verbose: bool = False,
) -> tuple[int, int]:
    """Validate TVL specification files.

    Args:
        files: List of TVL file paths to validate.
        strict: If True, treat warnings as errors.
        verbose: If True, print success messages.

    Returns:
        Tuple of (files_passed, files_failed).
    """
    passed = 0
    failed = 0

    for file_path in files:
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}", file=sys.stderr)
            failed += 1
            continue

        try:
            # First validate schema (early validation)
            import yaml

            with open(file_path, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)

            if raw_data is None:
                print(f"ERROR: Empty TVL spec: {file_path}", file=sys.stderr)
                failed += 1
                continue

            # Validate schema structure
            issues = validate_tvl_schema(raw_data, strict=strict)

            if issues:
                if strict:
                    print(f"FAILED: {file_path}", file=sys.stderr)
                    for issue in issues:
                        print(f"  - {issue}", file=sys.stderr)
                    failed += 1
                    continue
                else:
                    # Warnings only in non-strict mode
                    if verbose:
                        print(f"WARNING: {file_path}")
                        for issue in issues:
                            print(f"  - {issue}")

            # Try full loading to catch deeper issues
            try:
                load_tvl_spec(spec_path=str(file_path), validate_schema=False)
            except TVLValidationError as e:
                print(f"FAILED: {file_path}", file=sys.stderr)
                print(f"  - {e}", file=sys.stderr)
                failed += 1
                continue
            except Exception as e:
                print(f"FAILED: {file_path}", file=sys.stderr)
                print(f"  - Load error: {e}", file=sys.stderr)
                failed += 1
                continue

            passed += 1
            if verbose:
                print(f"OK: {file_path}")

        except yaml.YAMLError as e:
            print(f"FAILED: {file_path}", file=sys.stderr)
            print(f"  - Invalid YAML: {e}", file=sys.stderr)
            failed += 1
        except Exception as e:
            print(f"FAILED: {file_path}", file=sys.stderr)
            print(f"  - Unexpected error: {e}", file=sys.stderr)
            failed += 1

    return passed, failed


def find_tvl_files(directory: Path, recursive: bool = True) -> list[Path]:
    """Find TVL files in a directory.

    Args:
        directory: Directory to search.
        recursive: If True, search recursively.

    Returns:
        List of TVL file paths.
    """
    patterns = ["*.tvl.yaml", "*.tvl.yml"]
    files: list[Path] = []

    for pattern in patterns:
        if recursive:
            files.extend(directory.rglob(pattern))
        else:
            files.extend(directory.glob(pattern))

    return sorted(files)


def main() -> int:
    """Main entry point for TVL validation CLI.

    Returns:
        Exit code (0 for success, 1 for failures).
    """
    parser = argparse.ArgumentParser(
        description="Validate TVL (Traigent Variable Language) specification files.",
        prog="python -m traigent.tvl",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="TVL files to validate. If none provided, searches current directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print success messages and warnings.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=True,
        help="Search directories recursively (default: True).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Disable recursive directory search.",
    )

    args = parser.parse_args()

    # Collect files to validate
    files_to_validate: list[Path] = []

    if args.files:
        for file_path in args.files:
            if file_path.is_dir():
                files_to_validate.extend(
                    find_tvl_files(file_path, recursive=args.recursive)
                )
            elif file_path.exists():
                files_to_validate.append(file_path)
            else:
                print(f"ERROR: File not found: {file_path}", file=sys.stderr)
    else:
        # Search current directory
        files_to_validate = find_tvl_files(Path.cwd(), recursive=args.recursive)

    if not files_to_validate:
        if args.verbose:
            print("No TVL files found to validate.")
        return 0

    # Validate files
    passed, failed = validate_tvl_files(
        files_to_validate,
        strict=args.strict,
        verbose=args.verbose,
    )

    # Print summary
    total = passed + failed
    if args.verbose or failed > 0:
        print(f"\nValidation complete: {passed}/{total} files passed")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
