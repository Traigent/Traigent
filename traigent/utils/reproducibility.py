"""Reproducibility metadata collection for Traigent optimization runs.

This module provides comprehensive metadata collection to ensure
optimization runs can be reproduced exactly, including environment,
dependencies, random seeds, and data checksums.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-STORAGE FUNC-ANALYTICS REQ-STOR-007 REQ-ANLY-011 SYNC-StorageLogging

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from traigent.utils.logging import get_logger
from traigent.utils.secure_path import validate_path

logger = get_logger(__name__)


class ReproducibilityMetadata:
    """Collects and manages reproducibility metadata for optimization runs."""

    def __init__(self, run_path: Path) -> None:
        """Initialize reproducibility metadata collector.

        Args:
            run_path: Path to the optimization run directory
        """
        self.run_path = run_path
        self.metadata: dict[str, Any] = {}
        self._collect_all_metadata()

    @staticmethod
    def _resolve_path(
        path: Path, base_dir: Path | None = None, must_exist: bool = False
    ) -> Path:
        if base_dir is None:
            base_dir = path.parent if path.is_absolute() else Path.cwd().resolve()
        return validate_path(path, base_dir, must_exist=must_exist)

    def _collect_all_metadata(self) -> None:
        """Collect all reproducibility metadata."""
        self.metadata = {
            "timestamp": datetime.now(UTC).isoformat(),
            "environment": self._collect_environment(),
            "python": self._collect_python_info(),
            "dependencies": self._collect_dependencies(),
            "random_state": self._collect_random_state(),
            "hardware": self._collect_hardware_info(),
            "git": self._collect_git_info(),
            "traigent": self._collect_traigent_info(),
        }

    def _collect_environment(self) -> dict[str, Any]:
        """Collect environment variables and system information."""
        # Only collect relevant environment variables (not secrets)
        safe_env_vars = [
            "PYTHONPATH",
            "PATH",
            "VIRTUAL_ENV",
            "CONDA_DEFAULT_ENV",
            "TRAIGENT_MOCK_MODE",
            "TRAIGENT_REAL_MODE",
            "TRAIGENT_RESULTS_FOLDER",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        ]

        env_info = {}
        for var in safe_env_vars:
            if var in os.environ:
                env_info[var] = os.environ[var]

        return {
            "variables": env_info,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "user": os.environ.get("USER", "unknown"),
            "cwd": str(Path.cwd()),
        }

    def _collect_python_info(self) -> dict[str, Any]:
        """Collect Python interpreter information."""
        return {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
                "releaselevel": sys.version_info.releaselevel,
                "serial": sys.version_info.serial,
            },
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        }

    def _collect_dependencies(self) -> dict[str, str]:
        """Collect installed package versions."""
        dependencies = {}

        # Core dependencies to track
        packages = [
            "numpy",
            "pandas",
            "scikit-learn",
            "torch",
            "tensorflow",
            "langchain",
            "openai",
            "anthropic",
            "transformers",
            "optuna",
            "ray",
        ]

        for package in packages:
            try:
                import importlib.metadata

                version = importlib.metadata.version(package)
                dependencies[package] = version
            except (ImportError, importlib.metadata.PackageNotFoundError):
                # Package not installed
                pass

        # Add Traigent version
        try:
            import traigent

            dependencies["traigent"] = traigent.__version__
        except (ImportError, AttributeError):
            dependencies["traigent"] = "unknown"

        return dependencies

    def _collect_random_state(self) -> dict[str, Any]:
        """Collect random state information for reproducibility."""
        random_info: dict[str, Any] = {
            "python_random_state": None,
            "numpy_random_state": None,
            "seeds_captured": False,
        }

        try:
            python_state = random.getstate()
            random_info["python_random_state"] = self._serialize_python_random_state(
                python_state
            )

            # Get legacy tuple format for serialization compatibility
            numpy_state = np.random.get_state(legacy=True)  # type: ignore[call-overload]
            # legacy=True returns tuple format, not dict
            if isinstance(numpy_state, tuple):
                random_info["numpy_random_state"] = self._serialize_numpy_random_state(
                    numpy_state
                )

            random_info["seeds_captured"] = True
        except Exception as e:
            logger.warning(f"Could not capture random state: {e}")

        return random_info

    def restore_random_state(self, state: dict[str, Any]) -> None:
        """Restore random state from previously captured metadata."""
        if not state:
            logger.warning("No random state provided to restore.")
            return

        try:
            if python_state := state.get("python_random_state"):
                random.setstate(self._deserialize_python_random_state(python_state))
        except Exception as exc:
            logger.warning(f"Failed to restore Python random state: {exc}")

        try:
            if numpy_state := state.get("numpy_random_state"):
                np.random.set_state(self._deserialize_numpy_random_state(numpy_state))
        except Exception as exc:
            logger.warning(f"Failed to restore NumPy random state: {exc}")

        try:
            if rng_state := state.get("numpy_rng_state"):
                if hasattr(np.random, "default_rng"):
                    np.random.default_rng().bit_generator.state = rng_state
        except Exception as exc:
            logger.warning(f"Failed to restore NumPy default RNG state: {exc}")

    def _serialize_python_random_state(self, state: tuple[Any, ...]) -> dict[str, Any]:
        """Serialize Python's random state into a JSON-friendly representation."""
        version, internal_state, gauss = state
        return {
            "version": version,
            "state": list(internal_state),
            "gauss": gauss,
        }

    def _deserialize_python_random_state(
        self, payload: dict[str, Any]
    ) -> tuple[Any, ...]:
        """Deserialize Python's random state from JSON-friendly representation."""
        return (
            payload["version"],
            tuple(payload["state"]),
            payload["gauss"],
        )

    def _serialize_numpy_random_state(self, state: tuple[Any, ...]) -> dict[str, Any]:
        """Serialize NumPy random state."""
        version, keys, pos, has_gauss, cached = state
        return {
            "version": version,
            "keys": np.asarray(keys, dtype=np.uint32).tolist(),
            "pos": pos,
            "has_gauss": has_gauss,
            "cached_gaussian": cached,
        }

    def _deserialize_numpy_random_state(
        self, payload: dict[str, Any]
    ) -> tuple[Any, ...]:
        """Deserialize NumPy random state."""
        return (
            payload["version"],
            np.array(payload["keys"], dtype=np.uint32),
            payload["pos"],
            payload["has_gauss"],
            payload["cached_gaussian"],
        )

    def _collect_hardware_info(self) -> dict[str, Any]:
        """Collect hardware information."""
        hardware: dict[str, Any] = {
            "processor": platform.processor() or platform.machine(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory": {},
            "gpu": [],
        }

        # Try to get memory info
        try:
            import psutil

            mem = psutil.virtual_memory()
            hardware["memory"] = {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent_used": mem.percent,
            }
        except ImportError:
            pass

        # Try to get GPU info
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_info = {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_gb": round(
                            torch.cuda.get_device_properties(i).total_memory
                            / (1024**3),
                            2,
                        ),
                    }
                    hardware["gpu"].append(gpu_info)
        except ImportError:
            pass

        return hardware

    def _collect_git_info(self) -> dict[str, Any]:
        """Collect git repository information if available."""
        git_info: dict[str, Any] = {
            "available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
            "remote": None,
        }

        try:
            import subprocess

            # Check if we're in a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                cwd=self.run_path,
                timeout=5,
            )

            if result.returncode == 0:
                git_info["available"] = True

                # Get current commit
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=self.run_path,
                    timeout=5,
                )
                if result.returncode == 0:
                    git_info["commit"] = result.stdout.strip()

                # Get current branch
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=self.run_path,
                    timeout=5,
                )
                if result.returncode == 0:
                    git_info["branch"] = result.stdout.strip()

                # Check if working directory is dirty
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    cwd=self.run_path,
                    timeout=10,
                )
                if result.returncode == 0:
                    git_info["dirty"] = bool(result.stdout.strip())

                # Get remote URL
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    capture_output=True,
                    text=True,
                    cwd=self.run_path,
                    timeout=5,
                )
                if result.returncode == 0:
                    git_info["remote"] = result.stdout.strip()

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            pass

        return git_info

    def _collect_traigent_info(self) -> dict[str, Any]:
        """Collect Traigent-specific information."""
        traigent_info = {
            "version": "unknown",
            "config_path": None,
            "cache_dir": None,
        }

        try:
            import traigent

            traigent_info["version"] = getattr(traigent, "__version__", "unknown")
        except ImportError:
            pass

        # Check for Traigent config file
        config_paths = [
            Path.home() / ".traigent" / "config.yaml",
            Path.cwd() / "traigent.yaml",
            Path.cwd() / ".traigent.yaml",
        ]

        for config_path in config_paths:
            if config_path.exists():
                traigent_info["config_path"] = str(config_path)
                break

        # Check for cache directory
        cache_dir = Path.home() / ".traigent" / "cache"
        if cache_dir.exists():
            traigent_info["cache_dir"] = str(cache_dir)

        return traigent_info

    def compute_dataset_checksum(self, dataset_path: Path) -> str:
        """Compute SHA256 checksum of a dataset file.

        Args:
            dataset_path: Path to the dataset file

        Returns:
            SHA256 hex digest of the file
        """
        sha256_hash = hashlib.sha256()

        try:
            resolved_path = self._resolve_path(dataset_path, must_exist=True)
            with open(resolved_path, "rb") as f:
                # Read in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute checksum for {dataset_path}: {e}")
            return "error"

    def add_dataset_info(self, dataset_path: Path | str) -> None:
        """Add dataset information to metadata.

        Args:
            dataset_path: Path to the dataset file
        """
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)

        dataset_info = {
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
            "size_bytes": 0,
            "checksum": None,
            "modified": None,
        }

        if dataset_path.exists():
            stat = dataset_path.stat()
            dataset_info["size_bytes"] = stat.st_size
            dataset_info["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            dataset_info["checksum"] = self.compute_dataset_checksum(dataset_path)

        self.metadata["dataset"] = dataset_info

    def add_custom_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata to the collection.

        Args:
            key: Metadata key
            value: Metadata value
        """
        if "custom" not in self.metadata:
            self.metadata["custom"] = {}
        self.metadata["custom"][key] = value

    def save(self, filename: str = "reproducibility.json") -> Path:
        """Save reproducibility metadata to file.

        Args:
            filename: Name of the metadata file

        Returns:
            Path to the saved metadata file
        """
        metadata_path = self._resolve_path(
            self.run_path / "meta" / filename,
            base_dir=self.run_path,
            must_exist=False,
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable objects to strings
        def make_serializable(obj):
            if isinstance(obj, (tuple, list)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        serializable_metadata = make_serializable(self.metadata)

        with open(metadata_path, "w") as f:
            json.dump(serializable_metadata, f, indent=2)

        logger.info(f"Saved reproducibility metadata to {metadata_path}")
        return metadata_path

    @classmethod
    def load(cls, metadata_path: Path) -> dict[str, Any]:
        """Load reproducibility metadata from file.

        Args:
            metadata_path: Path to the metadata file

        Returns:
            Loaded metadata dictionary
        """
        resolved_path = cls._resolve_path(
            metadata_path, base_dir=metadata_path.parent, must_exist=True
        )
        with open(resolved_path) as f:
            return cast(dict[str, Any], json.load(f))

    def validate_environment(self, other_metadata: dict[str, Any]) -> dict[str, Any]:
        """Validate if current environment matches saved metadata.

        Args:
            other_metadata: Previously saved metadata to compare against

        Returns:
            Validation report with differences
        """
        report: dict[str, Any] = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "differences": {},
        }

        # Check Python version
        current_python = self.metadata["python"]["version_info"]
        saved_python = other_metadata.get("python", {}).get("version_info", {})

        if current_python.get("major") != saved_python.get("major"):
            report["errors"].append(
                f"Python major version mismatch: current={current_python.get('major')}, "
                f"saved={saved_python.get('major')}"
            )
            report["compatible"] = False
        elif current_python.get("minor") != saved_python.get("minor"):
            report["warnings"].append(
                f"Python minor version mismatch: current={current_python.get('minor')}, "
                f"saved={saved_python.get('minor')}"
            )

        # Check critical dependencies
        current_deps = self.metadata["dependencies"]
        saved_deps = other_metadata.get("dependencies", {})

        critical_packages = ["traigent", "numpy", "pandas"]
        for package in critical_packages:
            if package in current_deps and package in saved_deps:
                if current_deps[package] != saved_deps[package]:
                    report["warnings"].append(
                        f"{package} version mismatch: current={current_deps[package]}, "
                        f"saved={saved_deps[package]}"
                    )
                    report["differences"][package] = {
                        "current": current_deps[package],
                        "saved": saved_deps[package],
                    }

        # Check hardware differences
        current_hw = self.metadata["hardware"]
        saved_hw = other_metadata.get("hardware", {})

        if current_hw.get("cpu_count") != saved_hw.get("cpu_count"):
            report["warnings"].append(
                f"CPU count differs: current={current_hw.get('cpu_count')}, "
                f"saved={saved_hw.get('cpu_count')}"
            )

        # Check GPU availability
        current_gpu_count = len(current_hw.get("gpu", []))
        saved_gpu_count = len(saved_hw.get("gpu", []))

        if current_gpu_count != saved_gpu_count:
            report["warnings"].append(
                f"GPU count differs: current={current_gpu_count}, saved={saved_gpu_count}"
            )

        return report


def ensure_reproducibility(
    run_path: Path,
    dataset_path: Path | str | None = None,
    custom_metadata: dict[str, Any] | None = None,
) -> Path:
    """Ensure reproducibility by collecting and saving all metadata.

    Args:
        run_path: Path to the optimization run directory
        dataset_path: Optional path to the dataset file
        custom_metadata: Optional custom metadata to include

    Returns:
        Path to the saved reproducibility metadata file
    """
    collector = ReproducibilityMetadata(run_path)

    if dataset_path:
        collector.add_dataset_info(dataset_path)

    if custom_metadata:
        for key, value in custom_metadata.items():
            collector.add_custom_metadata(key, value)

    return collector.save()
