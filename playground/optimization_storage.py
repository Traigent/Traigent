"""
Optimization Results Storage Module
===================================

This module provides file-based storage functionality for TraiGent optimization results.
It handles saving, loading, and managing optimization runs across different problems.

Features:
- Persistent storage of optimization results
- Organized directory structure by problem name
- JSON-based storage format
- Version control for future compatibility
- Error handling and recovery
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class OptimizationStorage:
    """Handles file-based storage of optimization results."""

    STORAGE_VERSION = "1.0"
    RESULTS_DIR = "optimization_results"

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the storage handler.

        Args:
            base_path: Base directory for storage. Defaults to examples/optimization_results
        """
        if base_path is None:
            base_path = Path(__file__).parent / self.RESULTS_DIR
        self.base_path = Path(base_path)
        self.ensure_storage_directories()

    def ensure_storage_directories(self):
        """Create the base storage directory if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_problem_directory(self, problem_name: str) -> Path:
        """
        Get the directory path for a specific problem.

        Args:
            problem_name: Name of the problem

        Returns:
            Path object for the problem directory
        """
        # Clean problem name for filesystem
        safe_name = "".join(
            c for c in problem_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_name = safe_name.replace(" ", "_").lower()
        return self.base_path / safe_name

    def generate_run_id(self, problem_name: str, strategy: str) -> str:
        """
        Generate a unique run ID.

        Args:
            problem_name: Name of the problem
            strategy: Optimization strategy used

        Returns:
            Unique run identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add microseconds for uniqueness
        microseconds = datetime.now().strftime("%f")[:3]
        return f"run_{timestamp}_{microseconds}_{strategy}"

    def get_result_filepath(self, problem_name: str, run_id: str) -> Path:
        """
        Get the file path for a specific optimization result.

        Args:
            problem_name: Name of the problem
            run_id: Unique run identifier

        Returns:
            Path object for the result file
        """
        problem_dir = self.get_problem_directory(problem_name)
        return problem_dir / f"{run_id}.json"

    def save_optimization_result(self, result: Dict[str, Any]) -> str:
        """
        Save an optimization result to disk.

        Args:
            result: Optimization result dictionary

        Returns:
            The run_id of the saved result

        Raises:
            ValueError: If required fields are missing
            IOError: If save operation fails
        """
        # Validate required fields
        required_fields = ["problem", "strategy", "timestamp"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Generate run ID if not present
        if "run_id" not in result:
            result["run_id"] = self.generate_run_id(
                result["problem"], result["strategy"]
            )

        # Add storage metadata
        result["storage_version"] = self.STORAGE_VERSION
        result["saved_at"] = datetime.now().isoformat()

        # Ensure problem directory exists
        problem_dir = self.get_problem_directory(result["problem"])
        problem_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        filepath = self.get_result_filepath(result["problem"], result["run_id"])
        try:
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2, default=str)
            return result["run_id"]
        except Exception as e:
            raise OSError(f"Failed to save optimization result: {str(e)}") from e

    def load_optimization_result(
        self, problem_name: str, run_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load a specific optimization result.

        Args:
            problem_name: Name of the problem
            run_id: Run identifier

        Returns:
            Result dictionary or None if not found
        """
        filepath = self.get_result_filepath(problem_name, run_id)
        if not filepath.exists():
            return None

        try:
            with open(filepath) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading result {run_id}: {str(e)}")
            return None

    def load_problem_results(self, problem_name: str) -> List[Dict[str, Any]]:
        """
        Load all optimization results for a specific problem.

        Args:
            problem_name: Name of the problem

        Returns:
            List of result dictionaries, sorted by timestamp (newest first)
        """
        problem_dir = self.get_problem_directory(problem_name)
        if not problem_dir.exists():
            return []

        results = []
        for file_path in problem_dir.glob("run_*.json"):
            try:
                with open(file_path) as f:
                    result = json.load(f)
                    # Ensure run_id is present
                    if "run_id" not in result:
                        result["run_id"] = file_path.stem
                    results.append(result)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

    def load_all_results(self) -> List[Dict[str, Any]]:
        """
        Load all optimization results across all problems.

        Returns:
            List of all result dictionaries, sorted by timestamp (newest first)
        """
        all_results = []

        if not self.base_path.exists():
            return []

        # Iterate through all problem directories
        for problem_dir in self.base_path.iterdir():
            if problem_dir.is_dir():
                # Get problem name from directory
                problem_name = problem_dir.name.replace("_", " ").title()

                # Load results for this problem
                for file_path in problem_dir.glob("run_*.json"):
                    try:
                        with open(file_path) as f:
                            result = json.load(f)
                            # Ensure run_id is present
                            if "run_id" not in result:
                                result["run_id"] = file_path.stem
                            # Ensure problem name matches
                            if "problem" not in result:
                                result["problem"] = problem_name
                            all_results.append(result)
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
                        continue

        # Sort by timestamp (newest first)
        all_results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_results

    def delete_result(self, problem_name: str, run_id: str) -> bool:
        """
        Delete a specific optimization result.

        Args:
            problem_name: Name of the problem
            run_id: Run identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        filepath = self.get_result_filepath(problem_name, run_id)
        if filepath.exists():
            try:
                filepath.unlink()
                return True
            except Exception as e:
                print(f"Error deleting result: {str(e)}")
                return False
        return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "total_problems": 0,
            "total_results": 0,
            "problems": {},
            "storage_size_mb": 0,
        }

        if not self.base_path.exists():
            return stats

        total_size = 0

        for problem_dir in self.base_path.iterdir():
            if problem_dir.is_dir():
                stats["total_problems"] += 1
                problem_name = problem_dir.name.replace("_", " ").title()
                result_count = len(list(problem_dir.glob("run_*.json")))
                stats["problems"][problem_name] = result_count
                stats["total_results"] += result_count

                # Calculate size
                for file_path in problem_dir.glob("*.json"):
                    total_size += file_path.stat().st_size

        stats["storage_size_mb"] = round(total_size / (1024 * 1024), 2)
        return stats


# Convenience functions for direct import
storage = OptimizationStorage()


def save_optimization_result(result: Dict[str, Any]) -> str:
    """Save an optimization result."""
    return storage.save_optimization_result(result)


def load_optimization_results(problem_name: str) -> List[Dict[str, Any]]:
    """Load all results for a problem."""
    return storage.load_problem_results(problem_name)


def load_all_optimization_results() -> List[Dict[str, Any]]:
    """Load all optimization results."""
    return storage.load_all_results()


def get_storage_stats() -> Dict[str, Any]:
    """Get storage statistics."""
    return storage.get_storage_stats()
