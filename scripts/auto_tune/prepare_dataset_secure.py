#!/usr/bin/env python3
"""
Secure dataset preparation for auto-tuning optimization.
Includes input validation, error handling, and audit logging.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from security_utils import (
    AuditLogger,
    retry_with_backoff,
    safe_file_read,
    safe_file_write,
    sanitize_input,
    setup_logging,
    timeout,
    validate_json_schema,
    validate_path,
)

# Initialize logging and audit
logger = setup_logging(__name__, "prepare_dataset.log")
audit_logger = AuditLogger("prepare_dataset_audit.jsonl")


def load_params() -> Dict[str, Any]:
    """Load and validate parameters from params.yaml."""
    try:
        params_path = Path("params.yaml")
        if not validate_path(params_path):
            raise ValueError("Invalid params.yaml path")

        content = safe_file_read(params_path)
        if not content:
            raise FileNotFoundError("params.yaml not found")

        params = yaml.safe_load(content)

        # Validate required parameters
        required = ["prepare", "optimize", "evaluate"]
        if not all(key in params for key in required):
            raise ValueError("Missing required parameters in params.yaml")

        logger.info("Parameters loaded successfully")
        return params

    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        audit_logger.log_event("params_load", {"error": str(e)}, success=False)
        raise


def validate_dataset_item(item: Dict[str, Any]) -> bool:
    """Validate a single dataset item."""
    required_keys = ["input", "expected", "type"]

    # Check required keys
    if not validate_json_schema(item, required_keys):
        return False

    # Validate input length
    if len(str(item["input"])) > 10000:
        logger.warning("Input too long, truncating")
        item["input"] = str(item["input"])[:10000]

    # Sanitize inputs
    item["input"] = sanitize_input(item["input"])

    # Validate type
    valid_types = ["classification", "summarization", "extraction", "generation", "qa"]
    if item["type"] not in valid_types:
        logger.warning(f"Invalid type: {item['type']}")
        return False

    return True


@retry_with_backoff(max_attempts=3)
def load_external_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load dataset from external source with retry logic."""
    if not validate_path(path):
        raise ValueError(f"Invalid dataset path: {path}")

    try:
        with timeout(30):  # 30 second timeout
            content = safe_file_read(path)
            if not content:
                raise FileNotFoundError(f"Dataset not found: {path}")

            data = json.loads(content)

            # Validate dataset structure
            if not isinstance(data, list):
                raise ValueError("Dataset must be a list")

            # Validate each item
            valid_items = []
            for item in data:
                if validate_dataset_item(item):
                    valid_items.append(item)
                else:
                    logger.warning(
                        f"Skipping invalid item: {item.get('id', 'unknown')}"
                    )

            logger.info(f"Loaded {len(valid_items)} valid items from {path}")
            return valid_items

    except TimeoutError:
        logger.error(f"Timeout loading dataset from {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def generate_sample_dataset() -> List[Dict[str, Any]]:
    """Generate a secure sample dataset for testing."""
    logger.info("Generating sample dataset")

    # Sample tasks with sanitized inputs
    tasks = [
        {
            "input": sanitize_input(
                "Classify the sentiment of: 'This product exceeded my expectations!'"
            ),
            "expected": "positive",
            "type": "classification",
            "id": "sample_class_1",
        },
        {
            "input": sanitize_input(
                "Summarize: 'AI is transforming industries through automation.'"
            ),
            "expected": "AI transforms industries via automation.",
            "type": "summarization",
            "id": "sample_summ_1",
        },
        {
            "input": sanitize_input(
                "Extract entities from: 'Apple Inc. announced new products.'"
            ),
            "expected": ["Apple Inc."],
            "type": "extraction",
            "id": "sample_ext_1",
        },
        {
            "input": sanitize_input("Translate to Spanish: 'Hello, how are you?'"),
            "expected": "Hola, ¿cómo estás?",
            "type": "generation",
            "id": "sample_gen_1",
        },
        {
            "input": sanitize_input("Answer: What is the capital of France?"),
            "expected": "Paris",
            "type": "qa",
            "id": "sample_qa_1",
        },
    ]

    # Expand dataset with variations
    expanded_dataset = []
    for task in tasks:
        for i in range(5):
            variant = task.copy()
            variant["id"] = f"{task['type']}_{i}"
            variant["variant"] = i
            expanded_dataset.append(variant)

    logger.info(f"Generated {len(expanded_dataset)} sample items")
    return expanded_dataset


def split_dataset(
    dataset: List[Dict[str, Any]], split_ratio: float, seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset with validation."""
    # Validate split ratio
    if not 0.1 <= split_ratio <= 0.9:
        raise ValueError(f"Invalid split ratio: {split_ratio}")

    # Validate seed
    if not 0 <= seed <= 2**32:
        raise ValueError(f"Invalid seed: {seed}")

    random.seed(seed)
    shuffled = dataset.copy()
    random.shuffle(shuffled)

    split_point = int(len(shuffled) * split_ratio)
    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]

    logger.info(f"Split dataset: {len(train_set)} train, {len(test_set)} test")

    return train_set, test_set


def prepare_for_traigent(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare dataset in TraiGent-compatible format with validation."""
    prepared = {
        "version": "1.0",
        "problem_type": "mixed",
        "dataset": dataset,
        "metadata": {
            "total_samples": len(dataset),
            "problem_types": list({d.get("type", "unknown") for d in dataset}),
            "prepared_at": os.environ.get("CI_COMMIT_SHA", "local"),
            "timestamp": os.environ.get("CI_COMMIT_TIMESTAMP", "local"),
            "secure": True,
        },
    }

    # Validate structure
    if not validate_json_schema(
        prepared, ["version", "problem_type", "dataset", "metadata"]
    ):
        raise ValueError("Invalid prepared dataset structure")

    return prepared


def main():
    """Main preparation pipeline with comprehensive error handling."""
    start_time = os.environ.get("CI_JOB_STARTED_AT", "local")

    try:
        print("📊 Securely preparing dataset for auto-tuning...")
        logger.info("Starting dataset preparation")

        # Audit log start
        audit_logger.log_event(
            "prepare_start",
            {
                "start_time": start_time,
                "environment": os.environ.get("CI_ENVIRONMENT_NAME", "development"),
            },
        )

        # Load parameters with validation
        params = load_params()
        split_ratio = float(params["prepare"]["split_ratio"])
        seed = int(params["prepare"]["seed"])

        # Validate parameters
        if not 0 < split_ratio < 1:
            raise ValueError(f"Invalid split ratio: {split_ratio}")

        # Create directories with validation
        raw_dir = Path("data/raw")
        prepared_dir = Path("data/prepared")

        for dir_path in [raw_dir, prepared_dir]:
            if not validate_path(dir_path):
                raise ValueError(f"Invalid directory path: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load or generate dataset
        raw_data_file = raw_dir / "dataset.json"

        if raw_data_file.exists():
            logger.info(f"Loading dataset from {raw_data_file}")
            dataset = load_external_dataset(raw_data_file)
        else:
            logger.info("Generating sample dataset")
            dataset = generate_sample_dataset()

            # Save for reproducibility
            safe_file_write(raw_data_file, json.dumps(dataset, indent=2), backup=True)

        # Split dataset with validation
        train_set, test_set = split_dataset(dataset, split_ratio, seed)

        # Prepare TraiGent-compatible format
        train_data = prepare_for_traigent(train_set)
        test_data = prepare_for_traigent(test_set)

        # Save prepared data with atomic writes
        safe_file_write(
            prepared_dir / "train.json", json.dumps(train_data, indent=2), backup=False
        )

        safe_file_write(
            prepared_dir / "test.json", json.dumps(test_data, indent=2), backup=False
        )

        print(f"✅ Dataset securely prepared and saved to {prepared_dir}")

        # Output statistics for DVC metrics
        stats = {
            "train_samples": len(train_set),
            "test_samples": len(test_set),
            "total_samples": len(dataset),
            "split_ratio": split_ratio,
            "secure": True,
        }

        safe_file_write(
            Path("data/prepared/stats.json"), json.dumps(stats, indent=2), backup=False
        )

        # Audit log success
        audit_logger.log_event(
            "prepare_complete", {"stats": stats, "duration": "completed"}
        )

        logger.info("Dataset preparation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        print(f"❌ Dataset preparation failed: {e}")

        # Audit log failure
        audit_logger.log_event(
            "prepare_failed", {"error": str(e), "type": type(e).__name__}, success=False
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
