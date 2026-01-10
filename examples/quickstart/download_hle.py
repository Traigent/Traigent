#!/usr/bin/env python3
"""Download HLE (Humanity's Last Exam) dataset and convert to JSONL format."""

import json
import os
from pathlib import Path

from datasets import load_dataset


def main():
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set")
        print("Usage: export HF_TOKEN=hf_xxxxx && python download_hle.py")
        return

    print("Loading HLE dataset from HuggingFace...")
    dataset = load_dataset("cais/hle", split="test", token=hf_token)

    # Print schema to understand the fields
    print(f"\nDataset has {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    print("\nFirst example:")
    print(json.dumps(dataset[0], indent=2, default=str))

    # Output path
    output_dir = Path(__file__).parent.parent / "datasets" / "quickstart"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "hle.jsonl"

    # Keywords indicating the question requires viewing an image
    image_keywords = [
        "diagram",
        "figure",
        "image",
        "shown",
        "above",
        "below",
        "picture",
        "graph",
        "chart",
        "table",
        "move",
        "position",
        "board",
        "drawing",
        "illustration",
        "visual",
        "see the",
        "look at",
        "observe",
        "displayed",
    ]

    # Transform and write
    count = 0
    skipped_image = 0

    with open(output_file, "w") as f:
        for example in dataset:
            question_text = example.get("question", "")
            question_lower = question_text.lower()

            # Skip if question references visual content
            needs_image = any(kw in question_lower for kw in image_keywords)
            if needs_image:
                skipped_image += 1
                continue

            # Build question text
            question_text = example.get("question", "")

            # If multiple choice, append the choices
            if example.get("choices"):
                choices = example["choices"]
                if isinstance(choices, list):
                    choice_letters = "ABCDEFGHIJ"
                    choices_text = "\n".join(
                        f"{choice_letters[i]}. {c}" for i, c in enumerate(choices)
                    )
                    question_text = f"{question_text}\n\nAnswer with just the letter of the correct option.\nOptions:\n{choices_text}"

            # Get answer
            answer = example.get("answer", "")

            # Get category/subject
            category = example.get("subject", example.get("category", "unknown"))

            # Build our format
            record = {
                "input": {"question": question_text},
                "output": answer,
                "source": "HLE",
                "category": category,
            }

            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"\nWrote {count} examples to {output_file}")
    print(f"Skipped {skipped_image} image-based questions")


if __name__ == "__main__":
    main()
