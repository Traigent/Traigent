"""HotpotQA-style dataset for multi-hop question answering optimization.

This module provides a small bundled dataset suitable for regression tests
and demonstrations without requiring external downloads.

To use the full HotpotQA benchmark, set the environment variable:
    HOTPOTQA_DATASET_PATH=/path/to/hotpot_dev_distractor_v1.json

The official dataset can be downloaded from:
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable

from traigent.evaluators.base import Dataset, EvaluationExample

__all__ = ["dataset_path", "load_case_study_dataset"]

# Environment variable to override with full HotpotQA benchmark
HOTPOTQA_DATASET_PATH_ENV = "HOTPOTQA_DATASET_PATH"


@dataclass(frozen=True)
class _RawExample:
    """Internal representation of a HotpotQA-style example."""

    question: str
    answer: str
    context: list[str]
    metadata: dict[str, Any]


# Custom multi-hop demo questions designed to demonstrate Traigent's optimization.
# These are bridge-type questions requiring reasoning across multiple passages.
# For the full HotpotQA benchmark, set HOTPOTQA_DATASET_PATH environment variable.
_RAW_EXAMPLES: Final[list[_RawExample]] = [
    _RawExample(
        question="What is the capital of the country where the Eiffel Tower is located?",
        answer="Paris",
        context=[
            "France: France is a country in Western Europe with several overseas territories.",
            "Eiffel Tower: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "Berlin: Berlin is the capital and largest city of Germany by both area and population.",
            "Paris: Paris is the capital and most populous city of France, with a population of over 2 million.",
            "Tower Bridge: Tower Bridge is a combined bascule and suspension bridge in London.",
        ],
        metadata={
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Eiffel Tower", 0], ["Paris", 0]],
        },
    ),
    _RawExample(
        question="Who founded the company that created the iPhone?",
        answer="Steve Jobs",
        context=[
            "iPhone: The iPhone is a smartphone made by Apple that combines a phone with an iPod.",
            "Apple Inc.: Apple Inc. is an American technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
            "Samsung: Samsung Electronics is a South Korean multinational electronics corporation.",
            "Microsoft: Microsoft Corporation is an American technology company founded by Bill Gates and Paul Allen.",
            "Steve Jobs: Steve Jobs was an American entrepreneur and co-founder of Apple Inc.",
        ],
        metadata={
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["iPhone", 0], ["Apple Inc.", 0]],
        },
    ),
    _RawExample(
        question="What is the nationality of the author who wrote 'A Tale of Two Cities'?",
        answer="English",
        context=[
            "Charles Dickens: Charles Dickens was an English novelist and social critic, one of the greatest writers of the Victorian era.",
            "A Tale of Two Cities: A Tale of Two Cities is a historical novel by Charles Dickens, set during the French Revolution, published in 1859.",
            "Victor Hugo: Victor Hugo was a French novelist who wrote 'Les Miserables', also set during revolutionary France.",
            "Oliver Twist: Oliver Twist is another famous novel by Charles Dickens, published in 1837.",
            "Mark Twain: Mark Twain was an American author known for 'The Adventures of Tom Sawyer'.",
        ],
        metadata={
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["A Tale of Two Cities", 0], ["Charles Dickens", 0]],
        },
    ),
    _RawExample(
        question="What river flows through the city where Mozart was born?",
        answer="Salzach",
        context=[
            "Wolfgang Amadeus Mozart: Mozart was born on January 27, 1756, in Salzburg, which was then part of the Holy Roman Empire.",
            "Salzburg: Salzburg is a city in Austria located on the banks of the Salzach River.",
            "Vienna: Vienna is the capital of Austria, situated on the Danube River.",
            "Salzach River: The Salzach is a river in Austria and Germany, flowing through Salzburg.",
            "Danube: The Danube is the second-longest river in Europe, flowing through Vienna.",
        ],
        metadata={
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Wolfgang Amadeus Mozart", 0], ["Salzburg", 0]],
        },
    ),
    _RawExample(
        question="What is the population of the country where Mount Fuji is located?",
        answer="125 million",
        context=[
            "Mount Fuji: Mount Fuji is an active stratovolcano and the highest peak in Japan at 3,776 meters.",
            "Japan: Japan is an island country in East Asia with a population of approximately 125 million people.",
            "Mount Everest: Mount Everest is the highest mountain in the world, located on the Nepal-China border.",
            "China: China is the world's most populous country with over 1.4 billion people.",
            "Tokyo: Tokyo is the capital city of Japan with a metropolitan population of over 37 million.",
        ],
        metadata={
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["Mount Fuji", 0], ["Japan", 0]],
        },
    ),
    _RawExample(
        question="Who wrote the national anthem of the country where the Amazon River originates?",
        answer="Jose de la Torre Ugarte",
        context=[
            "Amazon River: The Amazon River originates in Peru near the city of Arequipa in the Andes Mountains.",
            "Peru: Peru is a country in South America. Its national anthem was composed by Jose de la Torre Ugarte.",
            "Brazil: Brazil is the largest country in South America through which most of the Amazon flows.",
            "National Anthem of Peru: The National Anthem of Peru was composed by Jose de la Torre Ugarte in 1821.",
            "Nile River: The Nile is the longest river in Africa, flowing through Egypt.",
        ],
        metadata={
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["Amazon River", 0], ["Peru", 0]],
        },
    ),
    _RawExample(
        question="What sport is played at the stadium where the 1966 World Cup final was held?",
        answer="Football",
        context=[
            "1966 FIFA World Cup: The 1966 FIFA World Cup final was held at Wembley Stadium in London, England.",
            "Wembley Stadium: Wembley Stadium is a football stadium in London, primarily used for association football.",
            "Wimbledon: Wimbledon is an annual tennis tournament held at the All England Lawn Tennis Club.",
            "Lords Cricket Ground: Lord's is a cricket ground in London, known as the home of cricket.",
            "Twickenham: Twickenham Stadium is a rugby union stadium in southwest London.",
        ],
        metadata={
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["1966 FIFA World Cup", 0], ["Wembley Stadium", 0]],
        },
    ),
    _RawExample(
        question="What language is primarily spoken in the country where the Nobel Prize ceremony is held?",
        answer="Swedish",
        context=[
            "Nobel Prize: The Nobel Prize ceremony is held annually in Stockholm, Sweden for most categories.",
            "Sweden: Sweden is a Nordic country in Northern Europe where Swedish is the official language.",
            "Norway: Norway is where the Nobel Peace Prize is awarded, in Oslo.",
            "Oslo: Oslo is the capital of Norway where Norwegian is the primary language.",
            "Stockholm: Stockholm is the capital of Sweden and hosts the Nobel Prize ceremony.",
        ],
        metadata={
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Nobel Prize", 0], ["Sweden", 0]],
        },
    ),
]

_DATASET: Dataset | None = None
_DATASETS_DIR = Path(__file__).parent / "datasets"


def dataset_path() -> Path:
    """Return the path to the bundled HotpotQA-style dataset file."""
    return _DATASETS_DIR / "hotpotqa_dev_subset.jsonl"


def _build_examples(raw_examples: Iterable[_RawExample]) -> list[EvaluationExample]:
    """Convert raw examples to EvaluationExample format."""
    examples: list[EvaluationExample] = []
    for raw in raw_examples:
        metadata = dict(raw.metadata)
        metadata.setdefault("case_study", "hotpotqa")
        metadata.setdefault("fallback_answer", raw.answer)
        examples.append(
            EvaluationExample(
                input_data={"question": raw.question, "context": raw.context},
                expected_output=raw.answer,
                metadata=metadata,
            )
        )
    return examples


def _ensure_dataset_file() -> None:
    """Generate the JSONL dataset file if it doesn't exist."""
    path = dataset_path()
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for raw in _RAW_EXAMPLES:
            record = {
                "input": {"question": raw.question, "context": raw.context},
                "output": raw.answer,
                "metadata": raw.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_official_hotpotqa(file_path: Path, max_examples: int = 100) -> list[_RawExample]:
    """Load examples from the official HotpotQA JSON file.

    Args:
        file_path: Path to hotpot_dev_distractor_v1.json
        max_examples: Maximum number of examples to load (default 100 for demo)

    Returns:
        List of _RawExample objects
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    examples: list[_RawExample] = []
    for item in data[:max_examples]:
        # HotpotQA format: each item has 'context' as list of [title, sentences]
        context_passages: list[str] = []
        for title, sentences in item.get("context", []):
            # Combine title with first sentence for context
            if sentences:
                context_passages.append(f"{title}: {sentences[0]}")

        examples.append(
            _RawExample(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                context=context_passages,
                metadata={
                    "id": item.get("_id", ""),
                    "type": item.get("type", "unknown"),
                    "level": item.get("level", "unknown"),
                    "supporting_facts": item.get("supporting_facts", []),
                },
            )
        )

    return examples


def load_case_study_dataset(
    *, force_refresh: bool = False, max_examples: int | None = None
) -> Dataset:
    """Return a cached dataset containing HotpotQA-style multi-hop questions.

    If HOTPOTQA_DATASET_PATH environment variable is set, loads from that file.
    Otherwise uses the bundled 8-example dataset.

    Args:
        force_refresh: Force reload of the dataset
        max_examples: Maximum examples to load from full benchmark (default: 100)

    Returns:
        Dataset object with HotpotQA examples
    """
    global _DATASET

    external_path = os.environ.get(HOTPOTQA_DATASET_PATH_ENV)

    if _DATASET is None or force_refresh:
        if external_path:
            # Load from official HotpotQA benchmark
            path = Path(external_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"HotpotQA dataset not found at {path}. "
                    f"Download from: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
                )
            raw_examples = _load_official_hotpotqa(path, max_examples or 100)
            description = f"Official HotpotQA benchmark ({len(raw_examples)} examples)"
        else:
            # Use bundled examples
            _ensure_dataset_file()
            raw_examples = list(_RAW_EXAMPLES)
            description = "Bundled HotpotQA-style examples for quick testing"

        _DATASET = Dataset(
            examples=_build_examples(raw_examples),
            name="hotpotqa_case_study",
            description=description,
            metadata={"task": "multi_hop_qa", "source": "official" if external_path else "bundled"},
        )

    return _DATASET
