"""Deterministic content feature extraction for server-side analytics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from traigent.evaluators.base import Dataset
from traigent.utils.example_id import compute_dataset_hash, generate_stable_example_id

_TOKEN_PATTERN = re.compile(r"\w+")
_SIMHASH_BITS = 64


class SimhashFeatureExtractor:
    """Build per-example simhash features for backend-side content analytics."""

    def extract_dataset_features(self, dataset: Dataset) -> list[dict[str, str]]:
        """Return stable example IDs paired with a 64-bit hex simhash."""
        dataset_name = getattr(dataset, "name", "dataset")
        dataset_hash = compute_dataset_hash(dataset_name)
        return [
            {
                "example_id": generate_stable_example_id(dataset_hash, index),
                "feature": self.compute_feature(example.input_data),
            }
            for index, example in enumerate(dataset.examples)
        ]

    def compute_feature(self, input_data: Any) -> str:
        """Compute a deterministic 64-bit simhash for example input data."""
        text = self._normalize_input(input_data)
        tokens = _TOKEN_PATTERN.findall(text.lower())
        weights = Counter(tokens) if tokens else Counter({text or "<empty>": 1})
        bit_scores = [0] * _SIMHASH_BITS

        for token, weight in weights.items():
            token_hash = hashlib.sha256(token.encode("utf-8")).digest()
            token_bits = int.from_bytes(token_hash[:8], byteorder="big", signed=False)
            for bit in range(_SIMHASH_BITS):
                mask = 1 << bit
                bit_scores[bit] += weight if token_bits & mask else -weight

        simhash = 0
        for bit, score in enumerate(bit_scores):
            if score >= 0:
                simhash |= 1 << bit

        return f"{simhash:016x}"

    @staticmethod
    def _normalize_input(input_data: Any) -> str:
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, dict):
            return json.dumps(input_data, sort_keys=True, ensure_ascii=True)
        try:
            return json.dumps(input_data, sort_keys=True, ensure_ascii=True)
        except TypeError:
            return str(input_data)
