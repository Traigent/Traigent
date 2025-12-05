"""
Configuration for safety guardrails optimization.

This module defines the search space for optimizing PII detection and hallucination
prevention while maintaining utility for legitimate queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PIIDetectionMethod(Enum):
    """PII detection approaches."""

    REGEX_RULES = "regex_rules"
    NER_MODEL = "ner_model"
    LLM_DETECTION = "llm_detection"
    HYBRID_ENSEMBLE = "hybrid_ensemble"
    PROGRESSIVE_CASCADE = "progressive_cascade"


class RedactionMethod(Enum):
    """PII redaction strategies."""

    MASK_TOKENS = "mask_tokens"
    ENTITY_TYPE_PLACEHOLDER = "entity_type_placeholder"
    SYNTHETIC_REPLACEMENT = "synthetic_replacement"
    CONTEXTUAL_PARAPHRASE = "contextual_paraphrase"


class HallucinationStrategy(Enum):
    """Hallucination prevention approaches."""

    CITATION_REQUIRED = "citation_required"
    CONFIDENCE_THRESHOLDS = "confidence_thresholds"
    FACT_CHECKING = "fact_checking"
    RETRIEVAL_AUGMENTED = "retrieval_augmented"
    UNCERTAINTY_EXPRESSION = "uncertainty_expression"


class SafetyLevel(Enum):
    """Safety configuration levels."""

    MINIMAL = "minimal"  # Basic protection, maximum utility
    BALANCED = "balanced"  # Balanced safety and utility
    STRICT = "strict"  # High safety, some utility impact
    MAXIMUM = "maximum"  # Maximum safety, significant utility impact


class VerificationMethod(Enum):
    """Output verification approaches."""

    NONE = "none"
    LLM_DOUBLE_CHECK = "llm_double_check"
    RULE_BASED_VERIFY = "rule_based_verify"
    HUMAN_IN_LOOP = "human_in_loop"


@dataclass
class SafetyConfig:
    """Configuration for safety guardrails."""

    # PII Detection
    pii_detection: str = "hybrid_ensemble"
    pii_threshold: float = 0.8
    redaction_method: str = "entity_type_placeholder"

    # Hallucination Prevention
    hallucination_strategy: str = "citation_required"
    confidence_threshold: float = 0.7
    fact_checking_enabled: bool = True

    # Safety Level
    safety_level: str = "balanced"

    # Verification
    verification_method: str = "llm_double_check"
    verification_threshold: float = 0.85

    # Advanced Settings
    allow_medical_info: bool = False
    allow_financial_info: bool = False
    context_awareness: bool = True
    whitelist_domains: list[str] | None = None

    # Performance Settings
    max_processing_time_ms: int = 500
    cache_results: bool = True
    batch_processing: bool = False

    def __post_init__(self):
        """Initialize default settings."""
        if self.whitelist_domains is None:
            self.whitelist_domains = []

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "pii_detection": self.pii_detection,
            "pii_threshold": self.pii_threshold,
            "redaction_method": self.redaction_method,
            "hallucination_strategy": self.hallucination_strategy,
            "confidence_threshold": self.confidence_threshold,
            "fact_checking_enabled": self.fact_checking_enabled,
            "safety_level": self.safety_level,
            "verification_method": self.verification_method,
            "verification_threshold": self.verification_threshold,
            "allow_medical_info": self.allow_medical_info,
            "allow_financial_info": self.allow_financial_info,
            "context_awareness": self.context_awareness,
            "whitelist_domains": self.whitelist_domains,
            "max_processing_time_ms": self.max_processing_time_ms,
            "cache_results": self.cache_results,
            "batch_processing": self.batch_processing,
        }


# TraiGent search space for safety guardrails optimization
SAFETY_SEARCH_SPACE = {
    # PII Detection Pipeline
    "pii_detection": [
        "regex_rules",  # Rule-based pattern matching
        "ner_model",  # Named entity recognition
        "llm_detection",  # LLM-based PII detection
        "hybrid_ensemble",  # Combine multiple methods
        "progressive_cascade",  # Sequential application
    ],
    # PII Detection Thresholds
    "pii_threshold": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
    # Redaction Strategies
    "redaction_method": [
        "mask_tokens",  # Replace with ***
        "entity_type_placeholder",  # [EMAIL], [NAME], etc.
        "synthetic_replacement",  # Generate fake alternatives
        "contextual_paraphrase",  # Rephrase without PII
    ],
    # Hallucination Prevention
    "hallucination_strategy": [
        "citation_required",  # Require source citations
        "confidence_thresholds",  # Block low-confidence claims
        "fact_checking",  # Verify against sources
        "retrieval_augmented",  # RAG-based verification
        "uncertainty_expression",  # Express uncertainty explicitly
    ],
    # Confidence Thresholds
    "confidence_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    # Fact Checking
    "fact_checking_enabled": [True, False],
    # Safety Levels
    "safety_level": [
        "minimal",  # Basic protection
        "balanced",  # Balanced approach
        "strict",  # High safety
        "maximum",  # Maximum protection
    ],
    # Verification Methods
    "verification_method": [
        "none",  # No verification
        "llm_double_check",  # Second LLM verification
        "rule_based_verify",  # Rule-based verification
        "human_in_loop",  # Human verification
    ],
    # Verification Thresholds
    "verification_threshold": [0.7, 0.75, 0.8, 0.85, 0.9],
    # Domain-Specific Settings
    "allow_medical_info": [True, False],
    "allow_financial_info": [True, False],
    "context_awareness": [True, False],
    # Performance Settings
    "max_processing_time_ms": [100, 250, 500, 750, 1000],
    "cache_results": [True, False],
    "batch_processing": [True, False],
}


def create_safety_config(**config_params) -> SafetyConfig:
    """Create a SafetyConfig from parameter dictionary."""
    return SafetyConfig(**config_params)


# PII Detection Patterns
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "address": r"\b\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b",
    # Additional patterns for names, dates, etc.
    "name": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Simple name pattern
    "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
}


def detect_pii_regex(
    text: str, patterns: dict[str, str] | None = None
) -> dict[str, list[str]]:
    """Detect PII using regex patterns."""
    if patterns is None:
        patterns = PII_PATTERNS

    detected = {}
    for pii_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            detected[pii_type] = matches

    return detected


def redact_pii(
    text: str,
    detected_pii: dict[str, list[str]],
    method: str = "entity_type_placeholder",
) -> str:
    """Redact PII from text using specified method."""

    redacted_text = text

    for pii_type, instances in detected_pii.items():
        for instance in instances:
            if method == "mask_tokens":
                replacement = "*" * len(instance)
            elif method == "entity_type_placeholder":
                replacement = f"[{pii_type.upper()}]"
            elif method == "synthetic_replacement":
                replacement = generate_synthetic_replacement(pii_type)
            elif method == "contextual_paraphrase":
                replacement = f"[redacted {pii_type}]"
            else:
                replacement = "[REDACTED]"

            redacted_text = redacted_text.replace(instance, replacement)

    return redacted_text


def generate_synthetic_replacement(pii_type: str) -> str:
    """Generate synthetic replacement for PII type."""

    replacements = {
        "email": "user@example.com",
        "phone": "(555) 123-4567",
        "ssn": "123-45-6789",
        "credit_card": "1234-5678-9012-3456",
        "name": "John Smith",
        "address": "123 Main Street",
        "date": "01/01/2024",
        "ip_address": "192.168.1.1",
    }

    return replacements.get(pii_type, "[SYNTHETIC]")


def assess_hallucination_risk(
    text: str,
    confidence_score: float,
    has_citations: bool = False,
    factual_claims_count: int = 0,
) -> dict[str, Any]:
    """Assess hallucination risk for generated text."""

    risk_factors = []
    risk_score = 0.0

    # Low confidence increases risk
    if confidence_score < 0.7:
        risk_factors.append("low_confidence")
        risk_score += 0.3

    # No citations for factual claims
    if factual_claims_count > 0 and not has_citations:
        risk_factors.append("no_citations")
        risk_score += 0.4

    # Absolute statements without hedging
    if any(
        phrase in text.lower()
        for phrase in ["definitely", "certainly", "always", "never", "all", "none"]
    ):
        risk_factors.append("absolute_statements")
        risk_score += 0.2

    # Specific numbers or dates without sources
    if re.search(r"\b\d{4}\b|\b\d+%\b|\$\d+", text) and not has_citations:
        risk_factors.append("unverified_statistics")
        risk_score += 0.3

    return {
        "risk_score": min(risk_score, 1.0),
        "risk_factors": risk_factors,
        "risk_level": (
            "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
        ),
    }


def get_baseline_configs() -> list[dict[str, Any]]:
    """Get baseline safety configurations."""

    return [
        {
            "name": "minimal_safety",
            "pii_detection": "regex_rules",
            "pii_threshold": 0.9,
            "redaction_method": "mask_tokens",
            "hallucination_strategy": "none",
            "confidence_threshold": 0.5,
            "fact_checking_enabled": False,
            "safety_level": "minimal",
            "verification_method": "none",
            "verification_threshold": 0.7,
            "allow_medical_info": True,
            "allow_financial_info": True,
            "context_awareness": False,
            "max_processing_time_ms": 100,
            "cache_results": True,
        },
        {
            "name": "balanced_safety",
            "pii_detection": "hybrid_ensemble",
            "pii_threshold": 0.8,
            "redaction_method": "entity_type_placeholder",
            "hallucination_strategy": "confidence_thresholds",
            "confidence_threshold": 0.7,
            "fact_checking_enabled": True,
            "safety_level": "balanced",
            "verification_method": "llm_double_check",
            "verification_threshold": 0.8,
            "allow_medical_info": False,
            "allow_financial_info": False,
            "context_awareness": True,
            "max_processing_time_ms": 500,
            "cache_results": True,
        },
        {
            "name": "strict_safety",
            "pii_detection": "progressive_cascade",
            "pii_threshold": 0.75,
            "redaction_method": "contextual_paraphrase",
            "hallucination_strategy": "citation_required",
            "confidence_threshold": 0.85,
            "fact_checking_enabled": True,
            "safety_level": "strict",
            "verification_method": "rule_based_verify",
            "verification_threshold": 0.9,
            "allow_medical_info": False,
            "allow_financial_info": False,
            "context_awareness": True,
            "max_processing_time_ms": 750,
            "cache_results": True,
        },
    ]


def calculate_safety_utility_tradeoff(
    safety_score: float,
    utility_score: float,
    pii_recall: float,
    false_positive_rate: float,
) -> dict[str, float]:
    """Calculate safety-utility tradeoff metrics."""

    # Balanced score considering both safety and utility
    balanced_score = safety_score * 0.6 + utility_score * 0.4

    # Penalty for high false positives
    false_positive_penalty = false_positive_rate * 0.3
    adjusted_score = balanced_score - false_positive_penalty

    # PII protection effectiveness
    pii_effectiveness = pii_recall * (1 - false_positive_rate)

    return {
        "balanced_score": balanced_score,
        "adjusted_score": max(0.0, adjusted_score),
        "safety_score": safety_score,
        "utility_score": utility_score,
        "pii_effectiveness": pii_effectiveness,
        "false_positive_penalty": false_positive_penalty,
        "overall_effectiveness": min(1.0, (pii_effectiveness + adjusted_score) / 2),
    }
