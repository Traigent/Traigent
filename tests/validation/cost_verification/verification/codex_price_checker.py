"""Use Codex SDK to verify provider pricing against their official URLs.

This module provides functionality to:
1. Fetch pricing from provider websites using Codex
2. Compare computed prices against official pricing
3. Return verification results with discrepancy details
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

# Provider pricing URLs
PROVIDER_PRICING_URLS = {
    "openai": "https://openai.com/api/pricing/",
    "anthropic": "https://www.anthropic.com/pricing",
    "groq": "https://groq.com/pricing/",
    "openrouter": "https://openrouter.ai/models",
    "google": "https://ai.google.dev/pricing",
    "cohere": "https://cohere.com/pricing",
}


@dataclass
class PriceVerificationResult:
    """Result of price verification against official source."""

    provider: str
    model: str
    computed_input_price: float
    computed_output_price: float
    official_input_price: float | None
    official_output_price: float | None
    input_matches: bool
    output_matches: bool
    source_url: str
    notes: str
    raw_response: str

    @property
    def verified(self) -> bool:
        """Check if both input and output prices match."""
        return self.input_matches and self.output_matches


def verify_pricing_with_codex(
    provider: str,
    model: str,
    computed_input_price: float,
    computed_output_price: float,
    reasoning_effort: str = "high",
) -> PriceVerificationResult:
    """
    Ask Codex to fetch and verify pricing from provider URL.

    Args:
        provider: Provider name (openai, anthropic, etc.)
        model: Model identifier
        computed_input_price: Our computed input price per token
        computed_output_price: Our computed output price per token
        reasoning_effort: Codex reasoning effort (high, xhigh)

    Returns:
        PriceVerificationResult with verification details
    """
    url = PROVIDER_PRICING_URLS.get(provider)
    if not url:
        return PriceVerificationResult(
            provider=provider,
            model=model,
            computed_input_price=computed_input_price,
            computed_output_price=computed_output_price,
            official_input_price=None,
            official_output_price=None,
            input_matches=False,
            output_matches=False,
            source_url="",
            notes=f"No pricing URL configured for provider: {provider}",
            raw_response="",
        )

    prompt = f"""Fetch the pricing page at {url} and find the per-token pricing for model '{model}'.

I computed:
- Input price: ${computed_input_price:.12f} per token
- Output price: ${computed_output_price:.12f} per token

Please:
1. Extract the official input/output pricing for {model} (or the closest matching model name)
2. Convert to per-token (not per-million-tokens)
3. Compare with my computed values
4. Report if they match (within 1% tolerance)

Respond ONLY with valid JSON in this exact format:
{{
    "model": "{model}",
    "official_input_price_per_token": <float or null if not found>,
    "official_output_price_per_token": <float or null if not found>,
    "computed_input_matches": <bool>,
    "computed_output_matches": <bool>,
    "notes": "<any discrepancies or context>"
}}"""

    try:
        result = subprocess.run(
            [
                "codex",
                "exec",
                "-m",
                "gpt-5.2",
                "-c",
                f'model_reasoning_effort="{reasoning_effort}"',
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        raw_response = result.stdout.strip()

        # Try to extract JSON from response
        try:
            # Handle case where response has extra text before/after JSON
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw_response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            return PriceVerificationResult(
                provider=provider,
                model=model,
                computed_input_price=computed_input_price,
                computed_output_price=computed_output_price,
                official_input_price=data.get("official_input_price_per_token"),
                official_output_price=data.get("official_output_price_per_token"),
                input_matches=data.get("computed_input_matches", False),
                output_matches=data.get("computed_output_matches", False),
                source_url=url,
                notes=data.get("notes", ""),
                raw_response=raw_response,
            )

        except (json.JSONDecodeError, ValueError) as e:
            return PriceVerificationResult(
                provider=provider,
                model=model,
                computed_input_price=computed_input_price,
                computed_output_price=computed_output_price,
                official_input_price=None,
                official_output_price=None,
                input_matches=False,
                output_matches=False,
                source_url=url,
                notes=f"Failed to parse Codex response: {e}",
                raw_response=raw_response,
            )

    except subprocess.TimeoutExpired:
        return PriceVerificationResult(
            provider=provider,
            model=model,
            computed_input_price=computed_input_price,
            computed_output_price=computed_output_price,
            official_input_price=None,
            official_output_price=None,
            input_matches=False,
            output_matches=False,
            source_url=url,
            notes="Codex request timed out",
            raw_response="",
        )
    except FileNotFoundError:
        return PriceVerificationResult(
            provider=provider,
            model=model,
            computed_input_price=computed_input_price,
            computed_output_price=computed_output_price,
            official_input_price=None,
            official_output_price=None,
            input_matches=False,
            output_matches=False,
            source_url=url,
            notes="Codex CLI not found - please install codex SDK",
            raw_response="",
        )
    except Exception as e:
        return PriceVerificationResult(
            provider=provider,
            model=model,
            computed_input_price=computed_input_price,
            computed_output_price=computed_output_price,
            official_input_price=None,
            official_output_price=None,
            input_matches=False,
            output_matches=False,
            source_url=url,
            notes=f"Codex verification failed: {e}",
            raw_response="",
        )


def batch_verify_pricing(
    verifications: list[dict[str, Any]], reasoning_effort: str = "high"
) -> list[PriceVerificationResult]:
    """
    Verify multiple model prices in batch.

    Args:
        verifications: List of dicts with keys: provider, model,
                      computed_input_price, computed_output_price
        reasoning_effort: Codex reasoning effort

    Returns:
        List of PriceVerificationResult
    """
    results = []
    for v in verifications:
        result = verify_pricing_with_codex(
            provider=v["provider"],
            model=v["model"],
            computed_input_price=v["computed_input_price"],
            computed_output_price=v["computed_output_price"],
            reasoning_effort=reasoning_effort,
        )
        results.append(result)
    return results


def get_pricing_url(provider: str) -> str:
    """Get the pricing URL for a provider."""
    return PROVIDER_PRICING_URLS.get(provider, "")
