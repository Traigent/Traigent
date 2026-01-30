"""Data models for cost verification results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CostVerificationResult:
    """Result of a cost verification test.

    Attributes:
        provider: Provider name (openai, anthropic, groq, etc.)
        model: Model identifier used
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        sdk_computed_cost: Cost calculated by Traigent SDK
        expected_cost: Expected cost from provider pricing
        fe_logged_cost: Cost logged in Frontend (if captured)
        be_stored_cost: Cost stored in Backend (if available)
        price_source_url: URL of provider pricing page
        codex_verified: Whether Codex confirmed the pricing
        timestamp: When the test was run
        raw_response: Raw API response for debugging
        notes: Additional notes or discrepancy details
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    sdk_computed_cost: float
    expected_cost: float
    price_source_url: str
    # Per-token prices for Codex verification (required for accurate price checking)
    expected_input_price_per_token: float | None = None
    expected_output_price_per_token: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    fe_logged_cost: float | None = None
    be_stored_cost: float | None = None
    codex_verified: bool | None = None  # None = not checked, True/False = result
    raw_response: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tolerance: float = 0.001  # Default 0.1% tolerance, can be overridden per-provider

    @property
    def cost_matches(self) -> bool:
        """Check if SDK computed cost matches expected cost within tolerance."""
        if self.expected_cost == 0:
            return self.sdk_computed_cost == 0
        # Use absolute floor for tiny costs (< $0.0001)
        abs_floor = 0.00001  # $0.00001 absolute tolerance floor
        rel_diff = abs(self.sdk_computed_cost - self.expected_cost)
        return rel_diff < max(self.expected_cost * self.tolerance, abs_floor)

    @property
    def fe_matches(self) -> bool | None:
        """Check if FE logged cost matches SDK cost."""
        if self.fe_logged_cost is None:
            return None
        if self.sdk_computed_cost == 0:
            return self.fe_logged_cost == 0
        tolerance = 0.001
        return (
            abs(self.fe_logged_cost - self.sdk_computed_cost) / self.sdk_computed_cost
            < tolerance
        )

    @property
    def be_matches(self) -> bool | None:
        """Check if BE stored cost matches SDK cost."""
        if self.be_stored_cost is None:
            return None
        if self.sdk_computed_cost == 0:
            return self.be_stored_cost == 0
        tolerance = 0.001
        return (
            abs(self.be_stored_cost - self.sdk_computed_cost) / self.sdk_computed_cost
            < tolerance
        )

    @property
    def all_verified(self) -> bool:
        """Check if all available verifications passed."""
        checks = [self.cost_matches]
        if self.codex_verified is not None:
            checks.append(self.codex_verified)
        if self.fe_matches is not None:
            checks.append(self.fe_matches)
        if self.be_matches is not None:
            checks.append(self.be_matches)
        return all(checks)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "sdk_computed_cost": self.sdk_computed_cost,
            "expected_cost": self.expected_cost,
            "fe_logged_cost": self.fe_logged_cost,
            "be_stored_cost": self.be_stored_cost,
            "price_source_url": self.price_source_url,
            "codex_verified": self.codex_verified,
            "timestamp": self.timestamp.isoformat(),
            "cost_matches": self.cost_matches,
            "fe_matches": self.fe_matches,
            "be_matches": self.be_matches,
            "all_verified": self.all_verified,
            "notes": self.notes,
        }


@dataclass
class VerificationReport:
    """Complete verification report across all providers."""

    results: list[CostVerificationResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    @property
    def total_cost(self) -> float:
        """Total cost across all tests."""
        return sum(r.sdk_computed_cost for r in self.results)

    @property
    def all_passed(self) -> bool:
        """Check if all verifications passed."""
        return all(r.all_verified for r in self.results)

    @property
    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.all_verified),
            "failed": sum(1 for r in self.results if not r.all_verified),
            "total_cost": self.total_cost,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else None
            ),
        }

    def finalize(self) -> None:
        """Mark the report as complete."""
        self.end_time = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Cost Verification Report - {self.start_time.strftime('%Y-%m-%d')}",
            "",
            "## Summary",
            "",
            "| Provider | Model | SDK Cost | Expected | Codex | FE | BE | Status |",
            "|----------|-------|----------|----------|-------|----|----|--------|",
        ]

        for r in self.results:
            codex_status = (
                "✅" if r.codex_verified else "❌" if r.codex_verified is False else "—"
            )
            fe_status = "✅" if r.fe_matches else "❌" if r.fe_matches is False else "—"
            be_status = "✅" if r.be_matches else "❌" if r.be_matches is False else "—"
            overall = "✅" if r.all_verified else "❌"

            lines.append(
                f"| {r.provider} | {r.model} | ${r.sdk_computed_cost:.6f} | "
                f"${r.expected_cost:.6f} | {codex_status} | {fe_status} | "
                f"{be_status} | {overall} |"
            )

        lines.extend(
            [
                "",
                f"**Total Cost:** ${self.total_cost:.4f}",
                f"**All Verifications Passed:** {'✅' if self.all_passed else '❌'}",
                "",
                "## Detailed Results",
                "",
            ]
        )

        for r in self.results:
            lines.extend(
                [
                    f"### {r.provider.title()} - {r.model}",
                    f"- **Prompt tokens:** {r.prompt_tokens}",
                    f"- **Completion tokens:** {r.completion_tokens}",
                    f"- **SDK computed cost:** ${r.sdk_computed_cost:.6f}",
                    f"- **Expected cost:** ${r.expected_cost:.6f}",
                    f"- **Price source:** {r.price_source_url}",
                    f"- **Cost match:** {'✅' if r.cost_matches else '❌'}",
                ]
            )
            if r.codex_verified is not None:
                lines.append(
                    f"- **Codex verification:** {'✅' if r.codex_verified else '❌'}"
                )
            if r.notes:
                lines.append(f"- **Notes:** {r.notes}")
            lines.append("")

        return "\n".join(lines)
