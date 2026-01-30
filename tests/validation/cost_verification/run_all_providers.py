#!/usr/bin/env python3
"""Main entry point for cost verification across all providers.

This script:
1. Runs cost verification tests for specified providers
2. Optionally verifies pricing against official sources via Codex
3. Optionally captures FE and BE evidence
4. Generates a comprehensive verification report

Usage:
    # Run basic validation (no external deps)
    python tests/validation/cost_verification/run_all_providers.py --no-codex

    # Run with Codex verification
    python tests/validation/cost_verification/run_all_providers.py

    # Run specific providers
    python tests/validation/cost_verification/run_all_providers.py --providers openai anthropic

    # Run with FE/BE integration
    python tests/validation/cost_verification/run_all_providers.py --with-fe --with-be
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
# From run_all_providers.py: cost_verification -> validation -> tests -> project_root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.validation.cost_verification.models import (
    CostVerificationResult,
    VerificationReport,
)
from tests.validation.cost_verification.provider_tests.test_anthropic_sdk import (
    run_anthropic_verification,
)
from tests.validation.cost_verification.provider_tests.test_groq import (
    run_groq_verification,
)
from tests.validation.cost_verification.provider_tests.test_langchain import (
    run_langchain_verification,
)
from tests.validation.cost_verification.provider_tests.test_litellm import (
    run_litellm_verification,
)

# Import provider test runners
from tests.validation.cost_verification.provider_tests.test_openai_sdk import (
    run_openai_verification,
)
from tests.validation.cost_verification.provider_tests.test_openrouter import (
    run_openrouter_verification,
)
from tests.validation.cost_verification.verification.be_evidence import (
    BECostRecord,
    check_be_available,
    fetch_be_cost_records,
)
from tests.validation.cost_verification.verification.codex_price_checker import (
    verify_pricing_with_codex,
)
from tests.validation.cost_verification.verification.fe_snapshot import (
    FECostSnapshot,
    capture_fe_cost_data,
    check_fe_available,
)

# All supported providers
ALL_PROVIDERS = ["openai", "anthropic", "langchain", "litellm", "groq", "openrouter"]

# Provider to test runner mapping
PROVIDER_RUNNERS = {
    "openai": run_openai_verification,
    "anthropic": run_anthropic_verification,
    "langchain": run_langchain_verification,
    "litellm": run_litellm_verification,
    "groq": run_groq_verification,
    "openrouter": run_openrouter_verification,
}


def run_provider_tests(
    providers: list[str] | None = None,
) -> list[CostVerificationResult]:
    """Run cost verification tests for specified providers.

    Args:
        providers: List of provider names, or None for all

    Returns:
        List of verification results
    """
    providers_to_run = providers or ALL_PROVIDERS
    results = []

    for provider in providers_to_run:
        runner = PROVIDER_RUNNERS.get(provider)
        if not runner:
            print(f"Warning: No test runner for provider '{provider}'")
            continue

        print(f"\n{'='*60}")
        print(f"Running {provider.upper()} verification tests...")
        print("=" * 60)

        try:
            provider_results = runner()
            results.extend(provider_results)
            print(f"✓ {provider}: {len(provider_results)} test(s) completed")
        except Exception as e:
            print(f"✗ {provider}: Failed with error: {e}")

    return results


def verify_with_codex(
    results: list[CostVerificationResult], reasoning_effort: str = "high"
) -> None:
    """Verify results against official pricing using Codex.

    Args:
        results: List of verification results to verify
        reasoning_effort: Codex reasoning effort level
    """
    print(f"\n{'='*60}")
    print("Verifying pricing with Codex...")
    print("=" * 60)

    for result in results:
        if result.prompt_tokens == 0:
            continue

        # Use actual per-token prices if available, otherwise skip Codex verification
        input_price = result.expected_input_price_per_token
        output_price = result.expected_output_price_per_token

        if input_price is None or output_price is None:
            print(
                f"  Skipping {result.provider}/{result.model} "
                "(no per-token prices available)"
            )
            continue

        print(f"  Checking {result.provider}/{result.model}...")

        verification = verify_pricing_with_codex(
            provider=result.provider,
            model=result.model,
            computed_input_price=input_price,
            computed_output_price=output_price,
            reasoning_effort=reasoning_effort,
        )

        result.codex_verified = verification.verified
        if verification.notes:
            result.notes = (
                result.notes + " | " if result.notes else ""
            ) + verification.notes

        status = "✓" if verification.verified else "✗"
        print(f"    {status} Codex verification: {verification.notes or 'OK'}")


def capture_fe_evidence(
    results: list[CostVerificationResult],
    fe_base_url: str,
    experiment_id: str | None = None,
    run_id: str | None = None,
) -> list[FECostSnapshot]:
    """Capture FE evidence for verification results.

    Args:
        results: List of verification results
        fe_base_url: Frontend base URL
        experiment_id: Optional experiment ID to query
        run_id: Optional run ID to query

    Returns:
        List of FE snapshots
    """
    if not check_fe_available(fe_base_url):
        print(f"Warning: FE not available at {fe_base_url}")
        return []

    if not experiment_id or not run_id:
        print("Warning: experiment_id and run_id required for FE capture")
        return []

    print(f"\n{'='*60}")
    print("Capturing FE evidence...")
    print("=" * 60)

    snapshot = capture_fe_cost_data(experiment_id, run_id, fe_base_url)

    # Update results with FE costs
    for result in results:
        for trial in snapshot.trials:
            if trial.get("model") == result.model:
                result.fe_logged_cost = trial.get("cost_usd", 0)
                break

    print(f"  ✓ Captured FE snapshot: ${snapshot.total_cost:.6f} total")
    return [snapshot]


def capture_be_evidence(
    results: list[CostVerificationResult],
    be_base_url: str,
    auth_token: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
) -> list[BECostRecord]:
    """Capture BE evidence for verification results.

    Args:
        results: List of verification results
        be_base_url: Backend base URL
        auth_token: Optional auth token
        experiment_id: Optional experiment ID to query
        run_id: Optional run ID to query

    Returns:
        List of BE records
    """
    if not check_be_available(be_base_url):
        print(f"Warning: BE not available at {be_base_url}")
        return []

    if not experiment_id or not run_id:
        print("Warning: experiment_id and run_id required for BE capture")
        return []

    print(f"\n{'='*60}")
    print("Capturing BE evidence...")
    print("=" * 60)

    record = fetch_be_cost_records(experiment_id, run_id, be_base_url, auth_token)

    # Update results with BE costs
    for result in results:
        for config in record.configuration_runs:
            if config.get("model") == result.model:
                result.be_stored_cost = config.get("cost", 0)
                break

    print(f"  ✓ Captured BE record: ${record.total_cost:.6f} total")
    return [record]


def generate_report(
    results: list[CostVerificationResult],
    output_dir: str,
    fe_snapshots: list[FECostSnapshot] | None = None,
    be_records: list[BECostRecord] | None = None,
) -> str:
    """Generate verification report.

    Args:
        results: List of verification results
        output_dir: Output directory for reports
        fe_snapshots: Optional FE snapshots
        be_records: Optional BE records

    Returns:
        Path to the generated report
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Create verification report
    report = VerificationReport(results=results)
    report.finalize()

    # Write markdown report
    md_path = Path(output_dir) / f"verification_{timestamp}.md"
    md_content = report.to_markdown()

    # Add FE section if available
    if fe_snapshots:
        md_content += "\n\n## FE Snapshots\n\n"
        for snap in fe_snapshots:
            md_content += f"### Experiment {snap.experiment_id} / Run {snap.run_id}\n"
            md_content += f"Total Cost: ${snap.total_cost:.6f}\n\n"
            if snap.trials:
                md_content += "| Trial | Model | Cost |\n|-------|-------|------|\n"
                for t in snap.trials:
                    md_content += f"| {t.get('trial_id', 'N/A')[:8]}... | {t.get('model', 'N/A')} | ${t.get('cost_usd', 0):.6f} |\n"
            md_content += "\n"

    # Add BE section if available
    if be_records:
        md_content += "\n\n## BE Evidence\n\n"
        for rec in be_records:
            md_content += f"### Experiment {rec.experiment_id} / Run {rec.run_id}\n"
            md_content += f"Total Cost: ${rec.total_cost:.6f}\n\n"
            if rec.configuration_runs:
                md_content += (
                    "| Config ID | Model | Cost |\n|-----------|-------|------|\n"
                )
                for c in rec.configuration_runs:
                    md_content += f"| {c.get('id', 'N/A')[:8]}... | {c.get('model', 'N/A')} | ${c.get('cost', 0):.6f} |\n"
            md_content += "\n"

    md_path.write_text(md_content)

    # Write JSON report
    json_path = Path(output_dir) / f"verification_{timestamp}.json"
    json_data = report.to_dict()
    if fe_snapshots:
        json_data["fe_snapshots"] = [s.to_dict() for s in fe_snapshots]
    if be_records:
        json_data["be_records"] = [r.to_dict() for r in be_records]
    json_path.write_text(json.dumps(json_data, indent=2))

    return str(md_path)


def run_cost_verification(
    providers: list[str] | None = None,
    verify_with_codex_flag: bool = True,
    capture_fe: bool = False,
    capture_be: bool = False,
    output_dir: str = "tests/validation/cost_verification/reports",
    fe_base_url: str = "http://localhost:3000",
    be_base_url: str = "http://localhost:8000",
    auth_token: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    codex_reasoning: str = "high",
) -> VerificationReport:
    """Run complete cost verification.

    Args:
        providers: List of providers to test, or None for all
        verify_with_codex_flag: Use Codex to verify against pricing URLs
        capture_fe: Capture FE snapshots (requires running FE)
        capture_be: Capture BE evidence (requires running BE)
        output_dir: Directory for reports
        fe_base_url: Frontend base URL
        be_base_url: Backend base URL
        auth_token: Auth token for BE API
        experiment_id: Experiment ID for FE/BE queries
        run_id: Run ID for FE/BE queries
        codex_reasoning: Codex reasoning effort level

    Returns:
        VerificationReport with all results
    """
    print("=" * 70)
    print("TRAIGENT COST VERIFICATION")
    print("=" * 70)
    print(f"Providers: {providers or 'ALL'}")
    print(f"Codex verification: {verify_with_codex_flag}")
    print(f"Capture FE: {capture_fe}")
    print(f"Capture BE: {capture_be}")
    print("=" * 70)

    # Run provider tests
    results = run_provider_tests(providers)

    if not results:
        print(
            "\n⚠ No test results generated. Check API keys and provider availability."
        )
        return VerificationReport()

    # Verify with Codex
    if verify_with_codex_flag:
        verify_with_codex(results, codex_reasoning)

    # Capture FE evidence
    fe_snapshots = []
    if capture_fe:
        fe_snapshots = capture_fe_evidence(results, fe_base_url, experiment_id, run_id)

    # Capture BE evidence
    be_records = []
    if capture_be:
        be_records = capture_be_evidence(
            results, be_base_url, auth_token, experiment_id, run_id
        )

    # Generate report
    report_path = generate_report(results, output_dir, fe_snapshots, be_records)

    # Print summary
    report = VerificationReport(results=results)
    report.finalize()

    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    summary = report.summary
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total cost: ${report.total_cost:.6f}")
    print(f"\n{'✅ ALL PASSED' if report.all_passed else '❌ SOME FAILED'}")
    print(f"\nReport saved to: {report_path}")

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run cost verification across LLM providers"
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=ALL_PROVIDERS,
        default=None,
        help="Providers to test (default: all)",
    )
    parser.add_argument(
        "--no-codex",
        action="store_true",
        help="Skip Codex verification",
    )
    parser.add_argument(
        "--with-fe",
        action="store_true",
        help="Capture FE snapshots",
    )
    parser.add_argument(
        "--with-be",
        action="store_true",
        help="Capture BE evidence",
    )
    parser.add_argument(
        "--fe-url",
        default="http://localhost:3000",
        help="Frontend base URL",
    )
    parser.add_argument(
        "--be-url",
        default="http://localhost:8000",
        help="Backend base URL",
    )
    parser.add_argument(
        "--experiment-id",
        help="Experiment ID for FE/BE queries",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID for FE/BE queries",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/validation/cost_verification/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--codex-reasoning",
        choices=["high", "xhigh"],
        default="high",
        help="Codex reasoning effort level",
    )

    args = parser.parse_args()

    # Get auth token from environment
    auth_token = os.environ.get("TRAIGENT_API_TOKEN")

    report = run_cost_verification(
        providers=args.providers,
        verify_with_codex_flag=not args.no_codex,
        capture_fe=args.with_fe,
        capture_be=args.with_be,
        output_dir=args.output_dir,
        fe_base_url=args.fe_url,
        be_base_url=args.be_url,
        auth_token=auth_token,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        codex_reasoning=args.codex_reasoning,
    )

    # Exit with error code if any tests failed
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
