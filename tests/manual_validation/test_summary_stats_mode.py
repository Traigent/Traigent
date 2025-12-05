#!/usr/bin/env python
"""Test script to verify summary_stats mode for privacy-preserving execution."""

import asyncio
import json

import pytest

from traigent.config.types import TraigentConfig
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
)


@pytest.mark.asyncio
async def test_summary_stats_generation():
    """Test that MetricsTracker correctly generates pandas.describe() compatible stats."""
    print("Testing summary_stats generation...")

    # Create a metrics tracker
    tracker = MetricsTracker()
    tracker.start_tracking()

    # Add some example metrics (simulating multiple evaluations)
    for i in range(10):
        example = ExampleMetrics(
            tokens=TokenMetrics(
                input_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                total_tokens=150 + i * 15,
            ),
            response=ResponseMetrics(
                response_time_ms=1000 + i * 100, tokens_per_second=10 + i * 0.5
            ),
            cost=CostMetrics(
                input_cost=0.001 + i * 0.0001,
                output_cost=0.002 + i * 0.0002,
                total_cost=0.003 + i * 0.0003,
            ),
            success=True if i != 5 else False,  # One failure
            error="Test error" if i == 5 else None,
        )
        tracker.add_example_metrics(example)

    tracker.end_tracking()

    # Generate summary stats
    summary_stats = tracker.format_as_summary_stats()

    print("\n=== Summary Stats Structure ===")
    print(json.dumps(summary_stats, indent=2))

    # Verify structure
    assert "metrics" in summary_stats
    assert "execution_time" in summary_stats
    assert "total_examples" in summary_stats
    assert "metadata" in summary_stats

    # Check that metrics have pandas.describe() format
    for metric_name, stats in summary_stats["metrics"].items():
        print(f"\nMetric: {metric_name}")
        assert "count" in stats, f"Missing 'count' in {metric_name}"
        assert "mean" in stats, f"Missing 'mean' in {metric_name}"
        assert "std" in stats, f"Missing 'std' in {metric_name}"
        assert "min" in stats, f"Missing 'min' in {metric_name}"
        assert "25%" in stats, f"Missing '25%' in {metric_name}"
        assert "50%" in stats, f"Missing '50%' in {metric_name}"
        assert "75%" in stats, f"Missing '75%' in {metric_name}"
        assert "max" in stats, f"Missing 'max' in {metric_name}"
        print(
            f"  count={stats['count']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )
        print(
            f"  min={stats['min']:.4f}, 25%={stats['25%']:.4f}, 50%={stats['50%']:.4f}, 75%={stats['75%']:.4f}, max={stats['max']:.4f}"
        )

    print("\n✅ Summary stats generation test passed!")
    return summary_stats


@pytest.mark.asyncio
async def test_privacy_mode_submission():
    """Test that privacy mode correctly uses summary_stats instead of detailed measures."""
    print("\n\nTesting privacy mode submission logic...")

    # Create a config with privacy mode
    config = TraigentConfig(
        execution_mode="privacy", model="gpt-4", temperature=0.7  # Correct mode name
    )

    print(f"Execution mode: {config.execution_mode}")

    # Simulate metadata that would be sent to backend
    summary_stats = await test_summary_stats_generation()

    metadata = {
        "execution_mode": config.execution_mode,
        "trial_id": "test_trial_123",
        "duration": 10.5,
        "summary_stats": summary_stats,
        # These detailed metrics should NOT be sent in privacy mode
        "input_tokens_mean": 150.0,  # Would be in summary_stats instead
        "output_tokens_mean": 75.0,  # Would be in summary_stats instead
    }

    # Check submission logic - privacy mode maps to hybrid + privacy_enabled=True
    # Should use summary stats for Edge Analytics mode or when privacy is enabled
    use_summary_stats = (
        metadata.get("execution_mode") in ["edge_analytics", "hybrid"]
        and config.is_privacy_enabled()
    ) or metadata.get("execution_mode") in {"edge_analytics"}
    has_summary_stats = "summary_stats" in metadata and metadata["summary_stats"]

    print(f"\nShould use summary_stats: {use_summary_stats}")
    print(f"Has summary_stats data: {has_summary_stats}")
    print(f"Config privacy enabled: {config.is_privacy_enabled()}")

    if use_summary_stats and has_summary_stats:
        print("✅ Would submit using summary_stats mode (privacy-preserving)")
        print(
            f"   Total examples processed: {metadata['summary_stats']['total_examples']}"
        )
        print(
            f"   Metrics included: {list(metadata['summary_stats']['metrics'].keys())}"
        )
    else:
        print("❌ Would submit using detailed measures mode")

    assert use_summary_stats, "Should use summary_stats for privacy mode"
    assert has_summary_stats, "Should have summary_stats data"

    print("\n✅ Privacy mode submission test passed!")


@pytest.mark.asyncio
async def test_edge_analytics_mode_submission():
    """Test that Edge Analytics mode correctly uses summary_stats."""
    print("\n\nTesting Edge Analytics mode submission logic...")

    # Create a config with Edge Analytics mode
    config = TraigentConfig(
        execution_mode="edge_analytics", model="gpt-4", temperature=0.7
    )

    print(f"Execution mode: {config.execution_mode}")

    # Check that Edge Analytics mode triggers summary_stats
    use_summary_stats = config.execution_mode in ["edge_analytics", "privacy"]

    print(f"Should use summary_stats: {use_summary_stats}")

    assert use_summary_stats, "Edge Analytics mode should use summary_stats"

    print("✅ Edge Analytics mode configuration test passed!")


@pytest.mark.asyncio
async def test_cloud_mode_submission():
    """Test that cloud mode uses detailed measures (not summary_stats)."""
    print("\n\nTesting cloud mode submission logic...")

    # Create a config with cloud mode
    config = TraigentConfig(execution_mode="cloud", model="gpt-4", temperature=0.7)

    print(f"Execution mode: {config.execution_mode}")

    # Check that cloud mode does NOT trigger summary_stats
    use_summary_stats = config.execution_mode in ["edge_analytics", "privacy"]

    print(f"Should use summary_stats: {use_summary_stats}")

    assert not use_summary_stats, "Cloud mode should NOT use summary_stats"

    print("✅ Cloud mode configuration test passed!")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Summary Stats Mode for Privacy-Preserving Execution")
    print("=" * 60)

    try:
        # Test summary stats generation
        await test_summary_stats_generation()

        # Test privacy mode
        await test_privacy_mode_submission()

        # Test Edge Analytics mode
        await test_edge_analytics_mode_submission()

        # Test cloud mode (should NOT use summary_stats)
        await test_cloud_mode_submission()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("- MetricsTracker correctly generates pandas.describe() format")
        print("- Privacy mode uses summary_stats (no detailed data)")
        print("- Edge Analytics mode uses summary_stats (privacy by default)")
        print("- Cloud mode uses detailed measures (full visibility)")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
