from pathlib import Path

from paper_experiments.case_study_fever.run_case_study import evaluate_configuration


def test_evaluate_configuration_respects_max_examples(tmp_path: Path) -> None:
    # Use mock mode to avoid external API calls and ensure we only process the capped slice.
    result = evaluate_configuration(
        {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "evidence_selector": "dense",
            "consistency_checker": "consistency",
            "retriever_k": 3,
            "verdict_threshold": 0.6,
        },
        mock_mode=True,
        max_examples=2,
    )

    assert result["sample_count"] == 2
    assert len(result["per_example"]) == 2
