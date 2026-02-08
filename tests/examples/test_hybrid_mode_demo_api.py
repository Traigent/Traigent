"""Regression tests for the hybrid mode demo Flask API."""

from examples.hybrid_mode_demo.app import app


def test_evaluate_accepts_null_target_and_id_only_payloads():
    """Evaluate endpoint should handle null target and ID-only items safely."""
    app.config.update(TESTING=True)

    with app.test_client() as client:
        response = client.post(
            "/traigent/v1/evaluate",
            json={
                "request_id": "req-1",
                "capability_id": "demo_agent",
                "evaluations": [
                    {
                        "input_id": "ex_1",
                        "output": {"response": "hello world"},
                        "target": None,
                    },
                    {
                        "input_id": "ex_2",
                        "output_id": "out_ex_2",
                        "target_id": "target_ex_2",
                    },
                ],
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "completed"
    assert len(payload["results"]) == 2
    assert payload["aggregate_metrics"]["accuracy"]["n"] == 2
