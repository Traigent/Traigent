import asyncio

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


class DummyRawResp:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
        class Usage:
            def __init__(self, p, c, t):
                self.prompt_tokens = p
                self.completion_tokens = c
                self.total_tokens = t

        self.usage = Usage(prompt_tokens, completion_tokens, total_tokens)


def _make_dataset(n=2):
    examples = []
    for i in range(n):
        examples.append(
            EvaluationExample(
                input_data={"text": f"Q{i}"},
                expected_output=None,
                metadata={"example_id": f"example_{i}"},
            )
        )
    return Dataset(examples=examples)


async def _run_eval(max_workers=2):
    def fn(text: str, **cfg):
        return {"text": "ok", "raw_response": DummyRawResp(5, 3, 8)}

    ev = LocalEvaluator(
        metrics=["accuracy"], timeout=5.0, max_workers=max_workers, detailed=True
    )
    ds = _make_dataset(2)
    res = await ev.evaluate(fn, config={}, dataset=ds)
    return res


def test_local_evaluator_parallel_raw_response_tokens():
    res = asyncio.run(_run_eval(max_workers=2))
    assert res.example_results
    # Each ExampleResult has metrics populated; token extraction happened via raw_response
    assert all(isinstance(er.metrics, dict) for er in res.example_results)
