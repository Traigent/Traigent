from traigent.evaluators.metrics_tracker import extract_llm_metrics


class DummyLCMessage:
    def __init__(self, usage_metadata=None, response_metadata=None, llm_output=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata
        self.llm_output = llm_output or {}


class DummyOpenAIResp:
    class Usage:
        def __init__(self, prompt_tokens, completion_tokens, total_tokens):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens

    def __init__(self, prompt, completion, total):
        self.usage = DummyOpenAIResp.Usage(prompt, completion, total)


class DummyAnthropicResp:
    def __init__(self, input_tokens, output_tokens, total_tokens=None):
        self.usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (
                total_tokens
                if total_tokens is not None
                else input_tokens + output_tokens
            ),
        }
        self.model = "claude-3-haiku-20240307"


def test_langchain_usage_metadata_tokens():
    msg = DummyLCMessage(
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    m = extract_llm_metrics(msg, model_name="gpt-4o-mini")
    assert m.tokens.input_tokens == 10
    assert m.tokens.output_tokens == 5
    assert m.tokens.total_tokens == 15


def test_langchain_response_metadata_token_usage():
    msg = DummyLCMessage(
        response_metadata={
            "token_usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            }
        }
    )
    m = extract_llm_metrics(msg, model_name="gpt-4o-mini")
    assert m.tokens.input_tokens == 7
    assert m.tokens.output_tokens == 3
    assert m.tokens.total_tokens == 10


def test_openai_usage_handler():
    resp = DummyOpenAIResp(prompt=12, completion=8, total=20)
    m = extract_llm_metrics(resp, model_name="gpt-4o-mini")
    assert m.tokens.input_tokens == 12
    assert m.tokens.output_tokens == 8
    assert m.tokens.total_tokens == 20


def test_anthropic_usage_handler():
    resp = DummyAnthropicResp(input_tokens=9, output_tokens=6)
    m = extract_llm_metrics(resp, model_name=resp.model)
    assert m.tokens.input_tokens == 9
    assert m.tokens.output_tokens == 6
    assert m.tokens.total_tokens == 15
