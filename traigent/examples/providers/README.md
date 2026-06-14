# Per-provider quickstart examples

Each script here optimizes a tiny Q&A agent across models and temperatures
for one LLM provider, using the same `@traigent.optimize` pattern as the
bundled [`traigent.examples.quickstart`](../quickstart/). They are designed
to be **copy-paste starting points**.

All of them **run with no API keys by default** (mock mode intercepts the
LLM calls — no provider spend), and make **real** calls when you set
`TRAIGENT_MOCK_LLM=false` and provide the provider's credentials.

```bash
# keyless demo (mock):
python -m traigent.examples.providers.openai

# real run:
OPENAI_API_KEY=sk-... TRAIGENT_MOCK_LLM=false python -m traigent.examples.providers.openai
```

## Providers

| Run | Provider | Implementation | Credentials |
|-----|----------|----------------|-------------|
| `...providers.openai`      | OpenAI        | LangChain `ChatOpenAI`        | `OPENAI_API_KEY` |
| `...providers.anthropic`   | Anthropic     | LangChain `ChatAnthropic`     | `ANTHROPIC_API_KEY` |
| `...providers.openrouter`  | OpenRouter    | LangChain `ChatOpenAI` + base_url | `OPENROUTER_API_KEY` |
| `...providers.azure`       | Azure OpenAI  | LiteLLM                       | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` (model = `azure/<deployment>`) |
| `...providers.gemini`      | Gemini (AI Studio) | LiteLLM                  | `GEMINI_API_KEY` |
| `...providers.groq`        | Groq          | LiteLLM                       | `GROQ_API_KEY` |
| `...providers.aws`         | AWS Bedrock   | LiteLLM                       | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` |
| `...providers.gcp`         | GCP Vertex AI | LiteLLM                       | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, ADC (`GOOGLE_APPLICATION_CREDENTIALS`) |
| `...providers.litellm_any` | Any provider  | LiteLLM                       | the key(s) for whichever model strings you list |

## Why LangChain for some, LiteLLM for others

We use **LangChain** for OpenAI / Anthropic / OpenRouter — they are the
idiomatic LangChain classes (`ChatOpenAI`, `ChatAnthropic`), they ship in
`traigent[recommended]`, and the SDK mock-intercepts them so the keyless
demo works.

We use **LiteLLM** for Azure / Gemini / Groq / Bedrock / Vertex / the
generic example. LiteLLM is a **core Traigent dependency** (always
installed), it covers every provider through one `litellm.completion`
call, and the SDK mock-intercepts it universally — so these run keyless
with **no extra packages**. (LangChain's `AzureChatOpenAI` and
`ChatGoogleGenerativeAI` are not mock-intercepted by the SDK, and
`langchain-aws` / `langchain-google-vertexai` / `langchain-groq` are not
bundled, so LiteLLM is both the simplest and the most robust path there.)

## Dependencies

`pip install "traigent[recommended]"` installs everything these examples
need. On a bare `pip install traigent`, the LangChain examples will detect
their missing package and offer to install it (interactive prompt only —
in CI / non-interactive shells, or with `TRAIGENT_EXAMPLES_NO_INSTALL=1`,
they just print the `pip install` command and exit). The LiteLLM examples
need nothing extra.

## Single source of truth

[`manifest.json`](./manifest.json) is the authoritative provider table
(implementation, import line, models, env vars, pip package). These
scripts, the SDK parity test (`tests/unit/examples/test_provider_examples.py`),
and the portal Quick Start page all derive from it.
