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
| `...providers.nous`        | Nous Portal (Hermes) | LangChain `ChatOpenAI` + base_url | `NOUS_API_KEY` (OAuth — see note below) |
| `...providers.azure`       | Azure OpenAI  | LiteLLM                       | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` (model = `azure/<deployment>`) |
| `...providers.gemini`      | Gemini (AI Studio) | LiteLLM                  | `GEMINI_API_KEY` |
| `...providers.groq`        | Groq          | LiteLLM                       | `GROQ_API_KEY` |
| `...providers.aws`         | AWS Bedrock   | LiteLLM                       | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` |
| `...providers.gcp`         | GCP Vertex AI | LiteLLM                       | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, ADC (`GOOGLE_APPLICATION_CREDENTIALS`) |
| `...providers.litellm_any` | Any provider  | LiteLLM                       | the key(s) for whichever model strings you list |

## Nous Portal uses OAuth, not a static API key

Every other provider here authenticates with a **static API key** read straight
from the environment. **Nous Portal is different:** it is OpenAI-compatible for
the *calls*, but its credential is a **short-lived JWT minted from a long-lived
refresh token** (OAuth), so the example calls
`traigent.integrations.llms.nous_auth.get_nous_api_key()` instead of reading a
key from `os.environ` directly. That helper resolves a *current* token, in
order:

1. `NOUS_API_KEY` — a pre-minted JWT, returned as-is (no auto-refresh; it will
   expire). This is also what makes the keyless mock demo run offline.
2. `NOUS_REFRESH_TOKEN` / `NOUS_PORTAL_REFRESH_TOKEN` — a refresh token,
   exchanged for a JWT and cached in-process (auto-refreshed before expiry).
3. `~/.hermes/auth.json` (override `TRAIGENT_NOUS_AUTH_FILE`) — the Hermes CLI's
   login file; the refresh token is read out of it.

Every failure (no credentials, malformed `auth.json`, a failed refresh) raises
`NousAuthError` naming the source — the helper never serves a stale or mock
token. **Cost objectives:** the SDK ships **no** Nous prices (nothing
fabricated), so cost tracking raises `UnknownModelError` until you set
`TRAIGENT_CUSTOM_MODEL_PRICING_JSON` / `_FILE` with your own per-token rates.

> The Nous token-endpoint URL, the `~/.hermes/auth.json` schema, and the exact
> portal model-ID strings are env-overridable placeholders pending Phase-0
> confirmation against real `hermes-agent` output — see the `OWNER:` comments in
> `traigent/integrations/llms/nous_auth.py` and `config/models.yaml`.

## Why LangChain for some, LiteLLM for others

We use **LangChain** for OpenAI / Anthropic / OpenRouter / Nous — they are the
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
