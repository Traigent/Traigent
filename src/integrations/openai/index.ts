import { registerFrameworkTarget } from '../registry.js';
import {
  ensurePositiveDuration,
  getFrameworkOverrides,
  recordProviderUsage,
} from '../shared.js';

const TRAIGENT_WRAPPED = Symbol.for('traigent.openai.wrapped');

type OpenAICompletionUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

type OpenAIChatCompletionsClient = {
  create: (params: Record<string, unknown>, ...args: unknown[]) => Promise<any>;
};

type OpenAIResponsesClient = {
  create: (params: Record<string, unknown>, ...args: unknown[]) => Promise<any>;
};

type OpenAIClientLike = {
  chat?: {
    completions?: OpenAIChatCompletionsClient;
  };
  responses?: OpenAIResponsesClient;
};

type WrappedCreate = OpenAIChatCompletionsClient['create'] & {
  [TRAIGENT_WRAPPED]?: true;
};

function isWrapped(
  create: OpenAIChatCompletionsClient['create'] | OpenAIResponsesClient['create'],
): boolean {
  return Boolean((create as WrappedCreate)[TRAIGENT_WRAPPED]);
}

function markWrapped<
  T extends OpenAIChatCompletionsClient['create'] | OpenAIResponsesClient['create'],
>(create: T): T {
  Object.defineProperty(create, TRAIGENT_WRAPPED, {
    value: true,
    enumerable: false,
    configurable: false,
    writable: false,
  });
  return create;
}

function wrapCreate(
  create: OpenAIChatCompletionsClient['create'],
): OpenAIChatCompletionsClient['create'] {
  return markWrapped(async function wrappedCreate(
    this: unknown,
    params: Record<string, unknown>,
    ...args: unknown[]
  ) {
    const overrides = getFrameworkOverrides('openai');
    const startedAt = Date.now();
    const response = await create.call(
      this,
      {
        ...params,
        ...overrides,
      },
      ...args,
    );

    const usage = response?.usage as OpenAICompletionUsage | undefined;
    const model = String(
      response?.model ?? overrides['model'] ?? params['model'] ?? 'unknown',
    );
    const inputTokens = usage?.prompt_tokens ?? 0;
    const outputTokens = usage?.completion_tokens ?? 0;

    recordProviderUsage(
      model,
      inputTokens,
      outputTokens,
      ensurePositiveDuration(Date.now() - startedAt),
    );

    return response;
  });
}

function wrapResponsesCreate(
  create: OpenAIResponsesClient['create'],
): OpenAIResponsesClient['create'] {
  return markWrapped(async function wrappedResponsesCreate(
    this: unknown,
    params: Record<string, unknown>,
    ...args: unknown[]
  ) {
    const overrides = getFrameworkOverrides('openai');
    const startedAt = Date.now();
    const response = await create.call(
      this,
      {
        ...params,
        model: overrides['model'] ?? params['model'],
        temperature: overrides['temperature'] ?? params['temperature'],
        max_output_tokens: overrides['max_tokens'] ?? params['max_output_tokens'],
      },
      ...args,
    );

    const usage = response?.usage as
      | { input_tokens?: number; output_tokens?: number }
      | undefined;
    const model = String(
      response?.model ?? overrides['model'] ?? params['model'] ?? 'unknown',
    );

    recordProviderUsage(
      model,
      usage?.input_tokens ?? 0,
      usage?.output_tokens ?? 0,
      ensurePositiveDuration(Date.now() - startedAt),
    );

    return response;
  });
}

export function createTraigentOpenAI<T extends OpenAIClientLike>(client: T): T {
  registerFrameworkTarget('openai');

  if (client.chat?.completions?.create && !isWrapped(client.chat.completions.create)) {
    client.chat.completions.create = wrapCreate(client.chat.completions.create);
  }

  if (client.responses?.create && !isWrapped(client.responses.create)) {
    client.responses.create = wrapResponsesCreate(client.responses.create);
  }

  return client;
}
