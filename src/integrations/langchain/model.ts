import { registerFrameworkTarget } from '../registry.js';
import {
  ensurePositiveDuration,
  getFrameworkOverrides,
  recordProviderUsage,
} from '../shared.js';

const TRAIGENT_WRAPPED = Symbol.for('traigent.langchain.wrapped');
const wrappedModelCache = new WeakMap<object, object>();

type BindableLangChainModel = {
  model?: string;
  modelName?: string;
  modelId?: string;
  bind?: (kwargs: Record<string, unknown>) => unknown;
  invoke?: (...args: unknown[]) => Promise<unknown>;
  stream?: (...args: unknown[]) => Promise<unknown>;
  batch?: (...args: unknown[]) => Promise<unknown>;
};

type WrappedLangChainModel = BindableLangChainModel & {
  [TRAIGENT_WRAPPED]?: true;
};

type TokenUsage = {
  inputTokens: number;
  outputTokens: number;
};

function getNumber(
  record: Record<string, unknown>,
  ...keys: string[]
): number | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }

  return undefined;
}

function getUsageFromSingleResult(result: unknown): TokenUsage {
  if (!result || typeof result !== 'object') {
    return { inputTokens: 0, outputTokens: 0 };
  }

  const record = result as Record<string, unknown>;
  const usageMetadata = record['usage_metadata'];
  if (usageMetadata && typeof usageMetadata === 'object') {
    const usage = usageMetadata as Record<string, unknown>;
    return {
      inputTokens: getNumber(usage, 'input_tokens', 'inputTokens') ?? 0,
      outputTokens: getNumber(usage, 'output_tokens', 'outputTokens') ?? 0,
    };
  }

  const responseMetadata = record['response_metadata'];
  if (responseMetadata && typeof responseMetadata === 'object') {
    const metadata = responseMetadata as Record<string, unknown>;
    const tokenUsage = metadata['tokenUsage'];
    if (tokenUsage && typeof tokenUsage === 'object') {
      const usage = tokenUsage as Record<string, unknown>;
      return {
        inputTokens: getNumber(usage, 'promptTokens', 'input_tokens', 'inputTokens') ?? 0,
        outputTokens:
          getNumber(usage, 'completionTokens', 'output_tokens', 'outputTokens') ?? 0,
      };
    }

    const usage = metadata['usage'];
    if (usage && typeof usage === 'object') {
      const usageRecord = usage as Record<string, unknown>;
      return {
        inputTokens:
          getNumber(usageRecord, 'input_tokens', 'promptTokens', 'inputTokens') ?? 0,
        outputTokens:
          getNumber(usageRecord, 'output_tokens', 'completionTokens', 'outputTokens') ?? 0,
      };
    }
  }

  const llmOutput = record['llmOutput'];
  if (llmOutput && typeof llmOutput === 'object') {
    const tokenUsage = (llmOutput as Record<string, unknown>)['tokenUsage'];
    if (tokenUsage && typeof tokenUsage === 'object') {
      const usage = tokenUsage as Record<string, unknown>;
      return {
        inputTokens: getNumber(usage, 'promptTokens', 'input_tokens', 'inputTokens') ?? 0,
        outputTokens:
          getNumber(usage, 'completionTokens', 'output_tokens', 'outputTokens') ?? 0,
      };
    }
  }

  return { inputTokens: 0, outputTokens: 0 };
}

function getUsageFromResult(result: unknown): TokenUsage {
  if (!Array.isArray(result)) {
    return getUsageFromSingleResult(result);
  }

  return result.reduce<TokenUsage>(
    (totals, entry) => {
      const usage = getUsageFromSingleResult(entry);
      totals.inputTokens += usage.inputTokens;
      totals.outputTokens += usage.outputTokens;
      return totals;
    },
    { inputTokens: 0, outputTokens: 0 },
  );
}

function getModelName(
  model: BindableLangChainModel,
  overrides: Record<string, unknown>,
): string {
  return String(
    overrides['model'] ?? model.model ?? model.modelName ?? model.modelId ?? 'unknown',
  );
}

function bindIfNeeded<T extends BindableLangChainModel>(model: T): T {
  const overrides = getFrameworkOverrides('langchain');
  if (Object.keys(overrides).length === 0 || typeof model.bind !== 'function') {
    return model;
  }

  return model.bind(overrides) as T;
}

export function withTraigentModel<T extends BindableLangChainModel>(model: T): T {
  const cached = wrappedModelCache.get(model as object);
  if (cached) {
    return cached as T;
  }

  if ((model as WrappedLangChainModel)[TRAIGENT_WRAPPED]) {
    return model;
  }

  registerFrameworkTarget('langchain');

  const wrapped = new Proxy(model, {
    get(target, property, receiver) {
      const originalValue = Reflect.get(target, property, receiver);
      if (
        property !== 'invoke' &&
        property !== 'stream' &&
        property !== 'batch'
      ) {
        return originalValue;
      }

      if (typeof originalValue !== 'function') {
        return originalValue;
      }

      return function wrappedLangChainMethod(this: unknown, ...args: unknown[]) {
        const activeModel = bindIfNeeded(target);
        const method = Reflect.get(activeModel, property, activeModel);
        if (typeof method !== 'function') {
          return originalValue.apply(this, args);
        }
        const overrides = getFrameworkOverrides('langchain');
        const startedAt = Date.now();

        return Promise.resolve(method.apply(activeModel, args)).then((result) => {
          const usage = getUsageFromResult(result);
          recordProviderUsage(
            getModelName(target, overrides),
            usage.inputTokens,
            usage.outputTokens,
            ensurePositiveDuration(Date.now() - startedAt),
          );
          return result;
        });
      };
    },
  }) as T;

  Object.defineProperty(wrapped, TRAIGENT_WRAPPED, {
    value: true,
    enumerable: false,
    configurable: false,
    writable: false,
  });

  wrappedModelCache.set(model as object, wrapped as object);

  return wrapped;
}
