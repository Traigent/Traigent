import { createTraigentOpenAI } from './openai/index.js';
import { withTraigentModel } from './langchain/index.js';
import { withTraigent } from './vercel-ai/index.js';

type UnknownRecord = Record<string, unknown>;

function isObjectLike(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null;
}

function isOpenAIClientLike(value: unknown): boolean {
  if (!isObjectLike(value)) {
    return false;
  }

  const chat = value['chat'];
  const responses = value['responses'];
  const completions =
    chat && isObjectLike(chat) ? (chat['completions'] as unknown) : undefined;

  return Boolean(
    (completions &&
      isObjectLike(completions) &&
      typeof completions['create'] === 'function') ||
    (responses &&
      isObjectLike(responses) &&
      typeof responses['create'] === 'function')
  );
}

function isLangChainModelLike(value: unknown): boolean {
  if (!isObjectLike(value)) {
    return false;
  }

  const hasMethod =
    typeof value['invoke'] === 'function' ||
    typeof value['stream'] === 'function' ||
    typeof value['batch'] === 'function';
  const hasModelIdentity =
    typeof value['bind'] === 'function' ||
    typeof value['model'] === 'string' ||
    typeof value['modelName'] === 'string' ||
    typeof value['modelId'] === 'string';

  return hasMethod && hasModelIdentity;
}

function isVercelLanguageModelLike(value: unknown): boolean {
  if (!isObjectLike(value)) {
    return false;
  }

  return (
    typeof value['modelId'] === 'string' &&
    (typeof value['doGenerate'] === 'function' ||
      typeof value['doStream'] === 'function')
  );
}

export function autoWrapFrameworkTarget<T>(value: T): T {
  if (isOpenAIClientLike(value)) {
    return createTraigentOpenAI(value as Parameters<typeof createTraigentOpenAI>[0]) as T;
  }

  if (isLangChainModelLike(value)) {
    return withTraigentModel(value as Parameters<typeof withTraigentModel>[0]) as T;
  }

  if (isVercelLanguageModelLike(value)) {
    return withTraigent(value as Parameters<typeof withTraigent>[0]) as T;
  }

  return value;
}

export function autoWrapFrameworkTargets<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((entry) => autoWrapFrameworkTarget(entry)) as T;
  }

  if (!isObjectLike(value)) {
    return value;
  }

  if (
    isOpenAIClientLike(value) ||
    isLangChainModelLike(value) ||
    isVercelLanguageModelLike(value)
  ) {
    return autoWrapFrameworkTarget(value);
  }

  const wrappedEntries = Object.entries(value).map(([key, entry]) => [
    key,
    autoWrapFrameworkTarget(entry),
  ]);

  return Object.fromEntries(wrappedEntries) as T;
}
