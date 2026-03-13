import { createTraigentOpenAI } from './openai/index.js';
import { withTraigentModel } from './langchain/index.js';
import type { FrameworkAutoOverrideStatus, FrameworkTarget } from '../optimization/types.js';
import { describeFrameworkAutoOverride } from './registry.js';
import { withTraigent } from './vercel-ai/index.js';

type UnknownRecord = Record<string, unknown>;

export interface DiscoveredFrameworkTarget {
  path: string;
  target: FrameworkTarget;
}

export interface PrepareFrameworkTargetsOptions {
  autoOverrideFrameworks?: boolean;
  frameworkTargets?: readonly FrameworkTarget[];
}

export interface PreparedFrameworkTargets<T> {
  wrapped: T;
  discovered: readonly DiscoveredFrameworkTarget[];
  autoOverrideStatus: FrameworkAutoOverrideStatus;
}

function isObjectLike(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null;
}

function isPlainObject(value: unknown): value is UnknownRecord {
  if (!isObjectLike(value)) {
    return false;
  }

  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isOpenAIClientLike(value: unknown): boolean {
  if (!isObjectLike(value)) {
    return false;
  }

  const chat = value['chat'];
  const responses = value['responses'];
  const completions = chat && isObjectLike(chat) ? (chat['completions'] as unknown) : undefined;

  return Boolean(
    (completions && isObjectLike(completions) && typeof completions['create'] === 'function') ||
    (responses && isObjectLike(responses) && typeof responses['create'] === 'function')
  );
}

function isLangChainModelLike(value: unknown): boolean {
  if (!isObjectLike(value)) {
    return false;
  }

  // This is intentionally pragmatic rather than exact runtime branding.
  // The goal is low-friction wrapping for common LangChain-style models, and
  // the worst case is an extra Proxy wrapper on a compatible-looking object.
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
    (typeof value['doGenerate'] === 'function' || typeof value['doStream'] === 'function')
  );
}

function detectFrameworkTarget(value: unknown): FrameworkTarget | undefined {
  if (isOpenAIClientLike(value)) {
    return 'openai';
  }

  if (isLangChainModelLike(value)) {
    return 'langchain';
  }

  if (isVercelLanguageModelLike(value)) {
    return 'vercel-ai';
  }

  return undefined;
}

function createObjectClone(value: UnknownRecord): UnknownRecord {
  const prototype = Object.getPrototypeOf(value);
  return prototype === null ? Object.create(null) : {};
}

function formatChildPath(parent: string, key: string | number): string {
  if (typeof key === 'number') {
    return `${parent}[${key}]`;
  }

  return parent === '<root>' ? key : `${parent}.${key}`;
}

function discoverFrameworkTargetsInternal(
  value: unknown,
  path: string,
  discovered: DiscoveredFrameworkTarget[],
  seen: WeakSet<object>
): void {
  if (!isObjectLike(value)) {
    return;
  }

  const target = detectFrameworkTarget(value);
  if (target) {
    discovered.push({ path, target });
    seen.add(value);
    return;
  }

  if (seen.has(value)) {
    return;
  }
  seen.add(value);

  if (Array.isArray(value)) {
    value.forEach((entry, index) => {
      discoverFrameworkTargetsInternal(entry, formatChildPath(path, index), discovered, seen);
    });
    return;
  }

  if (!isPlainObject(value)) {
    return;
  }

  Object.entries(value).forEach(([key, entry]) => {
    discoverFrameworkTargetsInternal(entry, formatChildPath(path, key), discovered, seen);
  });
}

/**
 * Discover supported framework targets inside an explicitly provided value.
 *
 * This is bounded discovery:
 * - direct framework targets are detected
 * - nested arrays and plain-object graphs are traversed
 * - arbitrary module state or non-plain container instances are not scanned
 */
export function discoverFrameworkTargets(value: unknown): DiscoveredFrameworkTarget[] {
  const discovered: DiscoveredFrameworkTarget[] = [];
  discoverFrameworkTargetsInternal(value, '<root>', discovered, new WeakSet());
  return discovered;
}

function autoWrapFrameworkTargetsInternal<T>(value: T, seen: WeakMap<object, unknown>): T {
  if (!isObjectLike(value)) {
    return value;
  }

  const existing = seen.get(value);
  if (existing !== undefined) {
    return existing as T;
  }

  const target = detectFrameworkTarget(value);
  if (target) {
    const wrapped = autoWrapFrameworkTarget(value);
    seen.set(value, wrapped);
    return wrapped;
  }

  if (Array.isArray(value)) {
    const wrappedArray: unknown[] = [];
    seen.set(value, wrappedArray);
    value.forEach((entry) => {
      wrappedArray.push(autoWrapFrameworkTargetsInternal(entry, seen));
    });
    return wrappedArray as T;
  }

  if (!isPlainObject(value)) {
    return value;
  }

  const wrappedObject = createObjectClone(value);
  seen.set(value, wrappedObject);
  Object.entries(value).forEach(([key, entry]) => {
    wrappedObject[key] = autoWrapFrameworkTargetsInternal(entry, seen);
  });
  return wrappedObject as T;
}

/**
 * Wrap a supported framework target for seamless interception.
 *
 * Important identity note:
 * - OpenAI clients are wrapped in place and preserve referential equality
 * - LangChain and Vercel AI targets return wrapped proxy/middleware objects
 *
 * All wrapper paths are idempotent, so calling this repeatedly is safe.
 */
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

/**
 * Auto-wrap supported framework targets in an array, a plain object map, or a
 * single direct framework object.
 */
export function autoWrapFrameworkTargets<T>(value: T): T {
  return autoWrapFrameworkTargetsInternal(value, new WeakMap());
}

/**
 * Discover, wrap, and describe supported framework targets inside an explicitly
 * provided object graph.
 */
export function prepareFrameworkTargets<T>(
  value: T,
  options: PrepareFrameworkTargetsOptions = {}
): PreparedFrameworkTargets<T> {
  const wrapped = autoWrapFrameworkTargets(value);
  return {
    wrapped,
    discovered: discoverFrameworkTargets(value),
    autoOverrideStatus: describeFrameworkAutoOverride(
      options.frameworkTargets,
      options.autoOverrideFrameworks ?? true
    ),
  };
}
