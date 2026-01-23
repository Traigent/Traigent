/**
 * Trial context management using AsyncLocalStorage.
 *
 * This mirrors Python's contextvars pattern used in traigent/config/context.py.
 * AsyncLocalStorage automatically propagates context through async call chains.
 */
import { AsyncLocalStorage, AsyncResource } from 'node:async_hooks';
import type { TrialConfig } from '../dtos/trial.js';

/**
 * Extended trial context that includes cancellation support.
 */
interface TrialContextData {
  config: TrialConfig;
  abortSignal?: AbortSignal;
}

/**
 * AsyncLocalStorage instance for trial context propagation.
 * This is the Node.js equivalent of Python's contextvars.
 */
const trialStorage = new AsyncLocalStorage<TrialContextData>();

/**
 * Error thrown when accessing trial context outside of a trial.
 */
export class TrialContextError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TrialContextError';
  }
}

/**
 * Error thrown when a trial is cancelled.
 */
export class TrialCancelledError extends Error {
  constructor(message: string = 'Trial was cancelled') {
    super(message);
    this.name = 'TrialCancelledError';
  }
}

/**
 * TrialContext provides methods for managing trial context during optimization.
 *
 * Usage:
 * ```typescript
 * await TrialContext.run(config, async () => {
 *   const cfg = TrialContext.getConfig();
 *   // ... use config
 * });
 * ```
 */
export const TrialContext = {
  /**
   * Run a function within a trial context.
   * The config will be available via getConfig() within the function and all async calls.
   *
   * @param config - Trial configuration from the orchestrator
   * @param fn - Async function to execute within the context
   * @param abortSignal - Optional AbortSignal for cancellation support
   * @returns Promise resolving to the function's return value
   */
  run<T>(config: TrialConfig, fn: () => T | Promise<T>, abortSignal?: AbortSignal): T | Promise<T> {
    const contextData: TrialContextData = { config, abortSignal };
    return trialStorage.run(contextData, fn);
  },

  /**
   * Get the current trial configuration.
   *
   * @throws {TrialContextError} If called outside of a trial context
   * @returns The current trial configuration
   */
  getConfig(): TrialConfig {
    const contextData = trialStorage.getStore();
    if (!contextData) {
      throw new TrialContextError(
        'TrialContext.getConfig() called outside of a trial. ' +
          'Ensure your code is running within TrialContext.run() or a Traigent optimization trial.'
      );
    }
    return contextData.config;
  },

  /**
   * Get the current trial configuration, or undefined if not in a trial.
   *
   * @returns The current trial configuration or undefined
   */
  getConfigOrUndefined(): TrialConfig | undefined {
    return trialStorage.getStore()?.config;
  },

  /**
   * Check if currently running within a trial context.
   *
   * @returns true if in a trial context, false otherwise
   */
  isInTrial(): boolean {
    return trialStorage.getStore() !== undefined;
  },

  /**
   * Get the current trial ID, or undefined if not in a trial.
   *
   * @returns The trial ID or undefined
   */
  getTrialId(): string | undefined {
    return trialStorage.getStore()?.config.trial_id;
  },

  /**
   * Get the current trial number, or undefined if not in a trial.
   *
   * @returns The trial number or undefined
   */
  getTrialNumber(): number | undefined {
    return trialStorage.getStore()?.config.trial_number;
  },

  /**
   * Check if the current trial has been cancelled.
   *
   * @returns true if the trial has been cancelled, false otherwise
   */
  isCancelled(): boolean {
    const contextData = trialStorage.getStore();
    return contextData?.abortSignal?.aborted ?? false;
  },

  /**
   * Check if cancelled and throw if so.
   * Use this for cooperative cancellation in long-running operations.
   *
   * @throws {TrialCancelledError} If the trial has been cancelled
   */
  checkCancellation(): void {
    if (this.isCancelled()) {
      throw new TrialCancelledError('Trial was cancelled');
    }
  },

  /**
   * Get the abort signal for the current trial, if any.
   *
   * @returns The AbortSignal or undefined
   */
  getAbortSignal(): AbortSignal | undefined {
    return trialStorage.getStore()?.abortSignal;
  },
};

/**
 * Convenience function to get the current trial's config parameters.
 * This is the JS equivalent of Python's traigent.get_trial_config().
 *
 * @throws {TrialContextError} If called outside of a trial context
 * @returns The config parameters object
 */
export function getTrialConfig(): Record<string, unknown> {
  return TrialContext.getConfig().config;
}

/**
 * Convenience function to get a specific config parameter.
 *
 * @param key - The parameter name
 * @param defaultValue - Default value if parameter is not set
 * @returns The parameter value or default
 */
export function getTrialParam<T>(key: string, defaultValue?: T): T | undefined {
  const config = TrialContext.getConfigOrUndefined();
  if (!config) {
    return defaultValue;
  }
  const value = config.config[key];
  return (value as T) ?? defaultValue;
}

/**
 * Wrap a callback function to preserve trial context.
 *
 * Use this for callback-based APIs that don't automatically preserve
 * AsyncLocalStorage context (e.g., some older libraries).
 *
 * @param fn - The callback function to wrap
 * @returns A wrapped function that preserves trial context
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function wrapCallback<T extends (...args: any[]) => any>(fn: T): T {
  const resource = new AsyncResource('TraigentCallback');
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return ((...args: any[]) => resource.runInAsyncScope(fn, null, ...args)) as T;
}

/**
 * Bind a function to the current trial context.
 * The returned function will always run in the context that was active
 * when bindContext was called.
 *
 * @param fn - The function to bind
 * @returns A bound function that preserves the current trial context
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function bindContext<T extends (...args: any[]) => any>(fn: T): T {
  const currentConfig = TrialContext.getConfigOrUndefined();
  const currentAbortSignal = TrialContext.getAbortSignal();
  if (!currentConfig) {
    return fn;
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return ((...args: any[]) =>
    TrialContext.run(currentConfig, () => fn(...args), currentAbortSignal)) as T;
}
