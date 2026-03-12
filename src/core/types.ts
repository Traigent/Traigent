import type { TrialConfig } from '../dtos/trial.js';

/**
 * Core type definitions for the Traigent JS SDK.
 */

/**
 * Callback for reporting trial progress.
 */
export type ProgressCallback = (progress: ProgressInfo) => void;

/**
 * Progress information during trial execution.
 */
export interface ProgressInfo {
  /** Number of examples processed */
  processed: number;
  /** Total number of examples */
  total: number;
  /** Current example index */
  currentIndex: number;
  /** Elapsed time in milliseconds */
  elapsedMs: number;
}

/**
 * Result from executing a user's trial function.
 */
export interface TrialFunctionResult {
  /** Computed metrics from the trial */
  metrics: Record<string, number | null>;
  /** Optional output data */
  output?: unknown;
  /** Optional metadata to include in results */
  metadata?: Record<string, unknown>;
  /** Optional execution duration in seconds */
  duration?: number;
}

/**
 * User-defined trial function signature.
 */
export type TrialFunction = (
  config: TrialConfig,
) => Promise<TrialFunctionResult>;

/**
 * Configuration for the Traigent JS SDK.
 */
export interface TraigentSDKConfig {
  /** Enable debug logging */
  readonly debug?: boolean;
  /** Path to data file for loading examples */
  readonly dataPath?: string;
  /** Custom data loader function */
  readonly dataLoader?: (indices: number[]) => Promise<unknown[]>;
  /** Timeout for trial execution in milliseconds */
  readonly timeoutMs?: number;
}

/**
 * Token usage information from LLM calls.
 */
export interface TokenUsage {
  /** Number of input/prompt tokens */
  inputTokens: number;
  /** Number of output/completion tokens */
  outputTokens: number;
  /** Total tokens (input + output) */
  totalTokens: number;
}

/**
 * Cost breakdown for LLM calls.
 */
export interface CostBreakdown {
  /** Cost for input tokens in USD */
  inputCost: number;
  /** Cost for output tokens in USD */
  outputCost: number;
  /** Total cost in USD */
  totalCost: number;
}

/**
 * Combined metrics from LLM execution.
 */
export interface LLMMetrics {
  /** Token usage */
  usage: TokenUsage;
  /** Cost breakdown */
  cost: CostBreakdown;
  /** Latency in milliseconds */
  latencyMs: number;
  /** Model used */
  model: string;
}
