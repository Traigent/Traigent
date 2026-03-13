/**
 * LangChain.js callback handler for Traigent optimization.
 *
 * This mirrors the Python SDK's TraigentHandler from traigent/integrations/langchain/handler.py.
 * It captures token usage, cost, and latency metrics from LangChain operations.
 *
 * @example
 * ```typescript
 * import { TraigentHandler } from '@traigent/sdk/langchain';
 * import { ChatOpenAI } from '@langchain/openai';
 *
 * const handler = new TraigentHandler();
 * const llm = new ChatOpenAI({ callbacks: [handler] });
 *
 * await llm.invoke("Hello!");
 *
 * const metrics = handler.toMeasuresDict();
 * // { langchain_total_cost: 0.001, langchain_input_tokens: 5, ... }
 * ```
 */
import type { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { Serialized } from '@langchain/core/load/serializable';
import type { LLMResult } from '@langchain/core/outputs';
import type { BaseMessage } from '@langchain/core/messages';
import type { MeasuresDict } from '../../dtos/measures.js';
import { prefixMeasures } from '../../dtos/measures.js';

/**
 * Token usage from a single LLM call.
 */
export interface LLMCallMetrics {
  /** Run ID for this call */
  runId: string;
  /** Model used */
  model: string;
  /** Input/prompt tokens */
  inputTokens: number;
  /** Output/completion tokens */
  outputTokens: number;
  /** Total tokens */
  totalTokens: number;
  /** Cost in USD */
  cost: number;
  /** Latency in milliseconds */
  latencyMs: number;
  /** Timestamp */
  timestamp: Date;
}

/**
 * Aggregated metrics from TraigentHandler.
 */
export interface TraigentHandlerMetrics {
  /** Total cost in USD */
  totalCost: number;
  /** Total latency in milliseconds */
  totalLatencyMs: number;
  /** Total input tokens */
  inputTokens: number;
  /** Total output tokens */
  outputTokens: number;
  /** Total tokens */
  totalTokens: number;
  /** Number of LLM calls */
  llmCallCount: number;
  /** Individual call metrics */
  calls: LLMCallMetrics[];
}

/**
 * Cost per 1M tokens for common models.
 * [inputCostPer1M, outputCostPer1M]
 */
const MODEL_COSTS: Record<string, [number, number]> = {
  'gpt-4o': [2.5, 10],
  'gpt-4o-mini': [0.15, 0.6],
  'gpt-4-turbo': [10, 30],
  'gpt-4': [30, 60],
  'gpt-3.5-turbo': [0.5, 1.5],
  'claude-3-opus': [15, 75],
  'claude-3-sonnet': [3, 15],
  'claude-3-haiku': [0.25, 1.25],
  'claude-3-5-sonnet': [3, 15],
};

/**
 * Estimate cost for a model based on token usage.
 * Sorts prefixes by length (longest first) to ensure specific models like
 * gpt-4o-mini match before generic prefixes like gpt-4o.
 */
function estimateCost(model: string, inputTokens: number, outputTokens: number): number {
  const modelLower = model.toLowerCase();

  // Sort by prefix length descending to match specific models first
  const sortedPrefixes = Object.keys(MODEL_COSTS).sort((a, b) => b.length - a.length);

  for (const prefix of sortedPrefixes) {
    if (modelLower.includes(prefix)) {
      const costs = MODEL_COSTS[prefix];
      if (costs) {
        const [inCost, outCost] = costs;
        return (inputTokens * inCost + outputTokens * outCost) / 1_000_000;
      }
    }
  }

  // Default fallback cost
  return (inputTokens * 1 + outputTokens * 3) / 1_000_000;
}

/**
 * LangChain callback handler for capturing Traigent optimization metrics.
 *
 * This handler tracks token usage, cost, and latency for all LLM calls
 * made during a trial execution.
 */
export class TraigentHandler implements Partial<BaseCallbackHandler> {
  readonly name = 'TraigentHandler';

  private readonly startTimes: Map<string, number> = new Map();
  private calls: LLMCallMetrics[] = [];
  private readonly metricPrefix: string;

  /**
   * Create a new TraigentHandler.
   *
   * @param options - Handler options
   * @param options.metricPrefix - Prefix for metric names (default: 'langchain_')
   */
  constructor(options: { metricPrefix?: string } = {}) {
    this.metricPrefix = options.metricPrefix ?? 'langchain_';
  }

  /**
   * Called at the start of an LLM call.
   */
  async handleLLMStart(_llm: Serialized, _prompts: string[], runId: string): Promise<void> {
    this.startTimes.set(runId, Date.now());
  }

  /**
   * Called at the start of a chat model call.
   */
  async handleChatModelStart(
    _llm: Serialized,
    _messages: BaseMessage[][],
    runId: string
  ): Promise<void> {
    this.startTimes.set(runId, Date.now());
  }

  /**
   * Called at the end of an LLM call.
   */
  async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
    const startTime = this.startTimes.get(runId);
    const latencyMs = startTime ? Date.now() - startTime : 0;
    this.startTimes.delete(runId);

    // Extract token usage from LLM output
    // Use bracket notation for index signature access (required for DTS generation)
    const usage = output.llmOutput?.['tokenUsage'] as
      | { promptTokens?: number; completionTokens?: number; totalTokens?: number }
      | undefined;

    const inputTokens = usage?.promptTokens ?? 0;
    const outputTokens = usage?.completionTokens ?? 0;
    const totalTokens = usage?.totalTokens ?? inputTokens + outputTokens;

    // Get model name (bracket notation for index signature)
    const model = (output.llmOutput?.['modelName'] as string) ?? 'unknown';

    // Calculate cost
    const cost = estimateCost(model, inputTokens, outputTokens);

    // Record metrics
    const metrics: LLMCallMetrics = {
      runId,
      model,
      inputTokens,
      outputTokens,
      totalTokens,
      cost,
      latencyMs,
      timestamp: new Date(),
    };

    this.calls.push(metrics);
  }

  /**
   * Called when an LLM call errors.
   */
  async handleLLMError(error: Error, runId: string): Promise<void> {
    // Clean up start time on error
    this.startTimes.delete(runId);
    // Optionally log the error
    console.error(`[TraigentHandler] LLM error in run ${runId}:`, error.message);
  }

  /**
   * Get aggregated metrics.
   */
  getMetrics(): TraigentHandlerMetrics {
    const totalCost = this.calls.reduce((sum, c) => sum + c.cost, 0);
    const totalLatencyMs = this.calls.reduce((sum, c) => sum + c.latencyMs, 0);
    const inputTokens = this.calls.reduce((sum, c) => sum + c.inputTokens, 0);
    const outputTokens = this.calls.reduce((sum, c) => sum + c.outputTokens, 0);
    const totalTokens = this.calls.reduce((sum, c) => sum + c.totalTokens, 0);

    return {
      totalCost,
      totalLatencyMs,
      inputTokens,
      outputTokens,
      totalTokens,
      llmCallCount: this.calls.length,
      calls: [...this.calls],
    };
  }

  /**
   * Convert metrics to MeasuresDict format with prefixed keys.
   *
   * @returns Record with keys like 'langchain_total_cost', 'langchain_input_tokens', etc.
   */
  toMeasuresDict(): MeasuresDict {
    const metrics = this.getMetrics();

    const raw: MeasuresDict = {
      total_cost: metrics.totalCost,
      total_latency_ms: metrics.totalLatencyMs,
      input_tokens: metrics.inputTokens,
      output_tokens: metrics.outputTokens,
      total_tokens: metrics.totalTokens,
      llm_call_count: metrics.llmCallCount,
    };

    return prefixMeasures(raw, this.metricPrefix);
  }

  /**
   * Reset all collected metrics.
   */
  reset(): void {
    this.startTimes.clear();
    this.calls = [];
  }

  /**
   * Get the number of LLM calls recorded.
   */
  get callCount(): number {
    return this.calls.length;
  }
}
