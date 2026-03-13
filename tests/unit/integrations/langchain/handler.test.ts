/**
 * Unit tests for LangChain TraigentHandler.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { TraigentHandler } from '../../../../src/integrations/langchain/handler.js';
import type { LLMResult } from '@langchain/core/outputs';
import type { Serialized } from '@langchain/core/load/serializable';
import type { BaseMessage } from '@langchain/core/messages';

// Mock LLM output factory
const createMockLLMResult = (
  options: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
    modelName?: string;
  } = {}
): LLMResult => ({
  generations: [[{ text: 'Response', generationInfo: {} }]],
  llmOutput: {
    tokenUsage: {
      promptTokens: options.promptTokens ?? 10,
      completionTokens: options.completionTokens ?? 20,
      totalTokens: options.totalTokens ?? 30,
    },
    modelName: options.modelName ?? 'gpt-4o-mini',
  },
});

// Mock serialized LLM
const mockSerializedLLM: Serialized = {
  lc: 1,
  type: 'constructor',
  id: ['langchain', 'llms', 'openai', 'ChatOpenAI'],
  kwargs: {},
};

describe('TraigentHandler', () => {
  let handler: TraigentHandler;

  beforeEach(() => {
    handler = new TraigentHandler();
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  describe('constructor', () => {
    it('should create handler with default prefix', () => {
      const h = new TraigentHandler();
      const metrics = h.toMeasuresDict();

      // Check that keys use default prefix
      expect('langchain_total_cost' in metrics).toBe(true);
    });

    it('should create handler with custom prefix', () => {
      const h = new TraigentHandler({ metricPrefix: 'custom_' });
      const metrics = h.toMeasuresDict();

      expect('custom_total_cost' in metrics).toBe(true);
      expect('langchain_total_cost' in metrics).toBe(false);
    });
  });

  describe('handleLLMStart', () => {
    it('should record start time for run', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');

      // Verify start time was recorded by checking that latency works
      await handler.handleLLMEnd(createMockLLMResult(), 'run-1');
      const metrics = handler.getMetrics();

      expect(metrics.llmCallCount).toBe(1);
      expect(metrics.totalLatencyMs).toBeGreaterThanOrEqual(0);
    });
  });

  describe('handleChatModelStart', () => {
    it('should record start time for chat model run', async () => {
      const messages: BaseMessage[][] = [[{ content: 'Hello' }]] as BaseMessage[][];
      await handler.handleChatModelStart(mockSerializedLLM, messages, 'chat-run-1');

      await handler.handleLLMEnd(createMockLLMResult(), 'chat-run-1');
      const metrics = handler.getMetrics();

      expect(metrics.llmCallCount).toBe(1);
    });
  });

  describe('handleLLMEnd', () => {
    it('should record metrics from LLM output', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 100,
          completionTokens: 200,
          totalTokens: 300,
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();

      expect(metrics.inputTokens).toBe(100);
      expect(metrics.outputTokens).toBe(200);
      expect(metrics.totalTokens).toBe(300);
    });

    it('should calculate totalTokens if not provided', async () => {
      const result: LLMResult = {
        generations: [[{ text: 'Response', generationInfo: {} }]],
        llmOutput: {
          tokenUsage: {
            promptTokens: 50,
            completionTokens: 100,
            // totalTokens not provided
          },
          modelName: 'gpt-4o-mini',
        },
      };

      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(result, 'run-1');

      const metrics = handler.getMetrics();
      expect(metrics.totalTokens).toBe(150);
    });

    it('should aggregate metrics across multiple calls', async () => {
      // First call
      await handler.handleLLMStart(mockSerializedLLM, ['prompt 1'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({ promptTokens: 10, completionTokens: 20 }),
        'run-1'
      );

      // Second call
      await handler.handleLLMStart(mockSerializedLLM, ['prompt 2'], 'run-2');
      await handler.handleLLMEnd(
        createMockLLMResult({ promptTokens: 15, completionTokens: 25 }),
        'run-2'
      );

      const metrics = handler.getMetrics();

      expect(metrics.llmCallCount).toBe(2);
      expect(metrics.inputTokens).toBe(25);
      expect(metrics.outputTokens).toBe(45);
    });

    it('should handle missing tokenUsage gracefully', async () => {
      const result: LLMResult = {
        generations: [[{ text: 'Response', generationInfo: {} }]],
        llmOutput: {},
      };

      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(result, 'run-1');

      const metrics = handler.getMetrics();
      expect(metrics.inputTokens).toBe(0);
      expect(metrics.outputTokens).toBe(0);
      expect(metrics.totalTokens).toBe(0);
    });

    it('should handle missing llmOutput gracefully', async () => {
      const result: LLMResult = {
        generations: [[{ text: 'Response', generationInfo: {} }]],
      };

      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(result, 'run-1');

      const metrics = handler.getMetrics();
      expect(metrics.inputTokens).toBe(0);
      expect(metrics.totalCost).toBeGreaterThanOrEqual(0);
    });

    it('should calculate latency without start time', async () => {
      // Don't call handleLLMStart, just handleLLMEnd
      await handler.handleLLMEnd(createMockLLMResult(), 'orphan-run');

      const metrics = handler.getMetrics();
      expect(metrics.llmCallCount).toBe(1);
      expect(metrics.totalLatencyMs).toBe(0);
    });
  });

  describe('handleLLMError', () => {
    it('should clean up start time on error', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMError(new Error('LLM failed'), 'run-1');

      // If we now end with the same run ID, latency should be 0 (start time was cleaned)
      await handler.handleLLMEnd(createMockLLMResult(), 'run-1');
      const metrics = handler.getMetrics();

      expect(metrics.llmCallCount).toBe(1);
      expect(metrics.totalLatencyMs).toBe(0);
    });

    it('should log error message', async () => {
      const consoleSpy = vi.spyOn(console, 'error');
      await handler.handleLLMError(new Error('Test error'), 'run-1');

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('[TraigentHandler]'),
        'Test error'
      );
    });
  });

  describe('cost estimation', () => {
    it('should calculate cost for gpt-4o-mini', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 1_000_000,
          completionTokens: 1_000_000,
          modelName: 'gpt-4o-mini',
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();
      // gpt-4o-mini: $0.15/1M input, $0.60/1M output
      expect(metrics.totalCost).toBeCloseTo(0.15 + 0.6, 2);
    });

    it('should calculate cost for gpt-4o', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 1_000_000,
          completionTokens: 1_000_000,
          modelName: 'gpt-4o',
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();
      // gpt-4o: $2.50/1M input, $10.00/1M output
      expect(metrics.totalCost).toBeCloseTo(2.5 + 10.0, 2);
    });

    it('should calculate cost for claude-3-5-sonnet', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 1_000_000,
          completionTokens: 1_000_000,
          modelName: 'claude-3-5-sonnet',
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();
      // claude-3-5-sonnet: $3.00/1M input, $15.00/1M output
      expect(metrics.totalCost).toBeCloseTo(3.0 + 15.0, 2);
    });

    it('should use fallback cost for unknown models', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 1_000_000,
          completionTokens: 1_000_000,
          modelName: 'unknown-model-xyz',
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();
      // Fallback: $1.00/1M input, $3.00/1M output
      expect(metrics.totalCost).toBeCloseTo(1.0 + 3.0, 2);
    });

    it('should match specific model before generic prefix', async () => {
      // gpt-4o-mini should match before gpt-4o
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 1_000_000,
          completionTokens: 1_000_000,
          modelName: 'gpt-4o-mini-2024-07-18', // Full model name
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();
      // Should use gpt-4o-mini pricing, not gpt-4o
      expect(metrics.totalCost).toBeCloseTo(0.15 + 0.6, 2);
    });

    it('should calculate cost for gpt-4-turbo', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 1_000_000,
          completionTokens: 1_000_000,
          modelName: 'gpt-4-turbo',
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();
      // gpt-4-turbo: $10.00/1M input, $30.00/1M output
      expect(metrics.totalCost).toBeCloseTo(10.0 + 30.0, 2);
    });

    it('should calculate cost for gpt-3.5-turbo', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 1_000_000,
          completionTokens: 1_000_000,
          modelName: 'gpt-3.5-turbo',
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();
      // gpt-3.5-turbo: $0.50/1M input, $1.50/1M output
      expect(metrics.totalCost).toBeCloseTo(0.5 + 1.5, 2);
    });

    it('should calculate cost for claude-3-opus', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 1_000_000,
          completionTokens: 1_000_000,
          modelName: 'claude-3-opus',
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();
      // claude-3-opus: $15.00/1M input, $75.00/1M output
      expect(metrics.totalCost).toBeCloseTo(15.0 + 75.0, 2);
    });

    it('should calculate cost for claude-3-haiku', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 1_000_000,
          completionTokens: 1_000_000,
          modelName: 'claude-3-haiku',
        }),
        'run-1'
      );

      const metrics = handler.getMetrics();
      // claude-3-haiku: $0.25/1M input, $1.25/1M output
      expect(metrics.totalCost).toBeCloseTo(0.25 + 1.25, 2);
    });
  });

  describe('getMetrics()', () => {
    it('should return aggregated metrics', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({ promptTokens: 100, completionTokens: 200 }),
        'run-1'
      );

      const metrics = handler.getMetrics();

      expect(metrics).toHaveProperty('totalCost');
      expect(metrics).toHaveProperty('totalLatencyMs');
      expect(metrics).toHaveProperty('inputTokens');
      expect(metrics).toHaveProperty('outputTokens');
      expect(metrics).toHaveProperty('totalTokens');
      expect(metrics).toHaveProperty('llmCallCount');
      expect(metrics).toHaveProperty('calls');
    });

    it('should return copy of calls array', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(createMockLLMResult(), 'run-1');

      const metrics1 = handler.getMetrics();
      const metrics2 = handler.getMetrics();

      // Should be different array instances
      expect(metrics1.calls).not.toBe(metrics2.calls);
      expect(metrics1.calls).toEqual(metrics2.calls);
    });

    it('should include individual call details', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(createMockLLMResult({ modelName: 'gpt-4' }), 'run-1');

      const metrics = handler.getMetrics();

      expect(metrics.calls).toHaveLength(1);
      expect(metrics.calls[0]).toHaveProperty('runId', 'run-1');
      expect(metrics.calls[0]).toHaveProperty('model', 'gpt-4');
      expect(metrics.calls[0]).toHaveProperty('timestamp');
    });
  });

  describe('toMeasuresDict()', () => {
    it('should return prefixed metrics dictionary', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(
        createMockLLMResult({
          promptTokens: 100,
          completionTokens: 200,
          totalTokens: 300,
        }),
        'run-1'
      );

      const dict = handler.toMeasuresDict();

      expect(dict['langchain_total_cost']).toBeDefined();
      expect(dict['langchain_total_latency_ms']).toBeDefined();
      expect(dict['langchain_input_tokens']).toBe(100);
      expect(dict['langchain_output_tokens']).toBe(200);
      expect(dict['langchain_total_tokens']).toBe(300);
      expect(dict['langchain_llm_call_count']).toBe(1);
    });

    it('should use custom prefix when configured', () => {
      const customHandler = new TraigentHandler({ metricPrefix: 'llm_' });
      const dict = customHandler.toMeasuresDict();

      expect(dict['llm_total_cost']).toBeDefined();
      expect(dict['llm_llm_call_count']).toBe(0);
    });
  });

  describe('reset()', () => {
    it('should clear all collected metrics', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(createMockLLMResult(), 'run-1');

      expect(handler.callCount).toBe(1);

      handler.reset();

      expect(handler.callCount).toBe(0);
      const metrics = handler.getMetrics();
      expect(metrics.llmCallCount).toBe(0);
      expect(metrics.totalTokens).toBe(0);
    });

    it('should clear pending start times', async () => {
      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');

      handler.reset();

      // Now end without start - should have 0 latency
      await handler.handleLLMEnd(createMockLLMResult(), 'run-1');
      const metrics = handler.getMetrics();
      expect(metrics.totalLatencyMs).toBe(0);
    });
  });

  describe('callCount', () => {
    it('should return number of recorded calls', async () => {
      expect(handler.callCount).toBe(0);

      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-1');
      await handler.handleLLMEnd(createMockLLMResult(), 'run-1');

      expect(handler.callCount).toBe(1);

      await handler.handleLLMStart(mockSerializedLLM, ['prompt'], 'run-2');
      await handler.handleLLMEnd(createMockLLMResult(), 'run-2');

      expect(handler.callCount).toBe(2);
    });
  });

  describe('name property', () => {
    it('should have correct name', () => {
      expect(handler.name).toBe('TraigentHandler');
    });
  });
});
