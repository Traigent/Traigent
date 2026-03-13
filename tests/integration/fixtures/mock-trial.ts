/**
 * Mock trial function for integration testing.
 * Supports various behaviors based on config parameters.
 */
import { getTrialParam } from '../../../src/core/context.js';

export async function runTrial(config: { trial_id: string; config: Record<string, unknown> }) {
  void config;
  const behavior = getTrialParam<string>('behavior', 'success');

  switch (behavior) {
    case 'success':
      return {
        metrics: {
          accuracy: 0.95,
          latency_ms: 100,
        },
      };

    case 'slow':
      // Simulate a slow trial that respects abort signal
      await new Promise((resolve) => setTimeout(resolve, 10_000));
      return { metrics: { accuracy: 0.5 } };

    case 'error':
      throw new Error('User function error');

    case 'bad_metrics':
      return {
        metrics: {
          valid: 1.0,
          nan_value: NaN,
          inf_value: Infinity,
          string_value: 'not a number',
        },
      };

    default:
      return { metrics: { accuracy: 1.0 } };
  }
}
