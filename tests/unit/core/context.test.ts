/**
 * Unit tests for TrialContext.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import {
  TrialContext,
  TrialContextError,
  getTrialConfig,
  getTrialParam,
  wrapCallback,
  bindContext,
} from '../../../src/core/context.js';
import type { TrialConfig } from '../../../src/dtos/trial.js';

const createMockConfig = (overrides: Partial<TrialConfig> = {}): TrialConfig => ({
  trial_id: 'test-trial-123',
  trial_number: 1,
  experiment_run_id: 'exp-456',
  config: { model: 'gpt-4o-mini', temperature: 0.7 },
  dataset_subset: { indices: [0, 1, 2], total: 10 },
  ...overrides,
});

describe('TrialContext', () => {
  describe('run()', () => {
    it('should execute function within trial context', async () => {
      const config = createMockConfig();

      const result = await TrialContext.run(config, async () => {
        expect(TrialContext.isInTrial()).toBe(true);
        return TrialContext.getConfig();
      });

      expect(result).toEqual(config);
    });

    it('should support nested runs with different configs', async () => {
      const outerConfig = createMockConfig({ trial_id: 'outer' });
      const innerConfig = createMockConfig({ trial_id: 'inner' });

      await TrialContext.run(outerConfig, async () => {
        expect(TrialContext.getTrialId()).toBe('outer');

        await TrialContext.run(innerConfig, async () => {
          expect(TrialContext.getTrialId()).toBe('inner');
        });

        // Should restore outer context
        expect(TrialContext.getTrialId()).toBe('outer');
      });
    });

    it('should work with synchronous functions', () => {
      const config = createMockConfig();

      const result = TrialContext.run(config, () => {
        return TrialContext.getConfig().trial_id;
      });

      expect(result).toBe('test-trial-123');
    });
  });

  describe('getConfig()', () => {
    it('should return config when in trial context', async () => {
      const config = createMockConfig();

      await TrialContext.run(config, async () => {
        const retrieved = TrialContext.getConfig();
        expect(retrieved).toEqual(config);
      });
    });

    it('should throw TrialContextError when not in trial context', () => {
      expect(() => TrialContext.getConfig()).toThrow(TrialContextError);
      expect(() => TrialContext.getConfig()).toThrow(
        'TrialContext.getConfig() called outside of a trial'
      );
    });
  });

  describe('getConfigOrUndefined()', () => {
    it('should return config when in trial context', async () => {
      const config = createMockConfig();

      await TrialContext.run(config, async () => {
        expect(TrialContext.getConfigOrUndefined()).toEqual(config);
      });
    });

    it('should return undefined when not in trial context', () => {
      expect(TrialContext.getConfigOrUndefined()).toBeUndefined();
    });
  });

  describe('isInTrial()', () => {
    it('should return true when in trial context', async () => {
      await TrialContext.run(createMockConfig(), async () => {
        expect(TrialContext.isInTrial()).toBe(true);
      });
    });

    it('should return false when not in trial context', () => {
      expect(TrialContext.isInTrial()).toBe(false);
    });
  });

  describe('getTrialId()', () => {
    it('should return trial ID when in context', async () => {
      await TrialContext.run(createMockConfig({ trial_id: 'my-trial' }), async () => {
        expect(TrialContext.getTrialId()).toBe('my-trial');
      });
    });

    it('should return undefined when not in context', () => {
      expect(TrialContext.getTrialId()).toBeUndefined();
    });
  });

  describe('getTrialNumber()', () => {
    it('should return trial number when in context', async () => {
      await TrialContext.run(createMockConfig({ trial_number: 5 }), async () => {
        expect(TrialContext.getTrialNumber()).toBe(5);
      });
    });

    it('should return undefined when not in context', () => {
      expect(TrialContext.getTrialNumber()).toBeUndefined();
    });
  });
});

describe('getTrialConfig()', () => {
  it('should return config.config when in trial', async () => {
    const config = createMockConfig({
      config: { model: 'gpt-4', temperature: 0.5 },
    });

    await TrialContext.run(config, async () => {
      const params = getTrialConfig();
      expect(params).toEqual({ model: 'gpt-4', temperature: 0.5 });
    });
  });

  it('should throw when not in trial', () => {
    expect(() => getTrialConfig()).toThrow(TrialContextError);
  });
});

describe('getTrialParam()', () => {
  it('should return parameter value when in trial', async () => {
    const config = createMockConfig({
      config: { model: 'gpt-4', temperature: 0.5 },
    });

    await TrialContext.run(config, async () => {
      expect(getTrialParam('model')).toBe('gpt-4');
      expect(getTrialParam<number>('temperature')).toBe(0.5);
    });
  });

  it('should return default value when param not set', async () => {
    const config = createMockConfig({ config: {} });

    await TrialContext.run(config, async () => {
      expect(getTrialParam('missing', 'default')).toBe('default');
    });
  });

  it('should return default value when not in trial', () => {
    expect(getTrialParam('model', 'fallback')).toBe('fallback');
    expect(getTrialParam('missing')).toBeUndefined();
  });
});

describe('wrapCallback()', () => {
  it('should preserve trial context in callback', async () => {
    const config = createMockConfig();

    await TrialContext.run(config, async () => {
      const callback = wrapCallback(() => TrialContext.getTrialId());

      // Simulate callback being called later (outside immediate context)
      const result = callback();
      expect(result).toBe('test-trial-123');
    });
  });
});

describe('bindContext()', () => {
  it('should bind function to current trial context', async () => {
    const config = createMockConfig({ trial_id: 'bound-trial' });

    let boundFn: () => string | undefined;

    await TrialContext.run(config, () => {
      boundFn = bindContext(() => TrialContext.getTrialId());
    });

    // Call outside of any trial context
    const result = boundFn!();
    expect(result).toBe('bound-trial');
  });

  it('should return original function when not in trial', () => {
    const fn = () => 'test';
    const bound = bindContext(fn);
    expect(bound).toBe(fn);
  });
});
