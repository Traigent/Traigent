import { beforeEach, describe, expect, it } from 'vitest';

import {
  clearRegisteredFrameworkTargets,
  describeFrameworkAutoOverride,
  getRegisteredFrameworkTargets,
  hasRegisteredFrameworkTarget,
  registerFrameworkTarget,
  resolveRegisteredFrameworkTargets,
} from '../../../src/integrations/registry.js';

describe('framework registry', () => {
  beforeEach(() => {
    clearRegisteredFrameworkTargets();
  });

  it('reports whether any framework target is registered', () => {
    expect(hasRegisteredFrameworkTarget(undefined)).toBe(false);

    registerFrameworkTarget('openai');

    expect(hasRegisteredFrameworkTarget(undefined)).toBe(true);
    expect(hasRegisteredFrameworkTarget([])).toBe(true);
  });

  it('checks specific target membership and clears state', () => {
    registerFrameworkTarget('langchain');

    expect(hasRegisteredFrameworkTarget(['langchain'])).toBe(true);
    expect(hasRegisteredFrameworkTarget(['openai'])).toBe(false);
    expect(hasRegisteredFrameworkTarget(['openai', 'langchain'])).toBe(true);

    clearRegisteredFrameworkTargets();

    expect(hasRegisteredFrameworkTarget(undefined)).toBe(false);
    expect(hasRegisteredFrameworkTarget(['langchain'])).toBe(false);
  });

  it('returns sorted active targets and resolves explicit filters', () => {
    registerFrameworkTarget('vercel-ai');
    registerFrameworkTarget('openai');

    expect(getRegisteredFrameworkTargets()).toEqual(['openai', 'vercel-ai']);
    expect(resolveRegisteredFrameworkTargets(undefined)).toEqual([
      'openai',
      'vercel-ai',
    ]);
    expect(resolveRegisteredFrameworkTargets(['langchain', 'openai'])).toEqual([
      'openai',
    ]);
  });

  it('describes auto-override state for active, filtered, disabled, and empty cases', () => {
    expect(describeFrameworkAutoOverride(undefined, true)).toEqual({
      autoOverrideFrameworks: true,
      requestedTargets: undefined,
      activeTargets: [],
      selectedTargets: [],
      enabled: false,
      reason:
        'No wrapped framework targets are currently registered for seamless interception.',
    });

    registerFrameworkTarget('vercel-ai');
    registerFrameworkTarget('openai');

    expect(describeFrameworkAutoOverride(undefined, true)).toEqual({
      autoOverrideFrameworks: true,
      requestedTargets: undefined,
      activeTargets: ['openai', 'vercel-ai'],
      selectedTargets: ['openai', 'vercel-ai'],
      enabled: true,
      reason:
        'Using all active registered framework targets for seamless interception.',
    });

    expect(describeFrameworkAutoOverride(['langchain'], true)).toEqual({
      autoOverrideFrameworks: true,
      requestedTargets: ['langchain'],
      activeTargets: ['openai', 'vercel-ai'],
      selectedTargets: [],
      enabled: false,
      reason: 'None of the requested framework targets are currently registered.',
    });

    expect(describeFrameworkAutoOverride(['openai'], false)).toEqual({
      autoOverrideFrameworks: false,
      requestedTargets: ['openai'],
      activeTargets: ['openai', 'vercel-ai'],
      selectedTargets: [],
      enabled: false,
      reason:
        'Framework auto-override is disabled for this seamless configuration.',
    });
  });
});
