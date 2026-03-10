import { beforeEach, describe, expect, it } from 'vitest';

import {
  clearRegisteredFrameworkTargets,
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
});
