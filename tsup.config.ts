import { defineConfig } from 'tsup';

// Main SDK and integrations (no CLI)
export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'integrations/langchain/index': 'src/integrations/langchain/index.ts',
    'integrations/vercel-ai/index': 'src/integrations/vercel-ai/index.ts',
    'integrations/openai/index': 'src/integrations/openai/index.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  treeshake: true,
  external: ['@langchain/core', 'ai', 'openai'],
});
