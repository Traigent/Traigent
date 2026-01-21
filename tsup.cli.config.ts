import { defineConfig } from 'tsup';

// CLI runner (source already has shebang, no banner needed)
export default defineConfig({
  entry: {
    'cli/runner': 'src/cli/runner.ts',
  },
  format: ['esm'],
  dts: false,
  splitting: false,
  sourcemap: true,
  clean: false, // Don't clean - SDK build already ran
  treeshake: true,
  // Note: no banner needed - src/cli/runner.ts already has shebang
});
