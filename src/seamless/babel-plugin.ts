import type { PluginObj, PluginPass } from '@babel/core';
import type { NodePath } from '@babel/traverse';
import type * as t from '@babel/types';

import { transformSeamlessProgram, type SeamlessDiagnostic } from './transform.js';

export function formatSeamlessDiagnosticPreview(
  diagnostic: SeamlessDiagnostic,
  fallbackFilename?: string
): string {
  const location =
    diagnostic.line !== undefined ? `:${diagnostic.line}:${(diagnostic.column ?? 0) + 1}` : '';
  return `- ${diagnostic.filename ?? fallbackFilename ?? '<unknown>'}${location} ${diagnostic.message}`;
}

export default function traigentSeamlessBabelPlugin(): PluginObj {
  return {
    name: 'traigent-seamless',
    visitor: {
      Program(path: NodePath<t.Program>, state: PluginPass) {
        const result = transformSeamlessProgram(path, {
          filename: state.filename,
        });

        const rejectedDiagnostics = result.diagnostics.filter(
          (diagnostic) => diagnostic.kind === 'rejected'
        );
        if (rejectedDiagnostics.length > 0) {
          const preview = rejectedDiagnostics
            .slice(0, 3)
            .map((diagnostic) => formatSeamlessDiagnosticPreview(diagnostic, state.filename))
            .join('\n');
          const suffix =
            rejectedDiagnostics.length > 3
              ? `\n- ...and ${rejectedDiagnostics.length - 3} more`
              : '';
          throw path.buildCodeFrameError(
            `[traigent-seamless] Refusing to emit a partial seamless transform.\n${preview}${suffix}\nUse explicit context/parameter injection or run the codemod to address rejected patterns first.`
          );
        }

        (state.file.metadata as Record<string, unknown>)['traigentSeamless'] = {
          changed: result.changed,
          rewrittenCount: result.rewrittenCount,
          diagnostics: result.diagnostics,
        };
      },
    },
  };
}
