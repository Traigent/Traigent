import { parse } from '@babel/parser';
import generatorModule from '@babel/generator';
import traverseModule, { type NodePath } from '@babel/traverse';
import * as t from '@babel/types';

import { ValidationError } from '../core/errors.js';
import { getTrialParam } from '../core/context.js';
import { type FrameworkTarget, type SeamlessResolution } from '../optimization/types.js';
import { resolveRegisteredFrameworkTargets } from '../integrations/registry.js';
import { transformSeamlessFunctionPath, type SeamlessDiagnostic } from './transform.js';

const ALLOWED_GLOBAL_IDENTIFIERS = new Set([
  'Array',
  'Boolean',
  'Date',
  'Error',
  'JSON',
  'Map',
  'Math',
  'Number',
  'Object',
  'Promise',
  'Reflect',
  'Set',
  'String',
  'console',
  'undefined',
]);

type AnyFunction = (...args: any[]) => any;
const traverse = traverseModule.default ?? traverseModule;
const generate = generatorModule.default ?? generatorModule;
const RUNTIME_SEAMLESS_OPT_IN_ENV = 'TRAIGENT_ENABLE_EXPERIMENTAL_RUNTIME_SEAMLESS';

export interface ResolvedSeamlessFunction<T extends AnyFunction> {
  fn: T;
  resolution: SeamlessResolution;
}

function isAlreadyTransformed(fn: AnyFunction): boolean {
  const source = Function.prototype.toString.call(fn);
  return source.includes('getTrialParam(') || source.includes('getTrialConfig(');
}

function collectFreeIdentifiers(
  fnPath: NodePath<t.FunctionExpression> | NodePath<t.FunctionDeclaration>
): string[] {
  const free = new Set<string>();

  fnPath.traverse({
    Function(path: NodePath<t.Function>) {
      if (path !== fnPath) {
        path.skip();
      }
    },
    Identifier(path: NodePath<t.Identifier>) {
      if (!path.isReferencedIdentifier()) {
        return;
      }

      const binding = path.scope.getBinding(path.node.name);
      if (binding) {
        return;
      }

      if (ALLOWED_GLOBAL_IDENTIFIERS.has(path.node.name)) {
        return;
      }

      if (path.node.name === 'getTrialParam') {
        return;
      }

      free.add(path.node.name);
    },
  });

  return [...free];
}

function createFunctionFromSource<T extends AnyFunction>(code: string): T {
  // This is an experimental convenience path for self-contained, developer-authored
  // local functions after AST-based rewrite validation. It is not intended as a
  // sandbox for untrusted code; prefer the codemod or build-time plugin in any
  // environment where the function source is not fully trusted.
  return new Function('getTrialParam', `return (${code});`)(getTrialParam) as T;
}

function isRuntimeSeamlessOptedIn(): boolean {
  return process.env[RUNTIME_SEAMLESS_OPT_IN_ENV] === '1';
}

export function resolveSeamlessFunction<T extends AnyFunction>(
  fn: T,
  configKeys: readonly string[],
  frameworkTargets: readonly FrameworkTarget[] | undefined,
  autoOverrideFrameworks: boolean
): ResolvedSeamlessFunction<T> {
  const resolvedTargets = autoOverrideFrameworks
    ? resolveRegisteredFrameworkTargets(frameworkTargets)
    : [];

  if (resolvedTargets.length > 0) {
    return {
      fn,
      resolution: {
        path: 'framework',
        reason:
          frameworkTargets && frameworkTargets.length > 0
            ? 'Using registered framework interception for the requested seamless targets.'
            : 'Using registered framework interception for all active seamless targets.',
        experimental: false,
        targets: resolvedTargets,
      },
    };
  }

  if (isAlreadyTransformed(fn)) {
    return {
      fn,
      resolution: {
        path: 'pretransformed',
        reason: 'Using a pre-transformed seamless function (codemod or build-time rewrite).',
        experimental: false,
      },
    };
  }

  const source = Function.prototype.toString.call(fn);
  const wrappedSource = `const __traigent_fn = ${source};`;
  const ast = parse(wrappedSource, {
    sourceType: 'module',
    plugins: ['typescript', 'jsx'],
  });

  let transformedCode: string | undefined;
  let rewrittenCount = 0;

  traverse(ast, {
    Program(path: NodePath<t.Program>) {
      const declaration = path
        .get('body')
        .find((entry: NodePath<t.Statement>) => entry.isVariableDeclaration());
      if (!declaration || !declaration.isVariableDeclaration()) {
        return;
      }

      const declarator = declaration.get('declarations.0');
      if (!declarator || Array.isArray(declarator)) {
        return;
      }

      const initPath = declarator.get('init');
      if (
        !initPath.isFunctionExpression() &&
        !initPath.isFunctionDeclaration() &&
        !initPath.isArrowFunctionExpression()
      ) {
        return;
      }

      if (initPath.isArrowFunctionExpression()) {
        throw new ValidationError(
          'Experimental runtime seamless rewriting does not support arrow functions. Use `traigent migrate seamless` or the Babel plugin instead.'
        );
      }

      const diagnostics: SeamlessDiagnostic[] = [];
      rewrittenCount = transformSeamlessFunctionPath(
        initPath,
        new Set(configKeys),
        'getTrialParam',
        '<runtime>',
        diagnostics
      );
      const freeIdentifiers = collectFreeIdentifiers(
        initPath as unknown as NodePath<t.FunctionExpression> | NodePath<t.FunctionDeclaration>
      );
      if (freeIdentifiers.length > 0) {
        throw new ValidationError(
          `Experimental runtime seamless rewriting only supports self-contained functions. Unsupported free identifiers: ${freeIdentifiers.join(', ')}.`
        );
      }

      transformedCode = generate(initPath.node).code;
      path.stop();
    },
  });

  if (!transformedCode || rewrittenCount === 0) {
    throw new ValidationError(
      'Seamless injection requires a wrapped framework target or transformed tuned-variable function. Run `traigent migrate seamless` or use the Babel plugin for non-framework tuned variables.'
    );
  }

  if (!isRuntimeSeamlessOptedIn()) {
    throw new ValidationError(
      `Experimental runtime seamless rewriting is disabled by default. Set ${RUNTIME_SEAMLESS_OPT_IN_ENV}=1 only for trusted local code, or use \`traigent migrate seamless\` / the Babel plugin instead.`
    );
  }

  return {
    fn: createFunctionFromSource<T>(transformedCode),
    resolution: {
      path: 'runtime',
      reason: 'Using experimental runtime seamless rewriting for a self-contained function.',
      experimental: true,
    },
  };
}
