import { readFileSync } from 'node:fs';

import { parse } from '@babel/parser';
import traverseModule, { type NodePath } from '@babel/traverse';
import * as t from '@babel/types';

import type { ParameterDefinition } from '../optimization/types.js';

const traverse = traverseModule.default ?? traverseModule;
const PARSER_PLUGINS: NonNullable<Parameters<typeof parse>[1]>['plugins'] = [
  'typescript',
  'jsx',
];

const COMMON_TUNABLE_NAMES = new Set([
  'model',
  'temperature',
  'maxTokens',
  'max_tokens',
  'topP',
  'top_p',
  'tone',
  'style',
  'promptStyle',
  'prompt_style',
  'retrievalK',
  'retrieval_k',
  'k',
  'threshold',
  'seed',
]);

const COMMON_TUNABLE_KEYS = new Set([
  'model',
  'temperature',
  'max_tokens',
  'maxTokens',
  'top_p',
  'topP',
  'frequency_penalty',
  'frequencyPenalty',
  'presence_penalty',
  'presencePenalty',
  'tone',
  'style',
  'k',
  'retrieval_k',
  'retrievalK',
]);

const STATISTICAL_SINKS = new Set([
  'create',
  'invoke',
  'ainvoke',
  'generate',
  'complete',
  'completion',
  'chat',
  'search',
  'query',
  'retrieve',
  'similaritySearch',
  'similarity_search',
  'embed',
  'embedQuery',
  'embed_query',
  'embedDocuments',
  'embed_documents',
  'run',
]);

export type TunedVariableConfidence = 'high' | 'medium' | 'low';
export type TunedVariableValueKind =
  | 'string'
  | 'int'
  | 'float'
  | 'boolean'
  | 'array'
  | 'object'
  | 'null';

export interface DiscoveredTunedVariable {
  name: string;
  confidence: TunedVariableConfidence;
  reason: string;
  kind: TunedVariableValueKind;
  defaultValue: unknown;
  supportedByConfigSpace: boolean;
  suggestedDefinition?: ParameterDefinition;
  line?: number;
  column?: number;
}

export interface TunedVariableDiscoveryResult {
  functionName: string;
  candidates: DiscoveredTunedVariable[];
  warnings: string[];
}

export interface DiscoverTunedVariablesOptions {
  functionName?: string;
  includeLowConfidence?: boolean;
  includePrivate?: boolean;
}

interface LiteralInfo {
  kind: TunedVariableValueKind;
  value: unknown;
}

function isSupportedLiteralExpression(
  node: t.Node | null | undefined,
): node is t.Expression {
  if (!node) {
    return false;
  }

  if (
    t.isStringLiteral(node) ||
    t.isNumericLiteral(node) ||
    t.isBooleanLiteral(node) ||
    t.isNullLiteral(node)
  ) {
    return true;
  }

  if (t.isUnaryExpression(node)) {
    return (
      (node.operator === '-' || node.operator === '+') &&
      t.isNumericLiteral(node.argument)
    );
  }

  if (t.isTemplateLiteral(node)) {
    return node.expressions.length === 0;
  }

  if (t.isArrayExpression(node)) {
    return node.elements.every((element) => {
      if (!element || t.isSpreadElement(element)) {
        return false;
      }
      return isSupportedLiteralExpression(element);
    });
  }

  if (t.isObjectExpression(node)) {
    return node.properties.every((property) => {
      if (!t.isObjectProperty(property) || property.computed) {
        return false;
      }
      if (
        !t.isIdentifier(property.key) &&
        !t.isStringLiteral(property.key) &&
        !t.isNumericLiteral(property.key)
      ) {
        return false;
      }
      return isSupportedLiteralExpression(property.value);
    });
  }

  return false;
}

function literalToInfo(node: t.Expression): LiteralInfo {
  if (t.isStringLiteral(node)) {
    return { kind: 'string', value: node.value };
  }
  if (t.isNumericLiteral(node)) {
    return {
      kind: Number.isInteger(node.value) ? 'int' : 'float',
      value: node.value,
    };
  }
  if (t.isBooleanLiteral(node)) {
    return { kind: 'boolean', value: node.value };
  }
  if (t.isNullLiteral(node)) {
    return { kind: 'null', value: null };
  }
  if (t.isUnaryExpression(node) && t.isNumericLiteral(node.argument)) {
    const value =
      node.operator === '-' ? -node.argument.value : node.argument.value;
    return {
      kind: Number.isInteger(value) ? 'int' : 'float',
      value,
    };
  }
  if (t.isTemplateLiteral(node)) {
    return {
      kind: 'string',
      value: node.quasis.map((quasi) => quasi.value.cooked ?? '').join(''),
    };
  }
  if (t.isArrayExpression(node)) {
    return {
      kind: 'array',
      value: node.elements.map((element) =>
        element && !t.isSpreadElement(element)
          ? literalToInfo(element).value
          : undefined,
      ),
    };
  }
  if (t.isObjectExpression(node)) {
    const value: Record<string, unknown> = {};
    for (const property of node.properties) {
      if (!t.isObjectProperty(property) || property.computed) {
        continue;
      }
      let key: string;
      if (t.isIdentifier(property.key)) {
        key = property.key.name;
      } else if (
        t.isStringLiteral(property.key) ||
        t.isNumericLiteral(property.key)
      ) {
        key = String(property.key.value);
      } else {
        continue;
      }
      value[key] = literalToInfo(property.value as t.Expression).value;
    }
    return { kind: 'object', value };
  }

  return { kind: 'null', value: null };
}

function inferSuggestedDefinition(info: LiteralInfo): {
  supportedByConfigSpace: boolean;
  suggestedDefinition?: ParameterDefinition;
} {
  switch (info.kind) {
    case 'boolean':
      return {
        supportedByConfigSpace: true,
        suggestedDefinition: {
          type: 'enum',
          values: [info.value as boolean, !(info.value as boolean)],
        },
      };
    case 'string':
      return {
        supportedByConfigSpace: true,
        suggestedDefinition: {
          type: 'enum',
          values: [info.value as string],
        },
      };
    case 'int': {
      const value = info.value as number;
      const delta = Math.max(1, Math.abs(value));
      return {
        supportedByConfigSpace: true,
        suggestedDefinition: {
          type: 'int',
          min: Math.floor(value - delta),
          max: Math.ceil(value + delta),
          step: 1,
        },
      };
    }
    case 'float': {
      const value = info.value as number;
      const delta = Math.max(0.1, Math.abs(value) * 0.5);
      return {
        supportedByConfigSpace: true,
        suggestedDefinition: {
          type: 'float',
          min: value - delta,
          max: value + delta,
        },
      };
    }
    default:
      return { supportedByConfigSpace: false };
  }
}

function getStaticPropertyKey(
  property: t.ObjectProperty | t.ObjectMethod | t.ClassMethod | t.ClassProperty,
): string | undefined {
  if (property.computed) {
    return undefined;
  }
  if (t.isIdentifier(property.key)) {
    return property.key.name;
  }
  if (
    t.isStringLiteral(property.key) ||
    t.isNumericLiteral(property.key)
  ) {
    return String(property.key.value);
  }
  return undefined;
}

function getCalleeName(callee: t.Expression | t.V8IntrinsicIdentifier): string | undefined {
  if (t.isIdentifier(callee)) {
    return callee.name;
  }
  if (t.isMemberExpression(callee) && !callee.computed && t.isIdentifier(callee.property)) {
    return callee.property.name;
  }
  return undefined;
}

function classifyCandidate(
  name: string,
  binding: NonNullable<ReturnType<NodePath<t.Identifier>['scope']['getBinding']>>,
): { confidence: TunedVariableConfidence; reason: string } | undefined {
  if (binding.referencePaths.length === 0) {
    return undefined;
  }

  let mediumReason: string | undefined;

  for (const referencePath of binding.referencePaths) {
    const propertyPath = referencePath.parentPath;
    if (
      propertyPath?.isObjectProperty() &&
      propertyPath.get('value') === referencePath
    ) {
      const key = getStaticPropertyKey(propertyPath.node);
      if (key && COMMON_TUNABLE_KEYS.has(key)) {
        return {
          confidence: 'high',
          reason: `Used as the "${key}" field in an options object passed into a call.`,
        };
      }
    }

    const callPath = referencePath.findParent((path) => path.isCallExpression());
    if (callPath?.isCallExpression()) {
      const calleeName = getCalleeName(callPath.node.callee);
      if (calleeName && STATISTICAL_SINKS.has(calleeName)) {
        return {
          confidence: COMMON_TUNABLE_NAMES.has(name) ? 'high' : 'medium',
          reason: `Flows into the ${calleeName}() call path.`,
        };
      }
      mediumReason ??= 'Flows into a function call.';
    }

    if (referencePath.findParent((path) => path.isTemplateLiteral())) {
      mediumReason ??= 'Interpolated into a template literal.';
    }

    if (
      referencePath.findParent(
        (path) => path.isBinaryExpression() || path.isTemplateLiteral(),
      )
    ) {
      mediumReason ??= 'Used in composed prompt or request content.';
    }
  }

  if (COMMON_TUNABLE_NAMES.has(name)) {
    return {
      confidence: mediumReason ? 'high' : 'medium',
      reason:
        mediumReason !== undefined
          ? `Name "${name}" matches a common tunable parameter and ${mediumReason.toLowerCase()}`
          : `Name "${name}" matches a common tunable parameter.`,
    };
  }

  if (mediumReason) {
    return {
      confidence: 'medium',
      reason: mediumReason,
    };
  }

  return {
    confidence: 'low',
    reason: 'Local literal configuration candidate with references inside the function.',
  };
}

function getFunctionName(
  functionPath:
    | NodePath<t.FunctionDeclaration>
    | NodePath<t.FunctionExpression>
    | NodePath<t.ArrowFunctionExpression>,
): string {
  if ('id' in functionPath.node && functionPath.node.id && t.isIdentifier(functionPath.node.id)) {
    return functionPath.node.id.name;
  }
  if (
    functionPath.parentPath.isVariableDeclarator() &&
    t.isIdentifier(functionPath.parentPath.node.id)
  ) {
    return functionPath.parentPath.node.id.name;
  }
  return 'anonymous';
}

function analyzeFunctionPath(
  functionPath:
    | NodePath<t.FunctionDeclaration>
    | NodePath<t.FunctionExpression>
    | NodePath<t.ArrowFunctionExpression>,
  options: DiscoverTunedVariablesOptions,
): TunedVariableDiscoveryResult {
  const functionName = getFunctionName(functionPath);
  const warnings: string[] = [];
  const candidates: DiscoveredTunedVariable[] = [];

  functionPath.traverse({
    Function(path) {
      if (path !== functionPath) {
        path.skip();
      }
    },
    VariableDeclarator(path) {
      if (!t.isIdentifier(path.node.id)) {
        return;
      }

      const name = path.node.id.name;
      if (!options.includePrivate && name.startsWith('_')) {
        return;
      }

      const binding = path.scope.getBinding(name);
      if (!binding || binding.path !== path) {
        return;
      }

      if (binding.constantViolations.length > 0) {
        warnings.push(
          `Skipped "${name}" because it is reassigned or mutated after declaration.`,
        );
        return;
      }

      if (!path.node.init || !isSupportedLiteralExpression(path.node.init)) {
        return;
      }

      const classification = classifyCandidate(name, binding);
      if (!classification) {
        return;
      }

      if (
        classification.confidence === 'low' &&
        !options.includeLowConfidence
      ) {
        return;
      }

      const literal = literalToInfo(path.node.init);
      const suggestion = inferSuggestedDefinition(literal);

      candidates.push({
        name,
        confidence: classification.confidence,
        reason: classification.reason,
        kind: literal.kind,
        defaultValue: literal.value,
        supportedByConfigSpace: suggestion.supportedByConfigSpace,
        suggestedDefinition: suggestion.suggestedDefinition,
        line: path.node.loc?.start.line,
        column: path.node.loc?.start.column,
      });
    },
  });

  candidates.sort((left, right) => {
    const confidenceOrder = { high: 0, medium: 1, low: 2 } as const;
    return (
      confidenceOrder[left.confidence] - confidenceOrder[right.confidence] ||
      left.name.localeCompare(right.name)
    );
  });

  return {
    functionName,
    candidates,
    warnings: [...new Set(warnings)],
  };
}

function parseSource(source: string) {
  return parse(source, {
    sourceType: 'module',
    plugins: PARSER_PLUGINS,
  });
}

function collectFunctionPaths(
  source: string,
  functionName?: string,
):
  | Array<
      | NodePath<t.FunctionDeclaration>
      | NodePath<t.FunctionExpression>
      | NodePath<t.ArrowFunctionExpression>
    >
  | never {
  const ast = parseSource(source);
  const functionPaths: Array<
    | NodePath<t.FunctionDeclaration>
    | NodePath<t.FunctionExpression>
    | NodePath<t.ArrowFunctionExpression>
  > = [];

  traverse(ast, {
    Program(path) {
      const registerFunctionPath = (
        pathToRegister:
          | NodePath<t.FunctionDeclaration>
          | NodePath<t.FunctionExpression>
          | NodePath<t.ArrowFunctionExpression>,
      ) => {
        if (!functionName || getFunctionName(pathToRegister) === functionName) {
          functionPaths.push(pathToRegister);
        }
      };

      const registerDeclaration = (
        declarationPath:
          | NodePath<t.FunctionDeclaration>
          | NodePath<t.VariableDeclaration>,
      ) => {
        if (declarationPath.isFunctionDeclaration()) {
          registerFunctionPath(declarationPath);
          return;
        }

        for (const declarator of declarationPath.get('declarations')) {
          const initPath = declarator.get('init');
          if (
            initPath.isFunctionExpression() ||
            initPath.isArrowFunctionExpression()
          ) {
            registerFunctionPath(initPath);
          }
        }
      };

      for (const childPath of path.get('body')) {
        if (childPath.isFunctionDeclaration() || childPath.isVariableDeclaration()) {
          registerDeclaration(childPath);
          continue;
        }

        if (!childPath.isExportNamedDeclaration()) {
          if (!childPath.isExportDefaultDeclaration()) {
            continue;
          }

          const declarationPath = childPath.get('declaration');
          if (
            declarationPath.isFunctionDeclaration() ||
            declarationPath.isFunctionExpression() ||
            declarationPath.isArrowFunctionExpression()
          ) {
            registerFunctionPath(declarationPath);
          }
          continue;
        }

        const declarationPath = childPath.get('declaration');
        if (
          declarationPath.isFunctionDeclaration() ||
          declarationPath.isVariableDeclaration()
        ) {
          registerDeclaration(declarationPath);
        }
      }
      path.stop();
    },
  });

  return functionPaths;
}

export function discoverTunedVariables(
  fn: (...args: any[]) => any,
  options: Omit<DiscoverTunedVariablesOptions, 'functionName'> = {},
): TunedVariableDiscoveryResult {
  const source = Function.prototype.toString.call(fn);
  const wrappedSource = `const __traigent_target = ${source};`;
  const [functionPath] = collectFunctionPaths(wrappedSource);

  if (!functionPath) {
    return {
      functionName: fn.name || 'anonymous',
      candidates: [],
      warnings: ['Could not analyze the provided function source.'],
    };
  }

  const result = analyzeFunctionPath(functionPath, options);
  return {
    ...result,
    functionName: fn.name || 'anonymous',
  };
}

export function discoverTunedVariablesFromSource(
  source: string,
  options: DiscoverTunedVariablesOptions = {},
): TunedVariableDiscoveryResult[] {
  const functionPaths = collectFunctionPaths(source, options.functionName);
  return functionPaths.map((functionPath) =>
    analyzeFunctionPath(functionPath, options),
  );
}

export function discoverTunedVariablesFromFile(
  filePath: string,
  options: DiscoverTunedVariablesOptions = {},
): TunedVariableDiscoveryResult[] {
  return discoverTunedVariablesFromSource(
    readFileSync(filePath, 'utf8'),
    options,
  );
}
