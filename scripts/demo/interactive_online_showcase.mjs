#!/usr/bin/env node

import { fileURLToPath } from 'node:url';

import {
  metadata as section1,
  runSection as runSection1,
} from '../../examples/core/online-showcase/01_seamless_openai_env.mjs';
import {
  metadata as section2,
  runSection as runSection2,
} from '../../examples/core/online-showcase/02_context_explicit.mjs';
import {
  metadata as section3,
  runSection as runSection3,
} from '../../examples/core/online-showcase/03_parameter_custom_evaluator.mjs';
import {
  metadata as section4,
  runSection as runSection4,
} from '../../examples/core/online-showcase/04_seamless_langchain_session.mjs';
import {
  metadata as section5,
  runSection as runSection5,
} from '../../examples/core/online-showcase/05_constraints_reporting.mjs';

const SECTIONS = [
  { ...section1, run: runSection1 },
  { ...section2, run: runSection2 },
  { ...section3, run: runSection3 },
  { ...section4, run: runSection4 },
  { ...section5, run: runSection5 },
];

function getArg(flag) {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function findSection(id) {
  return SECTIONS.find((section) => section.id === String(id));
}

function printUsage() {
  console.log(
    [
      'Usage:',
      '  node scripts/demo/interactive_online_showcase.mjs --list',
      '  node scripts/demo/interactive_online_showcase.mjs --describe <id>',
      '  node scripts/demo/interactive_online_showcase.mjs --section <id>',
    ].join('\n')
  );
}

if (process.argv.includes('--list')) {
  console.log(
    JSON.stringify(
      SECTIONS.map(({ id, title, description, codePath }) => ({
        id,
        title,
        description,
        codePath,
      })),
      null,
      2
    )
  );
  process.exit(0);
}

const describedId = getArg('--describe');
if (describedId) {
  const section = findSection(describedId);
  if (!section) {
    console.error(`Unknown section "${describedId}".`);
    process.exit(1);
  }

  console.log(
    JSON.stringify(
      {
        id: section.id,
        title: section.title,
        description: section.description,
        codePath: section.codePath,
      },
      null,
      2
    )
  );
  process.exit(0);
}

const sectionId = getArg('--section');
if (!sectionId) {
  printUsage();
  process.exit(1);
}

const section = findSection(sectionId);
if (!section) {
  console.error(`Unknown section "${sectionId}".`);
  process.exit(1);
}

const startedAt = Date.now();
const result = await section.run();
console.log(
  JSON.stringify(
    {
      section: {
        id: section.id,
        title: section.title,
        description: section.description,
        codePath: section.codePath,
      },
      durationMs: Date.now() - startedAt,
      result,
    },
    null,
    2
  )
);

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  process.exit(0);
}
