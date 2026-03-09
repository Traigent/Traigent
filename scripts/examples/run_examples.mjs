#!/usr/bin/env node

import { readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { spawn } from "node:child_process";

function parseArgs(argv) {
  const args = {
    base: "examples",
    pattern: ".mjs",
    timeout: 60,
    verbose: false,
  };

  for (let index = 2; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--base" && argv[index + 1]) {
      args.base = argv[++index];
    } else if (value === "--pattern" && argv[index + 1]) {
      args.pattern = argv[++index];
    } else if (value === "--timeout" && argv[index + 1]) {
      args.timeout = Number(argv[++index]);
    } else if (value === "--verbose") {
      args.verbose = true;
    }
  }

  return args;
}

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    if (entry.name === "archive" || entry.name === "shared") continue;
    const fullPath = resolve(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walk(fullPath)));
    } else {
      files.push(fullPath);
    }
  }
  return files;
}

async function runExample(file, timeout, verbose) {
  return new Promise((resolveRun) => {
    const child = spawn(process.execPath, [file], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        TRAIGENT_OFFLINE_MODE: process.env.TRAIGENT_OFFLINE_MODE ?? "true",
      },
      stdio: verbose ? "inherit" : "pipe",
    });

    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      resolveRun({ file, success: false, error: `Timed out after ${timeout}s` });
    }, timeout * 1000);

    child.on("exit", (code) => {
      clearTimeout(timer);
      resolveRun({
        file,
        success: code === 0,
        error: code === 0 ? "" : `Exited with code ${code}`,
      });
    });
  });
}

const args = parseArgs(process.argv);
const files = (await walk(resolve(process.cwd(), args.base))).filter((file) =>
  file.endsWith(args.pattern),
);

if (files.length === 0) {
  console.error(`No example files found under ${args.base} matching ${args.pattern}`);
  process.exit(1);
}

let failed = 0;
for (const file of files.sort()) {
  const result = await runExample(file, args.timeout, args.verbose);
  const prefix = result.success ? "OK" : "FAIL";
  console.log(`[${prefix}] ${file}`);
  if (!result.success) {
    failed += 1;
    console.log(`  ${result.error}`);
  }
}

if (failed > 0) {
  process.exit(1);
}

console.log(`Completed ${files.length} example runs successfully.`);
