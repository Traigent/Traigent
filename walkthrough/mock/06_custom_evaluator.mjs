#!/usr/bin/env node
import { runScenario } from "../utils/scenario_runner.mjs";
await runScenario("custom_evaluator", "mock");
