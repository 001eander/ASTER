#!/usr/bin/env node
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { continueBudget, parseArgs, parseBudget, parseProposalMs, parseSandboxMs } from "./budget.js";
import { runInnerLoop } from "./inner-loop.js";
import { parseProposalRewrites } from "./proposal-rewrite.js";
import { createPiContext, createPiProposal } from "./pi-agents.js";
import { DEFAULT_CONTEXT_MODEL, DEFAULT_PROPOSAL_MODEL } from "./role-model.js";
import { createDockerSandbox } from "./sandbox.js";
import { startStatusServer } from "./status-server.js";
import { loadTask } from "./task.js";

async function main(argv: string[]): Promise<void> {
  const args = parseArgs(argv);
  const command = String(args.get("_") ?? "");
  if (command === "run") {
    await runCommand(args);
    return;
  }
  if (command === "status") {
    await statusCommand(args);
    return;
  }
  if (command === "version" || args.get("version") === true) {
    const pkg = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")) as {
      version: string;
    };
    console.log(pkg.version);
    return;
  }
  console.log(`hyra-pi run --task <task-dir> --run <run-dir> --proposals 3 --budget 30m
hyra-pi run --task <task-dir> --no-limits
hyra-pi status --run <run-dir>
hyra-pi version`);
}

async function runCommand(args: Map<string, string | boolean>): Promise<void> {
  const taskFlag = args.get("task");
  if (typeof taskFlag !== "string") throw new Error("missing --task");
  const task = await loadTask(path.resolve(taskFlag));
  const runsRoot = path.resolve(String(args.get("runs") ?? "runs"));
  const runFlag = args.get("run");
  const runDir =
    typeof runFlag === "string"
      ? path.resolve(runFlag)
      : path.join(runsRoot, new Date().toISOString().replaceAll(":", "-"));
  await mkdir(runDir, { recursive: true });
  const saved = await readJson<{
    maxProposals?: number;
    maxSandboxes?: number;
    maxSolutions?: number;
    maxMs?: number;
  }>(path.join(runDir, "run.json"));
  const live = await readJson<{ consumedMs?: number }>(path.join(runDir, "live.json"));
  const noLimits = args.get("no-limits") === true;
  const proposals = Number(args.get("proposals") ?? saved?.maxProposals ?? 3);
  const sandboxes = Number(args.get("sandboxes") ?? saved?.maxSandboxes ?? 2);
  const budget = saved
    ? continueBudget(
        { maxSolutions: saved.maxSolutions, maxMs: saved.maxMs },
        live?.consumedMs ?? 0,
        {
          budget: typeof args.get("budget") === "string" ? String(args.get("budget")) : undefined,
          solutions: typeof args.get("solutions") === "string" ? String(args.get("solutions")) : undefined,
          unlimited: noLimits,
        },
      )
    : parseBudget(
        typeof args.get("budget") === "string"
          ? String(args.get("budget"))
          : noLimits
            ? undefined
            : "30m",
        typeof args.get("solutions") === "string" ? String(args.get("solutions")) : undefined,
        { unlimited: noLimits },
      );
  const proposalRewrites = parseProposalRewrites(
    typeof args.get("rewrites") === "string"
      ? String(args.get("rewrites"))
      : process.env.HYRA_PI_PROPOSAL_REWRITES,
  );
  const port = Number(args.get("port") ?? 8787);

  const sandboxMs = parseSandboxMs(
    typeof args.get("sandbox") === "string" ? String(args.get("sandbox")) : undefined,
    process.env.HYRA_PI_SANDBOX_MS,
    { unlimited: noLimits },
  );
  const proposalMs = parseProposalMs(
    typeof args.get("write") === "string" ? String(args.get("write")) : undefined,
    process.env.HYRA_PI_PROPOSAL_MS,
    { unlimited: noLimits },
  );
  const image = process.env.HYRA_PI_IMAGE ?? "debian:bookworm-slim";
  // 额外挂载（如数据只读挂载）：分号分隔多条，单条同 docker -v 语法。
  // Windows 宿主路径自带盘符冒号，故分隔符用分号。
  const extraMounts = (process.env.HYRA_PI_EXTRA_MOUNTS ?? "")
    .split(";")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
  const sandbox = createDockerSandbox({
    taskDir: task.dir,
    image,
    timeoutMs: sandboxMs,
    extraMounts,
  });
  const contextModel =
    typeof args.get("context-model") === "string"
      ? String(args.get("context-model"))
      : DEFAULT_CONTEXT_MODEL;
  const proposalModel =
    typeof args.get("proposal-model") === "string"
      ? String(args.get("proposal-model"))
      : DEFAULT_PROPOSAL_MODEL;
  const context = await createPiContext({
    runDir,
    task: task.description,
    model: contextModel,
  });
  const proposal = await createPiProposal({
    task: task.description,
    model: proposalModel,
    timeoutMs: proposalMs,
    runDir,
  });
  console.log(`context ${contextModel}`);
  console.log(`proposal ${proposalModel}`);
  console.log(`rewrites ${proposalRewrites}`);
  console.log(`write ${proposalMs === undefined ? "unlimited" : `${proposalMs}ms`}`);
  console.log(`sandbox ${sandboxMs === undefined ? "unlimited" : `${sandboxMs}ms`}`);
  if (noLimits) console.log("limits off");

  const server = await startStatusServer(runDir, port);
  console.log(`status ${server.url}`);
  openBrowser(server.url);

  try {
    const result = await runInnerLoop({
      runDir,
      maxProposals: proposals,
      maxSandboxes: sandboxes,
      lowWater: 1,
      highWater: Math.max(3, proposals * 2),
      budget,
      proposalRewrites,
      context,
      proposal,
      sandbox,
    });
    console.log(`stopped: ${result.stopReason}`);
    if (result.best?.score) console.log(`best: ${result.best.score.score}`);
    else console.log("best: none");
    console.log(`run dir: ${runDir}`);
  } finally {
    await server.close();
  }
}

async function statusCommand(args: Map<string, string | boolean>): Promise<void> {
  const runFlag = args.get("run");
  if (typeof runFlag !== "string") throw new Error("missing --run");
  const runDir = path.resolve(runFlag);
  const port = Number(args.get("port") ?? 8787);
  const server = await startStatusServer(runDir, port);
  console.log(`status ${server.url}`);
  openBrowser(server.url);
  console.log("press Ctrl+C to stop");
  await new Promise(() => undefined);
}

function openBrowser(url: string): void {
  // 跨平台打开浏览器；打不开不影响主流程（仅状态页入口）。
  const command =
    process.platform === "darwin"
      ? "open"
      : process.platform === "win32"
        ? "rundll32"
        : "xdg-open";
  const args = process.platform === "win32" ? ["url.dll,FileProtocolHandler", url] : [url];
  const child = spawn(command, args, { stdio: "ignore", detached: true });
  child.on("error", () => {});
  child.unref();
}

async function readJson<T>(file: string): Promise<T | undefined> {
  try {
    return JSON.parse(await readFile(file, "utf8")) as T;
  } catch {
    return undefined;
  }
}

main(process.argv.slice(2)).catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exitCode = 1;
});
