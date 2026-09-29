import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { ContextPort, ProposalPort } from "./inner-loop.js";
import {
  DEFAULT_CONTEXT_MODEL,
  DEFAULT_PROPOSAL_MODEL,
  parseRoleModel,
  type RoleModel,
} from "./role-model.js";
import { ActivitySink } from "./activity.js";
import { CONTEXT7_TOOLS, context7PiRoot } from "./context7-pi.js";
import { finishProposalWrite } from "./proposal-result.js";
import { withTimeout } from "./timeout.js";
import { parseLooseJson } from "./loose-json.js";

const rootDir = fileURLToPath(new URL("..", import.meta.url));
const PROPOSAL_REVIEW_MIN_MS = 60_000;

function msLeft(deadline: number): number {
  return Math.max(0, deadline - Date.now());
}

// Context7 is the official Pi port of the Context7 MCP: current library docs only.
export const PROPOSAL_TOOLS = ["read", "write", "edit", "ls", "bash", ...CONTEXT7_TOOLS] as const;
export const CONTEXT_TOOLS = ["read", "grep", "find", "ls", "write"] as const;

export async function createPiContext(opts: {
  runDir: string;
  task: string;
  model?: string;
}): Promise<ContextPort> {
  const sdk = await loadSdk();
  const system = await readFile(path.join(rootDir, "prompts", "context.md"), "utf8");
  const session = await openSession(sdk, {
    cwd: opts.runDir,
    system,
    tools: [...CONTEXT_TOOLS],
    roleModel: parseRoleModel(opts.model ?? DEFAULT_CONTEXT_MODEL),
  });

  return {
    async produce(input) {
      const outFile = path.join(opts.runDir, "context-out.json");
      const stopWatch = watchSession(
        session,
        new ActivitySink({ runDir: opts.runDir, actor: "context", id: "context" }),
      );
      try {
        await session.prompt(
          [
            `题目：\n${opts.task}`,
            `经验库代数：${input.generation}`,
            `当前最好：${JSON.stringify(input.best ?? null)}`,
            `全部记录：${JSON.stringify(input.records)}`,
            `队列需要补货：${input.needsMore}。队列已满：${input.mustStopProducing}。`,
            `已经用过的方向（不要再交）：${JSON.stringify(input.takenDirections)}`,
            "先读各条记录的 logPath、最好方案的 solutionDir（或 best/）和 queue/，再按系统提示做诊断、出实验。",
            `把结果写到 ${outFile}，JSON：{"inspirations":[{"direction":"...","context":"..."}],"stop":false}。只有当前最好已经达到题目过关线才把 stop 设为 true；否则 stop 必须是 false，且 inspirations 不能空。`,
            "写完本轮即停。",
          ].join("\n\n"),
        );
        const text = await readFile(outFile, "utf8");
        let raw: { inspirations?: Array<{ direction: string; context: string }>; stop?: boolean };
        try {
          raw = parseLooseJson(text) as typeof raw;
        } catch (err) {
          throw new Error(
            `无法解析 ${outFile}：${err instanceof Error ? err.message : String(err)}。` +
              "Context 必须输出合法 JSON，字符串内部的换行要写成 \\n。",
          );
        }
        return { inspirations: raw.inspirations ?? [], stop: raw.stop };
      } finally {
        await stopWatch();
      }
    },
  };
}

export async function createPiProposal(opts: {
  task: string;
  model?: string;
  timeoutMs?: number;
  runDir: string;
}): Promise<ProposalPort> {
  const sdk = await loadSdk();
  const system = await readFile(path.join(rootDir, "prompts", "proposal.md"), "utf8");
  const roleModel = parseRoleModel(opts.model ?? DEFAULT_PROPOSAL_MODEL);
  const timeoutMs = opts.timeoutMs;
  return {
    async write({ inspiration, workDir, lastError }) {
      const session = await openSession(sdk, {
        cwd: workDir,
        system,
        tools: [...PROPOSAL_TOOLS],
        roleModel,
        context7: true,
      });
      const stopWatch = watchSession(
        session,
        new ActivitySink({ runDir: opts.runDir, actor: "proposal", id: inspiration.id }),
      );
      try {
        const parts = [
          `题目：\n${opts.task}`,
          `灵感 ${inspiration.id}（经验库 v${inspiration.ebGeneration}）：${inspiration.direction}`,
          inspiration.context,
        ];
        if (lastError) {
          parts.push(
            `上一轮崩溃了。先读 ${workDir} 里已有文件，只修这个错误，不要推倒重来。`,
            `错误：\n${lastError}`,
          );
        }
        parts.push(
          `在 ${workDir} 写出能在沙盒里直接跑通的完整流水线：solve.sh 必须真正启动读数据、训练或推断、写出题目要求的预测文件。按灵感规格改。没有查到的库参数不要写。自己不要评分。写通本轮即停。`,
        );
        const deadline = timeoutMs === undefined ? undefined : Date.now() + timeoutMs;
        let err: unknown;
        try {
          await withTimeout(
            session.prompt(parts.join("\n\n")),
            deadline === undefined ? undefined : msLeft(deadline),
            "proposal write",
          );
        } catch (caught) {
          err = caught;
        }
        const reviewMs = deadline === undefined ? undefined : msLeft(deadline);
        if (reviewMs === undefined || reviewMs >= PROPOSAL_REVIEW_MIN_MS) {
          try {
            await withTimeout(
              session.prompt(
                [
                  `复读 ${workDir} 里每一个文件，把方案修到沙盒能直接跑通。`,
                  "从 solve.sh 走到读入、训练或推断、写出预测；缺的补上，假实现和 TODO 删掉。",
                  "用 bash 做语法检查。每一处第三方库调用必须已经对照过文档。只修漏洞，不要换实验主轴。修完再停。",
                ].join("\n\n"),
              ),
              reviewMs,
              "proposal review",
            );
          } catch (caught) {
            err = err ?? caught;
          }
        }
        return finishProposalWrite(workDir, err);
      } finally {
        await stopWatch();
        session.dispose();
      }
    },
  };
}

type Sdk = typeof import("@earendil-works/pi-coding-agent");

async function loadSdk(): Promise<Sdk> {
  try {
    return await import("@earendil-works/pi-coding-agent");
  } catch {
    throw new Error("缺少依赖 @earendil-works/pi-coding-agent，请先安装后再跑 hyra-pi run");
  }
}

async function openSession(
  sdk: Sdk,
  opts: { cwd: string; system: string; tools: string[]; roleModel: RoleModel; context7?: boolean },
) {
  const loader = new sdk.DefaultResourceLoader({
    cwd: opts.cwd,
    agentDir: sdk.getAgentDir(),
    systemPromptOverride: () => opts.system,
    appendSystemPromptOverride: () => [],
    additionalExtensionPaths: opts.context7 ? [context7PiRoot()] : [],
  });
  await loader.reload();
  const modelRuntime = await sdk.ModelRuntime.create();
  const model = await findModel(modelRuntime, opts.roleModel);
  const created = await sdk.createAgentSession({
    cwd: opts.cwd,
    tools: opts.tools,
    resourceLoader: loader,
    sessionManager: sdk.SessionManager.inMemory(opts.cwd),
    modelRuntime,
    model,
    thinkingLevel: opts.roleModel.thinkingLevel,
  });
  return created.session;
}

function watchSession(
  session: { subscribe: (listener: (event: unknown) => void) => () => void },
  sink: ActivitySink,
): () => Promise<void> {
  const unsub = session.subscribe((event) => sink.handle(event));
  return async () => {
    await sink.flush();
    unsub();
  };
}

async function findModel(modelRuntime: Awaited<ReturnType<Sdk["ModelRuntime"]["create"]>>, spec: RoleModel) {
  const direct = modelRuntime.getModel(spec.provider, spec.id);
  if (direct) return direct;
  const available = await modelRuntime.getAvailable();
  const match = available.find((item) => item.provider === spec.provider && item.id === spec.id);
  if (match) return match;
  const listed = available.map((item) => `${item.provider}/${item.id}`).join(", ");
  throw new Error(
    `找不到模型 ${spec.provider}/${spec.id}。当前已登录可用：${listed || "无"}`,
  );
}
