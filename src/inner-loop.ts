import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { ExperienceBank, type ExperienceRecord, type Score } from "./experience-bank.js";
import {
  DEFAULT_DIRECTION_QUOTA,
  InspirationQueue,
  normalizeDirection,
  type Inspiration,
} from "./inspiration-queue.js";
import { writeJsonFile } from "./json-file.js";
import { DEFAULT_PROPOSAL_REWRITES, needsProposalRewrite } from "./proposal-rewrite.js";
import { ResourceGates } from "./resource-gates.js";

export type ContextDraft = {
  direction: string;
  context: string;
};

export type ContextPort = {
  produce(input: {
    generation: number;
    needsMore: boolean;
    mustStopProducing: boolean;
    best?: ExperienceRecord;
    records: ExperienceRecord[];
    takenDirections: string[];
    // 上一轮被方向配额挡下的方向：不算用过，名额空出后可以再交。
    deferredDirections: string[];
  }): Promise<{ inspirations?: ContextDraft[]; stop?: boolean }>;
};

export type ProposalPort = {
  write(input: {
    inspiration: Inspiration;
    workDir: string;
    lastError?: string;
  }): Promise<{ solutionDir: string } | { error: string }>;
};

export type SandboxPort = {
  evaluate(solutionDir: string): Promise<{
    ok: boolean;
    score: Score | null;
    log: string;
  }>;
};

export type LoopPhase = "starting" | "running" | "draining" | "stopped";

export type LiveState = {
  phase: LoopPhase;
  stopReason?: StopReason;
  writers: Array<{ id: string; direction: string }>;
  sandboxes: Array<{ id: string; direction: string }>;
  contextRunning: boolean;
  lastContextAt?: number;
  startedAt: number;
  consumedMs?: number;
};

export type StopReason = "budget" | "context-stop" | "context-idle";

export type LoopResult = {
  best?: ExperienceRecord;
  records: ExperienceRecord[];
  stopReason: StopReason;
};

export type LoopOptions = {
  runDir: string;
  maxProposals: number;
  maxSandboxes: number;
  lowWater: number;
  highWater: number;
  // 同一方向组（「信号源：」前缀相同）的并发灵感上限，缺省 DEFAULT_DIRECTION_QUOTA。
  directionQuota?: number;
  budget: { maxSolutions?: number; maxMs?: number };
  context: ContextPort;
  proposal: ProposalPort;
  sandbox: SandboxPort;
  proposalRewrites?: number;
  now?: () => number;
};

export async function runInnerLoop(opts: LoopOptions): Promise<LoopResult> {
  const now = opts.now ?? Date.now;
  const sessionStartedAt = now();
  const previousLive = await readLive(opts.runDir);
  const consumedBefore = previousLive?.consumedMs ?? 0;
  const startedAt = previousLive?.startedAt ?? sessionStartedAt;
  await writeFile(
    path.join(opts.runDir, "run.json"),
    JSON.stringify(
      {
        maxProposals: opts.maxProposals,
        maxSandboxes: opts.maxSandboxes,
        lowWater: opts.lowWater,
        highWater: opts.highWater,
        directionQuota: opts.directionQuota ?? DEFAULT_DIRECTION_QUOTA,
        maxSolutions: opts.budget.maxSolutions,
        maxMs: opts.budget.maxMs,
        proposalRewrites: opts.proposalRewrites ?? DEFAULT_PROPOSAL_REWRITES,
        contextMaxIdleMs: Number(process.env.HYRA_PI_CONTEXT_IDLE_MS ?? 180_000),
      },
      null,
      2,
    ),
    "utf8",
  );
  const bank = await ExperienceBank.open(opts.runDir);
  const committed = new Set((await bank.list()).map((row) => row.inspirationId));
  const queue = await InspirationQueue.open(opts.runDir, {
    lowWater: opts.lowWater,
    highWater: opts.highWater,
    directionQuota: opts.directionQuota,
    knownIds: [...committed],
  });
  await queue.requeueOrphans(committed);
  const gates = new ResourceGates({
    maxProposals: opts.maxProposals,
    maxSandboxes: opts.maxSandboxes,
  });

  const live: LiveState = {
    phase: "starting",
    writers: [],
    sandboxes: [],
    contextRunning: false,
    startedAt,
    consumedMs: consumedBefore,
  };
  if (previousLive?.lastContextAt) live.lastContextAt = previousLive.lastContextAt;

  const jobs = new Set<Promise<void>>();
  let stopRequested = false;
  let stopReason: StopReason | undefined;
  // 被方向配额挡下的 direction：短暂饱和而已，交回 Context 等名额空出再提。
  const deferredDirections = new Set<string>();

  const consumedNow = () => consumedBefore + Math.max(0, now() - sessionStartedAt);

  const persist = async () => {
    live.phase = phaseOf();
    live.consumedMs = consumedNow();
    await writeJsonFile(path.join(opts.runDir, "live.json"), live);
  };

  const phaseOf = (): LoopPhase => {
    if (stopReason) return "stopped";
    const remaining = remainingWork();
    if (remaining <= 0 || stopRequested) return live.phase === "starting" ? "starting" : "draining";
    if (bank.generation() === 0) return "starting";
    return "running";
  };

  const remainingWork = () =>
    opts.budget.maxSolutions === undefined
      ? Number.POSITIVE_INFINITY
      : opts.budget.maxSolutions - bank.generation() - jobs.size;

  const budgetTimeUp = () =>
    opts.budget.maxMs !== undefined && consumedNow() >= opts.budget.maxMs;

  const runOne = async (insp: Inspiration) => {
    const workDir = path.join(opts.runDir, "workspaces", insp.id);
    await mkdir(workDir, { recursive: true });
    const maxRewrites = opts.proposalRewrites ?? DEFAULT_PROPOSAL_REWRITES;
    let lastError: string | undefined;
    let solutionDir = workDir;
    let heldProposal = true;

    try {
      for (let attempt = 0; attempt <= maxRewrites; attempt += 1) {
        if (!heldProposal) {
          await gates.acquireProposal();
          heldProposal = true;
        }

        live.writers.push({ id: insp.id, direction: insp.direction });
        await persist();
        let writeError: string | undefined;
        try {
          const written = await opts.proposal.write({ inspiration: insp, workDir, lastError });
          if ("error" in written) writeError = written.error;
          else solutionDir = written.solutionDir;
        } catch (err) {
          writeError = err instanceof Error ? err.message : String(err);
        } finally {
          live.writers = live.writers.filter((row) => row.id !== insp.id);
          gates.finishProposal();
          heldProposal = false;
          await persist();
        }

        if (writeError) {
          lastError = writeError;
          if (attempt < maxRewrites) continue;
          await bank.commit({
            inspirationId: insp.id,
            solutionDir,
            ok: false,
            log: writeError,
            score: null,
          });
          return;
        }

        await gates.acquireSandbox();
        live.sandboxes.push({ id: insp.id, direction: insp.direction });
        await persist();
        try {
          const evaluated = await opts.sandbox.evaluate(solutionDir);
          if (
            !needsProposalRewrite({ ok: evaluated.ok, score: evaluated.score }) ||
            attempt === maxRewrites
          ) {
            await bank.commit({
              inspirationId: insp.id,
              solutionDir,
              ok: evaluated.ok,
              log: evaluated.log,
              score: evaluated.score,
            });
            return;
          }
          lastError = evaluated.log;
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          if (attempt === maxRewrites) {
            await bank.commit({
              inspirationId: insp.id,
              solutionDir,
              ok: false,
              log: message,
              score: null,
            });
            return;
          }
          lastError = message;
        } finally {
          live.sandboxes = live.sandboxes.filter((row) => row.id !== insp.id);
          gates.finishSandbox();
          await persist();
        }
      }
    } finally {
      if (heldProposal) gates.finishProposal();
      queue.releaseHold(insp.id);
      await persist();
    }
  };

  const track = (job: Promise<void>) => {
    jobs.add(job);
    job.finally(() => {
      jobs.delete(job);
    });
  };

  try {
    while (!stopReason) {
      if (budgetTimeUp()) stopRequested = true;
      await persist();

      if (jobs.size === 0 && (remainingWork() <= 0 || (stopRequested && queue.waitingCount() === 0))) {
        stopReason =
          stopRequested &&
          (opts.budget.maxSolutions === undefined || bank.generation() < opts.budget.maxSolutions)
            ? "context-stop"
            : "budget";
        break;
      }

      let progressed = false;

      if (
        !stopRequested &&
        remainingWork() > 0 &&
        queue.needsMore() &&
        gates.tryStartContext()
      ) {
        live.contextRunning = true;
        await persist();
        try {
          const taken = queue.takenDirections();
          const takenSet = new Set(taken);
          const out = await opts.context.produce({
            generation: bank.generation(),
            needsMore: queue.needsMore(),
            mustStopProducing: queue.mustStopProducing(),
            best: await bank.best(),
            records: await bank.list(),
            takenDirections: taken,
            deferredDirections: [...deferredDirections].filter(
              (direction) => !takenSet.has(normalizeDirection(direction)),
            ),
          });
          if (out.stop) stopRequested = true;
          let added = 0;
          for (const draft of out.inspirations ?? []) {
            if (queue.mustStopProducing()) break;
            const put = await queue.enqueue({
              direction: draft.direction,
              context: draft.context,
              ebGeneration: bank.generation(),
            });
            if (put.ok) added += 1;
            else if (put.reason === "quota") deferredDirections.add(draft.direction);
          }
          if (
            added === 0 &&
            !stopRequested &&
            queue.waitingCount() === 0 &&
            jobs.size === 0
          ) {
            stopReason = "context-idle";
          }
        } finally {
          live.contextRunning = false;
          live.lastContextAt = now();
          gates.finishContext();
          await persist();
        }
        progressed = true;
        if (stopReason) break;
      }

      while (
        !stopRequested &&
        remainingWork() > 0 &&
        queue.waitingCount() > 0 &&
        gates.tryStartProposal()
      ) {
        const insp = await queue.claim();
        if (!insp) {
          gates.finishProposal();
          break;
        }
        track(runOne(insp));
        progressed = true;
      }

      if (jobs.size > 0) {
        await Promise.race(jobs);
        continue;
      }

      if (!progressed) {
        stopReason = stopRequested ? "context-stop" : "context-idle";
      }
    }
  } finally {
    live.phase = "stopped";
    live.stopReason = stopReason;
    await persist();
  }

  return {
    best: await bank.best(),
    records: await bank.list(),
    stopReason: stopReason ?? "budget",
  };
}

async function readLive(runDir: string): Promise<LiveState | undefined> {
  try {
    return JSON.parse(await readFile(path.join(runDir, "live.json"), "utf8")) as LiveState;
  } catch {
    return undefined;
  }
}
