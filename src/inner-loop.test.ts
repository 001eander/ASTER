import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { ExperienceBank } from "./experience-bank.js";
import { runInnerLoop, type ContextPort, type ProposalPort, type SandboxPort } from "./inner-loop.js";

async function runDir(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), "hyra-pi-loop-"));
}

function writingProposal(): ProposalPort {
  return {
    async write({ workDir }) {
      await mkdir(workDir, { recursive: true });
      await writeFile(path.join(workDir, "solve.sh"), "#!/bin/sh\necho ok\n", "utf8");
      return { solutionDir: workDir };
    },
  };
}

function scriptedContext(turns: Array<{ directions?: string[]; stop?: boolean }>): ContextPort {
  let i = 0;
  return {
    async produce() {
      const turn = turns[Math.min(i, turns.length - 1)]!;
      i += 1;
      return {
        stop: turn.stop,
        inspirations: (turn.directions ?? []).map((direction) => ({
          direction,
          context: `try ${direction}`,
        })),
      };
    },
  };
}

function scriptedSandbox(
  results: Array<{ ok: boolean; score: number | null; log: string }>,
): SandboxPort {
  let i = 0;
  return {
    async evaluate() {
      const row = results[Math.min(i, results.length - 1)]!;
      i += 1;
      return {
        ok: row.ok,
        log: row.log,
        score:
          row.score === null
            ? null
            : { score: row.score, higherIsBetter: true, notes: row.log },
      };
    },
  };
}

describe("inner loop", () => {
  it("writes a scored solution into the experience bank and returns it as best", async () => {
    const root = await runDir();
    const result = await runInnerLoop({
      runDir: root,
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 2,
      budget: { maxSolutions: 1 },
      context: scriptedContext([{ directions: ["first-idea"] }]),
      proposal: writingProposal(),
      sandbox: scriptedSandbox([{ ok: true, score: 6, log: "score 6" }]),
    });

    expect(result.stopReason).toBe("budget");
    expect(result.best?.score?.score).toBe(6);
    expect(result.records).toHaveLength(1);
    expect(result.records[0]?.inspirationId).toBeDefined();
    expect(result.records[0]?.ok).toBe(true);
  });

  it("records a failed and a timed-out evaluation instead of dropping them", async () => {
    const root = await runDir();
    const result = await runInnerLoop({
      runDir: root,
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 3,
      budget: { maxSolutions: 2 },
      context: scriptedContext([{ directions: ["a", "b"] }]),
      proposal: writingProposal(),
      sandbox: scriptedSandbox([
        { ok: false, score: null, log: "eval crashed" },
        { ok: false, score: null, log: "timeout" },
      ]),
    });

    expect(result.records).toHaveLength(2);
    expect(result.records.map((row) => row.ok)).toEqual([false, false]);
    expect(result.records.map((row) => row.score)).toEqual([null, null]);
    expect(result.best).toBeUndefined();
    expect(result.records.map((row) => row.logPath).every(Boolean)).toBe(true);
  });

  it("stops when Context asks to stop and still returns the best so far", async () => {
    const root = await runDir();
    const result = await runInnerLoop({
      runDir: root,
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 2,
      budget: { maxSolutions: 99 },
      context: scriptedContext([
        { directions: ["keep"] },
        { directions: [], stop: true },
      ]),
      proposal: writingProposal(),
      sandbox: scriptedSandbox([{ ok: true, score: 3, log: "ok" }]),
    });

    expect(result.stopReason).toBe("context-stop");
    expect(result.best?.score?.score).toBe(3);
  });

  it("rewrites the same inspiration after a crash and commits only the final result", async () => {
    const root = await runDir();
    const errors: Array<string | undefined> = [];
    const proposal: ProposalPort = {
      async write({ workDir, lastError }) {
        errors.push(lastError);
        await mkdir(workDir, { recursive: true });
        await writeFile(path.join(workDir, "solve.sh"), "#!/bin/sh\necho ok\n", "utf8");
        return { solutionDir: workDir };
      },
    };

    const result = await runInnerLoop({
      runDir: root,
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 2,
      budget: { maxSolutions: 1 },
      proposalRewrites: 2,
      context: scriptedContext([{ directions: ["lgbm"] }]),
      proposal,
      sandbox: scriptedSandbox([
        { ok: false, score: null, log: "TypeError: unexpected keyword argument 'verbose'" },
        { ok: true, score: 7, log: "score 7" },
      ]),
    });

    expect(errors).toEqual([undefined, "TypeError: unexpected keyword argument 'verbose'"]);
    expect(result.records).toHaveLength(1);
    expect(result.records[0]?.ok).toBe(true);
    expect(result.best?.score?.score).toBe(7);
  });

  it("does not rewrite a scored solution just because the score is low", async () => {
    const root = await runDir();
    let writes = 0;
    const proposal: ProposalPort = {
      async write({ workDir }) {
        writes += 1;
        await mkdir(workDir, { recursive: true });
        await writeFile(path.join(workDir, "solve.sh"), "#!/bin/sh\necho ok\n", "utf8");
        return { solutionDir: workDir };
      },
    };

    const result = await runInnerLoop({
      runDir: root,
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 2,
      budget: { maxSolutions: 1 },
      proposalRewrites: 2,
      context: scriptedContext([{ directions: ["fallback"] }]),
      proposal,
      sandbox: scriptedSandbox([{ ok: true, score: 0.5, log: "constant" }]),
    });

    expect(writes).toBe(1);
    expect(result.records).toHaveLength(1);
    expect(result.best?.score?.score).toBe(0.5);
  });

  it("resumes a crashed run: scored work stays, orphaned holds run again", async () => {
    const root = await runDir();
    await seedCrashedRun(root, { consumedMs: 5_000, startedAt: 1_000 });
    const evaluated: string[] = [];
    const result = await runInnerLoop({
      runDir: root,
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 6,
      budget: { maxSolutions: 3 },
      context: scriptedContext([{ directions: [], stop: false }]),
      proposal: writingProposal(),
      sandbox: {
        async evaluate(solutionDir) {
          evaluated.push(path.basename(solutionDir));
          return {
            ok: true,
            log: "resumed",
            score: { score: 2, higherIsBetter: true, notes: "resumed" },
          };
        },
      },
    });

    expect(evaluated.sort()).toEqual(["insp-002", "insp-003"]);
    expect(result.records.map((row) => row.inspirationId).sort()).toEqual([
      "insp-001",
      "insp-002",
      "insp-003",
    ]);
    expect(result.records.find((row) => row.inspirationId === "insp-001")?.score?.score).toBe(4);
    const live = JSON.parse(await readFile(path.join(root, "live.json"), "utf8")) as {
      consumedMs: number;
      writers: unknown[];
      sandboxes: unknown[];
      stopReason?: string;
    };
    expect(live.consumedMs).toBeGreaterThanOrEqual(5_000);
    expect(live.writers).toEqual([]);
    expect(live.sandboxes).toEqual([]);
  });

  it("stops immediately on resume when the remaining clock is already used up", async () => {
    const root = await runDir();
    await seedCrashedRun(root, { consumedMs: 2_000, startedAt: 1_000 });
    const evaluated: string[] = [];
    const result = await runInnerLoop({
      runDir: root,
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 6,
      budget: { maxSolutions: 10, maxMs: 2_000 },
      now: () => 10_000,
      context: scriptedContext([{ directions: ["should-not-run"] }]),
      proposal: writingProposal(),
      sandbox: {
        async evaluate(solutionDir) {
          evaluated.push(path.basename(solutionDir));
          return { ok: true, log: "no", score: { score: 1, higherIsBetter: true, notes: "" } };
        },
      },
    });

    expect(evaluated).toEqual([]);
    expect(result.records).toHaveLength(1);
    expect(result.stopReason === "budget" || result.stopReason === "context-stop").toBe(true);
  });

  it("keeps dispatching after eight scored solutions when the budget has no caps", async () => {
    const root = await runDir();
    const incoming = path.join(root, "incoming");
    await mkdir(incoming, { recursive: true });
    await writeFile(path.join(incoming, "solve.sh"), "#!/bin/sh\necho seed\n", "utf8");
    const bank = await ExperienceBank.open(root);
    for (let i = 1; i <= 8; i += 1) {
      await bank.commit({
        inspirationId: `insp-${String(i).padStart(3, "0")}`,
        solutionDir: incoming,
        ok: true,
        log: "seed",
        score: { score: 1, higherIsBetter: true, notes: "seed" },
      });
    }
    const evaluated: string[] = [];
    const result = await runInnerLoop({
      runDir: root,
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 4,
      budget: {},
      context: scriptedContext([{ directions: ["ninth"] }, { directions: [], stop: true }]),
      proposal: writingProposal(),
      sandbox: {
        async evaluate(solutionDir) {
          evaluated.push(path.basename(solutionDir));
          return { ok: true, log: "ninth", score: { score: 2, higherIsBetter: true, notes: "" } };
        },
      },
    });

    expect(evaluated).toEqual(["insp-009"]);
    expect(result.records).toHaveLength(9);
    expect(result.stopReason).toBe("context-stop");
  });
});

async function seedCrashedRun(root: string, live: { consumedMs: number; startedAt: number }): Promise<void> {
  const incoming = path.join(root, "incoming");
  await mkdir(incoming, { recursive: true });
  await writeFile(path.join(incoming, "solve.sh"), "#!/bin/sh\necho seed\n", "utf8");
  const bank = await ExperienceBank.open(root);
  await bank.commit({
    inspirationId: "insp-001",
    solutionDir: incoming,
    ok: true,
    log: "scored",
    score: { score: 4, higherIsBetter: true, notes: "seed" },
  });
  await mkdir(path.join(root, "queue"), { recursive: true });
  await writeFile(
    path.join(root, "queue", "insp-001.json"),
    JSON.stringify({
      id: "insp-001",
      direction: "done-dir",
      context: "done",
      ebGeneration: 0,
      state: "held",
    }),
    "utf8",
  );
  await writeFile(
    path.join(root, "queue", "insp-002.json"),
    JSON.stringify({
      id: "insp-002",
      direction: "orphan-dir",
      context: "orphan",
      ebGeneration: 0,
      state: "held",
    }),
    "utf8",
  );
  await writeFile(
    path.join(root, "queue", "insp-003.json"),
    JSON.stringify({
      id: "insp-003",
      direction: "waiting-dir",
      context: "wait",
      ebGeneration: 0,
      state: "waiting",
    }),
    "utf8",
  );
  await writeFile(
    path.join(root, "live.json"),
    JSON.stringify({
      phase: "stopped",
      stopReason: "context-stop",
      writers: [{ id: "insp-002", direction: "orphan-dir" }],
      sandboxes: [{ id: "insp-002", direction: "orphan-dir" }],
      contextRunning: true,
      startedAt: live.startedAt,
      consumedMs: live.consumedMs,
    }),
    "utf8",
  );
}
