import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { runInnerLoop, type ContextPort, type ProposalPort, type SandboxPort } from "./inner-loop.js";

async function runDir(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), "hyra-pi-qlt-"));
}

const proposal: ProposalPort = {
  async write({ workDir, inspiration }) {
    await mkdir(workDir, { recursive: true });
    await writeFile(path.join(workDir, "solve.sh"), `#!/bin/sh\n# ${inspiration.context}\n`, "utf8");
    return { solutionDir: workDir };
  },
};

const sandboxFromAim: SandboxPort = {
  async evaluate(solutionDir) {
    const text = await (await import("node:fs/promises")).readFile(
      path.join(solutionDir, "solve.sh"),
      "utf8",
    );
    const match = /aim (\d+)/.exec(text);
    const score = match ? Number(match[1]) : 0;
    return {
      ok: true,
      log: `score ${score}`,
      score: { score, higherIsBetter: true, notes: "" },
    };
  },
};

describe("quality loop", () => {
  it("shows Context the new bank version after a result is stored", async () => {
    const seen: number[] = [];
    const context: ContextPort = {
      async produce({ generation }) {
        seen.push(generation);
        if (generation >= 2) return { stop: true };
        return {
          inspirations: [{ direction: `g${generation}`, context: `aim ${generation + 1}` }],
        };
      },
    };

    await runInnerLoop({
      runDir: await runDir(),
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 3,
      budget: { maxSolutions: 4 },
      context,
      proposal,
      sandbox: sandboxFromAim,
    });

    expect(seen[0]).toBe(0);
    expect(seen).toContain(1);
  });

  it("raises the best score when Context aims one point above the current best", async () => {
    const context: ContextPort = {
      async produce({ best }) {
        const current = best?.score?.score ?? 0;
        return {
          inspirations: [{ direction: `nudge ${current + 1}`, context: `aim ${current + 1}` }],
        };
      },
    };

    const result = await runInnerLoop({
      runDir: await runDir(),
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 2,
      budget: { maxSolutions: 4 },
      context,
      proposal,
      sandbox: sandboxFromAim,
    });

    expect(result.best?.score?.score).toBe(4);
  });

  it("does not raise the best score when Context keeps aiming at 1", async () => {
    const context: ContextPort = {
      async produce() {
        return {
          inspirations: [{ direction: "frozen", context: "aim 1" }],
        };
      },
    };

    const result = await runInnerLoop({
      runDir: await runDir(),
      maxProposals: 1,
      maxSandboxes: 1,
      lowWater: 0,
      highWater: 2,
      budget: { maxSolutions: 4 },
      context,
      proposal,
      sandbox: sandboxFromAim,
    });

    expect(result.best?.score?.score).toBe(1);
  });
});
