import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { ExperienceBank } from "./experience-bank.js";

async function emptyRunDir(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), "hyra-pi-eb-"));
}

async function writeSolution(root: string, body: string): Promise<string> {
  const solutionDir = path.join(root, "incoming-solution");
  await mkdir(solutionDir, { recursive: true });
  await writeFile(path.join(solutionDir, "solve.sh"), body, "utf8");
  return solutionDir;
}

describe("ExperienceBank", () => {
  it("stores a scored solution so it can be read back with the same files", async () => {
    const runDir = await emptyRunDir();
    const bank = await ExperienceBank.open(runDir);
    const solutionDir = await writeSolution(runDir, "#!/bin/sh\necho unique-payload-17\n");

    const stored = await bank.commit({
      inspirationId: "insp-a",
      solutionDir,
      ok: true,
      log: "ran solve.sh\nscore 4.5\n",
      score: { score: 4.5, higherIsBetter: true, notes: "fixed cases passed" },
    });

    const loaded = await bank.get(stored.id);
    expect(loaded?.inspirationId).toBe("insp-a");
    expect(loaded?.ok).toBe(true);
    expect(loaded?.score).toEqual({
      score: 4.5,
      higherIsBetter: true,
      notes: "fixed cases passed",
    });
    expect(await readFile(path.join(loaded!.solutionDir, "solve.sh"), "utf8")).toBe(
      "#!/bin/sh\necho unique-payload-17\n",
    );
    expect(await readFile(loaded!.logPath, "utf8")).toBe("ran solve.sh\nscore 4.5\n");
  });

  it("keeps the higher score as best when higher is better", async () => {
    const runDir = await emptyRunDir();
    const bank = await ExperienceBank.open(runDir);
    const first = await writeSolution(path.join(runDir, "a"), "#!/bin/sh\necho first\n");
    const second = await writeSolution(path.join(runDir, "b"), "#!/bin/sh\necho second\n");
    const third = await writeSolution(path.join(runDir, "c"), "#!/bin/sh\necho third\n");

    await bank.commit({
      inspirationId: "one",
      solutionDir: first,
      ok: true,
      log: "",
      score: { score: 10, higherIsBetter: true, notes: "" },
    });
    await bank.commit({
      inspirationId: "two",
      solutionDir: second,
      ok: true,
      log: "",
      score: { score: 7, higherIsBetter: true, notes: "" },
    });
    const winner = await bank.commit({
      inspirationId: "three",
      solutionDir: third,
      ok: true,
      log: "",
      score: { score: 12, higherIsBetter: true, notes: "" },
    });

    const best = await bank.best();
    expect(best?.id).toBe(winner.id);
    expect(best?.score?.score).toBe(12);
    expect(await readFile(path.join(runDir, "best", "solve.sh"), "utf8")).toBe(
      "#!/bin/sh\necho third\n",
    );
  });

  it("keeps the lower score as best when lower is better", async () => {
    const runDir = await emptyRunDir();
    const bank = await ExperienceBank.open(runDir);

    await bank.commit({
      inspirationId: "one",
      solutionDir: await writeSolution(path.join(runDir, "a"), "a\n"),
      ok: true,
      log: "",
      score: { score: 5, higherIsBetter: false, notes: "" },
    });
    await bank.commit({
      inspirationId: "two",
      solutionDir: await writeSolution(path.join(runDir, "b"), "b\n"),
      ok: true,
      log: "",
      score: { score: 8, higherIsBetter: false, notes: "" },
    });
    const winner = await bank.commit({
      inspirationId: "three",
      solutionDir: await writeSolution(path.join(runDir, "c"), "c\n"),
      ok: true,
      log: "",
      score: { score: 3, higherIsBetter: false, notes: "" },
    });

    const best = await bank.best();
    expect(best?.id).toBe(winner.id);
    expect(best?.score?.score).toBe(3);
  });

  it("does not let a failed evaluation become the best solution", async () => {
    const runDir = await emptyRunDir();
    const bank = await ExperienceBank.open(runDir);

    const ok = await bank.commit({
      inspirationId: "ok",
      solutionDir: await writeSolution(path.join(runDir, "ok"), "ok\n"),
      ok: true,
      log: "",
      score: { score: 2, higherIsBetter: true, notes: "" },
    });
    await bank.commit({
      inspirationId: "fail",
      solutionDir: await writeSolution(path.join(runDir, "fail"), "fail\n"),
      ok: false,
      log: "eval crashed",
      score: { score: 99, higherIsBetter: true, notes: "ignored" },
    });

    expect((await bank.best())?.id).toBe(ok.id);
  });

  it("bumps the bank version by one after each stored result", async () => {
    const runDir = await emptyRunDir();
    const bank = await ExperienceBank.open(runDir);
    expect(bank.generation()).toBe(0);

    await bank.commit({
      inspirationId: "n1",
      solutionDir: await writeSolution(path.join(runDir, "n1"), "n1\n"),
      ok: true,
      log: "",
      score: { score: 1, higherIsBetter: true, notes: "" },
    });
    expect(bank.generation()).toBe(1);

    await bank.commit({
      inspirationId: "n2",
      solutionDir: await writeSolution(path.join(runDir, "n2"), "n2\n"),
      ok: false,
      log: "timeout",
      score: null,
    });
    expect(bank.generation()).toBe(2);
    expect((await bank.list()).map((row) => row.inspirationId)).toEqual(["n1", "n2"]);
  });
});
