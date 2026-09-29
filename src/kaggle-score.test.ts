import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  loadKaggleConfig,
  parseSubmissions,
  pickSubmission,
  scoreOnKaggle,
} from "./kaggle-score.js";

describe("parseSubmissions", () => {
  it("reads a completed public score from JSON", () => {
    const rows = parseSubmissions(
      JSON.stringify([
        {
          description: "hyra-pi 1",
          status: "complete",
          publicScore: "0.94105",
        },
      ]),
    );
    expect(rows).toEqual([{ description: "hyra-pi 1", status: "complete", publicScore: 0.94105 }]);
  });

  it("treats an empty listing as no rows", () => {
    expect(parseSubmissions("No submissions found")).toEqual([]);
  });
});

describe("pickSubmission", () => {
  it("prefers the row whose message matches", () => {
    const row = pickSubmission(
      [
        { description: "other", status: "complete", publicScore: 0.1 },
        { description: "hyra-pi 9", status: "complete", publicScore: 0.94 },
      ],
      "hyra-pi 9",
    );
    expect(row?.publicScore).toBe(0.94);
  });
});

describe("loadKaggleConfig", () => {
  it("returns null when the task is not a Kaggle task", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "hyra-pi-kg-"));
    await expect(loadKaggleConfig(dir)).resolves.toBeNull();
  });

  it("reads competition and submission file from kaggle.json", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "hyra-pi-kg-"));
    await writeFile(
      path.join(dir, "kaggle.json"),
      JSON.stringify({ competition: "playground-series-s6e9", submission: "submission.csv" }),
    );
    await expect(loadKaggleConfig(dir)).resolves.toEqual({
      competition: "playground-series-s6e9",
      submission: "submission.csv",
      higherIsBetter: true,
      folds: 5,
      minSaveScore: 0.9,
      targetScore: 0.95,
    });
  });
});

describe("scoreOnKaggle", () => {
  it("submits then uses the matching public score", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "hyra-pi-kg-"));
    await writeFile(path.join(dir, "submission.csv"), "id,Will_Buy_EV\n1,0.2\n", "utf8");
    const calls: string[][] = [];
    const result = await scoreOnKaggle({
      workDir: dir,
      config: { competition: "playground-series-s6e9", submission: "submission.csv", higherIsBetter: true },
      message: "hyra-pi 9",
      sleep: async () => undefined,
      cli: async (args) => {
        calls.push(args);
        if (args[1] === "submit") return { code: 0, stdout: "ok\n", stderr: "" };
        return {
          code: 0,
          stdout: JSON.stringify([
            { description: "hyra-pi 9", status: "complete", publicScore: "0.94601" },
          ]),
          stderr: "",
        };
      },
    });
    expect(result.ok).toBe(true);
    expect(result.score).toEqual({
      score: 0.94601,
      higherIsBetter: true,
      notes: "kaggle public 0.94601",
    });
    expect(calls[0]?.slice(0, 4)).toEqual(["competitions", "submit", "-c", "playground-series-s6e9"]);
  });

  it("keeps a quota or submit error as a failed eval", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "hyra-pi-kg-"));
    await mkdir(dir, { recursive: true });
    await writeFile(path.join(dir, "submission.csv"), "id,Will_Buy_EV\n1,0.2\n", "utf8");
    const result = await scoreOnKaggle({
      workDir: dir,
      config: { competition: "playground-series-s6e9", submission: "submission.csv", higherIsBetter: true },
      message: "hyra-pi 9",
      cli: async () => ({ code: 1, stdout: "", stderr: "429 daily submission limit\n" }),
    });
    expect(result.ok).toBe(false);
    expect(result.score).toBeNull();
    expect(result.log).toContain("daily submission limit");
  });
});
