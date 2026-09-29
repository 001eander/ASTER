import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { loadTask } from "./task.js";

async function dir(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), "hyra-pi-task-"));
}

describe("loadTask", () => {
  it("loads a task that has a description and a scoring script", async () => {
    const root = await dir();
    await writeFile(path.join(root, "TASK.md"), "Sort the numbers.\n", "utf8");
    await writeFile(path.join(root, "eval.sh"), "#!/bin/sh\necho ok\n", "utf8");
    const task = await loadTask(root);
    expect(task.dir).toBe(root);
    expect(task.description).toBe("Sort the numbers.\n");
    expect(task.evalPath).toBe(path.join(root, "eval.sh"));
  });

  it("rejects a task that has no scoring script", async () => {
    const root = await dir();
    await writeFile(path.join(root, "TASK.md"), "No scorer.\n", "utf8");
    await expect(loadTask(root)).rejects.toThrow("eval.sh");
  });

  it("rejects a task that has no description", async () => {
    const root = await dir();
    await mkdir(root, { recursive: true });
    await writeFile(path.join(root, "eval.sh"), "#!/bin/sh\n", "utf8");
    await expect(loadTask(root)).rejects.toThrow("TASK.md");
  });
});
