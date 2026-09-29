import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { createDockerSandbox, type DockerRun } from "./sandbox.js";

async function tmp(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), "hyra-pi-sb-"));
}

describe("docker sandbox", () => {
  it("runs solve then eval and reads score.json", async () => {
    const root = await tmp();
    const taskDir = path.join(root, "task");
    const solutionDir = path.join(root, "solution");
    await mkdir(taskDir, { recursive: true });
    await mkdir(solutionDir, { recursive: true });
    await writeFile(path.join(taskDir, "eval.sh"), "#!/bin/sh\n", "utf8");
    await writeFile(path.join(solutionDir, "solve.sh"), "#!/bin/sh\necho hi\n", "utf8");

    const docker: DockerRun = async (_args, opts) => {
      await writeFile(
        path.join(opts.workDir, "score.json"),
        JSON.stringify({ score: 11, higher_is_better: true, notes: "correct" }),
      );
      return { code: 0, stdout: "ok\n", stderr: "" };
    };

    const sandbox = createDockerSandbox({
      taskDir,
      image: "hyra-pi-eval:test",
      timeoutMs: 5000,
      docker,
    });
    const result = await sandbox.evaluate(solutionDir);
    expect(result.ok).toBe(true);
    expect(result.score).toEqual({ score: 11, higherIsBetter: true, notes: "correct" });
    expect(result.log).toContain("ok");
    expect(await readFile(path.join(solutionDir, "eval.log"), "utf8")).toContain("ok");
    expect(await readFile(path.join(solutionDir, "eval.stdout"), "utf8")).toBe("ok\n");
    expect(await readFile(path.join(solutionDir, "eval.stderr"), "utf8")).toBe("");
  });

  it("streams stdout and stderr into separate files", async () => {
    const root = await tmp();
    const taskDir = path.join(root, "task");
    const solutionDir = path.join(root, "solution");
    await mkdir(taskDir, { recursive: true });
    await mkdir(solutionDir, { recursive: true });
    await writeFile(path.join(taskDir, "eval.sh"), "#!/bin/sh\n", "utf8");
    await writeFile(path.join(solutionDir, "solve.sh"), "#!/bin/sh\n", "utf8");

    const docker: DockerRun = async (_args, opts) => {
      opts.onChunk?.("fold 1\n", "stdout");
      opts.onChunk?.("warn\n", "stderr");
      await writeFile(path.join(opts.workDir, "score.json"), JSON.stringify({ score: 2, higher_is_better: true }));
      return { code: 0, stdout: "fold 1\n", stderr: "warn\n" };
    };

    await createDockerSandbox({ taskDir, image: "hyra-pi-eval:test", timeoutMs: 5000, docker }).evaluate(solutionDir);
    expect(await readFile(path.join(solutionDir, "eval.stdout"), "utf8")).toBe("fold 1\n");
    expect(await readFile(path.join(solutionDir, "eval.stderr"), "utf8")).toBe("warn\n");
    expect(await readFile(path.join(solutionDir, "eval.log"), "utf8")).toContain("fold 1");
  });

  it("keeps a failed eval as a failed result instead of dropping it", async () => {
    const root = await tmp();
    const taskDir = path.join(root, "task");
    const solutionDir = path.join(root, "solution");
    await mkdir(taskDir, { recursive: true });
    await mkdir(solutionDir, { recursive: true });
    await writeFile(path.join(taskDir, "eval.sh"), "#!/bin/sh\n", "utf8");
    await writeFile(path.join(solutionDir, "solve.sh"), "#!/bin/sh\n", "utf8");

    const sandbox = createDockerSandbox({
      taskDir,
      image: "hyra-pi-eval:test",
      timeoutMs: 5000,
      docker: async () => ({ code: 2, stdout: "", stderr: "eval crashed\n" }),
    });
    const result = await sandbox.evaluate(solutionDir);
    expect(result.ok).toBe(false);
    expect(result.score).toBeNull();
    expect(result.log).toContain("eval crashed");
  });

  it("tells the operator to start Docker when the daemon is down", async () => {
    const root = await tmp();
    const taskDir = path.join(root, "task");
    const solutionDir = path.join(root, "solution");
    await mkdir(taskDir, { recursive: true });
    await mkdir(solutionDir, { recursive: true });
    await writeFile(path.join(taskDir, "eval.sh"), "#!/bin/sh\n", "utf8");
    await writeFile(path.join(solutionDir, "solve.sh"), "#!/bin/sh\n", "utf8");

    const sandbox = createDockerSandbox({
      taskDir,
      image: "hyra-pi-eval:test",
      timeoutMs: 5000,
      docker: async () => {
        throw new Error("Cannot connect to the Docker daemon");
      },
    });
    const result = await sandbox.evaluate(solutionDir);
    expect(result.ok).toBe(false);
    expect(result.log).toContain("请先启动 Docker");
  });
});
