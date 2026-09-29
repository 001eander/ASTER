import { spawn } from "node:child_process";
import { appendFile, cp, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { SandboxPort } from "./inner-loop.js";

export type LogStream = "stdout" | "stderr";

export type DockerRun = (
  args: string[],
  opts: { workDir: string; timeoutMs?: number; onChunk?: (chunk: string, stream?: LogStream) => void },
) => Promise<{ code: number; stdout: string; stderr: string }>;

export function createDockerSandbox(opts: {
  taskDir: string;
  image: string;
  timeoutMs?: number;
  /** 额外挂载，单条同 docker -v 语法（host:container[:ro]），如数据只读挂载。 */
  extraMounts?: string[];
  docker?: DockerRun;
}): SandboxPort {
  const docker = opts.docker ?? runDocker;
  return {
    async evaluate(solutionDir) {
      const workDir = await mkdtemp(path.join(tmpdir(), "hyra-pi-work-"));
      await cp(solutionDir, path.join(workDir, "solution"), { recursive: true });
      await cp(opts.taskDir, path.join(workDir, "task"), { recursive: true });
      await cp(path.join(opts.taskDir, "eval.sh"), path.join(workDir, "eval.sh"));

      const mountArgs: string[] = [];
      for (const mount of opts.extraMounts ?? []) {
        mountArgs.push("-v", mount);
      }

      const evalLog = path.join(solutionDir, "eval.log");
      const evalOut = path.join(solutionDir, "eval.stdout");
      const evalErr = path.join(solutionDir, "eval.stderr");
      let streamed = false;
      const onChunk = (chunk: string, stream: LogStream = "stdout") => {
        streamed = true;
        void appendFile(evalLog, chunk, "utf8");
        void appendFile(stream === "stderr" ? evalErr : evalOut, chunk, "utf8");
      };

      let ran;
      try {
        ran = await docker(
          [
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            `${workDir}:/work`,
            ...mountArgs,
            "-w",
            "/work",
            opts.image,
            "bash",
            "-lc",
            "bash /work/eval.sh /work/solution",
          ],
          { workDir, timeoutMs: opts.timeoutMs, onChunk },
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        const log = message.includes("Docker daemon")
          ? `${message}\n请先启动 Docker`
          : message;
        if (!streamed) {
          await writeFile(evalLog, log, "utf8");
          await writeFile(evalOut, "", "utf8");
          await writeFile(evalErr, log, "utf8");
        }
        return {
          ok: false,
          score: null,
          log,
        };
      }

      const log = `${ran.stdout}${ran.stderr}`;
      if (!streamed) {
        await writeFile(evalLog, log, "utf8");
        await writeFile(evalOut, ran.stdout, "utf8");
        await writeFile(evalErr, ran.stderr, "utf8");
      }
      if (ran.code !== 0) {
        return { ok: false, score: null, log };
      }
      try {
        const raw = JSON.parse(await readFile(path.join(workDir, "score.json"), "utf8")) as {
          score: number;
          higher_is_better?: boolean;
          higherIsBetter?: boolean;
          notes?: string;
        };
        return {
          ok: true,
          log,
          score: {
            score: raw.score,
            higherIsBetter: raw.higherIsBetter ?? raw.higher_is_better ?? true,
            notes: raw.notes ?? "",
          },
        };
      } catch {
        return { ok: false, score: null, log: `${log}\nmissing score.json` };
      }
    },
  };
}

function runDocker(
  args: string[],
  opts: { workDir: string; timeoutMs?: number; onChunk?: (chunk: string, stream?: LogStream) => void },
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn("docker", args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      const text = String(chunk);
      stdout += text;
      opts.onChunk?.(text, "stdout");
    });
    child.stderr.on("data", (chunk) => {
      const text = String(chunk);
      stderr += text;
      opts.onChunk?.(text, "stderr");
    });
    const timer =
      opts.timeoutMs === undefined
        ? undefined
        : setTimeout(() => {
            child.kill("SIGKILL");
            reject(new Error(`sandbox timed out after ${opts.timeoutMs}ms`));
          }, opts.timeoutMs);
    child.on("error", (err) => {
      if (timer) clearTimeout(timer);
      reject(err);
    });
    child.on("close", (code) => {
      if (timer) clearTimeout(timer);
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}
