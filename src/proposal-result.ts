import { access, readdir, readFile } from "node:fs/promises";
import path from "node:path";

const PLACEHOLDER = /\bTODO\b|\bFIXME\b|\bNotImplementedError\b|待实现/;
const SOURCE = /\.(sh|py|r|js|ts|jl)$/;

export async function inspectSolution(workDir: string): Promise<string | undefined> {
  const solve = path.join(workDir, "solve.sh");
  try {
    await access(solve);
  } catch {
    return "missing solve.sh";
  }
  const solveText = await readFile(solve, "utf8");
  if (!hasRunnableCommand(solveText)) return "solve.sh has no commands";
  const texts = await collectSource(workDir);
  if (texts.some((text) => PLACEHOLDER.test(text))) return "solution still has placeholders";
  return undefined;
}

export async function finishProposalWrite(
  workDir: string,
  err?: unknown,
): Promise<{ solutionDir: string } | { error: string }> {
  const inspection = await inspectSolution(workDir);
  if (inspection) {
    if (!err) return { error: inspection };
    const message = err instanceof Error ? err.message : String(err);
    return { error: message.includes("timed out") ? `${message}; ${inspection}` : message };
  }
  if (!err) return { solutionDir: workDir };
  const message = err instanceof Error ? err.message : String(err);
  if (message.includes("timed out")) return { solutionDir: workDir };
  return { error: message };
}

function hasRunnableCommand(text: string): boolean {
  return text.split("\n").some((line) => {
    const trimmed = line.trim();
    return trimmed.length > 0 && !trimmed.startsWith("#");
  });
}

async function collectSource(dir: string): Promise<string[]> {
  const out: string[] = [];
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const next = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "node_modules" || entry.name === ".git") continue;
      out.push(...(await collectSource(next)));
      continue;
    }
    if (entry.name === "solve.sh" || SOURCE.test(entry.name)) {
      out.push(await readFile(next, "utf8"));
    }
  }
  return out;
}
