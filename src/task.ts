import { access, readFile } from "node:fs/promises";
import path from "node:path";

export type Task = {
  dir: string;
  description: string;
  evalPath: string;
};

export async function loadTask(dir: string): Promise<Task> {
  const descriptionPath = path.join(dir, "TASK.md");
  const evalPath = path.join(dir, "eval.sh");
  try {
    await access(descriptionPath);
  } catch {
    throw new Error(`题目目录缺少 TASK.md: ${dir}`);
  }
  try {
    await access(evalPath);
  } catch {
    throw new Error(`题目目录缺少 eval.sh: ${dir}`);
  }
  return {
    dir,
    description: await readFile(descriptionPath, "utf8"),
    evalPath,
  };
}
