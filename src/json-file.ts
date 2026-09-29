import { readFile, rename, writeFile } from "node:fs/promises";

export async function readJsonFile<T>(file: string, attempts = 8): Promise<T> {
  let last: unknown;
  for (let i = 0; i < attempts; i += 1) {
    const text = await readFile(file, "utf8");
    try {
      return JSON.parse(text) as T;
    } catch (err) {
      last = err;
      if (!(err instanceof SyntaxError) || i === attempts - 1) throw err;
      await delay(15 * (i + 1));
    }
  }
  throw last;
}

export async function writeJsonFile(file: string, value: unknown): Promise<void> {
  const tmp = `${file}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`;
  await writeFile(tmp, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(tmp, file);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
