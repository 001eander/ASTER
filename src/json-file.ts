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
  // Windows 上 rename 需要目标句柄允许 delete 共享；状态页轮询 live.json 的读句柄
  // 不带 FILE_SHARE_DELETE，读取与 rename 撞上时抛 EPERM。读是瞬时的，退避重试即可。
  let last: unknown;
  for (let i = 0; i < 8; i += 1) {
    try {
      await rename(tmp, file);
      return;
    } catch (err) {
      last = err;
      const code = (err as NodeJS.ErrnoException).code;
      if (!["EPERM", "EBUSY", "EACCES"].includes(code ?? "") || i === 7) throw err;
      await delay(15 * (i + 1));
    }
  }
  throw last;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
