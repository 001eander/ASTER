import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { readJsonFile, writeJsonFile } from "./json-file.js";

describe("json-file", () => {
  it("reads a file that is briefly two JSON objects stuck together", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "hyra-pi-json-"));
    const file = path.join(dir, "live.json");
    await writeFile(file, '{\n  "phase": "running"\n}\n{\n  "phase": "running"\n}\n', "utf8");
    const pending = readJsonFile<{ phase: string }>(file);
    setTimeout(() => {
      void writeJsonFile(file, { phase: "running" });
    }, 20);
    await expect(pending).resolves.toEqual({ phase: "running" });
  });

  it("replaces a longer file instead of leaving leftover JSON behind", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "hyra-pi-json-"));
    const file = path.join(dir, "live.json");
    await writeJsonFile(file, { phase: "running", writers: [{ id: "insp-001", direction: "long" }] });
    await writeJsonFile(file, { phase: "stopped" });
    await expect(readJsonFile(file)).resolves.toEqual({ phase: "stopped" });
  });

  it("retries rename when the destination is transiently locked", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "hyra-pi-json-"));
    const file = path.join(dir, "live.json");
    await writeFile(file, "{}\n", "utf8");
    // Windows 上并发读持有不带 delete 共享的句柄会让 rename 抛 EPERM。
    // 用持续读制造真实锁窗口：读取期间写必须经重试最终成功。
    const stop = { done: false };
    const reader = (async () => {
      while (!stop.done) {
        try {
          await readFile(file, "utf8");
        } catch {
          // 文件被替换的瞬间可能读不到，忽略继续。
        }
      }
    })();
    await writeJsonFile(file, { phase: "running" });
    stop.done = true;
    await reader;
    await expect(readJsonFile<{ phase: string }>(file)).resolves.toEqual({ phase: "running" });
  });
});
