import { mkdtemp, writeFile } from "node:fs/promises";
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
});
