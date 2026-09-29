import { access } from "node:fs/promises";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { CONTEXT7_TOOLS, context7PiRoot } from "./context7-pi.js";

describe("context7-pi", () => {
  it("points at the official Pi extension that registers the Context7 doc tools", async () => {
    const root = context7PiRoot();
    await access(path.join(root, "extensions", "context7.ts"));
    await access(path.join(root, "skills", "context7-docs", "SKILL.md"));
    expect(CONTEXT7_TOOLS).toEqual(["resolve-library-id", "query-docs"]);
  });
});
