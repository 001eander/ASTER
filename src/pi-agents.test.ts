import { describe, expect, it } from "vitest";
import { PROPOSAL_TOOLS } from "./pi-agents.js";

describe("PROPOSAL_TOOLS", () => {
  it("gives Proposal bash so it can check the files it wrote", () => {
    expect(PROPOSAL_TOOLS).toContain("bash");
    expect(PROPOSAL_TOOLS).toEqual([
      "read",
      "write",
      "edit",
      "ls",
      "bash",
      "resolve-library-id",
      "query-docs",
    ]);
  });
});
