import { describe, expect, it } from "vitest";
import { needsProposalRewrite, parseProposalRewrites } from "./proposal-rewrite.js";

describe("needsProposalRewrite", () => {
  it("rewrites a missing solve.sh or a crashed eval", () => {
    expect(needsProposalRewrite({ writeError: "missing solve.sh", score: null })).toBe(true);
    expect(needsProposalRewrite({ ok: false, score: null })).toBe(true);
    expect(needsProposalRewrite({ ok: true, score: null })).toBe(true);
  });

  it("does not rewrite a scored run, even if the score is low", () => {
    expect(
      needsProposalRewrite({
        ok: true,
        score: { score: 0.5, higherIsBetter: true, notes: "constant fallback" },
      }),
    ).toBe(false);
  });
});

describe("parseProposalRewrites", () => {
  it("defaults to two rewrites and accepts zero", () => {
    expect(parseProposalRewrites(undefined)).toBe(2);
    expect(parseProposalRewrites("0")).toBe(0);
    expect(() => parseProposalRewrites("-1")).toThrow(/rewrites/);
  });
});
