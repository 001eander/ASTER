import { describe, expect, it } from "vitest";
import { continueBudget, parseArgs, parseBudget, parseProposalMs, parseSandboxMs } from "./budget.js";

describe("parseBudget", () => {
  it("reads a 30 minute cap as 1800000 ms", () => {
    expect(parseBudget("30m", "8")).toEqual({ maxSolutions: 8, maxMs: 1_800_000 });
  });

  it("omits the clock and solution cap in no-limits mode unless flags are given", () => {
    expect(parseBudget(undefined, undefined, { unlimited: true })).toEqual({});
    expect(parseBudget("2h", "20", { unlimited: true })).toEqual({
      maxSolutions: 20,
      maxMs: 7_200_000,
    });
  });
});

describe("parseSandboxMs", () => {
  it("defaults to 30 minutes and lets --sandbox override the env", () => {
    expect(parseSandboxMs(undefined, undefined)).toBe(1_800_000);
    expect(parseSandboxMs(undefined, "900000")).toBe(900_000);
    expect(parseSandboxMs("30m", "900000")).toBe(1_800_000);
    expect(parseSandboxMs(undefined, undefined, { unlimited: true })).toBeUndefined();
    expect(parseSandboxMs(undefined, "900000", { unlimited: true })).toBeUndefined();
    expect(parseSandboxMs("45m", undefined, { unlimited: true })).toBe(2_700_000);
  });
});

describe("parseProposalMs", () => {
  it("defaults to 30 minutes and lets --write override the env", () => {
    expect(parseProposalMs(undefined, undefined)).toBe(1_800_000);
    expect(parseProposalMs(undefined, "360000")).toBe(360_000);
    expect(parseProposalMs("30m", "360000")).toBe(1_800_000);
    expect(parseProposalMs(undefined, undefined, { unlimited: true })).toBeUndefined();
    expect(parseProposalMs(undefined, "360000", { unlimited: true })).toBeUndefined();
    expect(parseProposalMs("10m", undefined, { unlimited: true })).toBe(600_000);
  });
});

describe("continueBudget", () => {
  it("keeps the saved solution cap and the original clock so leftover time stays leftover", () => {
    expect(continueBudget({ maxSolutions: 100, maxMs: 3_600_000 }, 1_200_000)).toEqual({
      maxSolutions: 100,
      maxMs: 3_600_000,
    });
  });

  it("adds --budget onto the remaining clock instead of replacing it", () => {
    expect(
      continueBudget({ maxSolutions: 100, maxMs: 3_600_000 }, 1_200_000, { budget: "30m" }),
    ).toEqual({ maxSolutions: 100, maxMs: 5_400_000 });
  });

  it("lets --solutions replace the cap and does not default to 8", () => {
    expect(continueBudget({ maxSolutions: 100, maxMs: 1_000 }, 0, { solutions: "120" })).toEqual({
      maxSolutions: 120,
      maxMs: 1_000,
    });
  });

  it("omits a clock when the saved run had none and no extra budget is given", () => {
    expect(continueBudget({ maxSolutions: 50 }, 999)).toEqual({ maxSolutions: 50 });
  });

  it("drops saved caps when resuming with no-limits", () => {
    expect(
      continueBudget({ maxSolutions: 100, maxMs: 3_600_000 }, 1_200_000, { unlimited: true }),
    ).toEqual({});
  });
});

describe("parseArgs", () => {
  it("reads the run command and flags", () => {
    const args = parseArgs(["run", "--task", "./examples/task", "--proposals", "3"]);
    expect(args.get("_")).toBe("run");
    expect(args.get("task")).toBe("./examples/task");
    expect(args.get("proposals")).toBe("3");
  });

  it("reads --run for continuing an existing directory", () => {
    const args = parseArgs(["run", "--task", "./examples/s6e9", "--run", "runs/old"]);
    expect(args.get("run")).toBe("runs/old");
  });

  it("reads --no-limits as a boolean flag", () => {
    const args = parseArgs(["run", "--task", "./examples/s6e9", "--no-limits"]);
    expect(args.get("no-limits")).toBe(true);
  });
});
