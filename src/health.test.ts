import { describe, expect, it } from "vitest";
import { assessHealth, type HealthSnapshot } from "./health.js";

function running(overrides: Partial<HealthSnapshot> = {}): HealthSnapshot {
  return {
    phase: "running",
    waitingCount: 1,
    lowWater: 1,
    highWater: 3,
    proposalsInFlight: 1,
    maxProposals: 2,
    sandboxesInFlight: 1,
    maxSandboxes: 2,
    contextRunning: false,
    lastContextAt: 1000,
    now: 1500,
    contextMaxIdleMs: 5000,
    heldInspirationIds: ["insp-001"],
    writers: ["insp-001"],
    ...overrides,
  };
}

describe("assessHealth", () => {
  it("treats an empty queue as fine while starting or stopping", () => {
    expect(
      assessHealth(
        running({
          phase: "starting",
          waitingCount: 0,
          proposalsInFlight: 0,
          sandboxesInFlight: 0,
          heldInspirationIds: [],
          writers: [],
          lastContextAt: undefined,
        }),
      ),
    ).toEqual({ healthy: true });
    expect(
      assessHealth(
        running({
          phase: "stopped",
          waitingCount: 0,
          proposalsInFlight: 0,
          sandboxesInFlight: 0,
          heldInspirationIds: [],
          writers: [],
        }),
      ),
    ).toEqual({ healthy: true });
  });

  it("flags a full queue while Context is still adding work", () => {
    expect(
      assessHealth(
        running({
          waitingCount: 3,
          highWater: 3,
          contextRunning: true,
          proposalsInFlight: 2,
          maxProposals: 2,
          writers: ["w1", "w2"],
          heldInspirationIds: ["w1", "w2"],
        }),
      ),
    ).toEqual({ healthy: false, reason: "队列已经满了，Context 还在往里放" });
  });

  it("flags waiting work when a proposal slot is free", () => {
    expect(
      assessHealth(
        running({
          waitingCount: 2,
          proposalsInFlight: 0,
          writers: [],
          heldInspirationIds: [],
        }),
      ),
    ).toEqual({ healthy: false, reason: "队列里有活，但没人领" });
  });

  it("does not flag waiting work while Context is still producing", () => {
    expect(
      assessHealth(
        running({
          waitingCount: 2,
          proposalsInFlight: 0,
          writers: [],
          heldInspirationIds: [],
          contextRunning: true,
        }),
      ),
    ).toEqual({ healthy: true });
  });

  it("stays healthy while proposals are still writing after Context last ran", () => {
    expect(
      assessHealth(
        running({
          waitingCount: 0,
          proposalsInFlight: 2,
          maxProposals: 2,
          writers: ["a", "b"],
          heldInspirationIds: ["a", "b"],
          lastContextAt: 1000,
          now: 8000,
          contextMaxIdleMs: 5000,
        }),
      ),
    ).toEqual({ healthy: true });
  });

  it("flags the same inspiration held twice", () => {
    expect(
      assessHealth(
        running({
          heldInspirationIds: ["insp-001", "insp-001"],
          writers: ["insp-001", "insp-001"],
          proposalsInFlight: 2,
        }),
      ),
    ).toEqual({ healthy: false, reason: "同一条灵感被两个人领走" });
  });

  it("flags a proposal slot held after writing is done", () => {
    expect(
      assessHealth(
        running({
          waitingCount: 0,
          proposalsInFlight: 1,
          writers: [],
          heldInspirationIds: ["insp-001"],
          sandboxesInFlight: 1,
        }),
      ),
    ).toEqual({ healthy: false, reason: "方案已经写完，还占着写代码的名额干等沙盒" });
  });

  it("accepts a busy running loop with work in every stage", () => {
    expect(
      assessHealth(
        running({
          waitingCount: 2,
          proposalsInFlight: 2,
          maxProposals: 2,
          sandboxesInFlight: 1,
          writers: ["w1", "w2"],
          heldInspirationIds: ["w1", "w2", "s1"],
          lastContextAt: 1400,
          now: 1500,
        }),
      ),
    ).toEqual({ healthy: true });
  });
});
