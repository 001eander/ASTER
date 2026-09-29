import type { LoopPhase } from "./inner-loop.js";

export type HealthSnapshot = {
  phase: LoopPhase;
  waitingCount: number;
  lowWater: number;
  highWater: number;
  proposalsInFlight: number;
  maxProposals: number;
  sandboxesInFlight: number;
  maxSandboxes: number;
  contextRunning: boolean;
  lastContextAt?: number;
  now: number;
  contextMaxIdleMs: number;
  heldInspirationIds: string[];
  writers: string[];
};

export type HealthReport = {
  healthy: boolean;
  reason?: string;
};

export function assessHealth(snap: HealthSnapshot): HealthReport {
  const uniqueHeld = new Set(snap.heldInspirationIds);
  if (uniqueHeld.size !== snap.heldInspirationIds.length) {
    return { healthy: false, reason: "同一条灵感被两个人领走" };
  }

  if (snap.proposalsInFlight > snap.writers.length) {
    return { healthy: false, reason: "方案已经写完，还占着写代码的名额干等沙盒" };
  }

  if (snap.waitingCount >= snap.highWater && snap.contextRunning) {
    return { healthy: false, reason: "队列已经满了，Context 还在往里放" };
  }

  if (
    snap.phase === "running" &&
    snap.waitingCount > 0 &&
    snap.proposalsInFlight < snap.maxProposals &&
    !snap.contextRunning
  ) {
    return { healthy: false, reason: "队列里有活，但没人领" };
  }

  return { healthy: true };
}
