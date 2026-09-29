import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { InspirationQueue } from "./inspiration-queue.js";

async function newQueue(opts?: { lowWater?: number; highWater?: number }) {
  const runDir = await mkdtemp(path.join(tmpdir(), "hyra-pi-q-"));
  const queue = await InspirationQueue.open(runDir, {
    lowWater: opts?.lowWater ?? 1,
    highWater: opts?.highWater ?? 3,
  });
  return { runDir, queue };
}

describe("InspirationQueue", () => {
  it("gives back the same inspiration that was put in", async () => {
    const { queue } = await newQueue();
    const put = await queue.enqueue({
      direction: "try insertion sort",
      context: "best score is 4, last log timed out",
      ebGeneration: 2,
    });
    expect(put.ok).toBe(true);
    if (!put.ok) return;

    const got = await queue.claim();
    expect(got).toEqual({
      id: put.id,
      direction: "try insertion sort",
      context: "best score is 4, last log timed out",
      ebGeneration: 2,
    });
  });

  it("refuses a new inspiration when the queue is already at its high mark", async () => {
    const { queue } = await newQueue({ highWater: 2, lowWater: 1 });
    expect((await queue.enqueue({ direction: "a", context: "a", ebGeneration: 0 })).ok).toBe(true);
    expect((await queue.enqueue({ direction: "b", context: "b", ebGeneration: 0 })).ok).toBe(true);
    expect(queue.mustStopProducing()).toBe(true);

    const third = await queue.enqueue({ direction: "c", context: "c", ebGeneration: 0 });
    expect(third).toEqual({ ok: false, reason: "full" });
    expect(queue.waitingCount()).toBe(2);
  });

  it("asks for more inspirations only while at or below the low mark", async () => {
    const { queue } = await newQueue({ highWater: 3, lowWater: 1 });
    expect(queue.needsMore()).toBe(true);

    await queue.enqueue({ direction: "a", context: "a", ebGeneration: 0 });
    expect(queue.waitingCount()).toBe(1);
    expect(queue.needsMore()).toBe(true);

    await queue.enqueue({ direction: "b", context: "b", ebGeneration: 0 });
    expect(queue.waitingCount()).toBe(2);
    expect(queue.needsMore()).toBe(false);
    expect(queue.mustStopProducing()).toBe(false);
  });

  it("refuses a direction that was already queued, even after it is claimed", async () => {
    const { queue } = await newQueue({ highWater: 4, lowWater: 1 });
    expect((await queue.enqueue({ direction: "LightGBM 单热基线", context: "a", ebGeneration: 0 })).ok).toBe(
      true,
    );
    await queue.claim();
    expect(await queue.enqueue({ direction: "  LightGBM 单热基线 ", context: "again", ebGeneration: 1 })).toEqual({
      ok: false,
      reason: "duplicate",
    });
    expect(queue.takenDirections()).toEqual(["LightGBM 单热基线"]);
  });

  it("does not let two callers hold the same inspiration", async () => {
    const { queue } = await newQueue({ highWater: 4, lowWater: 1 });
    const first = await queue.enqueue({ direction: "one", context: "one", ebGeneration: 1 });
    const second = await queue.enqueue({ direction: "two", context: "two", ebGeneration: 1 });
    expect(first.ok && second.ok).toBe(true);
    if (!first.ok || !second.ok) return;

    const claimedA = await queue.claim();
    const claimedB = await queue.claim();
    const claimedC = await queue.claim();

    expect(claimedA?.id).not.toBe(claimedB?.id);
    expect([claimedA?.id, claimedB?.id].sort()).toEqual([first.id, second.id].sort());
    expect(claimedC).toBeUndefined();
    expect(queue.isHeld(first.id)).toBe(true);
    expect(queue.isHeld(second.id)).toBe(true);
  });

  it("reloads waiting, held, seen, and the next id from the run folder", async () => {
    const { runDir, queue } = await newQueue({ highWater: 6, lowWater: 1 });
    const claimed = await queue.enqueue({ direction: "was claimed", context: "h", ebGeneration: 1 });
    expect(claimed.ok).toBe(true);
    if (!claimed.ok) return;
    await queue.claim();
    const waiting = await queue.enqueue({ direction: "keep waiting", context: "w", ebGeneration: 1 });
    expect(waiting.ok).toBe(true);
    if (!waiting.ok) return;

    const again = await InspirationQueue.open(runDir, {
      lowWater: 1,
      highWater: 6,
      knownIds: ["insp-009"],
    });
    expect(again.peekWaiting().map((row) => row.id)).toEqual([waiting.id]);
    expect(again.peekHeld().map((row) => row.id)).toEqual([claimed.id]);
    expect(again.takenDirections()).toEqual(["was claimed", "keep waiting"]);
    expect(again.isHeld(claimed.id)).toBe(true);

    const next = await again.enqueue({ direction: "brand new", context: "n", ebGeneration: 2 });
    expect(next).toEqual({ ok: true, id: "insp-010" });
  });

  it("puts held inspirations without a score back at the front of the waiting list", async () => {
    const { runDir, queue } = await newQueue({ highWater: 6, lowWater: 1 });
    const scored = await queue.enqueue({ direction: "already scored", context: "s", ebGeneration: 0 });
    const orphan = await queue.enqueue({ direction: "orphaned write", context: "o", ebGeneration: 0 });
    const later = await queue.enqueue({ direction: "still waiting", context: "w", ebGeneration: 0 });
    expect(scored.ok && orphan.ok && later.ok).toBe(true);
    if (!scored.ok || !orphan.ok || !later.ok) return;
    await queue.claim();
    await queue.claim();

    const again = await InspirationQueue.open(runDir, { lowWater: 1, highWater: 6 });
    await again.requeueOrphans(new Set([scored.id]));

    expect(again.isHeld(scored.id)).toBe(true);
    expect(again.isHeld(orphan.id)).toBe(false);
    expect(again.peekWaiting().map((row) => row.id)).toEqual([orphan.id, later.id]);
    expect((await again.claim())?.id).toBe(orphan.id);
  });
});
