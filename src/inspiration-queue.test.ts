import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { directionGroup, InspirationQueue } from "./inspiration-queue.js";

async function newQueue(opts?: { lowWater?: number; highWater?: number; directionQuota?: number }) {
  const runDir = await mkdtemp(path.join(tmpdir(), "hyra-pi-q-"));
  const queue = await InspirationQueue.open(runDir, {
    lowWater: opts?.lowWater ?? 1,
    highWater: opts?.highWater ?? 3,
    directionQuota: opts?.directionQuota,
  });
  return { runDir, queue };
}

function draft(direction: string, context = `try ${direction}`) {
  return { direction, context, ebGeneration: 0 };
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

  it("caps how many inspirations of one direction group wait and run at the same time", async () => {
    const { queue } = await newQueue({ highWater: 6, lowWater: 1, directionQuota: 2 });
    expect((await queue.enqueue(draft("量：缩量阴跌后的放量反弹"))).ok).toBe(true);
    // 半角冒号、冒号前带空格都算同一组。
    expect((await queue.enqueue(draft("量: 换手率抬升做反转"))).ok).toBe(true);

    const claimed = await queue.claim();
    expect(claimed?.direction).toBe("量：缩量阴跌后的放量反弹");
    expect(queue.isHeld(claimed?.id ?? "")).toBe(true);
    expect(queue.waitingCount()).toBe(1);

    // waiting 1 条 + held 1 条 = 2，到顶；第三条同组被 quota 挡下。
    expect(await queue.enqueue(draft("量 ：价量背离"))).toEqual({ ok: false, reason: "quota" });
    expect(queue.waitingCount()).toBe(1);
  });

  it("counts each direction group on its own", async () => {
    const { queue } = await newQueue({ highWater: 6, lowWater: 1, directionQuota: 1 });
    expect((await queue.enqueue(draft("量：放量反弹"))).ok).toBe(true);
    expect((await queue.enqueue(draft("价：跨期基差套利"))).ok).toBe(true);

    expect(await queue.enqueue(draft("量：缩量反转"))).toEqual({ ok: false, reason: "quota" });
    expect(await queue.enqueue(draft("价：日内动量"))).toEqual({ ok: false, reason: "quota" });

    // 没出现过的组照常放行。
    expect((await queue.enqueue(draft("波动：隔夜跳空"))).ok).toBe(true);
  });

  it("treats a direction without a colon as one group, ignoring case and spacing", async () => {
    const { queue } = await newQueue({ highWater: 6, lowWater: 1, directionQuota: 1 });
    expect((await queue.enqueue(draft("RankIC 截面反转"))).ok).toBe(true);
    expect((await queue.enqueue(draft("动量分层"))).ok).toBe(true);

    expect(await queue.enqueue(draft("rankic 截面反转"))).toEqual({ ok: false, reason: "quota" });
  });

  it("frees a group slot when the inspiration is released", async () => {
    const { queue } = await newQueue({ highWater: 6, lowWater: 1, directionQuota: 1 });
    const first = await queue.enqueue(draft("量：放量反弹"));
    expect(first.ok).toBe(true);
    if (!first.ok) return;
    expect(await queue.enqueue(draft("量：缩量反转"))).toEqual({ ok: false, reason: "quota" });

    const held = await queue.claim();
    expect(held?.id).toBe(first.id);
    // held 也占名额，点名没空出来前同组还是进不来。
    expect(await queue.enqueue(draft("量：缩量反转"))).toEqual({ ok: false, reason: "quota" });

    queue.releaseHold(first.id);
    expect((await queue.enqueue(draft("量：缩量反转"))).ok).toBe(true);

    // 打分结束（releaseHold）后名额继续轮转。
    const second = await queue.claim();
    expect(second?.direction).toBe("量：缩量反转");
    queue.releaseHold(second?.id ?? "");
    expect((await queue.enqueue(draft("量：价量背离"))).ok).toBe(true);
  });

  it("rebuilds the group counts from the run folder on open", async () => {
    const { runDir, queue } = await newQueue({ highWater: 6, lowWater: 1, directionQuota: 2 });
    const waiting = await queue.enqueue(draft("量：缩量反转"));
    const laterWaiting = await queue.enqueue(draft("量：放量反弹"));
    expect(waiting.ok && laterWaiting.ok).toBe(true);
    if (!waiting.ok || !laterWaiting.ok) return;
    const held = await queue.claim();
    expect(held?.id).toBe(waiting.id);
    expect(await queue.enqueue(draft("量：价量背离"))).toEqual({ ok: false, reason: "quota" });

    const again = await InspirationQueue.open(runDir, {
      lowWater: 1,
      highWater: 6,
      directionQuota: 2,
    });
    expect(again.isHeld(waiting.id)).toBe(true);
    expect(again.peekWaiting().map((row) => row.id)).toEqual([laterWaiting.id]);
    // 重启后 held 1 条 + waiting 1 条，组计数仍是 2。
    expect(await again.enqueue(draft("量：价量背离"))).toEqual({ ok: false, reason: "quota" });

    again.releaseHold(waiting.id);
    expect((await again.enqueue(draft("量：价量背离"))).ok).toBe(true);
    expect(await again.enqueue(draft("量：隔夜跳空"))).toEqual({ ok: false, reason: "quota" });
  });

  it("reports a duplicate before it reports a quota", async () => {
    const { queue } = await newQueue({ highWater: 6, lowWater: 1, directionQuota: 1 });
    expect((await queue.enqueue(draft("量：放量反弹"))).ok).toBe(true);

    expect(await queue.enqueue(draft("量：放量反弹"))).toEqual({ ok: false, reason: "duplicate" });
    expect(await queue.enqueue(draft("  量：放量反弹 "))).toEqual({
      ok: false,
      reason: "duplicate",
    });
    expect(await queue.enqueue(draft("量：缩量反转"))).toEqual({ ok: false, reason: "quota" });
  });
});

describe("directionGroup", () => {
  it("takes the prefix before the first colon as the group key", () => {
    expect(directionGroup("量：缩量阴跌后的放量反弹")).toBe("量");
    expect(directionGroup("量: 换手率抬升做反转")).toBe("量");
    expect(directionGroup("  量 ：价量背离  ")).toBe("量");
    expect(directionGroup("Volume：换手率抬升")).toBe("volume");
  });

  it("falls back to the whole normalized sentence when there is no colon", () => {
    expect(directionGroup("RankIC 截面反转")).toBe("rankic 截面反转");
    expect(directionGroup("  RankIC   截面反转 ")).toBe("rankic 截面反转");
  });
});
