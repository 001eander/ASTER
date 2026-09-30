import { describe, expect, it } from "vitest";
import type { FactorRegistry } from "./direction-map.js";
import { buildPeerDigest, formatPeerDigest } from "./factor-digest.js";

function registryOf(
  factors: Array<{
    id: string;
    source: string;
    scale: string;
    mechanism: string;
    hypothesis?: string;
    status?: string;
  }>,
): FactorRegistry {
  return {
    version: 1,
    factors: factors.map((factor) => ({
      factorId: factor.id,
      hypothesis: factor.hypothesis ?? "",
      direction: {
        signal_source: factor.source,
        time_scale: factor.scale,
        mechanism: factor.mechanism,
      },
      rankIc: null,
      status: factor.status ?? "pool",
    })),
  };
}

describe("buildPeerDigest", () => {
  it("matches Chinese aliases of both volume and reversal axes", () => {
    const digest = buildPeerDigest(
      registryOf([
        {
          id: "vr_rev",
          source: "volume",
          scale: "short",
          mechanism: "reversal",
          hypothesis: "缩量阴跌后放量反弹做反转",
        },
        { id: "mom_5", source: "price", scale: "long", mechanism: "momentum" },
      ]),
      "量：缩量阴跌后的放量反弹做反转",
    );

    expect(digest?.matched.map((peer) => peer.factorId)).toEqual(["vr_rev"]);
    expect(digest?.matched[0]?.hits).toBe(2);
    expect(digest?.others.map((peer) => peer.factorId)).toEqual(["mom_5"]);
  });

  it("sorts matched factors by hit count descending", () => {
    const digest = buildPeerDigest(
      registryOf([
        { id: "two", source: "volume", scale: "short", mechanism: "reversal" },
        { id: "three", source: "price_volume", scale: "long", mechanism: "reversal" },
      ]),
      "量价长期反转",
    );

    expect(digest?.matched.map((peer) => peer.factorId)).toEqual(["three", "two"]);
    expect(digest?.matched.map((peer) => peer.hits)).toEqual([3, 2]);
  });

  it("leaves matched empty when no axis alias appears in the direction text", () => {
    const digest = buildPeerDigest(
      registryOf([{ id: "mom_5", source: "price", scale: "short", mechanism: "momentum" }]),
      "基本面：现金流质量",
    );

    expect(digest?.matched).toEqual([]);
    expect(digest?.others.map((peer) => peer.factorId)).toEqual(["mom_5"]);
  });

  it("treats an empty direction text as no match", () => {
    const digest = buildPeerDigest(
      registryOf([{ id: "mom_5", source: "price", scale: "short", mechanism: "momentum" }]),
      "   ",
    );
    expect(digest?.matched).toEqual([]);
  });

  it("keeps graveyard factors out of the digest", () => {
    const digest = buildPeerDigest(
      registryOf([
        { id: "live", source: "volume", scale: "short", mechanism: "reversal" },
        {
          id: "dead",
          source: "volume",
          scale: "short",
          mechanism: "reversal",
          status: "graveyard",
        },
      ]),
      "量：短期反转",
    );

    expect(digest?.matched.map((peer) => peer.factorId)).toEqual(["live"]);
    expect(digest?.others).toEqual([]);
  });

  it("returns null for a missing registry or an empty pool", () => {
    expect(buildPeerDigest(null, "量：短期反转")).toBeNull();
    expect(
      buildPeerDigest(
        registryOf([{ id: "dead", source: "volume", scale: "short", mechanism: "reversal", status: "graveyard" }]),
        "量：短期反转",
      ),
    ).toBeNull();
  });
});

describe("formatPeerDigest", () => {
  it("returns an empty string when there is nothing to inject", () => {
    expect(formatPeerDigest(null)).toBe("");
  });

  it("renders matched hypotheses and only index rows for other directions", () => {
    const digest = buildPeerDigest(
      registryOf([
        {
          id: "vr_rev",
          source: "volume",
          scale: "short",
          mechanism: "reversal",
          hypothesis: "缩量阴跌后放量反弹做反转",
        },
        {
          id: "mom_5",
          source: "price",
          scale: "long",
          mechanism: "momentum",
          hypothesis: "长期复权收盘涨幅延续",
        },
      ]),
      "量：缩量阴跌后的放量反弹做反转",
    );

    const text = formatPeerDigest(digest);
    expect(text).toContain("已入库因子摘要");
    expect(text).toContain("同方向");
    expect(text).toContain("vr_rev  volume/short/reversal");
    expect(text).toContain("假设：缩量阴跌后放量反弹做反转");
    expect(text).toContain("mom_5  price/long/momentum");
    expect(text).not.toContain("长期复权收盘涨幅延续");
  });

  it("says so when no pool factor shares the direction", () => {
    const digest = buildPeerDigest(
      registryOf([{ id: "mom_5", source: "price", scale: "short", mechanism: "momentum" }]),
      "基本面：现金流质量",
    );
    const text = formatPeerDigest(digest);
    expect(text).toContain("同方向暂无已入库因子");
    expect(text).not.toContain("假设：");
  });

  it("keeps the block under 30 lines even with a full library", () => {
    const digest = buildPeerDigest(
      registryOf(
        Array.from({ length: 4 }, (_, index) => ({
          id: `vol_rev_${index}`,
          source: "volume",
          scale: "short",
          mechanism: "reversal",
          hypothesis: "缩量阴跌后放量反弹做反转，按成交量分位分组再取反转信号。".repeat(3),
        })).concat(
          Array.from({ length: 9 }, (_, index) => ({
            id: `other_${index}`,
            source: "price",
            scale: "long",
            mechanism: "momentum",
          })),
        ),
      ),
      "量：缩量阴跌后的放量反弹做反转",
    );

    const text = formatPeerDigest(digest);
    expect(text.split("\n").length).toBeLessThanOrEqual(30);
    expect(text).toContain("…");
  });
});
