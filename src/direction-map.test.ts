import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { ExperienceRecord } from "./experience-bank.js";
import {
  buildDirectionMap,
  formatDirectionMap,
  loadFactorRegistry,
  readQueueDirections,
  type FactorRegistry,
} from "./direction-map.js";

async function tempRoot(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), "aster-direction-map-"));
}

function registryOf(
  factors: Array<{
    id: string;
    source: string;
    scale: string;
    mechanism: string;
    rankIc?: number | null;
    status?: string;
  }>,
): FactorRegistry {
  return {
    version: 1,
    factors: factors.map((factor) => ({
      factorId: factor.id,
      direction: {
        signal_source: factor.source,
        time_scale: factor.scale,
        mechanism: factor.mechanism,
      },
      rankIc: factor.rankIc ?? null,
      status: factor.status ?? "pool",
    })),
  };
}

function record(overrides: Partial<ExperienceRecord>): ExperienceRecord {
  return {
    id: "0001",
    inspirationId: "insp-001",
    ok: true,
    score: { score: 1, higherIsBetter: true, notes: "" },
    solutionDir: "solutions/0001",
    logPath: "solutions/0001/run.log",
    ...overrides,
  };
}

describe("loadFactorRegistry", () => {
  it("returns null when registry.json is missing instead of throwing", async () => {
    const root = await tempRoot();
    expect(await loadFactorRegistry(root)).toBeNull();
  });

  it("returns null when registry.json is not valid JSON", async () => {
    const root = await tempRoot();
    await mkdir(path.join(root, "factor_library"), { recursive: true });
    await writeFile(path.join(root, "factor_library", "registry.json"), "{ not json", "utf8");
    expect(await loadFactorRegistry(root)).toBeNull();
  });

  it("parses pool factors, tolerating null metrics and skipping malformed entries", async () => {
    const root = await tempRoot();
    await mkdir(path.join(root, "factor_library"), { recursive: true });
    await writeFile(
      path.join(root, "factor_library", "registry.json"),
      JSON.stringify({
        version: 1,
        factors: [
          {
            factor_id: "a",
            metrics: { rank_ic: 0.04 },
            direction: { signal_source: "price", time_scale: "short", mechanism: "momentum" },
            status: "pool",
          },
          {
            factor_id: "b",
            metrics: { rank_ic: null },
            direction: { signal_source: "volume", time_scale: "medium", mechanism: "liquidity" },
            status: "graveyard",
          },
          { factor_id: "broken", direction: { signal_source: "price" }, status: "pool" },
        ],
      }),
      "utf8",
    );
    const registry = await loadFactorRegistry(root);
    expect(registry?.factors).toHaveLength(2);
    expect(registry?.factors[0]).toMatchObject({ factorId: "a", rankIc: 0.04, status: "pool" });
    expect(registry?.factors[1]?.rankIc).toBeNull();
  });
});

describe("readQueueDirections", () => {
  it("returns an empty list when the queue dir is missing", async () => {
    const root = await tempRoot();
    expect(await readQueueDirections(path.join(root, "queue"))).toEqual([]);
  });

  it("reads id and direction, skipping broken files", async () => {
    const root = await tempRoot();
    const queueDir = path.join(root, "queue");
    await mkdir(queueDir, { recursive: true });
    await writeFile(
      path.join(queueDir, "insp-001.json"),
      JSON.stringify({ id: "insp-001", direction: "量：缩量反弹" }),
      "utf8",
    );
    await writeFile(path.join(queueDir, "insp-002.json"), "{ broken", "utf8");
    expect(await readQueueDirections(queueDir)).toEqual([
      { id: "insp-001", direction: "量：缩量反弹" },
    ]);
  });
});

describe("buildDirectionMap", () => {
  it("aggregates pool factors by direction triple and excludes graveyard", () => {
    const map = buildDirectionMap(
      registryOf([
        { id: "a", source: "price", scale: "short", mechanism: "momentum", rankIc: 0.03 },
        { id: "b", source: "price", scale: "short", mechanism: "momentum", rankIc: 0.07 },
        { id: "c", source: "price", scale: "short", mechanism: "momentum", rankIc: null },
        { id: "d", source: "volume", scale: "medium", mechanism: "liquidity" },
        { id: "e", source: "volume", scale: "medium", mechanism: "liquidity", rankIc: 0.5, status: "graveyard" },
      ]),
      [],
      [],
    );

    expect(map.poolCount).toBe(4);
    expect(map.structured).toHaveLength(2);
    const momentum = map.structured.find((row) => row.key === "price/short/momentum");
    expect(momentum?.quality).toEqual({
      count: 3,
      evaluated: 2,
      meanRankIc: 0.05,
      bestRankIc: 0.07,
    });
    const liquidity = map.structured.find((row) => row.key === "volume/medium/liquidity");
    expect(liquidity?.quality.evaluated).toBe(0);
    expect(liquidity?.quality.meanRankIc).toBeNull();
  });

  it("sorts low-density directions first and counts axes", () => {
    const map = buildDirectionMap(
      registryOf([
        { id: "a", source: "price", scale: "short", mechanism: "momentum" },
        { id: "b", source: "price", scale: "short", mechanism: "momentum" },
        { id: "c", source: "volume", scale: "long", mechanism: "liquidity" },
      ]),
      [],
      [],
    );
    expect(map.structured.map((row) => row.key)).toEqual([
      "volume/long/liquidity",
      "price/short/momentum",
    ]);
    expect(map.sources).toEqual([
      { value: "price", count: 2 },
      { value: "volume", count: 1 },
    ]);
    expect(map.mechanisms).toEqual([
      { value: "momentum", count: 2 },
      { value: "liquidity", count: 1 },
    ]);
  });

  it("counts rejection from ok=false or score=null and tolerates unknown inspiration ids", () => {
    const map = buildDirectionMap(
      registryOf([]),
      [
        record({ inspirationId: "insp-001", ok: true, score: { score: 5, higherIsBetter: true, notes: "" } }),
        record({ inspirationId: "insp-001", ok: false, score: null }),
        record({ inspirationId: "insp-002", ok: true, score: null }),
        record({ inspirationId: "insp-999", ok: false, score: null }),
      ],
      [
        { id: "insp-001", direction: "量：缩量反弹" },
        { id: "insp-002", direction: "价：短期反转" },
      ],
    );

    const first = map.recent.find((row) => row.direction === "量：缩量反弹");
    expect(first).toMatchObject({ total: 2, rejected: 1, rejectedRate: 0.5 });
    const second = map.recent.find((row) => row.direction === "价：短期反转");
    expect(second).toMatchObject({ total: 1, rejected: 1, rejectedRate: 1 });
    expect(map.unregisteredRecords).toBe(1);
  });

  it("returns an empty map when the registry is null", () => {
    const map = buildDirectionMap(null, [], []);
    expect(map).toMatchObject({ poolCount: 0, structured: [], recent: [], unregisteredRecords: 0 });
    expect(formatDirectionMap(map)).toBe("");
  });
});

describe("formatDirectionMap", () => {
  it("renders structured quality, low-density markers and rejection text", () => {
    const map = buildDirectionMap(
      registryOf([
        { id: "a", source: "price", scale: "short", mechanism: "momentum", rankIc: 0.021 },
        { id: "b", source: "price", scale: "short", mechanism: "momentum", rankIc: 0.049 },
        { id: "c", source: "value", scale: "long", mechanism: "fundamental" },
      ]),
      [
        record({ inspirationId: "insp-001", ok: false, score: null }),
        record({ inspirationId: "insp-001", ok: true, score: null }),
      ],
      [{ id: "insp-001", direction: "量：缩量阴跌后的放量反弹做反转" }],
    );

    const text = formatDirectionMap(map);
    expect(text).toContain("方向分布地图");
    expect(text).toContain("price/short/momentum");
    expect(text).toContain("IC均0.035 最好0.049");
    expect(text).toContain("←低密度");
    expect(text).toContain("质量未评估");
    expect(text).toContain("被拒2 100%");
    expect(text.split("\n").length).toBeLessThan(40);
  });

  it("still renders the recent section when only records exist", () => {
    const map = buildDirectionMap(
      null,
      [record({ inspirationId: "insp-001", ok: false, score: null })],
      [{ id: "insp-001", direction: "价：日内反转" }],
    );
    const text = formatDirectionMap(map);
    expect(text).toContain("本轮灵感战绩");
    expect(text).toContain("价：日内反转");
    expect(text).not.toContain("因子库方向分布");
  });
});
