import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import type { ExperienceRecord } from "./experience-bank.js";

// 方向分布地图：把因子库的结构化方向、本轮灵感的战绩压缩成一小段文字，
// 注入 Context 的 user prompt，让新灵感往低密度/空白方向分流。
// 读取一律容错：缺文件、坏 JSON 都返回空值，调用方降级跳过注入。

export type FactorDirection = {
  signal_source: string;
  time_scale: string;
  mechanism: string;
};

export type FactorRegistryEntry = {
  factorId: string;
  // 因子的经济假设原文；Proposal 摘要注入要用，registry 里缺失时给空串。
  hypothesis: string;
  direction: FactorDirection;
  rankIc: number | null;
  status: string;
};

export type FactorRegistry = {
  version: number;
  factors: FactorRegistryEntry[];
};

export type InspirationDirection = {
  id: string;
  direction: string;
};

export type DirectionQuality = {
  count: number;
  evaluated: number;
  meanRankIc: number | null;
  bestRankIc: number | null;
};

export type StructuredDirectionStat = {
  key: string;
  direction: FactorDirection;
  quality: DirectionQuality;
};

export type AxisStat = {
  value: string;
  count: number;
};

export type DirectionTextStat = {
  direction: string;
  total: number;
  rejected: number;
  rejectedRate: number;
};

export type DirectionMap = {
  poolCount: number;
  structured: StructuredDirectionStat[];
  sources: AxisStat[];
  mechanisms: AxisStat[];
  recent: DirectionTextStat[];
  unregisteredRecords: number;
};

// 仅 1 个因子的方向视为低密度；渲染时排序在前并打标。
const LOW_DENSITY_COUNT = 1;
const RECENT_LIMIT = 8;
const DIRECTION_TEXT_LIMIT = 28;

export function directionKey(direction: FactorDirection): string {
  return `${direction.signal_source}/${direction.time_scale}/${direction.mechanism}`;
}

export async function loadFactorRegistry(rootDir: string): Promise<FactorRegistry | null> {
  let text: string;
  try {
    text = await readFile(path.join(rootDir, "factor_library", "registry.json"), "utf8");
  } catch {
    return null;
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  return parseRegistry(raw);
}

// 读 queue 目录下每个 `insp-*.json` 的 id 与 direction，坏的条目跳过。
export async function readQueueDirections(queueDir: string): Promise<InspirationDirection[]> {
  let names: string[];
  try {
    names = await readdir(queueDir);
  } catch {
    return [];
  }
  const rows: InspirationDirection[] = [];
  for (const name of names) {
    if (!name.endsWith(".json")) continue;
    let raw: unknown;
    try {
      raw = JSON.parse(await readFile(path.join(queueDir, name), "utf8"));
    } catch {
      continue;
    }
    if (typeof raw !== "object" || raw === null) continue;
    const row = raw as Record<string, unknown>;
    if (typeof row.id !== "string" || typeof row.direction !== "string") continue;
    rows.push({ id: row.id, direction: row.direction });
  }
  return rows;
}

export function buildDirectionMap(
  registry: FactorRegistry | null,
  records: ExperienceRecord[],
  queueDirections: InspirationDirection[],
): DirectionMap {
  const pool = registry?.factors.filter((factor) => factor.status === "pool") ?? [];
  const grouped = new Map<
    string,
    { direction: FactorDirection; count: number; values: number[] }
  >();
  for (const factor of pool) {
    const key = directionKey(factor.direction);
    const bucket = grouped.get(key) ?? { direction: factor.direction, count: 0, values: [] };
    bucket.count += 1;
    if (typeof factor.rankIc === "number" && Number.isFinite(factor.rankIc)) {
      bucket.values.push(factor.rankIc);
    }
    grouped.set(key, bucket);
  }

  const structured: StructuredDirectionStat[] = [...grouped.entries()]
    .map(([key, bucket]) => ({
      key,
      direction: bucket.direction,
      quality: qualityOf(bucket.count, bucket.values),
    }))
    .sort((a, b) => a.quality.count - b.quality.count || a.key.localeCompare(b.key));

  const byId = new Map(queueDirections.map((item) => [item.id, normalizeText(item.direction)]));
  const textStats = new Map<string, { total: number; rejected: number }>();
  let unregisteredRecords = 0;
  for (const record of records) {
    const direction = byId.get(record.inspirationId);
    if (!direction) {
      unregisteredRecords += 1;
      continue;
    }
    const bucket = textStats.get(direction) ?? { total: 0, rejected: 0 };
    bucket.total += 1;
    if (!record.ok || record.score === null) bucket.rejected += 1;
    textStats.set(direction, bucket);
  }
  const recent: DirectionTextStat[] = [...textStats.entries()]
    .map(([direction, bucket]) => ({
      direction,
      total: bucket.total,
      rejected: bucket.rejected,
      rejectedRate: bucket.total > 0 ? bucket.rejected / bucket.total : 0,
    }))
    .sort(
      (a, b) =>
        b.rejectedRate - a.rejectedRate ||
        b.total - a.total ||
        a.direction.localeCompare(b.direction),
    );

  return {
    poolCount: pool.length,
    structured,
    sources: axisStats(pool, (factor) => factor.direction.signal_source),
    mechanisms: axisStats(pool, (factor) => factor.direction.mechanism),
    recent,
    unregisteredRecords,
  };
}

export function formatDirectionMap(map: DirectionMap): string {
  const lines: string[] = [];
  if (map.structured.length > 0) {
    lines.push(`一、因子库方向分布（结构化，pool 因子 ${map.poolCount} 个，信号源/时间尺度/机制，密度升序）：`);
    for (const row of map.structured) {
      const marker = row.quality.count <= LOW_DENSITY_COUNT ? "  ←低密度" : "";
      lines.push(`  ${row.key}  ×${row.quality.count}  ${renderQuality(row.quality)}${marker}`);
    }
    lines.push(
      `  已覆盖信号源：${renderAxes(map.sources)}；已覆盖机制：${renderAxes(map.mechanisms)}。`,
    );
  }
  if (map.recent.length > 0) {
    lines.push("二、本轮灵感战绩（按方向文本，被拒 = 崩溃或未评分）：");
    for (const row of map.recent.slice(0, RECENT_LIMIT)) {
      lines.push(
        `  ${truncate(row.direction)}  提交${row.total} 被拒${row.rejected} ${Math.round(
          row.rejectedRate * 100,
        )}%`,
      );
    }
  }
  if (map.unregisteredRecords > 0) {
    lines.push(`未登记队列方向的记录：${map.unregisteredRecords} 条（历史遗留，可忽略）。`);
  }
  if (lines.length === 0) return "";
  return [
    "【方向分布地图】新灵感优先走低密度/空白方向；同质方向（机制无差异）不要重复提交。",
    ...lines,
  ].join("\n");
}

function parseRegistry(raw: unknown): FactorRegistry | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (!Array.isArray(obj.factors)) return null;
  const factors: FactorRegistryEntry[] = [];
  for (const item of obj.factors) {
    const entry = parseFactorEntry(item);
    if (entry) factors.push(entry);
  }
  return { version: typeof obj.version === "number" ? obj.version : 1, factors };
}

function parseFactorEntry(raw: unknown): FactorRegistryEntry | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (typeof obj.factor_id !== "string") return null;
  const direction = parseDirection(obj.direction);
  if (!direction) return null;
  const metrics = (typeof obj.metrics === "object" && obj.metrics !== null ? obj.metrics : {}) as Record<
    string,
    unknown
  >;
  const rankIc = typeof metrics.rank_ic === "number" && Number.isFinite(metrics.rank_ic)
    ? metrics.rank_ic
    : null;
  return {
    factorId: obj.factor_id,
    hypothesis: typeof obj.hypothesis === "string" ? obj.hypothesis : "",
    direction,
    rankIc,
    status: typeof obj.status === "string" ? obj.status : "pool",
  };
}

function parseDirection(raw: unknown): FactorDirection | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  const source = obj.signal_source;
  const scale = obj.time_scale;
  const mechanism = obj.mechanism;
  if (typeof source !== "string" || typeof scale !== "string" || typeof mechanism !== "string") {
    return null;
  }
  return { signal_source: source, time_scale: scale, mechanism };
}

function qualityOf(count: number, values: number[]): DirectionQuality {
  if (values.length === 0) {
    return { count, evaluated: 0, meanRankIc: null, bestRankIc: null };
  }
  const sum = values.reduce((acc, value) => acc + value, 0);
  return {
    count,
    evaluated: values.length,
    meanRankIc: sum / values.length,
    bestRankIc: Math.max(...values),
  };
}

function axisStats(
  pool: FactorRegistryEntry[],
  pick: (factor: FactorRegistryEntry) => string,
): AxisStat[] {
  const counts = new Map<string, number>();
  for (const factor of pool) {
    const value = pick(factor);
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count || a.value.localeCompare(b.value));
}

function renderQuality(quality: DirectionQuality): string {
  if (quality.meanRankIc === null || quality.bestRankIc === null) return "质量未评估";
  return `IC均${quality.meanRankIc.toFixed(3)} 最好${quality.bestRankIc.toFixed(3)}`;
}

function renderAxes(stats: AxisStat[]): string {
  if (stats.length === 0) return "无";
  return stats.map((item) => `${item.value}(${item.count})`).join(" ");
}

function normalizeText(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

function truncate(text: string): string {
  return text.length > DIRECTION_TEXT_LIMIT ? `${text.slice(0, DIRECTION_TEXT_LIMIT)}…` : text;
}
