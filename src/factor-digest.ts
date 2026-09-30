import { directionKey, type FactorDirection, type FactorRegistry } from "./direction-map.js";

// 同方向已入库因子摘要：给 Proposal 的 user prompt 附上「这个方向已经有什么」，
// 只给 factorId、方向三元组与假设原文，绝不附因子代码。Proposal 据此在机制层面
// 与已有因子拉开距离，避免换皮复刻被相关性查重拒掉。
// 灵感方向是 Context 写的自由文本，与 registry 的结构化三元组没有精确映射，
// 因此用「轴取值 + 中文别名」的包含匹配做启发式对齐。

// 方向轴取值 → 中文别名。命中任一别名即该轴命中，命中轴数越多越可能同方向。
const AXIS_ALIASES: Readonly<Record<string, readonly string[]>> = {
  price: ["价", "价格", "收盘", "股价"],
  volume: ["量", "成交量", "成交额"],
  price_volume: ["量价", "价量"],
  short: ["短", "短期", "短线"],
  medium: ["中", "中期", "中线"],
  long: ["长", "长期", "长线"],
  momentum: ["动量", "趋势"],
  reversal: ["反转"],
  volatility: ["波动", "振幅"],
  liquidity: ["换手", "流动性", "缩量", "放量"],
  volume_ratio: ["量比", "均量"],
  ma_bias: ["乖离", "均线偏离"],
  range: ["波幅"],
  vwap_bias: ["均价", "vwap"],
  price_volume_corr: ["量价相关", "价量相关"],
  fundamental: ["基本面"],
};

const HYPOTHESIS_LIMIT = 80;

export type PeerFactor = {
  factorId: string;
  direction: FactorDirection;
  hypothesis: string;
  // 命中的轴数；0 表示与灵感方向无别名交集。
  hits: number;
};

export type PeerDigest = {
  matched: PeerFactor[];
  others: PeerFactor[];
};

// pool 因子为空或 registry 缺失时返回 null，调用方跳过注入。
export function buildPeerDigest(
  registry: FactorRegistry | null,
  directionText: string,
): PeerDigest | null {
  const pool = registry?.factors.filter((factor) => factor.status === "pool") ?? [];
  if (pool.length === 0) return null;
  const text = normalizeDirectionText(directionText);
  const scored: PeerFactor[] = pool.map((factor) => ({
    factorId: factor.factorId,
    direction: factor.direction,
    hypothesis: factor.hypothesis,
    hits: text.length === 0 ? 0 : countAxisHits(text, factor.direction),
  }));
  return {
    matched: scored
      .filter((peer) => peer.hits > 0)
      .sort((a, b) => b.hits - a.hits || a.factorId.localeCompare(b.factorId)),
    others: scored
      .filter((peer) => peer.hits === 0)
      .sort((a, b) => a.factorId.localeCompare(b.factorId)),
  };
}

// 库为空或 registry 缺失时返回空串。同方向逐个列出假设，其他方向只给索引行防膨胀。
export function formatPeerDigest(digest: PeerDigest | null): string {
  if (!digest || (digest.matched.length === 0 && digest.others.length === 0)) return "";
  const lines: string[] = [
    "【已入库因子摘要】同方向已有因子的假设与思路如下（只给文字，不给代码）。新因子必须在机制层面与它们拉开距离：构造方式、口径或触发条件要有本质区别，换参数或换窗口不算。相关性查重会拒绝与库内因子 |max_corr| > 0.7 的近似因子。",
  ];
  if (digest.matched.length > 0) {
    lines.push(`同方向（按方向文本启发式命中方向轴，共 ${digest.matched.length} 个）：`);
    for (const peer of digest.matched) {
      lines.push(`  ${peer.factorId}  ${directionKey(peer.direction)}  命中${peer.hits}轴`);
      lines.push(`    假设：${peer.hypothesis ? truncate(peer.hypothesis) : "未记录"}`);
    }
  } else {
    lines.push("同方向暂无已入库因子。");
  }
  if (digest.others.length > 0) {
    lines.push("其他方向（只列索引，同样要避开）：");
    for (const peer of digest.others) {
      lines.push(`  ${peer.factorId}  ${directionKey(peer.direction)}`);
    }
  }
  return lines.join("\n");
}

function countAxisHits(text: string, direction: FactorDirection): number {
  let hits = 0;
  for (const value of [
    direction.signal_source,
    direction.time_scale,
    direction.mechanism,
  ]) {
    if (axisCandidates(value).some((candidate) => text.includes(candidate))) hits += 1;
  }
  return hits;
}

function axisCandidates(value: string): string[] {
  const key = value.trim().toLowerCase();
  const aliases = AXIS_ALIASES[key] ?? [];
  return key.length > 0 ? [key, ...aliases] : [...aliases];
}

function normalizeDirectionText(text: string): string {
  return text.toLowerCase().replace(/\s+/g, "");
}

function truncate(text: string): string {
  return text.length > HYPOTHESIS_LIMIT ? `${text.slice(0, HYPOTHESIS_LIMIT)}…` : text;
}
