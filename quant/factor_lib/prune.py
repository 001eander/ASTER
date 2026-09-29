"""因子库定期整库：冗余聚簇降级与 pool 容量上限。

目标
----
自动挖掘会让注册表只增不减：同一信号的变体（同簇冗余）与低质量尾部因子都会
稀释建模数据集。这里给出唯一的口径，把「池内留谁」从隐含约定变成可复算的规则：

1. **聚簇降级**：对库内全部 pool 因子两两算逐日截面相关的时间序列均值
   （:func:`quant.factor_lib.correlation.cross_section_corr`），``|corr|``
   严格大于 :func:`cluster_corr_threshold`（复用入库查重阈值
   :data:`quant.eval.factor.MAX_CORR_REJECT`，不另造数值）的因子连边，
   连通分量为一个簇。每个簇只留质量最高的一个，其余降级进 graveyard。
   负相关同样入簇（绝对值口径），与入库查重的语义一致。
2. **容量上限**：pool 因子数不得超过库内总条目数（pool 加 graveyard）的
   :data:`CAPACITY_RATIO`（50%）。聚簇降级后仍超限时，按同一质量排序把最弱的
   继续降级直到达标。graveyard 条目留在库里、保留血统，但不占 pool 名额，
   只计入总条目数。

质量排序在 :func:`quality_key` 中集中定义：先比 ``metrics.rank_ic``（降序，
None 视为最弱排在最后），并列比 ``metrics.icir``（同样 None 最后），再并列比
``factor_id`` 字典序。排序键唯一，同一输入两次跑出的计划完全一致。

容量上限的边界
--------------
``floor(total_entries × 0.5)`` 在库内总条目数不足 2 时为 0，此时按规则 pool 会被
清空。这是规则的固有边界，不做例外：脚本默认干跑提示，真写盘前请先看计划。
:func:`capacity_limit` 单独暴露，便于调用方判断该边界。

幂等
----
:func:`plan_prune` 只看「当前 pool」与「库内已有 graveyard」：整库后的 registry
（每个簇只剩一个、pool 已达标）再跑计划为空，所以脚本可以反复执行。

落盘由调用方决定
----------------
:func:`plan_prune` 与 :func:`apply_prune` 都是纯函数，不碰 IO；写回走
:mod:`quant.factor_lib.registry` 的原子写。CLI 见 ``scripts/prune_factor_library.py``，
按需手动运行，不挂在 daily pipeline 上。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import polars as pl

from quant.factor_lib.correlation import cross_section_corr
from quant.factor_lib.registry import pool_factors
from quant.factor_lib.schema import (
    STATUS_GRAVEYARD,
    STATUS_POOL,
    FactorEntry,
    FactorLibError,
    Registry,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 容量上限比例：pool 因子数 ≤ ``floor(库内总条目数 × 此值)``。
CAPACITY_RATIO: float = 0.5

#: 降级原因：与库内其他 pool 因子同簇（同簇只留质量最高的一个）。
REASON_CLUSTER: str = "cluster"

#: 降级原因：pool 数超出容量上限。
REASON_CAPACITY: str = "capacity"

#: 质量排序用的指标键，取自 registry ``metrics`` 的既有键名。
RANK_IC_KEY: str = "rank_ic"

#: 质量排序的次级指标键。
ICIR_KEY: str = "icir"


def cluster_corr_threshold() -> float:
    """聚簇相关性阈值：``|corr|`` 严格大于此值即连边，与入库查重共用同一口径。

    阈值数值只在 :data:`quant.eval.factor.MAX_CORR_REJECT` 定义一次。这里惰性
    import，避免 ``quant.factor_lib.prune`` 在模块导入期反向依赖
    ``quant.eval.factor``（后者又经 ``quant.factor_lib`` 触达本模块）造成的循环导入。
    """
    from quant.eval.factor import MAX_CORR_REJECT

    return MAX_CORR_REJECT


# ---------------------------------------------------------------------------
# 值对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PruneDemotion:
    """单条降级记录。

    :param factor_id: 被降级进 graveyard 的 pool 因子。
    :param reason: :data:`REASON_CLUSTER` 或 :data:`REASON_CAPACITY`。
    :param kept: 聚簇降级时同簇保留下来的因子 id；容量降级为 None。
    """

    factor_id: str
    reason: str
    kept: str | None


@dataclass(frozen=True, slots=True)
class PrunePlan:
    """一次整库的完整计划，尚未落盘。

    :param demotions: 降级记录，先聚簇（按簇的最小 factor_id 排序）后容量。
    :param pool_before: 计划前的 pool 因子数。
    :param capacity_limit: 容量上限 ``floor(总条目数 × CAPACITY_RATIO)``。
    """

    demotions: tuple[PruneDemotion, ...]
    pool_before: int
    capacity_limit: int

    @property
    def is_empty(self) -> bool:
        """计划为空表示无需整库。"""
        return not self.demotions

    @property
    def pool_after(self) -> int:
        """计划执行后的 pool 因子数。"""
        return self.pool_before - len(self.demotions)

    def demoted_ids(self) -> frozenset[str]:
        """计划降级的 factor_id 集合。"""
        return frozenset(demotion.factor_id for demotion in self.demotions)


@dataclass(frozen=True, slots=True)
class PruneResult:
    """整库结果：执行后的新 registry 与所用计划。

    :param registry: 应用计划后的新 :class:`Registry`；计划为空时与原对象相同。
    :param plan: 本次使用的计划。
    :param changed: registry 是否真的发生了变化（计划非空但他项已全在 graveyard 时为 False）。
    """

    registry: Registry
    plan: PrunePlan
    changed: bool


# ---------------------------------------------------------------------------
# 质量排序
# ---------------------------------------------------------------------------


def _metric(entry: FactorEntry, key: str) -> float | None:
    """取条目 ``metrics[key]``，非有限数值一律按不可得（None）。"""
    value = entry.metrics.get(key)
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def quality_key(entry: FactorEntry) -> tuple[int, float, int, float, str]:
    """质量排序键，升序即「从强到弱」。

    ``rank_ic`` 降序、None 排最后；并列比 ``icir`` 降序、None 排最后；再并列比
    ``factor_id`` 字典序。键唯一，保证计划确定性。
    """
    rank_ic = _metric(entry, RANK_IC_KEY)
    icir = _metric(entry, ICIR_KEY)
    return (
        0 if rank_ic is not None else 1,
        -(rank_ic if rank_ic is not None else 0.0),
        0 if icir is not None else 1,
        -(icir if icir is not None else 0.0),
        entry.factor_id,
    )


def capacity_limit(total_entries: int) -> int:
    """容量上限 ``floor(total_entries × CAPACITY_RATIO)``。"""
    return math.floor(total_entries * CAPACITY_RATIO)


# ---------------------------------------------------------------------------
# 相关矩阵与聚簇
# ---------------------------------------------------------------------------


def build_corr_matrix(values: Mapping[str, pl.DataFrame]) -> dict[tuple[str, str], float]:
    """pool 因子两两相关矩阵，只存上三角。

    :param values: ``{factor_id: 值长表}``，长表列为 ``(date, instrument, value)``。
    :return: ``{(较小 id, 较大 id): 相关均值}``；任一侧不可得（None）或非有限时跳过该对。
    """
    ids = sorted(values)
    matrix: dict[tuple[str, str], float] = {}
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            corr = cross_section_corr(values[left], values[right])
            if corr is None or not math.isfinite(corr):
                continue
            matrix[(left, right)] = corr
    return matrix


def _cluster_members(
    factor_ids: list[str],
    matrix: Mapping[tuple[str, str], float],
    *,
    threshold: float | None = None,
) -> list[tuple[str, ...]]:
    """按 ``|corr| > threshold`` 连边取连通分量，返回按最小 id 排序的簇列表。

    ``threshold`` 为 None 时取 :func:`cluster_corr_threshold`。
    """
    if threshold is None:
        threshold = cluster_corr_threshold()
    parent: dict[str, str] = {factor_id: factor_id for factor_id in factor_ids}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # 路径压缩
            parent[node], node = root, parent[node]
        return root

    for (left, right), corr in matrix.items():
        if left not in parent or right not in parent or abs(corr) <= threshold:
            continue
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # 固定以字典序较小者为根，使结果与遍历顺序无关。
            low, high = sorted((left_root, right_root))
            parent[high] = low

    grouped: dict[str, list[str]] = {}
    for factor_id in factor_ids:
        grouped.setdefault(find(factor_id), []).append(factor_id)
    return [tuple(sorted(members)) for _, members in sorted(grouped.items())]


# ---------------------------------------------------------------------------
# 计划与执行
# ---------------------------------------------------------------------------


def plan_prune(
    registry: Registry,
    values: Mapping[str, pl.DataFrame],
) -> PrunePlan:
    """由注册表与 pool 因子取值算出整库计划（纯函数，不碰 IO）。

    ``values`` 里缺失的 pool 因子（取值加载失败等）不参与聚簇，但仍算 pool 名额，
    也会被容量规则降级——口径上「算不出相关性」不构成留池理由。graveyard 条目
    不参与聚簇，也不占 pool 名额，只计入总条目数。
    """
    pool = pool_factors(registry)
    pool_ids = {entry.factor_id for entry in pool}
    affordable = sorted(factor_id for factor_id in values if factor_id in pool_ids)
    matrix = build_corr_matrix(
        {factor_id: values[factor_id] for factor_id in affordable}
    )

    by_id = {entry.factor_id: entry for entry in pool}
    demotions: list[PruneDemotion] = []
    demoted: set[str] = set()

    for cluster in _cluster_members(affordable, matrix):
        if len(cluster) < 2:
            continue
        ranked = sorted((by_id[factor_id] for factor_id in cluster), key=quality_key)
        kept = ranked[0].factor_id
        for entry in ranked[1:]:
            demotions.append(PruneDemotion(entry.factor_id, REASON_CLUSTER, kept))
            demoted.add(entry.factor_id)

    limit = capacity_limit(len(registry.factors))
    remaining = [entry for entry in pool if entry.factor_id not in demoted]
    if len(remaining) > limit:
        for entry in sorted(remaining, key=quality_key)[limit:]:
            demotions.append(PruneDemotion(entry.factor_id, REASON_CAPACITY, None))

    return PrunePlan(
        demotions=tuple(demotions),
        pool_before=len(pool),
        capacity_limit=limit,
    )


def apply_prune(registry: Registry, plan: PrunePlan) -> Registry:
    """把计划作用到 registry，返回新的 :class:`Registry`（旧对象不变）。

    计划里的 ``factor_id`` 不在 registry 中时抛 :class:`FactorLibError`；条目本就
    是 graveyard 时原样保留（幂等）。计划为空时返回原对象。
    """
    if plan.is_empty:
        return registry
    targets = plan.demoted_ids()
    unknown = sorted(targets - {entry.factor_id for entry in registry.factors})
    if unknown:
        raise FactorLibError(f"整库计划含未知 factor_id：{unknown}")
    factors = tuple(
        replace(entry, status=STATUS_GRAVEYARD)
        if entry.factor_id in targets and entry.status == STATUS_POOL
        else entry
        for entry in registry.factors
    )
    return Registry(version=registry.version, factors=factors)


def prune(registry: Registry, values: Mapping[str, pl.DataFrame]) -> PruneResult:
    """跑一遍「计划 + 执行」，返回 :class:`PruneResult`（不落盘）。"""
    plan = plan_prune(registry, values)
    updated = apply_prune(registry, plan)
    return PruneResult(registry=updated, plan=plan, changed=updated != registry)


__all__ = [
    "CAPACITY_RATIO",
    "ICIR_KEY",
    "RANK_IC_KEY",
    "REASON_CAPACITY",
    "REASON_CLUSTER",
    "PruneDemotion",
    "PrunePlan",
    "PruneResult",
    "apply_prune",
    "build_corr_matrix",
    "capacity_limit",
    "cluster_corr_threshold",
    "plan_prune",
    "prune",
    "quality_key",
]
