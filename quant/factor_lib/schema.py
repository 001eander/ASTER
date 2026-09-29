"""因子库注册表 schema：字段定义、严格解析与校验。

设计意图
--------
``factor_library/registry.json`` 是因子元数据的唯一真源，取代「扫描目录即因子集」
的隐式约定。每个因子条目同时承载经济假设、评估指标、方向标签与血统链，
使 M3 之后的自动挖掘（变异 / 交叉）、入库准入与 graveyard 淘汰都能在同一份
结构上表达。

registry 形状（单文件）：

.. code-block:: json

    {"version": 1, "factors": [<FactorEntry>, ...]}

解析一律严格：缺键、多键、类型不符、``status`` 非法、``factor_id`` 重复
都抛 :class:`FactorLibError`，避免半成品元数据悄悄进入建模流程。

数据类全部 frozen：registry 以不可变值对象在模块间传递，落盘只经
:mod:`quant.factor_lib.registry` 的原子写入。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# 配置（键名与取值集中在此）
# ---------------------------------------------------------------------------

#: 支持的 registry 版本。结构变更时递增，旧版本加载即报错。
REGISTRY_VERSION: int = 1

#: 因子在池内，参与建模。
STATUS_POOL: str = "pool"

#: 因子被淘汰，留在库里但不再参与建模。
STATUS_GRAVEYARD: str = "graveyard"

#: 合法 ``status`` 取值。
VALID_STATUSES: frozenset[str] = frozenset({STATUS_POOL, STATUS_GRAVEYARD})

#: registry 顶层键（恰好这些，多一个都不允许）。
REGISTRY_KEYS: tuple[str, ...] = ("version", "factors")

#: 因子条目键（恰好这些）。
ENTRY_KEYS: tuple[str, ...] = (
    "factor_id",
    "hypothesis",
    "code_path",
    "metrics",
    "direction",
    "lineage",
    "status",
)

#: ``direction`` 的键（恰好这些）。
DIRECTION_KEYS: tuple[str, ...] = ("signal_source", "time_scale", "mechanism")

#: ``lineage`` 的键（恰好这些）。
LINEAGE_KEYS: tuple[str, ...] = ("op", "parents", "run_id", "generation")

#: ``lineage.op`` 的合法取值；新增演化操作时在此登记。
KNOWN_LINEAGE_OPS: tuple[str, ...] = ("seed", "mutation", "crossover")


class FactorLibError(ValueError):
    """registry 缺失、格式非法或违反约束。"""


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------


def _as_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorLibError(f"{where} 应为对象，实际 {type(value).__name__}")
    return value


def _check_keys(
    mapping: Mapping[str, Any], required: Sequence[str], *, where: str
) -> None:
    """要求键集合与 ``required`` 完全一致：缺键或多键都报错。"""
    keys = set(mapping)
    expected = set(required)
    missing = [key for key in required if key not in keys]
    if missing:
        raise FactorLibError(f"{where} 缺少字段：{missing}")
    extra = sorted(keys - expected)
    if extra:
        raise FactorLibError(f"{where} 含未知字段：{extra}")


def _require_str(mapping: Mapping[str, Any], key: str, *, where: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value:
        raise FactorLibError(
            f"{where} 字段 {key!r} 应为非空字符串，实际 {type(value).__name__}"
        )
    return value


def _require_int(mapping: Mapping[str, Any], key: str, *, where: str) -> int:
    value = mapping[key]
    # bool 是 int 的子类，需显式排除。
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactorLibError(
            f"{where} 字段 {key!r} 应为整数，实际 {type(value).__name__}"
        )
    return value


def _optional_str(mapping: Mapping[str, Any], key: str, *, where: str) -> str | None:
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise FactorLibError(
            f"{where} 字段 {key!r} 应为非空字符串或 null，实际 {type(value).__name__}"
        )
    return value


# ---------------------------------------------------------------------------
# 值对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Direction:
    """因子方向标签：信号来源、时间尺度、经济机制。"""

    signal_source: str
    time_scale: str
    mechanism: str

    @classmethod
    def from_dict(cls, raw: Any, *, where: str = "direction") -> Direction:
        mapping = _as_mapping(raw, where=where)
        _check_keys(mapping, DIRECTION_KEYS, where=where)
        return cls(
            signal_source=_require_str(mapping, "signal_source", where=where),
            time_scale=_require_str(mapping, "time_scale", where=where),
            mechanism=_require_str(mapping, "mechanism", where=where),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "signal_source": self.signal_source,
            "time_scale": self.time_scale,
            "mechanism": self.mechanism,
        }


@dataclass(frozen=True, slots=True)
class Lineage:
    """血统链：本因子由哪次 run 的什么操作、从哪些父因子演化而来。"""

    op: str
    parents: tuple[str, ...]
    run_id: str | None
    generation: int

    @classmethod
    def from_dict(cls, raw: Any, *, where: str = "lineage") -> Lineage:
        mapping = _as_mapping(raw, where=where)
        _check_keys(mapping, LINEAGE_KEYS, where=where)

        parents_raw = mapping["parents"]
        if not isinstance(parents_raw, list):
            raise FactorLibError(
                f"{where} 字段 'parents' 应为列表，实际 {type(parents_raw).__name__}"
            )
        parents: list[str] = []
        for index, parent in enumerate(parents_raw):
            if not isinstance(parent, str) or not parent:
                raise FactorLibError(
                    f"{where} 字段 'parents' 第 {index} 项应为非空字符串，"
                    f"实际 {type(parent).__name__}"
                )
            parents.append(parent)

        generation = _require_int(mapping, "generation", where=where)
        if generation < 0:
            raise FactorLibError(f"{where} 字段 'generation' 不应为负：{generation}")

        op = _require_str(mapping, "op", where=where)
        if op not in KNOWN_LINEAGE_OPS:
            raise FactorLibError(
                f"{where} 字段 'op' 取值未知：{op!r}，已知 {list(KNOWN_LINEAGE_OPS)}"
            )
        return cls(
            op=op,
            parents=tuple(parents),
            run_id=_optional_str(mapping, "run_id", where=where),
            generation=generation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "parents": list(self.parents),
            "run_id": self.run_id,
            "generation": self.generation,
        }


def _parse_metrics(raw: Any, *, where: str) -> dict[str, float | None]:
    """``metrics`` 允许任意键（rank_ic / icir / max_corr 等），值为数值或 null。"""
    mapping = _as_mapping(raw, where=where)
    parsed: dict[str, float | None] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise FactorLibError(f"{where} 的指标键应为非空字符串：{key!r}")
        if value is None:
            parsed[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FactorLibError(
                f"{where} 指标 {key!r} 应为数值或 null，实际 {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise FactorLibError(f"{where} 指标 {key!r} 应为有限数值，实际 {value!r}")
        parsed[key] = number
    return parsed


@dataclass(frozen=True, slots=True)
class FactorEntry:
    """单个因子在注册表中的元数据条目。"""

    factor_id: str
    hypothesis: str
    code_path: str
    metrics: dict[str, float | None]
    direction: Direction
    lineage: Lineage
    status: str

    @classmethod
    def from_dict(cls, raw: Any, *, where: str = "因子条目") -> FactorEntry:
        mapping = _as_mapping(raw, where=where)
        _check_keys(mapping, ENTRY_KEYS, where=where)

        status = _require_str(mapping, "status", where=where)
        if status not in VALID_STATUSES:
            raise FactorLibError(
                f"{where} 字段 'status' 非法：{status!r}，"
                f"可选 {sorted(VALID_STATUSES)}"
            )

        return cls(
            factor_id=_require_str(mapping, "factor_id", where=where),
            hypothesis=_require_str(mapping, "hypothesis", where=where),
            code_path=_require_str(mapping, "code_path", where=where),
            metrics=_parse_metrics(mapping["metrics"], where=f"{where}.metrics"),
            direction=Direction.from_dict(
                mapping["direction"], where=f"{where}.direction"
            ),
            lineage=Lineage.from_dict(mapping["lineage"], where=f"{where}.lineage"),
            status=status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "hypothesis": self.hypothesis,
            "code_path": self.code_path,
            "metrics": dict(self.metrics),
            "direction": self.direction.to_dict(),
            "lineage": self.lineage.to_dict(),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class Registry:
    """整份因子注册表。"""

    version: int
    factors: tuple[FactorEntry, ...]

    @classmethod
    def from_dict(cls, raw: Any, *, where: str = "registry") -> Registry:
        mapping = _as_mapping(raw, where=where)
        _check_keys(mapping, REGISTRY_KEYS, where=where)

        version = _require_int(mapping, "version", where=where)
        if version != REGISTRY_VERSION:
            raise FactorLibError(
                f"{where} 版本不支持：{version}，当前支持 {REGISTRY_VERSION}"
            )

        factors_raw = mapping["factors"]
        if not isinstance(factors_raw, list):
            raise FactorLibError(
                f"{where} 字段 'factors' 应为列表，实际 {type(factors_raw).__name__}"
            )

        entries: list[FactorEntry] = []
        seen: set[str] = set()
        for index, item in enumerate(factors_raw):
            entry = FactorEntry.from_dict(item, where=f"{where}.factors[{index}]")
            if entry.factor_id in seen:
                raise FactorLibError(f"{where} 中 factor_id 重复：{entry.factor_id}")
            seen.add(entry.factor_id)
            entries.append(entry)
        return cls(version=version, factors=tuple(entries))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "factors": [entry.to_dict() for entry in self.factors],
        }

    def get(self, factor_id: str) -> FactorEntry | None:
        """按 ``factor_id`` 取条目，不存在返回 None。"""
        for entry in self.factors:
            if entry.factor_id == factor_id:
                return entry
        return None


__all__ = [
    "DIRECTION_KEYS",
    "ENTRY_KEYS",
    "KNOWN_LINEAGE_OPS",
    "LINEAGE_KEYS",
    "REGISTRY_KEYS",
    "REGISTRY_VERSION",
    "STATUS_GRAVEYARD",
    "STATUS_POOL",
    "VALID_STATUSES",
    "Direction",
    "FactorEntry",
    "FactorLibError",
    "Lineage",
    "Registry",
]
