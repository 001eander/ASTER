"""因子库行为相关性查重：新因子与库内 pool 因子的逐日截面相关。

目标
----
新因子即便 IC 达标，若与已在池内的因子高度相关，也只是同一信号的换皮，
不带来增量。这里给出唯一的口径：

- 对每个库内 ``status == "pool"`` 的因子，取「逐日截面 Pearson 相关系数」，
  再对该逐日序列取时间序列均值，得到该库因子的相关系数 ``corr_i``（可正可负）。
- 逐日截面先按 ``(date, instrument)`` inner join 对齐新因子与库因子，只保留双方都有
  值的证券；当日有效证券数低于 ``MIN_CROSS_SECTION_COUNT``（默认 30）时该日相关系数
  记 null 并跳过，全部日子都不足则该库因子相关系数为 None。
- 最终 ``max_corr = max(|corr_i|)``：负相关（-0.9）与正相关同样冗余，一律取绝对值。

相关性口径直接复用指标内核 ``quant.eval.metrics`` 的 :func:`ic_series`
（``method="pearson"``）与 :func:`summarize_ic`，不另造指标语义（AGENTS.md 量化纪律第 2 条）。
库因子取值用 :func:`quant.factor_api.loader.load_factor` 按 registry 的 ``code_path``
动态加载，在同一份输入面板上现算，仓库根为因子库目录的父目录。
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from quant.eval.metrics import (
    DATE_COL,
    FACTOR_COL,
    INSTRUMENT_COL,
    LABEL_COL,
    MIN_CROSS_SECTION_COUNT,
    ic_series,
    summarize_ic,
)
from quant.factor_api.loader import load_factor
from quant.factor_lib.registry import load_registry, pool_factors, registry_path

logger = logging.getLogger(__name__)

#: 因子值长表的列名。
VALUE_COL: str = "value"

#: 相关系数口径：Pearson。
CORR_METHOD: str = "pearson"


# ---------------------------------------------------------------------------
# 值对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrelationReport:
    """新因子对整库的相关性查重结果。

    :param per_factor: ``{库因子 id: 相关系数时间序列均值}``，不可得为 None，保留正负号。
    :param max_corr: ``max(|per_factor 值|)``，库内无可比因子时为 None。
    """

    per_factor: dict[str, float | None]
    max_corr: float | None


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _require_value_columns(df: pl.DataFrame, where: str) -> None:
    """校验值长表含 ``(date, instrument, value)``，缺列抛 ``ValueError``。"""
    missing = [col for col in (DATE_COL, INSTRUMENT_COL, VALUE_COL) if col not in df.columns]
    if missing:
        raise ValueError(f"{where} 缺少必需列：{missing}；实际列：{df.columns}")


def _align(new_values: pl.DataFrame, library_values: pl.DataFrame) -> pl.DataFrame:
    """按 ``(date, instrument)`` inner join，返回 ``(date, instrument, factor, label)``。

    库因子值改名为 ``label`` 只是复用 :func:`ic_series` 的输入契约；语义上是「待查因子」
    与「库因子」两组截面值。
    """
    _require_value_columns(new_values, "新因子值")
    _require_value_columns(library_values, "库因子值")
    return new_values.select(DATE_COL, INSTRUMENT_COL, VALUE_COL).rename(
        {VALUE_COL: FACTOR_COL}
    ).join(
        library_values.select(DATE_COL, INSTRUMENT_COL, VALUE_COL).rename(
            {VALUE_COL: LABEL_COL}
        ),
        on=[DATE_COL, INSTRUMENT_COL],
        how="inner",
    )


# ---------------------------------------------------------------------------
# 相关性计算
# ---------------------------------------------------------------------------


def cross_section_corr(
    new_values: pl.DataFrame,
    library_values: pl.DataFrame,
    *,
    min_count: int = MIN_CROSS_SECTION_COUNT,
) -> float | None:
    """新因子与单个库因子的「逐日截面 Pearson 相关的时间序列均值」。

    :param new_values: 新因子值长表 ``(date, instrument, value)``。
    :param library_values: 库因子值长表，列同 ``new_values``。
    :param min_count: 逐日截面最少有效证券数，不足的日子跳过。
    :return: 相关系数均值；全部日子样本不足（或无可比日期）时为 None。
    """
    sample = _align(new_values, library_values)
    if sample.height == 0:
        return None
    return summarize_ic(ic_series(sample, method=CORR_METHOD, min_count=min_count)).mean


def max_library_corr(
    new_values: pl.DataFrame,
    library_values: Mapping[str, pl.DataFrame],
    *,
    min_count: int = MIN_CROSS_SECTION_COUNT,
) -> CorrelationReport:
    """对整库逐个算 :func:`cross_section_corr`，汇总绝对值最大者。

    库为空（``library_values`` 没有条目）或所有库因子的相关系数都不可得时，
    ``max_corr`` 为 None。
    """
    per_factor = {
        factor_id: cross_section_corr(new_values, values, min_count=min_count)
        for factor_id, values in library_values.items()
    }
    magnitudes = [
        abs(value)
        for value in per_factor.values()
        if isinstance(value, float) and math.isfinite(value)
    ]
    return CorrelationReport(
        per_factor=per_factor,
        max_corr=max(magnitudes) if magnitudes else None,
    )


# ---------------------------------------------------------------------------
# 库因子取值
# ---------------------------------------------------------------------------


def load_library_values(
    factor_library_dir: str | Path,
    data: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """加载库内 pool 因子并在 ``data`` 上现算，返回 ``{factor_id: 值长表}``。

    - ``factor_library_dir`` 下没有 ``registry.json`` 时返回空 dict（跳过查重）。
    - ``code_path`` 相对路径按因子库目录的父目录（仓库根）解析。
    - 单个因子文件缺失、加载失败或计算抛错时跳过该因子并写 warning，不阻断整次评估；
      registry 本身非法仍由 :func:`load_registry` 抛 ``FactorLibError``。
    """
    root = Path(factor_library_dir)
    if not registry_path(root).is_file():
        return {}

    registry = load_registry(root)
    repo_root = root.resolve().parent
    values: dict[str, pl.DataFrame] = {}
    for entry in pool_factors(registry):
        code_path = Path(entry.code_path)
        if not code_path.is_absolute():
            code_path = repo_root / code_path
        if not code_path.is_file():
            logger.warning("库因子 %s 的文件缺失，跳过查重：%s", entry.factor_id, code_path)
            continue
        try:
            output = load_factor(code_path)(data)
            _require_value_columns(output, f"库因子 {entry.factor_id} 输出")
        except Exception as exc:  # noqa: BLE001 - 单个库因子失败不阻断本次评估
            logger.warning("库因子 %s 计算失败，跳过查重：%s", entry.factor_id, exc)
            continue
        values[entry.factor_id] = output.select(DATE_COL, INSTRUMENT_COL, VALUE_COL)
    return values


__all__ = [
    "CORR_METHOD",
    "CorrelationReport",
    "VALUE_COL",
    "cross_section_corr",
    "load_library_values",
    "max_library_corr",
]
