"""automl 数据集构造：因子面板 → 单窗口训练集。

职责范围
--------
把行情与一组因子 ``compute`` 的输出拼成「每票每日一行」的宽表，供 AutoML 直接消费：

- 特征列：``factors`` 字典的键，值为该因子在 ``(date, instrument)`` 上的取值。
- 标签列：来自 :func:`quant.labels.attach_label`（T+1 开盘 → T+2 开盘），本模块
  不另造标签。
- 元信息列：``date`` / ``instrument`` / ``delay_days``。

特征预处理口径
--------------
截面 z-score 标准化**按日**进行：同一交易日内对每个特征列做
``(x - mean) / std``，``std`` 为样本标准差（``ddof=1``）。截面只有一只证券或
``std == 0`` 时该值记 null（无可比性）。``±inf`` 一律先转 null，再参与统计。
本模块不做丢弃整列、填充等操作；缺测率由 :func:`missing_rate` 单独统计，交给报告层。

前视约束
--------
因子 ``compute`` 只允许看到 cutoff（T 日收盘）之前的数据，标签由 labels 模块按
T+1 开盘口径生成。本模块只做拼接与标准化，不改变行列的时间语义。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import polars as pl

from quant.labels.open_to_open import DEFAULT_HORIZON, attach_label

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 因子接口：输入行情长表，输出 ``(date, instrument, value)``。
FactorCompute = Callable[[pl.DataFrame], pl.DataFrame]

#: 元信息列名。
DATE_COL: str = "date"
INSTRUMENT_COL: str = "instrument"

#: 标签与标签辅助列名。
LABEL_COL: str = "label"
DELAY_COL: str = "delay_days"

#: 因子输出表的固定列名。
_VALUE_COL: str = "value"

#: 不允许作为特征名的保留列。
_RESERVED_COLUMNS: frozenset[str] = frozenset(
    {DATE_COL, INSTRUMENT_COL, LABEL_COL, DELAY_COL}
)


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------


def build_dataset(
    bars: pl.DataFrame,
    factors: Mapping[str, FactorCompute],
    *,
    horizon: int = DEFAULT_HORIZON,
) -> pl.DataFrame:
    """把行情与因子输出拼成训练宽表。

    参数
    ----
    bars
        日线行情长表，至少含 ``date`` / ``instrument`` / ``open`` / ``adjfactor``。
        额外列会被忽略。
    factors
        ``{特征名: compute}`` 的有序映射（用普通 ``dict`` 即可，Python 3.7+ 保证
        插入序）。每个 ``compute(bars)`` 必须返回 ``(date, instrument, value)``。
    horizon
        传给 :func:`quant.labels.attach_label` 的持有期，默认 1（T+1 开盘 → T+2 开盘）。

    返回
    ----
    ``date, instrument, <特征列...>, label, delay_days`` 的宽表，按
    ``(instrument, date)`` 排序。尾部无未来行情的记录 ``label`` 为 null，**不丢行**。

    异常
    ----
    特征名与保留列冲突、``compute`` 返回列不符时抛 :class:`ValueError`。
    """
    for name in factors:
        if not isinstance(name, str) or not name:
            raise ValueError(f"因子名必须是非空字符串，收到 {name!r}")
        if name in _RESERVED_COLUMNS:
            raise ValueError(f"因子名 {name!r} 与保留列冲突：{sorted(_RESERVED_COLUMNS)}")

    labelled = attach_label(bars, horizon=horizon)

    dataset = labelled.select(DATE_COL, INSTRUMENT_COL, LABEL_COL, DELAY_COL)
    for name, compute in factors.items():
        factor_frame = _compute_factor(name, compute, bars)
        dataset = dataset.join(
            factor_frame, on=[DATE_COL, INSTRUMENT_COL], how="left"
        )

    dataset = _cross_section_zscore(dataset, tuple(factors))
    feature_columns = list(factors)
    return dataset.select(
        DATE_COL, INSTRUMENT_COL, *feature_columns, LABEL_COL, DELAY_COL
    ).sort([INSTRUMENT_COL, DATE_COL])


def _compute_factor(
    name: str, compute: FactorCompute, bars: pl.DataFrame
) -> pl.DataFrame:
    """执行单个因子并把输出规整为 ``(date, instrument, <name>)``。"""
    result = compute(bars)
    missing = [
        col for col in (DATE_COL, INSTRUMENT_COL, _VALUE_COL) if col not in result.columns
    ]
    if missing:
        raise ValueError(f"因子 {name!r} 的输出缺少列：{missing}")
    return result.select(
        DATE_COL,
        INSTRUMENT_COL,
        pl.col(_VALUE_COL).cast(pl.Float64).alias(name),
    )


def _cross_section_zscore(
    dataset: pl.DataFrame, feature_columns: tuple[str, ...]
) -> pl.DataFrame:
    """逐日截面 z-score：``inf → null`` 后按 ``date`` 分组标准化。

    ``std`` 为样本标准差；截面有效证券少于 2 只或 ``std == 0`` 时该值记 null。
    """
    expressions: list[pl.Expr] = []
    for name in feature_columns:
        finite = pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None)
        mean = finite.mean().over(DATE_COL)
        std = finite.std().over(DATE_COL)
        expressions.append(
            pl.when(std.is_not_null() & (std > 0.0))
            .then((finite - mean) / std)
            .otherwise(None)
            .alias(name)
        )
    if not expressions:
        return dataset
    return dataset.with_columns(expressions)


# ---------------------------------------------------------------------------
# 缺测率报告
# ---------------------------------------------------------------------------


def missing_rate(dataset: pl.DataFrame, feature_columns: tuple[str, ...]) -> pl.DataFrame:
    """统计各特征列的缺测率，返回 ``(feature, n, n_missing, missing_rate)``。

    ``missing_rate`` = 缺失行数 / 总行数（``n == 0`` 时记 null）。按输入
    ``feature_columns`` 顺序输出，不丢列，供报告层呈现。
    """
    missing_columns = [col for col in feature_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"待统计的特征列不存在：{missing_columns}")
    n = dataset.height
    expressions = [
        pl.col(name).is_null().sum().cast(pl.Int64).alias(f"{name}__missing")
        for name in feature_columns
    ]
    if not expressions:
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "n": pl.Int64,
                "n_missing": pl.Int64,
                "missing_rate": pl.Float64,
            }
        )
    counts = dataset.select(expressions).row(0, named=True)
    return pl.DataFrame(
        {
            "feature": list(feature_columns),
            "n": [n] * len(feature_columns),
            "n_missing": [int(counts[f"{name}__missing"]) for name in feature_columns],
            "missing_rate": [
                (None if n == 0 else int(counts[f"{name}__missing"]) / n)
                for name in feature_columns
            ],
        }
    )


__all__ = [
    "DATE_COL",
    "DELAY_COL",
    "INSTRUMENT_COL",
    "LABEL_COL",
    "FactorCompute",
    "build_dataset",
    "missing_rate",
]
