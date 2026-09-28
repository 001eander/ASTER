"""turnover_proxy：换手活跃度代理。

经济假设：成交额相对自身近期均值放大，代表换手活跃度上升，往往对应信息流
与关注度的变化。A 股真实换手率需要流通股本，本仓库 schema 不含该字段，故用
「当日成交额 / 过去 20 日平均成交额」作为活跃度代理：值大于 1 表示成交额高于
近期常态。成交额类因子无需复权。

使用字段：amount。
窗口：20。
复权口径：成交额不复权。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 基准窗口（交易日）。
WINDOW: int = 20

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "amount")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """换手活跃度代理：``amount(T) / mean(amount, 20)``。

    输出 ``(date, instrument, value)``，前 19 个交易日为 null；均值为 0 或 null
    时为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"turnover_proxy 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            pl.col("amount")
            .rolling_mean(WINDOW, min_samples=WINDOW)
            .over("instrument")
            .alias("_amount_base")
        )
        .with_columns(
            pl.when(pl.col("_amount_base") > 0)
            .then(pl.col("amount") / pl.col("_amount_base"))
            .otherwise(None)
            .alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
