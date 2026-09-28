"""range_pct：日内相对振幅的 10 日均值。

经济假设：``(high - low) / close`` 衡量日内价格波动幅度，10 日均值刻画近期
震荡强度。振幅放大常伴随分歧加剧与情绪升温。
使用字段：high, low, close。
窗口：10。
复权口径：同一交易日内复权因子在比值中约去，未复权价与复权价结果相同。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 平滑窗口（交易日）。
WINDOW: int = 10

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "high", "low", "close")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """日内相对振幅的 10 日均值：``mean((high - low) / close, 10)``。

    输出 ``(date, instrument, value)``，前 9 个交易日为 null；``close`` 非正时
    当日振幅记 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"range_pct 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            pl.when(pl.col("close") > 0)
            .then((pl.col("high") - pl.col("low")) / pl.col("close"))
            .otherwise(None)
            .alias("_range")
        )
        .with_columns(
            pl.col("_range")
            .rolling_mean(WINDOW, min_samples=WINDOW)
            .over("instrument")
            .alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
