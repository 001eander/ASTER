"""mom_20：20 日动量。

经济假设：过去 20 个交易日的中期复权收盘涨幅有延续性（趋势效应）。
使用字段：close, adjfactor。
窗口：20。
复权口径：close × adjfactor（后复权），收益对除权除息不敏感。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 回看窗口（交易日）。
WINDOW: int = 20

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "adjfactor")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """20 日动量：``adj_close(T) / adj_close(T-20) - 1``。

    输出 ``(date, instrument, value)``，前 20 个交易日为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"mom_20 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            (pl.col("close") * pl.col("adjfactor")).alias("_adj_close")
        )
        .with_columns(
            (
                pl.col("_adj_close")
                / pl.col("_adj_close").shift(WINDOW).over("instrument")
                - 1.0
            ).alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
