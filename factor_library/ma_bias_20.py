"""ma_bias_20：20 日均线乖离。

经济假设：复权收盘价相对 20 日均线的偏离度衡量中期超买超卖，偏离过大时有向
均线回归的倾向。
使用字段：close, adjfactor。
窗口：20。
复权口径：close × adjfactor（后复权），均线与价格同口径，对除权除息不敏感。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 均线窗口（交易日）。
WINDOW: int = 20

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "adjfactor")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """20 日均线乖离：``adj_close / mean(adj_close, 20) - 1``。

    输出 ``(date, instrument, value)``，前 19 个交易日为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"ma_bias_20 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            (pl.col("close") * pl.col("adjfactor")).alias("_adj_close")
        )
        .with_columns(
            (
                pl.col("_adj_close")
                / pl.col("_adj_close")
                .rolling_mean(WINDOW, min_samples=WINDOW)
                .over("instrument")
                - 1.0
            ).alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
