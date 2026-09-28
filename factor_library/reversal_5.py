"""reversal_5：5 日反转（开盘口径）。

经济假设：过去 5 日开盘价的累计涨幅在短期内容易被过度反应，随后回落，
故取负的 5 日收益作为反转信号。用开盘价而非收盘价，与 ``mom_5`` 的口径区分开，
避免两个因子完全共线。

使用字段：open, adjfactor。
窗口：5。
复权口径：open × adjfactor（后复权），收益对除权除息不敏感。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 回看窗口（交易日）。
WINDOW: int = 5

#: 本因子依赖的输入列。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "open", "adjfactor")


def compute(data: pl.DataFrame) -> pl.DataFrame:
    """5 日反转：``adj_open(T-5) / adj_open(T) - 1``。

    即 ``-(adj_open(T) / adj_open(T-5) - 1)``，值越大越看多。
    输出 ``(date, instrument, value)``，前 5 个交易日为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"reversal_5 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            (pl.col("open") * pl.col("adjfactor")).alias("_adj_open")
        )
        .with_columns(
            (
                pl.col("_adj_open").shift(WINDOW).over("instrument")
                / pl.col("_adj_open")
                - 1.0
            ).alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
