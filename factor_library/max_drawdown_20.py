"""max_drawdown_20：20 日高点回撤。

经济假设：当前复权收盘价相对过去 20 日高点的回撤幅度，衡量近期下行风险与
套牢盘压力。回撤越深（值越负）越看空。
使用字段：close, adjfactor。
窗口：20。
复权口径：close × adjfactor（后复权），回撤对除权除息不敏感。
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
    """20 日高点回撤：``adj_close(T) / max(adj_close, 20) - 1``。

    即窗口内最高点到当日的最小收益，值域 ``[-1, 0]``，0 表示处于窗口新高。
    输出 ``(date, instrument, value)``，前 19 个交易日为 null。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"max_drawdown_20 缺少输入列：{missing}")
    ordered = data.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    return (
        ordered.with_columns(
            (pl.col("close") * pl.col("adjfactor")).alias("_adj_close")
        )
        .with_columns(
            (
                pl.col("_adj_close")
                / pl.col("_adj_close")
                .rolling_max(WINDOW, min_samples=WINDOW)
                .over("instrument")
                - 1.0
            ).alias("value")
        )
        .select("date", "instrument", pl.col("value").cast(pl.Float64))
    )
